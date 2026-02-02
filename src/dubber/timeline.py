"""
Audio timeline building for different alignment modes.
"""

import logging
import os
from collections.abc import Callable

from pydub import AudioSegment
from tqdm import tqdm

from .io_ffmpeg import ensure_dir, time_stretch_wav_ffmpeg
from .models import Block, Segment
from .tts import _hash_for_cache

logger = logging.getLogger("dubber")


def build_timeline_wav(
    segments: list[Segment],
    tmp_dir: str,
    voice_tag: str,
    synth_func: Callable[[str, str], None],
    sample_rate: int = 24000,
    strict_timing: bool = False,
    tolerance_ms: int = 30,
    fit_mode: str = "pad-or-speedup",
    cache_sig: tuple[str, str, str] | None = None,
    no_tts: bool = False,
) -> AudioSegment:
    """Build timeline per individual segments."""
    ensure_dir(tmp_dir)
    timeline = AudioSegment.silent(duration=0)
    cursor_ms = 0
    failures: list[int] = []
    provider, model, voice = cache_sig or ("prov", "model", "voice")

    def find_cached(tmp_dir: str, voice_tag: str, i: int, sig: str) -> str | None:
        cands = [
            os.path.join(tmp_dir, f"{voice_tag}_{i:04d}_{sig}.wav"),
            os.path.join(tmp_dir, f"{voice_tag}_seg_{i:04d}.wav"),
            os.path.join(tmp_dir, f"seg_{i:04d}.wav"),
        ]
        for p in cands:
            if os.path.exists(p):
                try:
                    if len(AudioSegment.from_wav(p)) > 50:
                        return p
                except Exception:
                    pass
        return None

    missing_cache: list[int] = []

    for i, seg in enumerate(tqdm(segments, desc=f"TTS {voice_tag}")):
        sig = _hash_for_cache(provider, model, voice, seg.text)
        seg_wav = os.path.join(tmp_dir, f"{voice_tag}_{i:04d}_{sig}.wav")

        cached = find_cached(tmp_dir, voice_tag, i, sig)
        if cached and cached != seg_wav:
            seg_wav = cached
        elif not cached and no_tts:
            missing_cache.append(i)
            AudioSegment.silent(duration=int((seg.end - seg.start) * 1000)).export(
                seg_wav, format="wav"
            )
        elif not cached:
            try:
                synth_func(seg.text, seg_wav)
            except Exception as e:
                logger.error(f"TTS failed for segment {i}: {e}")
                failures.append(i)
                AudioSegment.silent(duration=int((seg.end - seg.start) * 1000)).export(
                    seg_wav, format="wav"
                )
        try:
            clip = AudioSegment.from_wav(seg_wav).set_frame_rate(sample_rate)
        except Exception as e:
            logger.error(f"Could not read synthesized clip for segment {i}: {e}")
            clip = AudioSegment.silent(duration=int((seg.end - seg.start) * 1000))

        tgt_ms = int((seg.end - seg.start) * 1000)
        if strict_timing and tgt_ms > 0:
            diff = len(clip) - tgt_ms
            if fit_mode == "pad-or-speedup":
                if diff > tolerance_ms:
                    ratio = len(clip) / max(tgt_ms, 1)
                    stretched = seg_wav.replace(".wav", "_fit.wav")
                    time_stretch_wav_ffmpeg(seg_wav, stretched, ratio=ratio)
                    clip = AudioSegment.from_wav(stretched).set_frame_rate(sample_rate)
                elif diff < -tolerance_ms:
                    clip += AudioSegment.silent(duration=(tgt_ms - len(clip)))
            else:
                ratio = len(clip) / max(tgt_ms, 1)
                if abs(diff) > tolerance_ms and ratio != 1.0:
                    stretched = seg_wav.replace(".wav", "_fit.wav")
                    time_stretch_wav_ffmpeg(seg_wav, stretched, ratio=ratio)
                    clip = AudioSegment.from_wav(stretched).set_frame_rate(sample_rate)

        start_ms = int(seg.start * 1000)
        if start_ms > cursor_ms:
            timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
            cursor_ms = start_ms
        timeline += clip
        cursor_ms += len(clip)

    if failures:
        logger.warning(
            f"TTS completed with {len(failures)} failed segments (rendered as silence): {failures}"
        )
    if no_tts and missing_cache:
        raise RuntimeError(
            "--no-tts was set, but cached clips were missing for segments: "
            + ", ".join(map(str, missing_cache))
            + "\nHint: drop --no-tts (to synthesize), or ensure previous run's tmp clips exist in workdir/tmp/main."
        )

    return timeline.set_frame_rate(sample_rate).set_channels(1)


def build_timeline_sentences(
    segments: list[Segment],
    sentence_groups: list[list[int]],
    tmp_dir: str,
    voice_tag: str,
    synth_func: Callable[[str, str], None],
    sample_rate: int = 24000,
    cache_sig: tuple[str, str, str] | None = None,
) -> AudioSegment:
    """Build timeline per sentence groups."""
    ensure_dir(tmp_dir)
    timeline = AudioSegment.silent(duration=0)
    cursor_ms = 0
    provider, model, voice = cache_sig or ("prov", "model", "voice")

    def cache_path(i: int, text: str) -> str:
        sig = _hash_for_cache(provider, model, voice, text)
        return os.path.join(tmp_dir, f"{voice_tag}_sent_{i:03d}_{sig}.wav")

    for gi, idxs in enumerate(tqdm(sentence_groups, desc=f"TTS {voice_tag} sentences")):
        text = " ".join((segments[k].text or "").strip() for k in idxs)
        text = " ".join(text.split())
        if not text:
            # just silence the whole window interval
            for k in idxs:
                start_ms = int(segments[k].start * 1000)
                end_ms = int(segments[k].end * 1000)
                if start_ms > cursor_ms:
                    timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
                    cursor_ms = start_ms
                timeline += AudioSegment.silent(duration=max(0, end_ms - start_ms))
                cursor_ms = end_ms
            continue

        out_wav = cache_path(gi, text)
        if not os.path.exists(out_wav):
            synth_func(text, out_wav)
        sent_audio = AudioSegment.from_wav(out_wav).set_frame_rate(sample_rate)
        pos_ms = 0

        for k in idxs:
            start_ms = int(segments[k].start * 1000)
            end_ms = int(segments[k].end * 1000)
            win_len = max(0, end_ms - start_ms)

            if start_ms > cursor_ms:
                timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
                cursor_ms = start_ms

            remain = max(0, len(sent_audio) - pos_ms)
            if remain <= 0:
                timeline += AudioSegment.silent(duration=win_len)
                cursor_ms += win_len
                continue

            put = min(win_len, remain)
            timeline += sent_audio[pos_ms : pos_ms + put]
            cursor_ms += put

            if put < win_len:
                timeline += AudioSegment.silent(duration=win_len - put)
                cursor_ms += win_len - put

            pos_ms += put

        if pos_ms < len(sent_audio):
            overflow = len(sent_audio) - pos_ms
            logger.warning(f"sentence {gi} overflow {overflow}ms trimmed")

    return timeline.set_frame_rate(sample_rate).set_channels(1)


def build_timeline_blocks(
    blocks: list[Block],
    tmp_dir: str,
    voice_tag: str,
    synth_func: Callable[[str, str], None],
    sample_rate: int = 24000,
    min_atempo: float = 0.90,
    max_atempo: float = 1.10,
    breath_pad_ms: int = 200,
    no_speedup: bool = False,
    cache_sig: tuple[str, str, str] | None = None,
) -> AudioSegment:
    """Build timeline by blocks (1 block → 1 TTS), limited atempo for naturalness."""
    ensure_dir(tmp_dir)
    timeline = AudioSegment.silent(duration=0)
    cursor_ms = 0
    provider, model, voice = cache_sig or ("prov", "model", "voice")

    def cached_block_path(i: int, text: str) -> str:
        sig = _hash_for_cache(provider, model, voice, text)
        return os.path.join(tmp_dir, f"{voice_tag}_blk_{i:03d}_{sig}.wav")

    for i, b in enumerate(tqdm(blocks, desc=f"TTS {voice_tag} blocks")):
        out_wav = cached_block_path(i, b.text)
        if not os.path.exists(out_wav) or len(AudioSegment.from_wav(out_wav)) <= 50:
            synth_func(b.text, out_wav)
        clip = AudioSegment.from_wav(out_wav).set_frame_rate(sample_rate)

        blk_ms = int((b.end - b.start) * 1000)
        tgt_ms = max(50, blk_ms - max(0, breath_pad_ms))
        if tgt_ms <= 0:
            tgt_ms = blk_ms

        ratio = len(clip) / max(tgt_ms, 1)
        clamped = min(max(ratio, min_atempo), max_atempo)

        stretched_path = out_wav.replace(".wav", "_fit.wav")
        if ratio > 1.0 and no_speedup:
            fitted = clip
        elif abs(clamped - 1.0) > 0.02:
            time_stretch_wav_ffmpeg(out_wav, stretched_path, ratio=clamped)
            fitted = AudioSegment.from_wav(stretched_path).set_frame_rate(sample_rate)
        else:
            fitted = clip

        if len(fitted) < tgt_ms:
            fitted = fitted + AudioSegment.silent(duration=(tgt_ms - len(fitted)))
        elif len(fitted) > tgt_ms + 20:
            fitted = fitted[:tgt_ms]

        start_ms = int(b.start * 1000)
        if start_ms > cursor_ms:
            timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
            cursor_ms = start_ms
        timeline += fitted
        cursor_ms += len(fitted)

        end_ms = int(b.end * 1000)
        if end_ms > cursor_ms:
            timeline += AudioSegment.silent(duration=end_ms - cursor_ms)
            cursor_ms = end_ms

    return timeline.set_frame_rate(sample_rate).set_channels(1)
