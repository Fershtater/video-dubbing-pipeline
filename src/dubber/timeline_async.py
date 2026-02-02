"""
Asynchronous audio timeline building for different alignment modes.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from typing import List

from pydub import AudioSegment
from tqdm.asyncio import tqdm

from .io_ffmpeg import ensure_dir, time_stretch_wav_ffmpeg
from .models import Segment
from .tts import _hash_for_cache


logger = logging.getLogger("dubber")


async def build_timeline_sentences_async(
    segments: list[Segment],
    sentence_groups: list[list[int]],
    tmp_dir: str,
    voice_tag: str,
    synth_func_async: Callable[[List[str], List[str]], List[str]],
    sample_rate: int = 24000,
    cache_sig: tuple[str, str, str] | None = None,
    no_tts: bool = False,
) -> AudioSegment:
    """Build timeline using sentence groups with async TTS processing."""
    ensure_dir(tmp_dir)

    if cache_sig:
        provider, model, voice = cache_sig
    else:
        provider, model, voice = "unknown", "unknown", "unknown"

    # Prepare all texts and output paths for batch processing
    texts = []
    output_paths = []
    segment_indices = []

    for gi, idxs in enumerate(sentence_groups):
        if not idxs:
            continue

        # Combine all segments in this sentence group
        combined_text = " ".join(segments[i].text for i in idxs if i < len(segments))
        if not combined_text.strip():
            continue

        sig = _hash_for_cache(provider, model, voice, combined_text)
        out_path = os.path.join(tmp_dir, f"{voice_tag}_sent_{gi:04d}_{sig}.wav")

        texts.append(combined_text)
        output_paths.append(out_path)
        segment_indices.append(idxs)

    # Process all TTS requests asynchronously
    if not no_tts and texts:
        logger.info(f"Processing {len(texts)} sentence groups asynchronously...")
        # Process in batches to avoid overwhelming the API
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_paths = output_paths[i:i+batch_size]

            # Process batch
            tasks = [
                synth_func_async([text], [out_path])
                for text, out_path in zip(batch_texts, batch_paths)
            ]
            await tqdm.gather(*tasks, desc=f"TTS {voice_tag} sentences (async)")

    # Build the final audio timeline with proper synchronization
    timeline = AudioSegment.silent(duration=0)
    cursor_ms = 0

    for gi, idxs in enumerate(sentence_groups):
        if not idxs:
            continue

        # Handle empty text segments with silence
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

        # Find the corresponding output file
        if gi < len(output_paths):
            out_path = output_paths[gi]
            if os.path.exists(out_path):
                try:
                    sent_audio = AudioSegment.from_wav(out_path).set_frame_rate(sample_rate)
                    pos_ms = 0

                    # Distribute audio across segments in this group
                    for k in idxs:
                        start_ms = int(segments[k].start * 1000)
                        end_ms = int(segments[k].end * 1000)
                        win_len = max(0, end_ms - start_ms)

                        # Add silence if needed before this segment
                        if start_ms > cursor_ms:
                            timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
                            cursor_ms = start_ms

                        # Place as much audio as fits in this window
                        remain = max(0, len(sent_audio) - pos_ms)
                        if remain <= 0:
                            timeline += AudioSegment.silent(duration=win_len)
                            cursor_ms += win_len
                            continue

                        put = min(remain, win_len)
                        timeline += sent_audio[pos_ms:pos_ms + put]
                        cursor_ms += put

                        # Add silence if we couldn't fill the window
                        if put < win_len:
                            timeline += AudioSegment.silent(duration=win_len - put)
                            cursor_ms += win_len - put

                        pos_ms += put

                except Exception as e:
                    logger.warning(f"Failed to load audio segment {out_path}: {e}")
                    # Add silence for all segments in this group
                    for k in idxs:
                        start_ms = int(segments[k].start * 1000)
                        end_ms = int(segments[k].end * 1000)
                        if start_ms > cursor_ms:
                            timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
                            cursor_ms = start_ms
                        timeline += AudioSegment.silent(duration=max(0, end_ms - start_ms))
                        cursor_ms = end_ms
                    continue
            else:
                logger.warning(f"Audio file not found: {out_path}")
                # Add silence for all segments in this group
                for k in idxs:
                    start_ms = int(segments[k].start * 1000)
                    end_ms = int(segments[k].end * 1000)
                    if start_ms > cursor_ms:
                        timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
                        cursor_ms = start_ms
                    timeline += AudioSegment.silent(duration=max(0, end_ms - start_ms))
                    cursor_ms = end_ms
        else:
            logger.warning(f"No output path for sentence group {gi}")
            # Add silence for all segments in this group
            for k in idxs:
                start_ms = int(segments[k].start * 1000)
                end_ms = int(segments[k].end * 1000)
                if start_ms > cursor_ms:
                    timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
                    cursor_ms = start_ms
                timeline += AudioSegment.silent(duration=max(0, end_ms - start_ms))
                cursor_ms = end_ms

    return timeline


def build_timeline_sentences_sync(
    segments: list[Segment],
    sentence_groups: list[list[int]],
    tmp_dir: str,
    voice_tag: str,
    synth_func: Callable[[str, str], None],
    sample_rate: int = 24000,
    cache_sig: tuple[str, str, str] | None = None,
    no_tts: bool = False,
) -> AudioSegment:
    """Sync wrapper for build_timeline_sentences_async."""

    # Create async synth function from sync one
    async def async_synth(texts: List[str], output_paths: List[str]) -> List[str]:
        for text, out_path in zip(texts, output_paths):
            synth_func(text, out_path)
        return output_paths

    # Run async version
    return asyncio.run(build_timeline_sentences_async(
        segments, sentence_groups, tmp_dir, voice_tag,
        async_synth, sample_rate, cache_sig, no_tts
    ))


async def build_timeline_wav_async(
    segments: list[Segment],
    tmp_dir: str,
    voice_tag: str,
    synth_func_async: Callable[[str, str], None],
    sample_rate: int = 24000,
    strict_timing: bool = False,
    tolerance_ms: int = 30,
    fit_mode: str = "pad-or-speedup",
    cache_sig: tuple[str, str, str] | None = None,
    no_tts: bool = False,
) -> AudioSegment:
    """Build timeline using individual segments with async processing."""
    ensure_dir(tmp_dir)

    if cache_sig:
        provider, model, voice = cache_sig
    else:
        provider, model, voice = "unknown", "unknown", "unknown"

    # Prepare all segments for processing
    tasks = []
    segment_files = []

    for i, seg in enumerate(segments):
        if not seg.text.strip():
            continue

        sig = _hash_for_cache(provider, model, voice, seg.text)
        out_path = os.path.join(tmp_dir, f"{voice_tag}_{i:04d}_{sig}.wav")
        segment_files.append(out_path)

        # Check if already cached
        if os.path.exists(out_path) and not no_tts:
            continue

        # Create async task for TTS
        if not no_tts:
            tasks.append(synth_func_async(seg.text, out_path))

    # Process all TTS requests asynchronously
    if tasks:
        logger.info(f"Processing {len(tasks)} segments asynchronously...")
        await tqdm.gather(*tasks, desc=f"TTS {voice_tag} (async)")

    # Build the final audio timeline with proper synchronization
    timeline = AudioSegment.silent(duration=0)
    cursor_ms = 0

    for i, seg in enumerate(segments):
        if not seg.text.strip():
            # Add silence for this segment
            start_ms = int(seg.start * 1000)
            end_ms = int(seg.end * 1000)
            if start_ms > cursor_ms:
                timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
                cursor_ms = start_ms
            timeline += AudioSegment.silent(duration=max(0, end_ms - start_ms))
            cursor_ms = end_ms
            continue

        if i < len(segment_files):
            out_path = segment_files[i]
            if os.path.exists(out_path):
                try:
                    seg_audio = AudioSegment.from_wav(out_path).set_frame_rate(sample_rate)
                    start_ms = int(seg.start * 1000)
                    end_ms = int(seg.end * 1000)
                    tgt_ms = max(0, end_ms - start_ms)

                    # Add silence if needed before this segment
                    if start_ms > cursor_ms:
                        timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
                        cursor_ms = start_ms

                    if strict_timing:
                        # Apply timing constraints
                        actual_duration = len(seg_audio)
                        diff = abs(actual_duration - tgt_ms)

                        if diff > tolerance_ms and tgt_ms > 0:
                            if fit_mode == "stretch-both":
                                # Stretch audio to fit
                                ratio = tgt_ms / actual_duration
                                seg_audio = time_stretch_wav_ffmpeg(seg_audio, ratio)
                            else:  # pad-or-speedup
                                if actual_duration < tgt_ms:
                                    # Pad with silence
                                    pad_duration = tgt_ms - actual_duration
                                    seg_audio += AudioSegment.silent(duration=pad_duration)
                                else:
                                    # Speed up
                                    ratio = tgt_ms / actual_duration
                                    seg_audio = time_stretch_wav_ffmpeg(seg_audio, ratio)
                    else:
                        # Non-strict timing: fit audio in available window
                        if len(seg_audio) > tgt_ms and tgt_ms > 0:
                            # Truncate if too long
                            seg_audio = seg_audio[:tgt_ms]
                        elif len(seg_audio) < tgt_ms:
                            # Pad with silence if too short
                            seg_audio += AudioSegment.silent(duration=tgt_ms - len(seg_audio))

                    timeline += seg_audio
                    cursor_ms += len(seg_audio)

                except Exception as e:
                    logger.warning(f"Failed to load audio segment {out_path}: {e}")
                    # Add silence for this segment
                    start_ms = int(seg.start * 1000)
                    end_ms = int(seg.end * 1000)
                    if start_ms > cursor_ms:
                        timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
                        cursor_ms = start_ms
                    timeline += AudioSegment.silent(duration=max(0, end_ms - start_ms))
                    cursor_ms = end_ms
                    continue
            else:
                # File not found, add silence
                start_ms = int(seg.start * 1000)
                end_ms = int(seg.end * 1000)
                if start_ms > cursor_ms:
                    timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
                    cursor_ms = start_ms
                timeline += AudioSegment.silent(duration=max(0, end_ms - start_ms))
                cursor_ms = end_ms

    return timeline
