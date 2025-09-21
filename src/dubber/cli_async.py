"""
Asynchronous command-line interface for the video dubbing pipeline.
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydub import AudioSegment

from .cost import estimate_costs
from .io_ffmpeg import (
    add_subtitles_to_video,
    ensure_dir,
    extract_audio,
    get_video_duration_ms,
    mix_tracks,
    mux_audio_to_video,
    run,
)
from .polish import polish_segments
from .sentences import (
    pack_sentences_into_chunks,
    split_segments_into_sentences,
    write_sentences_manifest,
)
from .srt_utils import normalize_segments_by_punct, parse_srt, write_srt
from .stt import transcribe_local_faster_whisper, transcribe_whisper_api
from .timeline_async import build_timeline_sentences_async
from .tts_async import make_synth_openai_async, make_synth_openai_batch_async
from .youtube import generate_youtube_assets, write_youtube_assets

logger = logging.getLogger("dubber")

# Optional OpenAI SDK
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Async Video dubbing pipeline")

    # Phase control
    ap.add_argument(
        "--only-burn-subs",
        action="store_true",
        help="Only embed subtitles into a given video and exit",
    )
    ap.add_argument(
        "--stage",
        choices=["prep", "synth", "burn", "youtube"],
        default="prep",
        help="prep: STT+polish+SRT+sentences; synth: TTS+mix; burn: embed subs only; youtube: generate YouTube assets",
    )

    # IO
    ap.add_argument("--input_video", required=True)
    ap.add_argument("--workdir", default=".work")
    ap.add_argument("--output", default="output_dubbed.mp4")

    # STT & GPT/TTS models
    ap.add_argument(
        "--stt", choices=["local", "openai"], default="local", help="Speech-to-text backend"
    )
    ap.add_argument("--gpt-model", default="gpt-4o-mini")
    ap.add_argument("--whisper-model", default="whisper-1")
    ap.add_argument("--skip-polish", action="store_true")
    ap.add_argument(
        "--segments-json",
        default=None,
        help="Use segments from JSON (skip STT). Fields: start,end,text",
    )
    ap.add_argument("--segments-srt", default=None, help="Use segments from SRT (skip STT)")

    # TTS provider & voices
    ap.add_argument("--tts-provider", choices=["openai", "elevenlabs"], default="openai")
    ap.add_argument(
        "--tts-model", default="gpt-4o-mini-tts", help="Used when --tts-provider=openai"
    )
    ap.add_argument("--voice-main", default="alloy", help="OpenAI TTS voice (when provider=openai)")
    ap.add_argument(
        "--voice-instructions",
        default=os.getenv("OPENAI_TTS_INSTRUCTIONS"),
        help="Optional TTS style instructions for OpenAI (not read aloud)",
    )

    # Async processing
    ap.add_argument("--async-tts", action="store_true", help="Use async TTS processing")
    ap.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent TTS requests")

    # Alignment / timing
    ap.add_argument(
        "--align-mode",
        choices=["segment", "sentence", "block"],
        default="block",
        help="Build VO per tiny segment, per sentences, or per larger blocks",
    )
    ap.add_argument(
        "--sentence-join-gap",
        type=float,
        default=1.2,
        help="Max gap (sec) to keep adjacent segments in one sentence",
    )
    ap.add_argument(
        "--sentences-per-chunk",
        type=int,
        default=2,
        help="How many full sentences to synthesize per chunk (no mid-sentence cuts)",
    )

    # YouTube generation
    ap.add_argument(
        "--youtube-outdir",
        default=None,
        help="Output directory for YouTube assets (default: <workdir>/youtube)",
    )
    ap.add_argument(
        "--youtube-source",
        choices=["sentences", "srt", "blocks"],
        default="sentences",
        help="Source for chapter generation",
    )
    ap.add_argument(
        "--youtube-gap-thresh",
        type=float,
        default=1.5,
        help="Gap threshold for chapter detection from SRT",
    )
    ap.add_argument("--youtube-ai", action="store_true", help="Use AI to improve description and tags")

    # Logging
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return ap.parse_args()


async def main_async() -> None:
    """Main async CLI entry point."""
    # Load environment variables from .env file
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    args = parse_args()
    setup_logging(args.verbose)

    # Phase B: only burn subs and exit
    if args.only_burn_subs:
        subs_path = getattr(args, 'subs_path', None)
        if not subs_path:
            raise RuntimeError("--subs-path is required when --only-burn-subs is set")
        add_subtitles_to_video(
            args.input_video,
            subs_path,
            args.output,
            mode=getattr(args, 'burn_mode', 'soft'),
            crf=getattr(args, 'burn_crf', 18),
            preset=getattr(args, 'burn_preset', 'medium'),
        )
        logger.info(f"Done (subs -> {args.output})")
        return

    ensure_dir(args.workdir)
    tmp = os.path.join(args.workdir, "tmp")
    ensure_dir(tmp)

    input_wav = os.path.join(args.workdir, "extracted.wav")
    extract_audio(args.input_video, input_wav, sample_rate=16000)

    # Decide services
    need_openai = (
        (args.stt == "openai")
        or (not args.skip_polish)
        or (args.tts_provider == "openai" and args.stage in ["synth", "burn"])
        or args.youtube_ai
        or (args.translate and args.stage == "prep")  # Translation needs OpenAI client
    )
    openai_key = os.getenv("OPENAI_API_KEY")
    client = None
    if need_openai:
        if not AsyncOpenAI:
            raise RuntimeError("openai package not installed. Install with: poetry add openai")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or environment.")
        client = AsyncOpenAI(api_key=openai_key)

    # Cost estimate (based on extracted audio length)
    audio_len_min = len(AudioSegment.from_wav(input_wav)) / 60000.0
    rates = dict(
        stt_openai_per_min=0.006,
        tts_openai_per_min=0.03,
        tts_elevenlabs_per_min=0.15,
        gpt_in_per_mtok=0.60,
        gpt_out_per_mtok=2.40,
        tokens_in_per_min=200.0,
        tokens_out_per_min=200.0,
    )
    est = estimate_costs(
        audio_len_min,
        args.stt,
        has_polish=(not args.skip_polish),
        tts_provider=args.tts_provider,
        commentary=False,
        commentary_ratio=0.0,
        rates=rates,
    )
    logger.info(f"=== Estimated costs (based on {audio_len_min:.2f} min) ===")
    logger.info(f"STT ({args.stt}): ${est['stt_cost']:.4f}")
    tts_str = "n/a" if est["tts_cost"] is None else f"${est['tts_cost']:.4f}"
    logger.info(f"TTS ({args.tts_provider} ~ {est['tts_minutes']:.2f} min): {tts_str}")
    logger.info(f"Polish (GPT): ${est['polish_cost']:.4f}")
    total_str = "n/a" if est["tts_cost"] is None else f"${est['total']:.4f}"
    logger.info(f"TOTAL: {total_str}")

    # Transcription -> segments
    if args.stage == "synth":
        # Read approved user SRT
        srt_path = getattr(args, 'subs_path', None) or os.path.join(args.workdir, "subs.srt")
        if not os.path.exists(srt_path):
            raise RuntimeError(f"SRT not found for synth stage: {srt_path}")
        segments = parse_srt(srt_path)
        logger.info(f"Loaded SRT -> {srt_path} ({len(segments)} segments)")
    else:
        # prep: do STT (+opt. polish) and write SRT
        if args.stt == "openai":
            try:
                segments = transcribe_whisper_api(client, input_wav, model=args.whisper_model)
            except Exception as e:
                logger.warning(
                    f"API transcription failed: {e}\nFalling back to local faster-whisper…"
                )
                segments = transcribe_local_faster_whisper(input_wav, local_model="base.en")
        else:
            segments = transcribe_local_faster_whisper(input_wav, local_model="base.en")

        if all(not s.text.strip() for s in segments):
            raise RuntimeError("Transcription returned empty text.")

        raw_json = os.path.join(args.workdir, "segments_raw.json")
        with open(raw_json, "w", encoding="utf-8") as f:
            json.dump([s.__dict__ for s in segments], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved raw segments -> {raw_json}")

        if not args.skip_polish:
            segments = polish_segments(client, segments, model=args.gpt_model)
            polished_json = os.path.join(args.workdir, "segments_polished.json")
            with open(polished_json, "w", encoding="utf-8") as f:
                json.dump([s.__dict__ for s in segments], f, ensure_ascii=False, indent=2)
            logger.info(f"Saved polished segments -> {polished_json}")

        # Normalize segments: split multi-sentence segments by punctuation
        segments = normalize_segments_by_punct(segments)

        srt_path = getattr(args, 'subs_path', None) or os.path.join(args.workdir, "subs.srt")
        write_srt(segments, srt_path)
        logger.info(f"Saved SRT -> {srt_path}")

    true_sents = split_segments_into_sentences(
        segments,
        join_gap_max=args.sentence_join_gap,
        max_sentence_secs=12.0,
        max_segments_per_sentence=6,
    )

    # Pack N sentences into chunk
    groups = pack_sentences_into_chunks(
        true_sents,
        segments,
        sentences_per_chunk=args.sentences_per_chunk,
        max_chunk_secs=20.0,
        max_segments_per_chunk=12,
    )

    # Write manifest
    sent_json = os.path.join(args.workdir, "sentences_groups.json")
    write_sentences_manifest(segments, groups, sent_json)
    logger.info(
        f"Saved sentences manifest -> {sent_json} ({len(groups)} chunks, {len(true_sents)} sentences)"
    )

    if args.stage == "prep":
        logger.info(
            "Stage 'prep' complete. Review subs.srt and sentences_groups.json, then run stage 'synth'."
        )
        return

    # YouTube generation stage
    if args.stage == "youtube":
        youtube_outdir = args.youtube_outdir or os.path.join(args.workdir, "youtube")
        assets = generate_youtube_assets(
            workdir=args.workdir,
            youtube_source=args.youtube_source,
            gap_thresh=args.youtube_gap_thresh,
            use_ai=args.youtube_ai,
            client=client,
            title_override=args.youtube_title,
        )
        write_youtube_assets(
            youtube_outdir, assets.title, assets.description, assets.chapters
        )
        logger.info(f"YouTube assets generated in {youtube_outdir}")
        return

    # TTS synth functions
    if args.tts_provider == "openai":
        if args.async_tts:
            synth_main = make_synth_openai_batch_async(
                client, args.tts_model, args.voice_main, args.voice_instructions, args.max_concurrent
            )
        else:
            synth_main = make_synth_openai_async(
                client, args.tts_model, args.voice_main, args.voice_instructions
            )
        cache_sig = ("openai", args.tts_model, args.voice_main)
    else:
        raise RuntimeError("Async TTS only supports OpenAI for now")

    # Build per-mode with async support
    if args.align_mode == "sentence" and args.async_tts:
        logger.info(f"Using async TTS processing with {args.max_concurrent} concurrent requests")
        main_track = await build_timeline_sentences_async(
            segments=segments,
            sentence_groups=groups,
            tmp_dir=os.path.join(args.workdir, "tmp", "main_sent"),
            voice_tag="main",
            synth_func_async=synth_main,
            sample_rate=24000,
            cache_sig=cache_sig,
        )
    else:
        # Fallback to sync processing
        from .timeline import build_timeline_sentences
        main_track = build_timeline_sentences(
            segments=segments,
            sentence_groups=groups,
            tmp_dir=os.path.join(args.workdir, "tmp", "main_sent"),
            voice_tag="main",
            synth_func=synth_main,
            sample_rate=24000,
            cache_sig=cache_sig,
        )

    # After building main_track
    vid_ms = get_video_duration_ms(args.input_video)
    logger.info(f"[dur] video = {vid_ms/1000:.3f}s, main_track = {len(main_track)/1000:.3f}s")
    final_audio = mix_tracks(main_track, None, main_db=0.0)

    # Force pad/trim to EXACT video length
    if vid_ms > 0:
        if len(final_audio) < vid_ms:
            pad = vid_ms - len(final_audio)
            logger.info(f"[dur] padding audio by {pad/1000:.3f}s to match video")
            final_audio = final_audio + AudioSegment.silent(duration=pad)
        elif len(final_audio) > vid_ms:
            cut = len(final_audio) - vid_ms
            logger.info(f"[dur] trimming audio by {cut/1000:.3f}s to match video")
            final_audio = final_audio[:vid_ms]

    logger.info(f"[dur] final_audio = {len(final_audio)/1000:.3f}s (target {vid_ms/1000:.3f}s)")

    final_wav = os.path.join(args.workdir, "new_audio.wav")
    final_audio.export(final_wav, format="wav")

    probe = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            os.path.join(args.workdir, "new_audio.wav"),
        ],
        check=False,
    ).strip()
    logger.info(f"[ffprobe] new_audio.wav duration ≈ {probe}s")

    logger.info(f"Exported mixed audio -> {final_wav}")

    mux_audio_to_video(args.input_video, final_wav, args.output)
    logger.info(f"Done (dubbed) -> {args.output}")


def main() -> None:
    """Main CLI entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
