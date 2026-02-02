"""
Command-line interface for the video dubbing pipeline.
"""

import argparse
import json
import logging
import os

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
from .scenes import (
    derive_block_boundaries,
    detect_scene_changes,
    merge_segments_into_blocks,
    write_block_srt,
)
from .sentences import (
    pack_sentences_into_chunks,
    split_segments_into_sentences,
    write_sentences_manifest,
)
from .srt_utils import normalize_segments_by_punct, parse_srt, write_srt
from .stt import transcribe_local_faster_whisper, transcribe_whisper_api, detect_language
from .translation import translate_segments_to_english, get_language_name
from .timeline import build_timeline_blocks, build_timeline_sentences, build_timeline_wav
from .tts import make_synth_elevenlabs, make_synth_openai, pick_elevenlabs_default_voice
from .youtube import generate_youtube_assets, write_youtube_assets

logger = logging.getLogger("dubber")

# Optional OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Video dubbing pipeline (block-synced)")

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

    # Multilingual support
    ap.add_argument("--source-language", default=None,
                   help="Source language code (e.g., 'ru', 'de', 'fr'). Auto-detected if not specified.")
    ap.add_argument("--translate", action="store_true",
                   help="Enable translation from source language to English")
    ap.add_argument("--translation-model", default="gpt-4o-mini",
                   help="GPT model to use for translation")

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
    ap.add_argument(
        "--elevenlabs-voice-id",
        default=None,
        help="Main ElevenLabs voice_id (defaults to $ELEVENLABS_VOICE_ID or auto-pick)",
    )
    ap.add_argument("--elevenlabs-model-id", default="eleven_multilingual_v2")

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
    ap.add_argument(
        "--scene-detect", action="store_true", help="Detect scene cuts via ffprobe to form blocks"
    )
    ap.add_argument(
        "--scene-thresh",
        type=float,
        default=0.3,
        help="Scene change threshold for ffprobe (0.1..0.6 typical)",
    )
    ap.add_argument(
        "--min-scene-gap",
        type=float,
        default=1.2,
        help="Audio pause >= this (sec) becomes a block boundary",
    )

    # Segment strict timing (legacy)
    ap.add_argument(
        "--strict-timing",
        action="store_true",
        help="Fit each TTS segment to its original duration (segment mode)",
    )
    ap.add_argument(
        "--timing-tolerance-ms", type=int, default=30, help="No stretch if |diff| <= tolerance"
    )
    ap.add_argument(
        "--timing-fit-mode", choices=["pad-or-speedup", "stretch-both"], default="pad-or-speedup"
    )

    # Block fitting controls
    ap.add_argument(
        "--min-atempo", type=float, default=0.90, help="Lower atempo clamp for block fitting"
    )
    ap.add_argument(
        "--max-atempo", type=float, default=1.10, help="Upper atempo clamp for block fitting"
    )
    ap.add_argument("--breath-pad-ms", type=int, default=200, help="Silence at end of each block")
    ap.add_argument(
        "--no-speedup", action="store_true", help="Do not accelerate speech (block mode)"
    )

    # SRT / burning
    ap.add_argument(
        "--emit-srt", action="store_true", help="Emit subs.srt from (polished) segments"
    )
    ap.add_argument(
        "--subs-path",
        default=None,
        help="Path to .srt for burning or override default workdir path",
    )
    ap.add_argument("--burn-mode", choices=["soft", "hard"], default="soft")
    ap.add_argument("--burn-crf", type=int, default=18)
    ap.add_argument("--burn-preset", default="medium")
    ap.add_argument("--block-max-lines", type=int, default=3, help="Max lines per SRT block cue")
    ap.add_argument(
        "--block-wrap-chars", type=int, default=42, help="Wrap width per SRT block line"
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
    ap.add_argument(
        "--youtube-ai", action="store_true", help="Use AI to improve description and tags"
    )
    ap.add_argument("--youtube-title", default=None, help="Override title for YouTube description")

    # Cost estimation
    ap.add_argument("--estimate-only", action="store_true", help="Print cost estimate and exit")
    ap.add_argument(
        "--commentary-ratio",
        type=float,
        default=0.0,
        help="Extra minutes proportion for commentary (kept 0 by default)",
    )
    ap.add_argument("--rate-stt-openai-per-min", type=float, default=0.006)
    ap.add_argument("--rate-tts-openai-per-min", type=float, default=0.03)
    ap.add_argument("--rate-gpt-in-per-mtok", type=float, default=0.60)
    ap.add_argument("--rate-gpt-out-per-mtok", type=float, default=2.40)
    ap.add_argument(
        "--rate-tts-elevenlabs-per-min",
        type=float,
        default=0.15,
        help="Set to 0 if included minutes cover your usage",
    )

    # Cache control
    ap.add_argument(
        "--no-tts",
        action="store_true",
        help="Do not call TTS; reuse cached segment/block WAVs only (error if missing)",
    )

    # Logging
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return ap.parse_args()


def main() -> None:
    """Main CLI entry point."""
    # Load environment variables from .env file
    # Look for .env in the project root (parent of src directory)
    import pathlib
    project_root = pathlib.Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Fallback: try loading from current directory
        load_dotenv()

    args = parse_args()
    setup_logging(args.verbose)

    # Phase B: only burn subs and exit
    if args.only_burn_subs:
        subs_path = args.subs_path
        if not subs_path:
            raise RuntimeError("--subs-path is required when --only-burn-subs is set")
        add_subtitles_to_video(
            args.input_video,
            subs_path,
            args.output,
            mode=args.burn_mode,
            crf=args.burn_crf,
            preset=args.burn_preset,
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
        or (args.youtube_ai and args.stage == "youtube")
        or (args.translate and args.stage == "prep")  # Translation needs OpenAI client
    )
    openai_key = os.getenv("OPENAI_API_KEY")
    client = None
    if need_openai:
        if not OpenAI:
            raise RuntimeError("openai package not installed. Install with: poetry add openai")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or environment.")
        client = OpenAI(api_key=openai_key)

    eleven_key = os.getenv("ELEVENLABS_API_KEY")

    # Cost estimate (based on extracted audio length)
    audio_len_min = len(AudioSegment.from_wav(input_wav)) / 60000.0
    rates = dict(
        stt_openai_per_min=args.rate_stt_openai_per_min,
        tts_openai_per_min=args.rate_tts_openai_per_min,
        tts_elevenlabs_per_min=(args.rate_tts_elevenlabs_per_min or None),
        gpt_in_per_mtok=args.rate_gpt_in_per_mtok,
        gpt_out_per_mtok=args.rate_gpt_out_per_mtok,
        tokens_in_per_min=200.0,
        tokens_out_per_min=200.0,
    )
    est = estimate_costs(
        audio_len_min,
        args.stt,
        has_polish=(not args.skip_polish),
        tts_provider=args.tts_provider,
        commentary=False,
        commentary_ratio=args.commentary_ratio,
        rates=rates,
    )
    logger.info(f"=== Estimated costs (based on {audio_len_min:.2f} min) ===")
    logger.info(f"STT ({args.stt}): ${est['stt_cost']:.4f}")
    tts_str = "n/a" if est["tts_cost"] is None else f"${est['tts_cost']:.4f}"
    logger.info(f"TTS ({args.tts_provider} ~ {est['tts_minutes']:.2f} min): {tts_str}")
    logger.info(f"Polish (GPT): ${est['polish_cost']:.4f}")
    total_str = "n/a" if est["tts_cost"] is None else f"${est['total']:.4f}"
    logger.info(f"TOTAL: {total_str}")
    if args.estimate_only:
        return

    # Transcription -> segments
    if args.stage == "synth":
        # Read approved user SRT
        srt_path = args.subs_path or os.path.join(args.workdir, "subs.srt")
        if not os.path.exists(srt_path):
            raise RuntimeError(f"SRT not found for synth stage: {srt_path}")
        segments = parse_srt(srt_path)
        logger.info(f"Loaded SRT -> {srt_path} ({len(segments)} segments)")
    else:
        # prep: do STT (+opt. polish) and write SRT
        detected_language = None

        # Detect language if not specified and translation is enabled
        if args.translate and not args.source_language and client:
            logger.info("Auto-detecting source language...")
            detected_language = detect_language(client, input_wav)
            if detected_language:
                logger.info(f"Detected source language: {get_language_name(detected_language)} ({detected_language})")

        source_language = args.source_language or detected_language

        if args.stt == "openai":
            try:
                segments = transcribe_whisper_api(
                    client, input_wav,
                    model=args.whisper_model,
                    language=source_language
                )
            except Exception as e:
                logger.warning(
                    f"API transcription failed: {e}\nFalling back to local faster-whisper…"
                )
                # Use multilingual model for local fallback
                local_model = "base" if source_language and source_language != "en" else "base.en"
                segments = transcribe_local_faster_whisper(
                    input_wav,
                    local_model=local_model,
                    language=source_language
                )
        else:
            # Use multilingual model for local transcription
            local_model = "base" if source_language and source_language != "en" else "base.en"
            segments = transcribe_local_faster_whisper(
                input_wav,
                local_model=local_model,
                language=source_language
            )

        if all(not s.text.strip() for s in segments):
            raise RuntimeError("Transcription returned empty text.")

        # Translate segments if needed
        if args.translate and source_language and source_language != "en":
            logger.info(f"Translating segments from {get_language_name(source_language)} to English...")
            if not client:
                raise RuntimeError("Translation requires OpenAI client. Set OPENAI_API_KEY environment variable.")

            try:
                segments = translate_segments_to_english(
                    client, segments,
                    source_language=source_language,
                    model=args.translation_model
                )
                logger.info(f"Translation completed: {len(segments)} segments translated")
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                raise RuntimeError(f"Translation failed: {e}")

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

        srt_path = args.subs_path or os.path.join(args.workdir, "subs.srt")
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
        logger.info("Starting YouTube generation stage")
        youtube_outdir = args.youtube_outdir or os.path.join(args.workdir, "youtube")
        logger.info(f"YouTube output directory: {youtube_outdir}")

        # For YouTube, we need OpenAI client if youtube_ai is True
        if args.youtube_ai and not client:
            logger.warning("YouTube AI requested but OpenAI client not available, falling back to SRT-based generation")
            youtube_client = None
            use_ai = False
        else:
            youtube_client = client if args.youtube_ai else None
            use_ai = args.youtube_ai

        logger.info(f"Using AI: {use_ai}, Client available: {youtube_client is not None}")

        assets = generate_youtube_assets(
            workdir=args.workdir,
            youtube_source=args.youtube_source,
            gap_thresh=args.youtube_gap_thresh,
            use_ai=use_ai,
            client=youtube_client,
            title_override=args.youtube_title,
        )
        logger.info(f"Generated assets: {len(assets.chapters)} chapters")

        write_youtube_assets(
            youtube_outdir, assets.title, assets.description, assets.chapters
        )
        logger.info(f"YouTube assets generated in {youtube_outdir}")
        return

    # SRT export (fine-grained)
    srt_path = args.subs_path or os.path.join(args.workdir, "subs.srt")
    write_srt(segments, srt_path)
    logger.info(f"Saved SRT -> {srt_path}")

    # TTS synth functions
    if args.tts_provider == "openai":
        synth_main = make_synth_openai(
            client, args.tts_model, args.voice_main, args.voice_instructions
        )
        cache_sig = ("openai", args.tts_model, args.voice_main)
    else:
        main_voice_id = args.elevenlabs_voice_id or os.getenv("ELEVENLABS_VOICE_ID")
        if not eleven_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set. Put it in .env or environment.")
        if not main_voice_id:
            auto_vid = pick_elevenlabs_default_voice(eleven_key)
            if auto_vid:
                main_voice_id = auto_vid
                logger.info(f"Using ElevenLabs voice_id (auto): {main_voice_id}")
        if not main_voice_id:
            raise RuntimeError(
                "ElevenLabs voice_id not provided. Set ELEVENLABS_VOICE_ID or pass --elevenlabs-voice-id."
            )
        synth_main = make_synth_elevenlabs(eleven_key, main_voice_id, args.elevenlabs_model_id)
        cache_sig = ("elevenlabs", args.elevenlabs_model_id, main_voice_id)

    # Build per-mode
    if args.align_mode == "segment":
        main_track = build_timeline_wav(
            segments=segments,
            tmp_dir=os.path.join(args.workdir, "tmp", "main_sent"),
            voice_tag="main",
            synth_func=synth_main,
            sample_rate=24000,
            cache_sig=cache_sig,
        )

    elif args.align_mode == "sentence":
        # Use already collected and "packed" groups
        logger.info(f"Grouped into {len(groups)} chunk(s) from {len(true_sents)} sentence(s)")
        main_track = build_timeline_sentences(
            segments=segments,
            sentence_groups=groups,
            tmp_dir=os.path.join(args.workdir, "tmp", "main_sent"),
            voice_tag="main",
            synth_func=synth_main,
            sample_rate=24000,
            cache_sig=cache_sig,
        )

    else:  # block
        scene_times: list[float] = []
        if args.scene_detect:
            try:
                scene_times = detect_scene_changes(args.input_video, thresh=args.scene_thresh)
                logger.info(f"Detected {len(scene_times)} scene cuts")
            except Exception as e:
                logger.warning(f"scene detection failed: {e}")
        boundaries = derive_block_boundaries(segments, scene_times, args.min_scene_gap)
        blocks = merge_segments_into_blocks(segments, boundaries)
        block_srt = os.path.join(args.workdir, "subs_block.srt")
        write_block_srt(
            blocks, block_srt, wrap_chars=args.block_wrap_chars, max_lines=args.block_max_lines
        )
        logger.info(f"Saved SRT (blocks) -> {block_srt}")

        main_track = build_timeline_blocks(
            blocks=blocks,
            tmp_dir=os.path.join(args.workdir, "tmp", "main_blocks"),
            voice_tag="main",
            synth_func=synth_main,
            sample_rate=24000,
            min_atempo=args.min_atempo,
            max_atempo=args.max_atempo,
            breath_pad_ms=args.breath_pad_ms,
            no_speedup=args.no_speedup,
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
    logger.info(
        "Review subs_block.srt for overlay, or use subs.srt if you prefer fine granularity."
    )
