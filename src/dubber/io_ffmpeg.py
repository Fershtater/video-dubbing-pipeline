"""
Audio and video processing utilities using ffmpeg/ffprobe.
"""

import logging
import subprocess
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger("dubber")


def run(cmd: list[str], *, check: bool = True) -> str:
    """Run a shell command and return stdout."""
    logger.debug("Running: %s", ' '.join(map(str, cmd)))
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    if proc.returncode != 0 and check:
        logger.error("Command failed with code %d: %s", proc.returncode, proc.stdout)
        msg = f"Command failed with code {proc.returncode}"
        raise RuntimeError(msg)
    return proc.stdout


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def time_stretch_wav_ffmpeg(in_wav: str, out_wav: str, ratio: float) -> None:
    """
    Fit audio duration via ffmpeg atempo chain.
    NOTE (correct semantics): atempo < 1.0 => slow down (longer),
    atempo > 1.0 => speed up (shorter).
    We chain multiple atempo filters to stay within 0.5..2.0 per step.
    """
    if ratio <= 0:
        ratio = 1.0
    steps: list[float] = []
    r = ratio
    MIN_ATEMPO = 0.5
    MAX_ATEMPO = 2.0
    while r < MIN_ATEMPO or r > MAX_ATEMPO:
        step = MIN_ATEMPO if r < 1.0 else MAX_ATEMPO
        steps.append(step)
        r /= step
    steps.append(r)
    filt = ",".join(f"atempo={s:.6f}" for s in steps)
    run(["ffmpeg", "-y", "-i", in_wav, "-filter:a", filt, out_wav])


def extract_audio(input_video: str, out_wav: str, sample_rate: int = 16000) -> None:
    """Extract audio from video file."""
    ensure_dir(str(Path(out_wav).parent))
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        out_wav,
    ]
    run(cmd)


def mux_audio_to_video(input_video: str, audio_wav: str, output_video: str) -> None:
    """Mux audio track into video (copy video stream)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-i",
        audio_wav,
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        output_video,
    ]
    run(cmd)


def get_video_duration_ms(input_video: str) -> int:
    """Get video duration in milliseconds."""
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_video,
        ]
    )
    try:
        seconds = float(out.strip())
    except ValueError:
        seconds = 0.0
    return int(seconds * 1000)


def add_subtitles_to_video(
    input_video: str,
    subs_path: str,
    output_video: str,
    mode: str = "soft",
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """Add subtitles to video (soft or hard)."""
    if mode == "soft":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_video,
            "-i",
            subs_path,
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-c:s",
            "mov_text",
            output_video,
        ]
        run(cmd)
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_video,
            "-vf",
            f"subtitles={subs_path}",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-c:a",
            "copy",
            output_video,
        ]
        run(cmd)


def mix_tracks(
    main: AudioSegment,
    commentary: AudioSegment | None = None,
    main_db: float = 0.0,
    comm_db: float = -8.0,
) -> AudioSegment:
    """Mix main and commentary audio tracks."""
    if commentary is None:
        return main
    dur = max(len(main), len(commentary))
    main_pad = main + AudioSegment.silent(duration=dur - len(main)) if len(main) < dur else main
    comm_pad = (
        commentary + AudioSegment.silent(duration=dur - len(commentary))
        if len(commentary) < dur
        else commentary
    )
    return main_pad.apply_gain(main_db).overlay(comm_pad.apply_gain(comm_db))
