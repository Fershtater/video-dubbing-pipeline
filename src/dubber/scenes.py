"""
Scene detection and block building utilities.
"""

import contextlib
import itertools
import logging
from pathlib import Path

from .io_ffmpeg import run
from .models import Block, Segment
from .srt_utils import wrap_lines

logger = logging.getLogger("dubber")


def detect_scene_changes(input_video: str, thresh: float = 0.3) -> list[float]:
    """Return list of seconds where a scene cut is detected via ffprobe+lavfi scene filter."""
    vf = f"movie={input_video},select=gt(scene\\,{thresh})"
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        vf,
        "-show_entries",
        "frame=pkt_pts_time",
        "-of",
        "csv=p=0",
    ]
    out = run(cmd)
    cuts: list[float] = []
    for line in out.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        with contextlib.suppress(ValueError):
            cuts.append(float(stripped_line))
    return sorted(set(cuts))


def derive_block_boundaries(
    segments: list[Segment], scene_times: list[float], min_scene_gap: float
) -> list[float]:
    """Combine scene cuts and large gaps between segments into boundary times.
    Returns sorted list including 0.0 and last end.
    """
    if not segments:
        return [0.0, 0.0]
    times = {0.0}
    # Large pauses from STT
    for i in range(len(segments) - 1):
        gap = segments[i + 1].start - segments[i].end
        if gap >= min_scene_gap:
            times.add(segments[i].end)
    # Scene cuts
    for t in scene_times:
        # snap to nearest segment boundary if within threshold
        SNAP_THRESHOLD = 0.25
        for s in segments:
            if abs(s.start - t) < SNAP_THRESHOLD:
                t = s.start
                break
            if abs(s.end - t) < SNAP_THRESHOLD:
                t = s.end
                break
        times.add(max(0.0, t))
    last_end = max(s.end for s in segments)
    times.add(last_end)
    return sorted(times)


def merge_segments_into_blocks(segments: list[Segment], boundaries: list[float]) -> list[Block]:
    """Merge segments into blocks based on boundaries."""
    blocks: list[Block] = []
    if not segments:
        return blocks
    for b_start, b_end in itertools.pairwise(boundaries):
        in_block = [s for s in segments if s.start < b_end and s.end > b_start]
        if not in_block:
            continue
        start = max(b_start, min(s.start for s in in_block))
        end = min(b_end, max(s.end for s in in_block))
        text = " ".join(s.text.strip() for s in in_block if s.text.strip())
        text = " ".join(text.split())
        if not text:
            continue
        blocks.append(Block(start=start, end=end, text=text))
    return blocks


def write_block_srt(
    blocks: list[Block], path: str, wrap_chars: int = 42, max_lines: int = 3
) -> None:
    """Write blocks to SRT file with text wrapping."""

    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02}:{m:02}:{int(s):02},{int((s-int(s))*1000):03}"

    content = ""
    for i, b in enumerate(blocks, 1):
        txt = wrap_lines(b.text, wrap_chars, max_lines)
        content += f"{i}\n{fmt(b.start)} --> {fmt(b.end)}\n{txt}\n\n"
    Path(path).write_text(content, encoding="utf-8")
