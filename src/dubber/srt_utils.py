"""
SRT parsing, writing, and text normalization utilities.
"""

import logging
import re

from .models import Segment

logger = logging.getLogger("dubber")

# Sentence splitting patterns
_SENT_END_RE = re.compile(r'[.!?]["\')\]]*\s*$')
_SENT_SPLIT_RE = re.compile(r"\s*(?<=[.!?])[\s\n]+")
ABBR_SET = {"e.g.", "i.e.", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "etc."}


def write_srt(segments: list[Segment], path: str) -> None:
    """Write segments to SRT file."""

    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02}:{m:02}:{int(s):02},{int((s-int(s))*1000):03}"

    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            f.write(f"{i}\n{fmt(s.start)} --> {fmt(s.end)}\n{s.text}\n\n")


def parse_srt(path: str) -> list[Segment]:
    """Parse SRT file into segments."""

    def parse_ts(ts: str) -> float:
        h, m, rest = ts.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    with open(path, encoding="utf-8") as f:
        raw = f.read()

    blocks = re.split(r"\n\s*\n", raw.strip(), flags=re.M)
    out: list[Segment] = []
    for b in blocks:
        lines = [ln for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        if re.match(r"^\d+$", lines[0].strip()):
            lines = lines[1:]
        if not lines:
            continue
        m = re.match(r"(\d\d:\d\d:\d\d,\d\d\d)\s+--\>\s+(\d\d:\d\d:\d\d,\d\d\d)", lines[0])
        if not m:
            continue
        start = parse_ts(m.group(1))
        end = parse_ts(m.group(2))
        text = " ".join(ln.strip() for ln in lines[1:])
        out.append(Segment(start=start, end=end, text=text))
    return out


def read_srt(path: str) -> list[Segment]:
    """Alternative SRT reader (legacy compatibility)."""
    import re

    ts = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")

    def to_sec(m):
        h, m_, s, ms = map(int, m)
        return h * 3600 + m_ * 60 + s + ms / 1000.0

    with open(path, encoding="utf-8") as f:
        data = f.read().splitlines()

    segs: list[Segment] = []
    i = 0
    while i < len(data):
        if data[i].strip().isdigit():
            i += 1
        if i >= len(data):
            break
        if "-->" not in data[i]:
            i += 1
            continue
        left, right = [x.strip() for x in data[i].split("-->")]
        i += 1
        left_match = ts.search(left)
        right_match = ts.search(right)
        if not left_match or not right_match:
            continue
        start = to_sec(left_match.groups())
        end = to_sec(right_match.groups())
        lines: list[str] = []
        while i < len(data) and data[i].strip() != "":
            lines.append(data[i].strip())
            i += 1
        while i < len(data) and data[i].strip() == "":
            i += 1
        text = " ".join(lines).strip()
        segs.append(Segment(start=start, end=end, text=text))
    return segs


def _split_segment_text_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter that keeps punctuation and avoids common abbreviations."""
    raw_parts = _SENT_SPLIT_RE.split(text.strip())
    parts: list[str] = []
    buf = []
    for p in raw_parts:
        chunk = p.strip()
        if not chunk:
            continue
        buf.append(chunk)
        joined = " ".join(buf)
        lw = joined.split()[-1] if joined.split() else ""
        if lw in ABBR_SET:
            continue
        if re.search(r"[.!?]$", joined):
            parts.append(joined)
            buf = []
    if buf:
        parts.append(" ".join(buf))
    return parts


def normalize_segments_by_punct(segments: list[Segment]) -> list[Segment]:
    """Split any single STT segment that contains multiple sentences into several
    shorter segments by punctuation, distributing duration proportionally by text length.
    """
    normalized: list[Segment] = []
    for seg in segments:
        txt = (seg.text or "").strip()
        if not txt:
            normalized.append(seg)
            continue
        parts = _split_segment_text_into_sentences(txt)
        if len(parts) <= 1:
            normalized.append(seg)
            continue
        total_chars = sum(max(1, len(p)) for p in parts)
        start = float(seg.start)
        total_dur = max(0.0, float(seg.end - seg.start))
        acc_time = 0.0
        for idx, p in enumerate(parts):
            share = max(1, len(p)) / total_chars
            dur = total_dur * share if idx < len(parts) - 1 else max(0.0, total_dur - acc_time)
            new_start = start + acc_time
            new_end = new_start + dur
            normalized.append(Segment(start=new_start, end=new_end, text=p.strip()))
            acc_time += dur
    return normalized


def wrap_lines(text: str, max_chars: int = 42, max_lines: int = 3) -> str:
    """Wrap text to specified character and line limits."""
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        if sum(len(x) for x in cur) + len(cur) + len(w) > max_chars and cur:
            lines.append(" ".join(cur))
            cur = []
            if len(lines) >= max_lines - 1:
                break
        cur.append(w)
    if cur and len(lines) < max_lines:
        lines.append(" ".join(cur))
    return "\n".join(lines)
