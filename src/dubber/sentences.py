"""
Sentence grouping and chunking logic for natural speech synthesis.
"""

import json
import logging
import re
from pathlib import Path

from .models import Segment

logger = logging.getLogger("dubber")

# Sentence ending pattern
_SENT_END_RE = re.compile(r'[.!?]["\')\]]*\s*$')


def split_segments_into_sentences(
    segments: list[Segment],
    *,
    join_gap_max: float = 1.2,
    max_sentence_secs: float = 12.0,
    max_segments_per_sentence: int = 6,
) -> list[list[int]]:
    """
    Group segments into natural sentences:
    - Close on . ! ? at end of accumulated text AND no short continuation\n      immediately (or large pause),
    - OR force cut on large pause (>= join_gap_max),
    - OR on duration/segment count limits (to avoid bloating tails without punctuation).
    Never break inside a segment.
    """

    res: list[list[int]] = []
    cur: list[int] = []
    acc = ""
    cur_start: float | None = None

    def sentence_done(i: int) -> bool:
        """Decide whether to close sentence at current segment i."""
        nonlocal acc, cur_start
        s = segments[i]
        # duration of current sentence
        dur = s.end - (cur_start if cur_start is not None else s.start)
        # pause to next
        # next_gap = None
        # if i + 1 < len(segments):
        #     next_gap = max(0.0, float(segments[i + 1].start - s.end))

        by_punct = bool(_SENT_END_RE.search(acc))
        # by_long_gap = next_gap is not None and next_gap >= join_gap_max
        by_limits = (dur >= max_sentence_secs) or (len(cur) >= max_segments_per_sentence)

        # Soft anti-break: if only limits trigger, but phrase clearly continues
        # (ends with service/linking words or English contractions with apostrophe),
        # then don't close, to avoid breaking inside "We'll also …" etc.
        if by_limits and not by_punct:
            tail = acc.strip().rstrip()
            last_word = tail.split()[-1] if tail else ""
            last_word = last_word.strip("\"')].,;:!?" "''").lower()
            continuation_words = {
                "and",
                "or",
                "but",
                "so",
                "also",
                "then",
                "thus",
                "therefore",
                "however",
                "moreover",
                "furthermore",
                "meanwhile",
                "besides",
                "because",
                "since",
                "though",
                "although",
                "unless",
                "until",
                "to",
                "of",
                "in",
                "on",
                "with",
                "for",
                "as",
                "that",
                "this",
                "these",
                "those",
                "we'll",
                "i'll",
                "they'll",
                "you'll",
                "he'll",
                "she'll",
                "we're",
                "i'm",
                "you're",
                "they're",
                "we've",
                "i've",
                "you've",
                "they've",
                "we'd",
                "i'd",
                "you'd",
                "they'd",
            }
            contraction_like = last_word.endswith(
                ("'ll", "'re", "'ve", "'d", "'m")
            )
            if last_word in continuation_words or contraction_like:
                return False

        # Close on final punctuation almost always, to avoid long sentence chains sticking together.
        # Exceptions: common abbreviations like "e.g.", "i.e.", "Mr.", "Dr.".
        if by_punct:
            tail = acc.strip().rstrip()
            last_word = tail.split()[-1] if tail else ""
            last_word_clean = last_word.strip("\"')].,;:!?" "''")
            lower = last_word_clean.lower()
            common_abbr = {"e.g", "i.e", "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc"}
            if lower in common_abbr:
                pass
            else:
                return True
        return by_limits

    for i, seg in enumerate(segments):
        piece = (seg.text or "").strip()
        if not cur:
            cur = [i]
            acc = piece
            cur_start = seg.start
        else:
            cur.append(i)
            acc = (acc + " " + piece).strip()

        if sentence_done(i):
            res.append(cur)
            cur, acc, cur_start = [], "", None

    if cur:
        res.append(cur)
    return res


def pack_sentences_into_chunks(
    sentence_groups: list[list[int]],
    segments: list[Segment],
    *,
    sentences_per_chunk: int = 2,
    max_chunk_secs: float = 20.0,
    max_segments_per_chunk: int = 12,
) -> list[list[int]]:
    """
    Take ready sentences (lists of segment indices) and pack N sentences into one chunk.
    Don't break inside a sentence. Additional safeguards: don't bloat\n    longer than max_chunk_secs and >max_segments_per_chunk.
    """
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_start: float | None = None
    cur_end: float | None = None
    cur_sent_count = 0

    def chunk_dur(start: float | None, end: float | None) -> float:
        if start is None or end is None:
            return 0.0
        return max(0.0, end - start)

    for sent in sentence_groups:
        # sentence duration
        s_start = segments[sent[0]].start
        s_end = segments[sent[-1]].end
        # s_dur = max(0.0, s_end - s_start)  # unused variable

        # if empty chunk — just put it
        if not cur:
            cur = sent[:]
            cur_start, cur_end = s_start, s_end
            cur_sent_count = 1
        else:
            # try to add another sentence
            cand = cur + sent
            cand_start = cur_start if cur_start is not None else s_start
            cand_end = s_end
            cand_dur = chunk_dur(cand_start, cand_end)
            cand_seg_count = len(cand)

            if (
                (cur_sent_count + 1) <= sentences_per_chunk
                and cand_dur <= max_chunk_secs
                and cand_seg_count <= max_segments_per_chunk
            ):
                # ok, add it
                cur = cand
                cur_end = cand_end
                cur_sent_count += 1
            else:
                # fix current chunk and start new
                chunks.append(cur)
                cur = sent[:]
                cur_start = s_start
                cur_end = s_end  # fix unused variable
                cur_sent_count = 1

    if cur:
        chunks.append(cur)
    return chunks


def write_sentences_manifest(segments: list[Segment], groups: list[list[int]], path: str) -> None:
    """Write sentences manifest to JSON file."""
    data = []
    for gi, idxs in enumerate(groups):
        text = " ".join((segments[k].text or "").strip() for k in idxs)
        text = " ".join(text.split())
        windows = [{"start": float(segments[k].start), "end": float(segments[k].end)} for k in idxs]
        data.append({"id": gi, "text": text, "windows": windows})
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
