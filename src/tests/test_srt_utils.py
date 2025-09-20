"""
Tests for SRT utilities.
"""

import tempfile

from src.dubber.models import Segment
from src.dubber.srt_utils import normalize_segments_by_punct, parse_srt, write_srt


def test_write_and_parse_srt():
    """Test SRT write/parse roundtrip."""
    segments = [
        Segment(start=0.0, end=2.5, text="Hello world."),
        Segment(start=2.5, end=5.0, text="This is a test."),
        Segment(start=5.0, end=7.5, text="Goodbye!"),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
        srt_path = f.name

    try:
        # Write SRT
        write_srt(segments, srt_path)

        # Parse SRT
        parsed_segments = parse_srt(srt_path)

        # Verify roundtrip
        assert len(parsed_segments) == 3
        assert parsed_segments[0].text == "Hello world."
        assert parsed_segments[1].text == "This is a test."
        assert parsed_segments[2].text == "Goodbye!"
        assert parsed_segments[0].start == 0.0
        assert parsed_segments[0].end == 2.5
    finally:
        import os

        os.unlink(srt_path)


def test_normalize_segments_by_punct():
    """Test segment normalization by punctuation."""
    segments = [
        Segment(start=0.0, end=5.0, text="Hello world. This is a test. Goodbye!"),
        Segment(start=5.0, end=7.0, text="Another sentence."),
    ]

    normalized = normalize_segments_by_punct(segments)

    # Should split the first segment into 3 parts
    assert len(normalized) == 4
    assert normalized[0].text == "Hello world."
    assert normalized[1].text == "This is a test."
    assert normalized[2].text == "Goodbye!"
    assert normalized[3].text == "Another sentence."

    # Check timing is distributed
    assert normalized[0].start == 0.0
    assert normalized[2].end == 5.0
    assert normalized[3].start == 5.0


def test_wrap_lines():
    """Test text wrapping functionality."""
    from src.dubber.srt_utils import wrap_lines

    text = "This is a very long line that should be wrapped into multiple lines"
    wrapped = wrap_lines(text, max_chars=20, max_lines=3)
    lines = wrapped.split("\n")

    assert len(lines) <= 3
    for line in lines:
        assert len(line) <= 20
