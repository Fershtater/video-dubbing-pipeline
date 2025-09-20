"""
Tests for YouTube generation functionality.
"""

from src.dubber.models import Segment
from src.dubber.youtube import build_chapters_from_srt, extract_keywords, format_time


def test_format_time():
    """Test time formatting."""

    assert format_time(0) == "00:00:00"
    assert format_time(65) == "00:01:05"
    assert format_time(3661) == "01:01:01"


def test_build_chapters_from_srt():
    """Test chapter building from SRT segments."""
    segments = [
        Segment(start=0.0, end=2.0, text="Introduction to the topic."),
        Segment(start=2.5, end=4.0, text="Let's get started."),
        Segment(start=5.0, end=7.0, text="Advanced techniques."),
    ]

    chapters = build_chapters_from_srt(segments, gap_thresh=1.5)

    # Should create chapters based on gaps and punctuation
    assert len(chapters) >= 1
    assert chapters[0].start_time == "00:00:00"
    assert "Introduction" in chapters[0].title


def test_extract_keywords():
    """Test keyword extraction."""
    segments = [
        Segment(start=0.0, end=2.0, text="Python programming tutorial for beginners."),
        Segment(start=2.0, end=4.0, text="Learn coding basics and advanced techniques."),
    ]

    keywords = extract_keywords(segments, use_gpt=False)

    # Should extract relevant keywords
    assert len(keywords) > 0
    # Common words should be filtered out
    assert "python" in [kw.lower() for kw in keywords] or "programming" in [
        kw.lower() for kw in keywords
    ]


def test_chapter_minimum_length():
    """Test that chapters meet minimum length requirements."""
    segments = [
        Segment(start=0.0, end=1.0, text="Short."),  # Too short
        Segment(start=1.1, end=2.0, text="Also short."),  # Too short
        Segment(
            start=2.1,
            end=25.0,
            text="This is a much longer segment that should create a proper chapter.",
        ),
    ]

    chapters = build_chapters_from_srt(segments, gap_thresh=0.5)

    # Should merge short chapters or ensure minimum length
    assert len(chapters) >= 1
    # Last chapter should be substantial
    final_chapter = chapters[-1]
    assert "longer segment" in final_chapter.title or "proper chapter" in final_chapter.title
