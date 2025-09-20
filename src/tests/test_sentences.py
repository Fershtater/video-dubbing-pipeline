"""
Tests for sentence processing.
"""

from src.dubber.models import Segment
from src.dubber.sentences import pack_sentences_into_chunks, split_segments_into_sentences


def test_split_segments_into_sentences():
    """Test sentence splitting logic."""
    segments = [
        Segment(start=0.0, end=1.0, text="Hello world."),
        Segment(start=1.5, end=2.5, text="This is a test."),
        Segment(start=3.0, end=4.0, text="Goodbye!"),
        Segment(start=4.5, end=5.5, text="Another sentence."),
    ]

    sentence_groups = split_segments_into_sentences(segments)

    # Should create separate sentences due to punctuation
    assert len(sentence_groups) == 4
    assert sentence_groups[0] == [0]
    assert sentence_groups[1] == [1]
    assert sentence_groups[2] == [2]
    assert sentence_groups[3] == [3]


def test_pack_sentences_into_chunks():
    """Test sentence chunking logic."""
    # Create mock sentence groups (lists of segment indices)
    sentence_groups = [[0], [1], [2], [3]]
    segments = [
        Segment(start=0.0, end=1.0, text="Hello world."),
        Segment(start=1.5, end=2.5, text="This is a test."),
        Segment(start=3.0, end=4.0, text="Goodbye!"),
        Segment(start=4.5, end=5.5, text="Another sentence."),
    ]

    chunks = pack_sentences_into_chunks(sentence_groups, segments, sentences_per_chunk=2)

    # Should pack 2 sentences per chunk
    assert len(chunks) == 2
    assert chunks[0] == [0, 1]
    assert chunks[1] == [2, 3]


def test_no_dangling_continuations():
    """Test that sentences don't end with dangling continuations."""
    segments = [
        Segment(start=0.0, end=1.0, text="We'll"),
        Segment(start=1.1, end=2.0, text="also"),
        Segment(start=2.1, end=3.0, text="cover"),
        Segment(start=3.1, end=4.0, text="advanced topics."),
    ]

    sentence_groups = split_segments_into_sentences(segments)

    # Should group "We'll also cover advanced topics." as one sentence
    assert len(sentence_groups) == 1
    assert sentence_groups[0] == [0, 1, 2, 3]
