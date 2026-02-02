"""
Data models for the video dubbing pipeline.
"""

from dataclasses import dataclass


@dataclass
class Segment:
    """A single transcribed segment with timing and text."""

    start: float  # seconds
    end: float  # seconds
    text: str


@dataclass
class Block:
    """A block of content (multiple segments) with timing and text."""

    start: float
    end: float
    text: str


@dataclass
class Chapter:
    """A YouTube chapter with timing and title."""

    start_time: str  # HH:MM:SS format
    title: str


@dataclass
class YouTubeAssets:
    """YouTube description and metadata."""

    title: str
    description: str
    chapters: list[Chapter]
