"""
Video Dubbing Pipeline - Modular video dubbing with TTS and subtitles.

A comprehensive pipeline for:
- Extracting audio from videos
- Transcribing speech (local faster-whisper or OpenAI Whisper)
- Polishing transcripts with GPT
- Synthesizing speech with OpenAI or ElevenLabs TTS
- Building aligned audio timelines
- Generating subtitles and YouTube descriptions
"""

__version__ = "0.1.0"
