# Changelog

All notable changes to the Video Dubbing Pipeline project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Async TTS processing with 3-5x performance improvement
- Real-time progress bars for TTS synthesis
- Configurable concurrency control (MAX_CONCURRENT)
- New async CLI interface (`cli_async.py`)
- Async TTS functions (`tts_async.py`)
- Async timeline building (`timeline_async.py`)
- Comprehensive async documentation
- MIT License
- Contributing guidelines

### Changed

- Improved console output with real-time progress
- Enhanced Makefile with async targets
- Updated documentation structure
- Moved documentation to `docs/` directory

### Fixed

- Output buffering issues in Makefile
- Python output buffering with PYTHONUNBUFFERED=1
- Progress bar display in terminal

## [0.1.0] - 2025-09-19

### Added

- Initial release
- Basic video dubbing pipeline
- Speech-to-text transcription with faster-whisper
- Text-to-speech synthesis with OpenAI TTS
- Text polishing with GPT
- Timeline building and audio alignment
- Subtitle generation and burning
- YouTube description and chapters generation
- Makefile automation
- Poetry dependency management
- Comprehensive documentation

### Features

- **Audio Extraction**: Extract audio from video files using ffmpeg
- **Speech-to-Text**: Local transcription with faster-whisper
- **Text Polishing**: GPT-powered text improvement
- **Text-to-Speech**: OpenAI TTS synthesis with caching
- **Timeline Building**: Multiple alignment modes (segment, sentence, block)
- **Subtitles**: Generate and embed SRT subtitles
- **YouTube Integration**: Generate descriptions, chapters, and tags
- **Scene Detection**: Automatic scene boundary detection

### Technical Details

- Python 3.11+ support
- Poetry for dependency management
- FFmpeg for audio/video processing
- OpenAI API integration
- Modular architecture
- Comprehensive error handling
- Caching system for TTS synthesis

## [0.0.1] - 2025-09-19

### Added

- Project initialization
- Basic project structure
- Initial dependencies
- Basic documentation

---

## Legend

- **Added** - New features
- **Changed** - Changes to existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security improvements
