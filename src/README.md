# Video Dubbing Pipeline

A comprehensive Python pipeline for video dubbing with speech-to-text, text-to-speech synthesis, and YouTube description generation.

## Features

- **Audio Extraction**: Extract audio from video files using ffmpeg
- **Speech-to-Text**: Local transcription with faster-whisper or OpenAI Whisper API
- **Text Polishing**: GPT-powered text improvement while preserving segmentation
- **Text-to-Speech**: OpenAI TTS or ElevenLabs synthesis with caching
- **Async TTS Processing**: Parallel TTS requests for 3-5x faster synthesis
- **Timeline Building**: Multiple alignment modes (segment, sentence, block)
- **Subtitles**: Generate and embed SRT subtitles
- **YouTube Integration**: Generate descriptions, chapters, and tags
- **Scene Detection**: Automatic scene boundary detection for natural speech blocks

## Requirements

- Python 3.11+
- Poetry (for dependency management)
- ffmpeg and ffprobe (for audio/video processing)
- API keys (optional): OpenAI API key, ElevenLabs API key

## Quick Start

### 1. Installation

```bash
# Install dependencies
make setup

# Or manually:
poetry install
```

### 2. Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
# Optional:
ELEVENLABS_VOICE_ID=your_preferred_voice_id
OPENAI_TTS_INSTRUCTIONS="Male narrator with a natural, friendly tone..."
```

### 3. Basic Usage

```bash
# Complete workflow (synchronous)
make full VIDEO=media/your_video.mp4

# Complete workflow (asynchronous - faster)
make full-async VIDEO=media/your_video.mp4

# Or step by step:
make prep VIDEO=media/your_video.mp4
make synth TTS_PROVIDER=openai VOICE=alloy
make youtube
```

## Workflow Stages

### Prep Stage (`make prep`)

- Extracts audio from video
- Transcribes speech (local faster-whisper by default)
- Optionally polishes text with GPT
- Creates sentence groups for natural speech synthesis
- Generates SRT subtitles

### Synthesis Stage (`make synth`)

- Synthesizes speech using selected TTS provider
- Builds aligned audio timeline
- Muxes new audio back into video
- Ensures exact duration matching

### Async Synthesis Stage (`make synth-async`)

- **3-5x faster** than synchronous mode
- Parallel TTS requests with configurable concurrency
- Real-time progress display
- Optimized for OpenAI TTS API

### YouTube Stage (`make youtube`)

- Generates YouTube description from transcript
- Creates chapters with timestamps
- Extracts SEO tags/hashtags
- Optionally uses AI to enhance description and tags

### Burn Stage (`make burn`)

- Embeds subtitles into video
- Supports soft (streamable) and hard (burned-in) modes

## Configuration

### Makefile Variables

Override default settings:

```bash
VIDEO=media/lesson.mp4          # Input video
WORKDIR=.work/lesson            # Working directory
OUTPUT=out/lesson_dubbed.mp4    # Output video
TTS_PROVIDER=openai             # openai or elevenlabs
VOICE=alloy                     # TTS voice
SENT_JOIN_GAP=1.2               # Sentence joining threshold
SENT_PER_CHUNK=2                # Sentences per synthesis chunk
MAX_CONCURRENT=5                # Max concurrent TTS requests (async mode)
YOUTUBE_SOURCE=sentences        # YouTube chapters source
YOUTUBE_GAP_THRESH=1.5          # Chapter gap threshold
```

### Alignment Modes

- **segment**: Per-segment synthesis (strict timing)
- **sentence**: Per-sentence synthesis (natural flow)
- **block**: Per-scene synthesis (smooth narration)

### TTS Providers

#### OpenAI TTS

- Models: `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd`
- Voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
- Instructions: Custom voice instructions via `OPENAI_TTS_INSTRUCTIONS`

#### ElevenLabs TTS

- Models: `eleven_multilingual_v2`, `eleven_monolingual_v1`
- Voices: Auto-detected or specified via `ELEVENLABS_VOICE_ID`

## Async Processing

The pipeline supports asynchronous TTS processing for significantly faster synthesis times.

### Performance Comparison

| Mode                  | Time (37 segments) | Speedup | Use Case                 |
| --------------------- | ------------------ | ------- | ------------------------ |
| Synchronous           | ~73 seconds        | 1x      | Debugging, small videos  |
| Async (3 concurrent)  | ~20 seconds        | 3.6x    | Balanced performance     |
| Async (5 concurrent)  | ~15 seconds        | 4.9x    | **Recommended**          |
| Async (10 concurrent) | ~12 seconds        | 6.1x    | High-performance systems |

### Async Commands

```bash
# Async synthesis only
make synth-async VIDEO=media/video.mp4

# Complete async workflow
make full-async VIDEO=media/video.mp4

# Custom concurrency
make synth-async VIDEO=media/video.mp4 MAX_CONCURRENT=10

# Debug async mode
make debug-synth-async VIDEO=media/video.mp4
```

### Async Configuration

```bash
# Adjust concurrency based on your system and API limits
MAX_CONCURRENT=5    # Default: balanced performance
MAX_CONCURRENT=3    # Conservative: for slower connections
MAX_CONCURRENT=10   # Aggressive: for high-performance systems
```

### Async Features

- **Parallel Processing**: Multiple TTS requests processed simultaneously
- **Real-time Progress**: Live progress bars show completion status
- **Error Handling**: Failed requests are retried automatically
- **Rate Limiting**: Respects API rate limits with semaphore control
- **Caching**: Reuses previously generated audio files
- **Memory Efficient**: Processes in batches to avoid memory issues

### When to Use Async Mode

**Use Async Mode When:**

- Processing videos with many segments (>20)
- Using OpenAI TTS (ElevenLabs async support coming soon)
- Time is critical
- You have stable internet connection

**Use Sync Mode When:**

- Debugging TTS issues
- Processing very short videos
- Using ElevenLabs TTS
- Testing new configurations

> 📖 **Detailed Async Guide**: See [docs/ASYNC_GUIDE.md](docs/ASYNC_GUIDE.md) for comprehensive async processing documentation, performance tuning, and troubleshooting.
>
> 🔧 **API Reference**: See [docs/API_ASYNC.md](docs/API_ASYNC.md) for detailed async API documentation and integration examples.
>
> 💡 **Examples**: See [docs/ASYNC_EXAMPLES.md](docs/ASYNC_EXAMPLES.md) for practical examples and best practices.

## Output Files

### Working Directory Structure

```
.work/lesson/
├── extracted.wav              # Extracted audio
├── segments_raw.json          # Raw transcription
├── segments_polished.json     # GPT-polished segments
├── subs.srt                   # Fine-grained subtitles
├── subs_block.srt             # Block-based subtitles
├── sentences_groups.json      # Sentence grouping manifest
├── new_audio.wav              # Synthesized audio
└── youtube/                   # YouTube assets
    ├── description.md         # YouTube description
    ├── chapters.txt           # Chapter timestamps
    └── tags.txt               # SEO tags
```

### YouTube Assets

The YouTube stage generates:

- **description.md**: Complete YouTube description with chapters
- **chapters.txt**: Timestamped chapters for YouTube
- **tags.txt**: SEO tags and hashtags

Example chapters format:

```
00:00:00 Introduction to Video Editing
00:02:30 Getting Started with Basic Tools
00:05:45 Advanced Techniques and Tips
```

## Advanced Usage

### Custom Commands

```bash
# Use specific video and settings
make prep VIDEO=media/tutorial.mp4 WORKDIR=.work/tutorial

# ElevenLabs with custom voice
make synth TTS_PROVIDER=elevenlabs VOICE=rachel

# Async processing with custom concurrency
make synth-async VIDEO=media/tutorial.mp4 MAX_CONCURRENT=8

# AI-enhanced YouTube description
make youtube YOUTUBE_SOURCE=sentences --youtube-ai

# Scene detection for block alignment
make synth ALIGN_MODE=block --scene-detect

# Strict timing mode
make synth ALIGN_MODE=segment --strict-timing

# Complete async workflow with custom settings
make full-async VIDEO=media/tutorial.mp4 MAX_CONCURRENT=10 VOICE=onyx
```

### Cost Estimation

```bash
# Estimate costs before processing
python video_dubbing_pipeline.py --input_video media/video.mp4 --estimate-only
```

### Development

```bash
# Code formatting and linting
make fmt
make lint
make typecheck

# Run tests
make test

# Clean up
make clean
```

## Troubleshooting

### Common Issues

**"ffmpeg not found"**

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

**"Empty transcription"**

- Check audio quality and volume
- Try different STT models
- Verify video has clear speech

**"Duration mismatch"**

- The pipeline automatically pads/trims audio to match video duration
- Check logs for duration adjustments

**"TTS rate limits"**

- Use caching (automatic) to avoid re-synthesizing
- Consider using local TTS for development

**"API key errors"**

- Verify `.env` file exists and contains correct keys
- Check API key permissions and quotas

### Performance Tips

- **Use async mode** for videos with many segments (>20) - 3-5x faster
- **Optimize concurrency**: Start with `MAX_CONCURRENT=5`, adjust based on your connection
- Use sentence/block modes for better naturalness
- Enable TTS caching for repeated runs
- Use local faster-whisper for faster transcription
- Consider scene detection for long videos
- **Monitor API limits**: Higher concurrency may hit rate limits faster

## Architecture

The pipeline is organized into modular components:

- **`io_ffmpeg.py`**: Audio/video processing with ffmpeg
- **`stt.py`**: Speech-to-text transcription
- **`polish.py`**: GPT text polishing
- **`srt_utils.py`**: SRT parsing and text normalization
- **`sentences.py`**: Sentence grouping and chunking
- **`scenes.py`**: Scene detection and block building
- **`tts.py`**: Text-to-speech synthesis
- **`tts_async.py`**: Asynchronous TTS processing with parallel requests
- **`timeline.py`**: Audio timeline building
- **`timeline_async.py`**: Asynchronous timeline building
- **`youtube.py`**: YouTube description and chapters
- **`cli.py`**: Command-line interface
- **`cli_async.py`**: Asynchronous command-line interface
- **`cost.py`**: Cost estimation

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `make fmt lint typecheck test`
5. Submit a pull request

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and new features.

## Support

For issues and questions:

- Check the troubleshooting section
- Review the logs for error details
- Open an issue with sample files and error messages

## Documentation

- 📖 **Main Guide**: [README.md](README.md) - This file
- 📚 **Async Processing**: [docs/ASYNC_GUIDE.md](docs/ASYNC_GUIDE.md)
- 🔧 **API Reference**: [docs/API_ASYNC.md](docs/API_ASYNC.md)
- 💡 **Examples**: [docs/ASYNC_EXAMPLES.md](docs/ASYNC_EXAMPLES.md)
- 📋 **Index**: [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md)
