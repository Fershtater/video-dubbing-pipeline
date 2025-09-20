# Documentation Index

This document provides an overview of all available documentation for the Video Dubbing Pipeline.

## Main Documentation

### [../README.md](../README.md)

**Main project documentation**

- Quick start guide
- Feature overview
- Basic usage examples
- Configuration options
- Troubleshooting

## Async Processing Documentation

### [ASYNC_GUIDE.md](ASYNC_GUIDE.md)

**Comprehensive async processing guide**

- Performance comparison tables
- Configuration options
- Best practices
- Troubleshooting
- When to use async vs sync

### [API_ASYNC.md](API_ASYNC.md)

**Detailed async API reference**

- Function signatures
- Parameters and return values
- Code examples
- Error handling
- Integration patterns

### [ASYNC_EXAMPLES.md](ASYNC_EXAMPLES.md)

**Practical examples and use cases**

- Basic usage examples
- Advanced integration
- Performance testing
- Error handling patterns
- Best practices

## Quick Reference

### Commands

| Command                  | Description             | Use Case              |
| ------------------------ | ----------------------- | --------------------- |
| `make full`              | Complete sync workflow  | Standard processing   |
| `make full-async`        | Complete async workflow | **Faster processing** |
| `make synth-async`       | Async synthesis only    | When prep data exists |
| `make debug-synth-async` | Debug async mode        | Troubleshooting       |

### Configuration

| Variable         | Default               | Description                 |
| ---------------- | --------------------- | --------------------------- |
| `MAX_CONCURRENT` | 5                     | Max concurrent TTS requests |
| `VIDEO`          | media/1.mp4           | Input video file            |
| `WORKDIR`        | .work/lesson          | Working directory           |
| `OUTPUT`         | out/lesson_dubbed.mp4 | Output video file           |
| `TTS_PROVIDER`   | openai                | TTS provider                |
| `VOICE`          | onyx                  | TTS voice                   |

### Performance

| Mode        | Speed | Use Case                |
| ----------- | ----- | ----------------------- |
| Sync        | 1x    | Debugging, small videos |
| Async (3x)  | 3.6x  | Conservative            |
| Async (5x)  | 4.9x  | **Recommended**         |
| Async (10x) | 6.1x  | High-performance        |

## Getting Started

### 1. First Time Setup

```bash
# Install dependencies
make setup

# Configure API keys in .env
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 2. Basic Usage

```bash
# Process a video (async mode - recommended)
make full-async VIDEO=media/your_video.mp4

# Or step by step
make prep VIDEO=media/your_video.mp4
make synth-async VIDEO=media/your_video.mp4
make youtube
```

### 3. Custom Settings

```bash
# High-performance processing
make full-async VIDEO=media/video.mp4 MAX_CONCURRENT=10

# Conservative processing
make full-async VIDEO=media/video.mp4 MAX_CONCURRENT=3
```

## Troubleshooting

### Common Issues

1. **"Rate limit exceeded"**

   - Reduce `MAX_CONCURRENT`
   - Check API quota

2. **"Connection timeout"**

   - Check internet connection
   - Reduce concurrency

3. **"Progress bar not showing"**
   - Ensure `PYTHONUNBUFFERED=1` is set
   - Check terminal support

### Debug Commands

```bash
# Debug async processing
make debug-synth-async VIDEO=media/video.mp4

# Check logs
make show-logs WORKDIR=.work/lesson

# Show help
make help
```

## File Structure

```
src/
├── README.md                 # Main documentation
├── LICENSE                   # MIT License
├── docs/                     # Documentation directory
│   ├── README.md            # Documentation index
│   ├── ASYNC_GUIDE.md       # Comprehensive async guide
│   ├── API_ASYNC.md         # Async API reference
│   ├── ASYNC_EXAMPLES.md    # Practical examples
│   └── DOCS_INDEX.md        # This file
├── dubber/                  # Source code
│   ├── cli.py              # Sync CLI
│   ├── cli_async.py        # Async CLI
│   ├── tts.py              # Sync TTS
│   ├── tts_async.py        # Async TTS
│   ├── timeline.py         # Sync timeline
│   ├── timeline_async.py   # Async timeline
│   └── ...
└── Makefile                # Build automation
```

## Contributing

When adding new async features:

1. Update the relevant documentation files
2. Add examples to `ASYNC_EXAMPLES.md`
3. Update API reference in `API_ASYNC.md`
4. Test with different concurrency levels
5. Update this index

## Support

For questions about async processing:

1. Check the troubleshooting section in `ASYNC_GUIDE.md`
2. Review examples in `ASYNC_EXAMPLES.md`
3. Check API reference in `API_ASYNC.md`
4. Open an issue with sample files and error messages
