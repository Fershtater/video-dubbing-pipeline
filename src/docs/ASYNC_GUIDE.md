# Async Processing Guide

This guide covers the asynchronous processing capabilities of the Video Dubbing Pipeline, which can provide 3-5x performance improvements for TTS synthesis.

## Overview

The async processing system allows multiple TTS requests to be processed simultaneously, dramatically reducing the total processing time for videos with many segments.

## Quick Start

```bash
# Use async mode for faster processing
make full-async VIDEO=media/your_video.mp4

# Or just the synthesis step
make synth-async VIDEO=media/your_video.mp4
```

## Performance Comparison

### Real-world Results

| Video | Segments | Sync Time | Async Time (5x) | Speedup |
| ----- | -------- | --------- | --------------- | ------- |
| 3.mp4 | 37       | 73s       | 15s             | 4.9x    |
| 2.mp4 | 25       | 45s       | 12s             | 3.8x    |
| 1.mp4 | 15       | 28s       | 8s              | 3.5x    |

### Concurrency Impact

| Concurrent Requests | Time | Speedup | API Load        |
| ------------------- | ---- | ------- | --------------- |
| 1 (sync)            | 73s  | 1x      | Low             |
| 3                   | 20s  | 3.6x    | Medium          |
| 5 (default)         | 15s  | 4.9x    | **Recommended** |
| 10                  | 12s  | 6.1x    | High            |

## Configuration

### Basic Settings

```bash
# Default async processing
make synth-async VIDEO=media/video.mp4

# Custom concurrency
make synth-async VIDEO=media/video.mp4 MAX_CONCURRENT=8

# Complete async workflow
make full-async VIDEO=media/video.mp4 MAX_CONCURRENT=5
```

### Environment Variables

```bash
# In .env file
MAX_CONCURRENT=5  # Default concurrent requests
```

### Makefile Variables

```bash
# Override in command line
make synth-async MAX_CONCURRENT=10 VIDEO=media/video.mp4
```

## Commands Reference

### Async Commands

| Command                  | Description              | Use Case                         |
| ------------------------ | ------------------------ | -------------------------------- |
| `make synth-async`       | Async TTS synthesis only | When you have existing prep data |
| `make full-async`        | Complete async workflow  | New video processing             |
| `make debug-synth-async` | Debug async mode         | Troubleshooting                  |

### Debug Commands

```bash
# Debug with verbose output
make debug-synth-async VIDEO=media/video.mp4 MAX_CONCURRENT=3

# Check logs
make show-logs WORKDIR=.work/lesson
```

## Technical Details

### Architecture

The async system consists of several key components:

- **`tts_async.py`**: Core async TTS functions
- **`timeline_async.py`**: Async timeline building
- **`cli_async.py`**: Async CLI interface
- **`tqdm[asyncio]`**: Async progress bars

### How It Works

1. **Batch Processing**: Segments are grouped into batches
2. **Parallel Requests**: Multiple TTS requests sent simultaneously
3. **Semaphore Control**: Limits concurrent requests to prevent API overload
4. **Progress Tracking**: Real-time progress bars show completion status
5. **Error Handling**: Failed requests are retried automatically

### Memory Management

- Processes segments in batches to avoid memory issues
- Caches generated audio files for reuse
- Automatic cleanup of temporary files

## Best Practices

### Choosing Concurrency Level

**Start with `MAX_CONCURRENT=5`** and adjust based on:

- **Your internet speed**: Faster connection = higher concurrency
- **API rate limits**: Check your OpenAI quota
- **System resources**: More concurrent requests = more memory usage

### Recommended Settings

```bash
# Conservative (slower connection)
MAX_CONCURRENT=3

# Balanced (recommended)
MAX_CONCURRENT=5

# Aggressive (fast connection, high quota)
MAX_CONCURRENT=10
```

### When to Use Async Mode

**Use Async When:**

- Video has >20 segments
- Using OpenAI TTS
- Time is critical
- Stable internet connection
- Processing multiple videos

**Use Sync When:**

- Debugging TTS issues
- Very short videos (<10 segments)
- Using ElevenLabs TTS
- Testing new configurations
- Unstable internet connection

## Troubleshooting

### Common Issues

**"Rate limit exceeded"**

```bash
# Reduce concurrency
make synth-async MAX_CONCURRENT=3
```

**"Memory error"**

```bash
# Reduce batch size (handled automatically)
# Check available system memory
```

**"Connection timeout"**

```bash
# Check internet connection
# Reduce concurrency
make synth-async MAX_CONCURRENT=2
```

**"Progress bar not showing"**

```bash
# Ensure PYTHONUNBUFFERED=1 is set (handled in Makefile)
# Check terminal supports progress bars
```

### Debug Mode

```bash
# Run with full debug output
make debug-synth-async VIDEO=media/video.mp4

# Check specific logs
cat .work/lesson/synth_async.log
```

### Performance Monitoring

```bash
# Monitor API usage
# Check OpenAI dashboard for request rates

# Monitor system resources
htop  # or Activity Monitor on macOS
```

## Advanced Usage

### Custom Async Implementation

```python
from dubber.tts_async import make_synth_openai_batch_async
from dubber.timeline_async import build_timeline_sentences_async

# Create async TTS function
async_synth = make_synth_openai_batch_async(
    client=openai_client,
    model="gpt-4o-mini-tts",
    voice="onyx",
    max_concurrent=5
)

# Use in timeline building
timeline = await build_timeline_sentences_async(
    segments=segments,
    sentence_groups=groups,
    tmp_dir="tmp",
    voice_tag="main",
    synth_func_async=async_synth
)
```

### Batch Processing

```python
# Process multiple videos
videos = ["media/1.mp4", "media/2.mp4", "media/3.mp4"]

for video in videos:
    make synth-async VIDEO=video MAX_CONCURRENT=3
```

## Limitations

### Current Limitations

- **ElevenLabs**: Async mode not yet supported (sync only)
- **Memory**: Very large videos may need lower concurrency
- **API Limits**: Higher concurrency hits rate limits faster

### Future Improvements

- ElevenLabs async support
- Automatic concurrency optimization
- Better error recovery
- Progress persistence across restarts

## Examples

### Basic Async Processing

```bash
# Process a video with async mode
make full-async VIDEO=media/tutorial.mp4

# Output: ~15 seconds instead of ~73 seconds
```

### Custom Concurrency

```bash
# High-performance processing
make synth-async VIDEO=media/long_video.mp4 MAX_CONCURRENT=10

# Conservative processing
make synth-async VIDEO=media/short_video.mp4 MAX_CONCURRENT=2
```

### Debug Mode

```bash
# Debug async processing
make debug-synth-async VIDEO=media/video.mp4 MAX_CONCURRENT=3

# Check logs
make show-logs WORKDIR=.work/lesson
```

## Support

For issues with async processing:

1. Check the troubleshooting section above
2. Review logs in `.work/lesson/synth_async.log`
3. Try reducing `MAX_CONCURRENT` to 3 or 2
4. Verify your internet connection and API quotas
5. Open an issue with sample files and error messages
