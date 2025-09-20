# Async API Reference

This document provides detailed API reference for the asynchronous processing components.

## Core Async Modules

### `tts_async.py`

#### `tts_speak_openai_async(client, text, model, voice, out_path, instructions=None)`

Asynchronously synthesize speech using OpenAI TTS.

**Parameters:**

- `client` (AsyncOpenAI): OpenAI async client
- `text` (str): Text to synthesize
- `model` (str): TTS model name
- `voice` (str): Voice identifier
- `out_path` (str): Output file path
- `instructions` (str, optional): Voice instructions

**Example:**

```python
import asyncio
from openai import AsyncOpenAI
from dubber.tts_async import tts_speak_openai_async

async def synthesize():
    client = AsyncOpenAI(api_key="your-key")
    await tts_speak_openai_async(
        client=client,
        text="Hello, world!",
        model="gpt-4o-mini-tts",
        voice="onyx",
        out_path="output.wav"
    )

asyncio.run(synthesize())
```

#### `process_tts_batch_async(client, texts, model, voice, output_paths, instructions=None, max_concurrent=5)`

Process multiple TTS requests asynchronously with concurrency control.

**Parameters:**

- `client` (AsyncOpenAI): OpenAI async client
- `texts` (List[str]): List of texts to synthesize
- `model` (str): TTS model name
- `voice` (str): Voice identifier
- `output_paths` (List[str]): List of output file paths
- `instructions` (str, optional): Voice instructions
- `max_concurrent` (int): Maximum concurrent requests

**Returns:**

- `List[str]`: List of output file paths

**Example:**

```python
texts = ["Hello", "World", "Async TTS"]
outputs = ["hello.wav", "world.wav", "async.wav"]

results = await process_tts_batch_async(
    client=client,
    texts=texts,
    model="gpt-4o-mini-tts",
    voice="onyx",
    output_paths=outputs,
    max_concurrent=3
)
```

#### `make_synth_openai_batch_async(client, model, voice, instructions=None, max_concurrent=5)`

Create an async batch TTS synthesis function for OpenAI.

**Parameters:**

- `client` (AsyncOpenAI): OpenAI async client
- `model` (str): TTS model name
- `voice` (str): Voice identifier
- `instructions` (str, optional): Voice instructions
- `max_concurrent` (int): Maximum concurrent requests

**Returns:**

- `Callable`: Async batch synthesis function

**Example:**

```python
async_synth = make_synth_openai_batch_async(
    client=client,
    model="gpt-4o-mini-tts",
    voice="onyx",
    max_concurrent=5
)

# Use with timeline building
results = await async_synth(texts, output_paths)
```

### `timeline_async.py`

#### `build_timeline_sentences_async(segments, sentence_groups, tmp_dir, voice_tag, synth_func_async, sample_rate=24000, cache_sig=None, no_tts=False)`

Build timeline using sentence groups with async TTS processing.

**Parameters:**

- `segments` (List[Segment]): List of segments
- `sentence_groups` (List[List[int]]): Sentence grouping indices
- `tmp_dir` (str): Temporary directory for audio files
- `voice_tag` (str): Voice identifier tag
- `synth_func_async` (Callable): Async synthesis function
- `sample_rate` (int): Audio sample rate
- `cache_sig` (tuple, optional): Cache signature
- `no_tts` (bool): Skip TTS processing

**Returns:**

- `AudioSegment`: Built audio timeline

**Example:**

```python
from dubber.timeline_async import build_timeline_sentences_async

timeline = await build_timeline_sentences_async(
    segments=segments,
    sentence_groups=groups,
    tmp_dir="tmp",
    voice_tag="main",
    synth_func_async=async_synth,
    sample_rate=24000
)
```

#### `build_timeline_wav_async(segments, tmp_dir, voice_tag, synth_func_async, sample_rate=24000, strict_timing=False, tolerance_ms=30, fit_mode="pad-or-speedup", cache_sig=None, no_tts=False)`

Build timeline using individual segments with async processing.

**Parameters:**

- `segments` (List[Segment]): List of segments
- `tmp_dir` (str): Temporary directory
- `voice_tag` (str): Voice identifier tag
- `synth_func_async` (Callable): Async synthesis function
- `sample_rate` (int): Audio sample rate
- `strict_timing` (bool): Enable strict timing
- `tolerance_ms` (int): Timing tolerance in milliseconds
- `fit_mode` (str): Timing fit mode
- `cache_sig` (tuple, optional): Cache signature
- `no_tts` (bool): Skip TTS processing

**Returns:**

- `AudioSegment`: Built audio timeline

### `cli_async.py`

#### `main_async()`

Main async CLI entry point.

**Usage:**

```bash
python -m dubber.cli_async --input_video video.mp4 --stage synth --async-tts
```

**Key Arguments:**

- `--async-tts`: Enable async TTS processing
- `--max-concurrent`: Set maximum concurrent requests
- `--input_video`: Input video file
- `--workdir`: Working directory
- `--stage`: Processing stage (prep, synth, burn, youtube)

## Error Handling

### Common Exceptions

#### `RuntimeError: asyncio.run() cannot be called from a running event loop`

**Cause:** Nested event loop calls
**Solution:** Use `asyncio.create_task()` instead of `asyncio.run()`

#### `TypeError: 'async for' requires an object with __aiter__ method`

**Cause:** Incorrect usage of `tqdm.as_completed()`
**Solution:** Use `for` instead of `async for` with `tqdm.as_completed()`

#### `httpx.ConnectError: Connection timeout`

**Cause:** Network issues or high concurrency
**Solution:** Reduce `max_concurrent` or check internet connection

### Error Recovery

```python
import asyncio
from dubber.tts_async import process_tts_batch_async

async def robust_tts_processing(texts, outputs, max_retries=3):
    for attempt in range(max_retries):
        try:
            results = await process_tts_batch_async(
                client=client,
                texts=texts,
                model="gpt-4o-mini-tts",
                voice="onyx",
                output_paths=outputs,
                max_concurrent=5
            )
            return results
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## Performance Tuning

### Concurrency Guidelines

| System                         | Recommended MAX_CONCURRENT | Notes                 |
| ------------------------------ | -------------------------- | --------------------- |
| Slow connection (<10 Mbps)     | 2-3                        | Conservative approach |
| Medium connection (10-50 Mbps) | 5                          | Default setting       |
| Fast connection (>50 Mbps)     | 8-10                       | Aggressive processing |
| High API quota                 | 10-15                      | Maximum throughput    |

### Memory Management

```python
# Process in smaller batches for large videos
batch_size = 50
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_outputs = outputs[i:i+batch_size]

    await process_tts_batch_async(
        client=client,
        texts=batch_texts,
        output_paths=batch_outputs,
        max_concurrent=5
    )
```

### Progress Monitoring

```python
from tqdm.asyncio import tqdm

async def monitor_progress(tasks):
    results = []
    async for result in tqdm.as_completed(tasks, desc="Processing"):
        results.append(await result)
    return results
```

## Integration Examples

### Custom Async Pipeline

```python
import asyncio
from openai import AsyncOpenAI
from dubber.tts_async import make_synth_openai_batch_async
from dubber.timeline_async import build_timeline_sentences_async

async def custom_async_pipeline(video_path, segments, groups):
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key="your-key")

    # Create async TTS function
    async_synth = make_synth_openai_batch_async(
        client=client,
        model="gpt-4o-mini-tts",
        voice="onyx",
        max_concurrent=5
    )

    # Build timeline asynchronously
    timeline = await build_timeline_sentences_async(
        segments=segments,
        sentence_groups=groups,
        tmp_dir="tmp",
        voice_tag="main",
        synth_func_async=async_synth
    )

    return timeline

# Usage
timeline = asyncio.run(custom_async_pipeline("video.mp4", segments, groups))
```

### Batch Video Processing

```python
import asyncio
from pathlib import Path

async def process_videos_async(video_paths):
    tasks = []

    for video_path in video_paths:
        task = asyncio.create_task(process_single_video(video_path))
        tasks.append(task)

    # Process all videos concurrently
    results = await asyncio.gather(*tasks)
    return results

async def process_single_video(video_path):
    # Your video processing logic here
    pass

# Usage
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = asyncio.run(process_videos_async(videos))
```

## Testing

### Unit Tests

```python
import pytest
import asyncio
from dubber.tts_async import tts_speak_openai_async

@pytest.mark.asyncio
async def test_async_tts():
    client = AsyncOpenAI(api_key="test-key")

    await tts_speak_openai_async(
        client=client,
        text="Test",
        model="gpt-4o-mini-tts",
        voice="onyx",
        out_path="test.wav"
    )

    assert Path("test.wav").exists()
```

### Performance Tests

```python
import time
import asyncio
from dubber.tts_async import process_tts_batch_async

async def benchmark_async_tts(texts, max_concurrent):
    start_time = time.time()

    results = await process_tts_batch_async(
        client=client,
        texts=texts,
        model="gpt-4o-mini-tts",
        voice="onyx",
        output_paths=[f"output_{i}.wav" for i in range(len(texts))],
        max_concurrent=max_concurrent
    )

    end_time = time.time()
    return end_time - start_time

# Benchmark different concurrency levels
for concurrency in [1, 3, 5, 10]:
    duration = asyncio.run(benchmark_async_tts(texts, concurrency))
    print(f"Concurrency {concurrency}: {duration:.2f}s")
```
