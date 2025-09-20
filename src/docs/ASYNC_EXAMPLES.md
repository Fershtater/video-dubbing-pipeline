# Async Processing Examples

This document provides practical examples of using the async processing capabilities.

## Basic Examples

### 1. Simple Async Processing

```bash
# Process a single video with async mode
make full-async VIDEO=media/tutorial.mp4

# Output: ~15 seconds instead of ~73 seconds for 37 segments
```

### 2. Custom Concurrency

```bash
# High-performance processing
make synth-async VIDEO=media/long_video.mp4 MAX_CONCURRENT=10

# Conservative processing for slower connections
make synth-async VIDEO=media/video.mp4 MAX_CONCURRENT=3
```

### 3. Debug Mode

```bash
# Debug async processing with verbose output
make debug-synth-async VIDEO=media/video.mp4 MAX_CONCURRENT=5

# Check logs
make show-logs WORKDIR=.work/lesson
```

## Advanced Examples

### 1. Batch Video Processing

```bash
#!/bin/bash
# Process multiple videos with async mode

videos=("media/lesson1.mp4" "media/lesson2.mp4" "media/lesson3.mp4")

for video in "${videos[@]}"; do
    echo "Processing $video..."
    make full-async VIDEO="$video" MAX_CONCURRENT=5
    echo "Completed $video"
done
```

### 2. Performance Testing

```bash
#!/bin/bash
# Test different concurrency levels

video="media/test_video.mp4"
concurrency_levels=(1 3 5 8 10)

for level in "${concurrency_levels[@]}"; do
    echo "Testing concurrency level: $level"
    time make synth-async VIDEO="$video" MAX_CONCURRENT="$level"
    echo "---"
done
```

### 3. Custom Workflow

```bash
#!/bin/bash
# Custom async workflow with specific settings

VIDEO="media/custom_video.mp4"
WORKDIR=".work/custom"
OUTPUT="out/custom_dubbed.mp4"
MAX_CONCURRENT=8
VOICE="onyx"

# Prep stage
make prep VIDEO="$VIDEO" WORKDIR="$WORKDIR"

# Async synthesis with custom settings
make synth-async \
    VIDEO="$VIDEO" \
    WORKDIR="$WORKDIR" \
    OUTPUT="$OUTPUT" \
    MAX_CONCURRENT="$MAX_CONCURRENT" \
    VOICE="$VOICE"

# YouTube assets
make youtube VIDEO="$VIDEO" WORKDIR="$WORKDIR"
```

## Python Integration Examples

### 1. Basic Async Usage

```python
import asyncio
from openai import AsyncOpenAI
from dubber.tts_async import process_tts_batch_async

async def basic_async_example():
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key="your-api-key")

    # Prepare texts and outputs
    texts = [
        "Hello, welcome to our tutorial.",
        "Today we'll learn about async processing.",
        "This is much faster than sync mode."
    ]
    outputs = ["hello.wav", "welcome.wav", "tutorial.wav"]

    # Process asynchronously
    results = await process_tts_batch_async(
        client=client,
        texts=texts,
        model="gpt-4o-mini-tts",
        voice="onyx",
        output_paths=outputs,
        max_concurrent=5
    )

    print(f"Generated {len(results)} audio files")
    return results

# Run the example
asyncio.run(basic_async_example())
```

### 2. Custom Timeline Building

```python
import asyncio
from openai import AsyncOpenAI
from dubber.tts_async import make_synth_openai_batch_async
from dubber.timeline_async import build_timeline_sentences_async
from dubber.models import Segment

async def custom_timeline_example():
    # Initialize client
    client = AsyncOpenAI(api_key="your-api-key")

    # Create async TTS function
    async_synth = make_synth_openai_batch_async(
        client=client,
        model="gpt-4o-mini-tts",
        voice="onyx",
        max_concurrent=5
    )

    # Sample segments
    segments = [
        Segment(start=0.0, end=2.0, text="Hello, world!"),
        Segment(start=2.0, end=4.0, text="This is async processing."),
        Segment(start=4.0, end=6.0, text="Much faster than sync mode.")
    ]

    # Sentence groups (each group is synthesized together)
    sentence_groups = [[0], [1], [2]]

    # Build timeline asynchronously
    timeline = await build_timeline_sentences_async(
        segments=segments,
        sentence_groups=sentence_groups,
        tmp_dir="tmp",
        voice_tag="main",
        synth_func_async=async_synth
    )

    # Export the timeline
    timeline.export("output.wav", format="wav")
    print("Timeline built successfully!")

asyncio.run(custom_timeline_example())
```

### 3. Error Handling and Retry Logic

```python
import asyncio
import time
from dubber.tts_async import process_tts_batch_async

async def robust_async_processing(texts, outputs, max_retries=3):
    client = AsyncOpenAI(api_key="your-api-key")

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}")

            results = await process_tts_batch_async(
                client=client,
                texts=texts,
                model="gpt-4o-mini-tts",
                voice="onyx",
                output_paths=outputs,
                max_concurrent=5
            )

            print("Success!")
            return results

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt == max_retries - 1:
                print("All attempts failed!")
                raise

            # Exponential backoff
            wait_time = 2 ** attempt
            print(f"Waiting {wait_time} seconds before retry...")
            await asyncio.sleep(wait_time)

# Usage
texts = ["Text 1", "Text 2", "Text 3"]
outputs = ["output1.wav", "output2.wav", "output3.wav"]

try:
    results = asyncio.run(robust_async_processing(texts, outputs))
    print(f"Generated {len(results)} files")
except Exception as e:
    print(f"Failed to process: {e}")
```

### 4. Performance Monitoring

```python
import asyncio
import time
from dubber.tts_async import process_tts_batch_async

async def monitor_performance(texts, outputs, max_concurrent):
    start_time = time.time()

    print(f"Starting async processing with {max_concurrent} concurrent requests")
    print(f"Processing {len(texts)} segments")

    client = AsyncOpenAI(api_key="your-api-key")

    results = await process_tts_batch_async(
        client=client,
        texts=texts,
        model="gpt-4o-mini-tts",
        voice="onyx",
        output_paths=outputs,
        max_concurrent=max_concurrent
    )

    end_time = time.time()
    duration = end_time - start_time

    print(f"Completed in {duration:.2f} seconds")
    print(f"Average time per segment: {duration/len(texts):.2f} seconds")
    print(f"Throughput: {len(texts)/duration:.2f} segments/second")

    return results

# Test different concurrency levels
async def benchmark_concurrency():
    texts = [f"Segment {i}" for i in range(20)]
    outputs = [f"output_{i}.wav" for i in range(20)]

    concurrency_levels = [1, 3, 5, 8, 10]

    for level in concurrency_levels:
        print(f"\n--- Testing concurrency level {level} ---")
        await monitor_performance(texts, outputs, level)

asyncio.run(benchmark_concurrency())
```

## Makefile Integration Examples

### 1. Custom Makefile Target

```makefile
# Add to your Makefile
.PHONY: process-all-async

process-all-async:
	@echo "Processing all videos with async mode..."
	@for video in media/*.mp4; do \
		echo "Processing $$video..."; \
		make full-async VIDEO="$$video" MAX_CONCURRENT=5; \
		echo "Completed $$video"; \
	done
	@echo "All videos processed!"

# Usage: make process-all-async
```

### 2. Performance Testing Target

```makefile
# Add to your Makefile
.PHONY: benchmark-async

benchmark-async:
	@echo "Benchmarking async performance..."
	@echo "Testing different concurrency levels..."
	@for level in 1 3 5 8 10; do \
		echo "--- Testing concurrency level $$level ---"; \
		time make synth-async VIDEO=media/test.mp4 MAX_CONCURRENT=$$level; \
	done
	@echo "Benchmark completed!"

# Usage: make benchmark-async
```

### 3. Custom Async Workflow

```makefile
# Add to your Makefile
.PHONY: custom-async-workflow

custom-async-workflow: check-env check-files
	@echo "=== Custom Async Workflow ==="
	@echo "Video: $(VIDEO)"
	@echo "Concurrency: $(MAX_CONCURRENT)"
	@echo "Voice: $(VOICE)"

	# Prep stage
	@echo "Running prep stage..."
	@make prep VIDEO="$(VIDEO)" WORKDIR="$(WORKDIR)"

	# Async synthesis
	@echo "Running async synthesis..."
	@make synth-async VIDEO="$(VIDEO)" WORKDIR="$(WORKDIR)" MAX_CONCURRENT="$(MAX_CONCURRENT)" VOICE="$(VOICE)"

	# YouTube assets
	@echo "Generating YouTube assets..."
	@make youtube VIDEO="$(VIDEO)" WORKDIR="$(WORKDIR)"

	@echo "=== Custom Async Workflow Completed ==="

# Usage: make custom-async-workflow VIDEO=media/video.mp4 MAX_CONCURRENT=8 VOICE=onyx
```

## Troubleshooting Examples

### 1. Debug Async Issues

```bash
#!/bin/bash
# Debug script for async processing issues

VIDEO="media/problem_video.mp4"
WORKDIR=".work/debug"

echo "=== Debugging Async Processing ==="
echo "Video: $VIDEO"
echo "Workdir: $WORKDIR"

# Check if prep data exists
if [ ! -f "$WORKDIR/subs.srt" ]; then
    echo "Running prep stage first..."
    make prep VIDEO="$VIDEO" WORKDIR="$WORKDIR"
fi

# Try with low concurrency first
echo "Testing with low concurrency (2)..."
make debug-synth-async VIDEO="$VIDEO" WORKDIR="$WORKDIR" MAX_CONCURRENT=2

# Check logs
echo "=== Recent Logs ==="
make show-logs WORKDIR="$WORKDIR"

# Check output
if [ -f "out/lesson_dubbed.mp4" ]; then
    echo "✓ Output video created successfully"
    ls -la out/lesson_dubbed.mp4
else
    echo "✗ Output video not created"
fi
```

### 2. Performance Analysis

```bash
#!/bin/bash
# Analyze async performance

VIDEO="media/test_video.mp4"
LOG_FILE="performance.log"

echo "=== Async Performance Analysis ===" > "$LOG_FILE"

# Test different concurrency levels
for level in 1 3 5 8 10; do
    echo "Testing concurrency level: $level" | tee -a "$LOG_FILE"

    start_time=$(date +%s)
    make synth-async VIDEO="$VIDEO" MAX_CONCURRENT="$level" 2>&1 | tee -a "$LOG_FILE"
    end_time=$(date +%s)

    duration=$((end_time - start_time))
    echo "Duration: ${duration}s" | tee -a "$LOG_FILE"
    echo "---" | tee -a "$LOG_FILE"
done

echo "Performance analysis completed. Check $LOG_FILE for details."
```

## Best Practices

### 1. Start Conservative

```bash
# Start with low concurrency and increase gradually
make synth-async MAX_CONCURRENT=3  # Start here
make synth-async MAX_CONCURRENT=5  # If stable
make synth-async MAX_CONCURRENT=8  # If still stable
```

### 2. Monitor Resources

```bash
# Monitor system resources during processing
htop &  # or Activity Monitor on macOS
make synth-async VIDEO=media/video.mp4 MAX_CONCURRENT=5
```

### 3. Use Appropriate Concurrency

```bash
# For short videos (<10 segments)
make synth-async MAX_CONCURRENT=3

# For medium videos (10-30 segments)
make synth-async MAX_CONCURRENT=5

# For long videos (>30 segments)
make synth-async MAX_CONCURRENT=8
```

### 4. Handle Errors Gracefully

```bash
# Use error handling in scripts
if ! make synth-async VIDEO="$VIDEO" MAX_CONCURRENT=5; then
    echo "Async processing failed, trying with lower concurrency..."
    make synth-async VIDEO="$VIDEO" MAX_CONCURRENT=3
fi
```
