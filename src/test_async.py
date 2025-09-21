#!/usr/bin/env python3
"""
Test script to demonstrate async TTS improvements.
"""

import asyncio
import time
from pathlib import Path

from dubber.cli_async import main_async
from dubber.cli import main as main_sync


def test_sync_vs_async():
    """Compare sync vs async TTS performance."""
    print("=== Testing Sync vs Async TTS Performance ===")

    # Test parameters
    video = "media/3.mp4"
    workdir = ".work/test_sync"
    workdir_async = ".work/test_async"
    output_sync = "out/test_sync.mp4"
    output_async = "out/test_async.mp4"

    # Ensure we have prep data
    print("Running prep stage...")
    import subprocess
    subprocess.run([
        "poetry", "run", "python", "-m", "dubber.cli",
        "--input_video", video,
        "--workdir", workdir,
        "--stage", "prep",
        "--stt", "local",
        "--skip-polish",
        "--align-mode", "sentence",
        "--sentence-join-gap", "1.2",
        "--sentences-per-chunk", "2",
        "--verbose"
    ])

    # Copy prep data to async workdir
    import shutil
    if Path(workdir).exists():
        shutil.copytree(workdir, workdir_async, dirs_exist_ok=True)

    print("\n=== Testing Sync TTS ===")
    start_time = time.time()

    # Run sync version
    import sys
    sys.argv = [
        "dubber.cli",
        "--input_video", video,
        "--workdir", workdir,
        "--stage", "synth",
        "--stt", "local",
        "--align-mode", "sentence",
        "--sentence-join-gap", "1.2",
        "--sentences-per-chunk", "2",
        "--tts-provider", "openai",
        "--tts-model", "gpt-4o-mini-tts",
        "--voice-main", "onyx",
        "--output", output_sync,
        "--verbose"
    ]

    main_sync()
    sync_time = time.time() - start_time
    print(f"Sync TTS completed in {sync_time:.2f} seconds")

    print("\n=== Testing Async TTS ===")
    start_time = time.time()

    # Run async version
    sys.argv = [
        "dubber.cli_async",
        "--input_video", video,
        "--workdir", workdir_async,
        "--stage", "synth",
        "--stt", "local",
        "--align-mode", "sentence",
        "--sentence-join-gap", "1.2",
        "--sentences-per-chunk", "2",
        "--tts-provider", "openai",
        "--tts-model", "gpt-4o-mini-tts",
        "--voice-main", "onyx",
        "--output", output_async,
        "--async-tts",
        "--max-concurrent", "5",
        "--verbose"
    ]

    asyncio.run(main_async())
    async_time = time.time() - start_time
    print(f"Async TTS completed in {async_time:.2f} seconds")

    print("\n=== Performance Comparison ===")
    print(f"Sync time:   {sync_time:.2f}s")
    print(f"Async time:  {async_time:.2f}s")
    print(f"Speedup:     {sync_time/async_time:.2f}x")
    print(f"Time saved:  {sync_time - async_time:.2f}s")


if __name__ == "__main__":
    test_sync_vs_async()
