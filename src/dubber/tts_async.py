"""
Asynchronous Text-to-speech synthesis with OpenAI and ElevenLabs.
"""

import asyncio
import hashlib
import logging
from collections.abc import Callable
from typing import List
from tqdm.asyncio import tqdm

logger = logging.getLogger("dubber")

# Optional OpenAI SDK
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


def _hash_for_cache(provider: str, model: str, voice: str, text: str) -> str:
    """Generate cache hash for TTS audio."""
    key = f"{provider}|{model}|{voice}|{text}".encode()
    return hashlib.sha1(key).hexdigest()[:12]


async def tts_speak_openai_async(
    client: AsyncOpenAI,
    text: str,
    model: str,
    voice: str,
    out_path: str,
    instructions: str | None = None,
) -> None:
    """Asynchronously synthesize speech using OpenAI TTS."""
    if client is None:
        raise RuntimeError("OpenAI client is not initialized (missing OPENAI_API_KEY)")

    try:
        response = await client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="wav",
            instructions=instructions,
        )

        # Write the response content to file
        with open(out_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    except Exception as e:
        logger.error(f"OpenAI TTS failed for text '{text[:50]}...': {e}")
        raise


async def process_tts_batch_async(
    client: AsyncOpenAI,
    texts: List[str],
    model: str,
    voice: str,
    output_paths: List[str],
    instructions: str | None = None,
    max_concurrent: int = 5,
) -> List[str]:
    """Process multiple TTS requests asynchronously with concurrency limit."""

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single(text: str, out_path: str) -> str:
        async with semaphore:
            await tts_speak_openai_async(
                client, text, model, voice, out_path, instructions
            )
            return out_path

    # Create tasks for all TTS requests
    tasks = [
        process_single(text, out_path)
        for text, out_path in zip(texts, output_paths)
    ]

    # Process with progress bar
    results = []
    for result in tqdm.as_completed(tasks, desc="TTS Batch Processing"):
        results.append(await result)

    return results


def make_synth_openai_async(
    client: AsyncOpenAI,
    model: str,
    voice: str,
    instructions: str | None = None,
    max_concurrent: int = 5,
) -> Callable:
    """Create an async TTS synthesis function for OpenAI."""

    async def synth_func_async(text: str, out_path: str) -> None:
        """Async synthesis function."""
        await tts_speak_openai_async(
            client, text, model, voice, out_path, instructions
        )

    def sync_wrapper(text: str, out_path: str) -> None:
        """Sync wrapper for compatibility with existing code."""
        asyncio.run(synth_func_async(text, out_path))

    return sync_wrapper


def make_synth_openai_batch_async(
    client: AsyncOpenAI,
    model: str,
    voice: str,
    instructions: str | None = None,
    max_concurrent: int = 5,
) -> Callable:
    """Create an async batch TTS synthesis function for OpenAI."""

    async def batch_synth_func_async(texts: List[str], output_paths: List[str]) -> List[str]:
        """Async batch synthesis function."""
        return await process_tts_batch_async(
            client, texts, model, voice, output_paths, instructions, max_concurrent
        )

    async def async_wrapper(texts: List[str], output_paths: List[str]) -> List[str]:
        """Async wrapper for batch processing."""
        return await batch_synth_func_async(texts, output_paths)

    return async_wrapper
