"""
Text-to-speech synthesis with OpenAI and ElevenLabs.
"""

import hashlib
import logging
from pathlib import Path
from collections.abc import Callable

import httpx
from pydub import AudioSegment

logger = logging.getLogger("dubber")

# Optional OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _hash_for_cache(provider: str, model: str, voice: str, text: str) -> str:
    """Generate cache hash for TTS audio."""
    key = f"{provider}|{model}|{voice}|{text}".encode()
    return hashlib.sha1(key).hexdigest()[:12]


def tts_speak_openai(
    client: OpenAI,
    text: str,
    model: str,
    voice: str,
    out_path: str,
    stream: bool = True,
    instructions: str | None = None,
) -> None:
    """Synthesize speech using OpenAI TTS."""
    if client is None:
        raise RuntimeError("OpenAI client is not initialized (missing OPENAI_API_KEY)")

    if stream:
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            response_format="wav",
            instructions=instructions,
        ) as resp:
            resp.stream_to_file(out_path)
    else:
        resp = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="wav",
            instructions=instructions,
        )
        with open(out_path, "wb") as f:
            f.write(resp.content)


def elevenlabs_tts_speak(
    api_key: str, voice_id: str, text: str, out_path: str, model_id: str = "eleven_multilingual_v2"
) -> None:
    """Synthesize speech using ElevenLabs TTS."""
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set.")
    if not voice_id:
        raise RuntimeError(
            "ElevenLabs voice_id is required (use --elevenlabs-voice-id or ELEVENLABS_VOICE_ID)."
        )

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/mpeg",
        "Content-Type": "application/json",
        "User-Agent": "video-dubbing-pipeline/1.0",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        r = client.post(url, json=payload, headers=headers)
        ctype = r.headers.get("content-type", "")
        if r.status_code != 200 or not ctype.startswith(("audio/", "application/octet-stream")):
            raise RuntimeError(f"ElevenLabs TTS failed: {r.status_code} {r.text[:300]}")
        tmp_mp3 = out_path.replace(".wav", ".mp3")
        with open(tmp_mp3, "wb") as f:
            f.write(r.content)
    clip = AudioSegment.from_file(tmp_mp3, format="mp3")
    clip.export(out_path, format="wav")
    try:
        Path(tmp_mp3).unlink()
    except OSError:
        pass


def make_synth_openai(
    client: OpenAI, tts_model: str, voice: str, instructions: str | None
) -> Callable[[str, str], None]:
    """Create OpenAI TTS synthesis function."""

    def _synth(text: str, out_path: str) -> None:
        tts_speak_openai(
            client, text, tts_model, voice, out_path, stream=True, instructions=instructions
        )

    return _synth


def make_synth_elevenlabs(api_key: str, voice_id: str, model_id: str) -> Callable[[str, str], None]:
    """Create ElevenLabs TTS synthesis function."""

    def _synth(text: str, out_path: str) -> None:
        elevenlabs_tts_speak(api_key, voice_id, text, out_path, model_id=model_id)

    return _synth


def pick_elevenlabs_default_voice(api_key: str) -> str | None:
    """Auto-pick first available ElevenLabs voice."""
    try:
        r = httpx.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={
                "xi-api-key": api_key,
                "accept": "application/json",
                "User-Agent": "video-dubbing-pipeline/1.0",
            },
            timeout=30.0,
        )
        HTTP_OK = 200
        if r.status_code == HTTP_OK:
            data = r.json()
            voices = data.get("voices", []) or []
            if voices and isinstance(voices, list):
                vid = voices[0].get("voice_id")
                return str(vid) if vid else None
        else:
            logger.warning("Could not fetch voices list (%d)", r.status_code)
    except Exception as e:
        logger.warning("Failed to auto-pick ElevenLabs voice: %s", e)
    return None
