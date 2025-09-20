"""
Speech-to-text transcription modules.
"""

import logging

from pydub import AudioSegment

from .models import Segment

logger = logging.getLogger("dubber")

# Optional OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def transcribe_whisper_api(
    client: OpenAI, wav_path: str, model: str = "whisper-1"
) -> list[Segment]:
    """Transcribe audio using OpenAI Whisper API."""
    if client is None:
        raise RuntimeError("OpenAI client is not initialized (missing OPENAI_API_KEY)")

    def _segments_from_response(resp) -> list[Segment] | None:
        segs = getattr(resp, "segments", None)
        if segs is None and isinstance(resp, dict):
            segs = resp.get("segments")
        if not segs:
            return None
        out: list[Segment] = []
        for seg in segs:
            start = (
                float(seg.get("start", 0.0))
                if isinstance(seg, dict)
                else float(getattr(seg, "start", 0.0))
            )
            end = (
                float(seg.get("end", 0.0))
                if isinstance(seg, dict)
                else float(getattr(seg, "end", 0.0))
            )
            text = (
                seg.get("text", "").strip()
                if isinstance(seg, dict)
                else str(getattr(seg, "text", "")).strip()
            )
            out.append(Segment(start=start, end=end, text=text))
        return out

    audio_len_s = len(AudioSegment.from_wav(wav_path)) / 1000.0

    try:
        with open(wav_path, "rb") as f:
            logger.info(f"Transcribing with {model} …")
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json",
            )
        segs = _segments_from_response(resp)
        if segs:
            return segs
        full_text = getattr(resp, "text", None)
        if full_text is None and isinstance(resp, dict):
            full_text = resp.get("text", "")
        if full_text:
            return [Segment(start=0.0, end=audio_len_s, text=str(full_text).strip())]
        logger.warning("No segments/text in response; will try gpt-4o-transcribe fallback.")
    except Exception as e:
        logger.warning(f"Primary transcription failed ({e}); trying gpt-4o-transcribe …")

    try:
        with open(wav_path, "rb") as f2:
            resp2 = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f2,
            )
        text2 = getattr(resp2, "text", None)
        if text2 is None and isinstance(resp2, dict):
            text2 = resp2.get("text", "")
        if text2:
            return [Segment(start=0.0, end=audio_len_s, text=str(text2 or "").strip())]
    except Exception as e:
        logger.error(f"Fallback transcription failed: {e}")

    return [Segment(start=0.0, end=audio_len_s, text="")]


def transcribe_local_faster_whisper(
    wav_path: str, local_model: str = "base.en", beam_size: int = 1
) -> list[Segment]:
    """Transcribe audio using local faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is not installed. Install with: poetry add faster-whisper"
        ) from e

    logger.info(f"Transcribing locally with faster-whisper ({local_model}) …")
    model = WhisperModel(local_model, device="cpu", compute_type="int8")

    segments_iter, _info = model.transcribe(
        wav_path,
        language="en",
        vad_filter=True,
        beam_size=beam_size,
        word_timestamps=False,
    )
    out: list[Segment] = []
    for s in segments_iter:
        out.append(Segment(start=float(s.start), end=float(s.end), text=str(s.text).strip()))
    if not out:
        audio_len_s = len(AudioSegment.from_wav(wav_path)) / 1000.0
        out = [Segment(start=0.0, end=audio_len_s, text="")]
    return out
