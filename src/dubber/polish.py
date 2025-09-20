"""
Text polishing with GPT while preserving segmentation.
"""

import json
import logging

from .models import Segment

logger = logging.getLogger("dubber")

# Optional OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def polish_segments(client: OpenAI, segments: list[Segment], model: str) -> list[Segment]:
    """Polish transcript with GPT while preserving segmentation."""
    if client is None:
        msg = "OpenAI client is not initialized (missing OPENAI_API_KEY)"
        raise RuntimeError(msg)

    logger.info("Polishing transcript with GPT …")
    input_text = [s.text for s in segments]

    system = (
        "You are a careful English editor for voice-over scripts."
        "Improve grammar, fluency, and clarity "
        "without changing meaning. Keep the number and order of lines exactly the same. "
        "Return ONLY a JSON array of strings."
    )
    user = json.dumps(input_text, ensure_ascii=False)

    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    content = chat.choices[0].message.content
    try:
        improved_lines = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            improved_lines = json.loads(content[start : end + 1])
        else:
            msg = "Model did not return valid JSON"
            raise RuntimeError(msg) from None

    if len(improved_lines) != len(segments):
        logger.warning(f"Segment count changed during polishing: {len(segments)} -> {len(improved_lines)}. Using original segments.")
        return segments

    out: list[Segment] = []
    for seg, line in zip(segments, improved_lines, strict=False):
        out.append(Segment(start=seg.start, end=seg.end, text=line.strip()))
    return out
