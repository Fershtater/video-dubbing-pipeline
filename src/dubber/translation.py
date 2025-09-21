"""
Translation module for converting text from any language to English.
"""

import logging
from typing import Optional

logger = logging.getLogger("dubber")

# Optional OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def translate_to_english(
    client: OpenAI,
    text: str,
    source_language: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Translate text from any language to English using OpenAI GPT.

    Args:
        client: OpenAI client instance
        text: Text to translate
        source_language: Source language code (optional, will be auto-detected)
        model: GPT model to use for translation

    Returns:
        Translated text in English
    """
    if client is None:
            raise RuntimeError("OpenAI client is not initialized (missing OPENAI_API_KEY)")

    if not text.strip():
        return text

    # Prepare the prompt
    if source_language:
        prompt = f"""Translate the following text from {source_language} to English.
Maintain the original tone, style, and meaning. Keep technical terms accurate.
Return only the translated text without any explanations or additional text.

Text to translate:
{text}"""
    else:
        prompt = f"""Translate the following text to English.
Detect the source language automatically and translate while maintaining the original tone, style, and meaning.
Keep technical terms accurate. Return only the translated text without any explanations or additional text.

Text to translate:
{text}"""

    try:
        logger.info(f"Translating text (source: {source_language or 'auto'}) using {model}...")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator specializing in technical and educational content. Always provide accurate, natural translations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent translation
            max_tokens=4000
        )

        translated_text = response.choices[0].message.content.strip()
        logger.info(f"Translation completed: {len(text)} -> {len(translated_text)} characters")

        return translated_text

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        # Return original text if translation fails
        return text


def translate_segments_to_english(
    client: OpenAI,
    segments: list,
    source_language: Optional[str] = None,
    model: str = "gpt-4o-mini",
    batch_size: int = 5
) -> list:
    """
    Translate a list of segments to English.

    Args:
        client: OpenAI client instance
        segments: List of segments with text to translate
        source_language: Source language code (optional)
        model: GPT model to use for translation
        batch_size: Number of segments to translate in one batch

    Returns:
        List of segments with translated text
    """
    if not segments:
        return segments

    # Import Segment class here to avoid circular imports
    from .models import Segment

    translated_segments = []

    # Process in batches to avoid token limits
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]

        # Combine text from batch
        batch_texts = []
        for j, segment in enumerate(batch):
            if hasattr(segment, 'text') and segment.text.strip():
                batch_texts.append(f"[{j}]: {segment.text}")

        if not batch_texts:
            translated_segments.extend(batch)
            continue

        combined_text = "\n".join(batch_texts)

        # Prepare prompt for batch translation
        if source_language:
            prompt = f"""Translate the following numbered texts from {source_language} to English.
Each text is numbered with [number]: format. Translate each one separately.
Maintain the original tone, style, and meaning. Keep technical terms accurate.
Return the translations in the same numbered format.

Texts to translate:
{combined_text}"""
        else:
            prompt = f"""Translate the following numbered texts to English.
Detect the source language automatically. Each text is numbered with [number]: format.
Translate each one separately while maintaining the original tone, style, and meaning.
Keep technical terms accurate. Return the translations in the same numbered format.

Texts to translate:
{combined_text}"""

        try:
            logger.info(f"Translating batch {i//batch_size + 1} ({len(batch)} segments)...")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator specializing in technical and educational content. Always provide accurate, natural translations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000
            )

            translated_batch_text = response.choices[0].message.content.strip()

            # Parse translated texts
            translated_lines = translated_batch_text.split('\n')
            translation_map = {}

            for line in translated_lines:
                if line.strip() and ': ' in line:
                    try:
                        idx_str, translated_text = line.split(': ', 1)
                        idx = int(idx_str.strip('[]'))
                        translation_map[idx] = translated_text.strip()
                    except (ValueError, IndexError):
                        continue

            # Apply translations to segments
            for j, segment in enumerate(batch):
                if hasattr(segment, 'text') and segment.text.strip():
                    translated_text = translation_map.get(j, segment.text)
                    # Create new segment with translated text
                    new_segment = Segment(
                        start=segment.start,
                        end=segment.end,
                        text=translated_text
                    )
                    translated_segments.append(new_segment)
                else:
                    translated_segments.append(segment)

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            # Fallback to individual translation
            for segment in batch:
                if hasattr(segment, 'text') and segment.text.strip():
                    translated_text = translate_to_english(client, segment.text, source_language, model)
                    new_segment = Segment(
                        start=segment.start,
                        end=segment.end,
                        text=translated_text
                    )
                    translated_segments.append(new_segment)
                else:
                    translated_segments.append(segment)

    return translated_segments


def get_language_name(language_code: str) -> str:
    """Get human-readable language name from language code."""
    language_names = {
        "ru": "Russian",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "ar": "Arabic",
        "hi": "Hindi",
        "en": "English",
        "uk": "Ukrainian",
        "pl": "Polish",
        "nl": "Dutch",
        "sv": "Swedish",
        "no": "Norwegian",
        "da": "Danish",
        "fi": "Finnish",
        "tr": "Turkish",
        "he": "Hebrew",
        "th": "Thai",
        "vi": "Vietnamese"
    }
    return language_names.get(language_code.lower(), language_code.upper())
