# 🌍 Multilingual Video Dubbing Guide

This guide explains how to use the multilingual features of the video dubbing pipeline to translate and dub videos from any language to English.

## 🚀 Quick Start

### Basic Translation Workflow

```bash
# Auto-detect language and translate to English
make full VIDEO=media/foreign_video.mp4 TRANSLATE=true

# Specify source language explicitly
make full VIDEO=media/russian_video.mp4 SOURCE_LANGUAGE=ru TRANSLATE=true

# Use async processing for faster results
make full-async VIDEO=media/german_video.mp4 SOURCE_LANGUAGE=de TRANSLATE=true
```

## 📋 Supported Languages

The system supports **99+ languages** including:

| Language | Code | Auto-detect | Transcription | Translation |
|----------|------|-------------|---------------|-------------|
| Russian | `ru` | ✅ | ✅ | ✅ |
| German | `de` | ✅ | ✅ | ✅ |
| French | `fr` | ✅ | ✅ | ✅ |
| Spanish | `es` | ✅ | ✅ | ✅ |
| Italian | `it` | ✅ | ✅ | ✅ |
| Portuguese | `pt` | ✅ | ✅ | ✅ |
| Japanese | `ja` | ✅ | ✅ | ✅ |
| Korean | `ko` | ✅ | ✅ | ✅ |
| Chinese | `zh` | ✅ | ✅ | ✅ |
| Arabic | `ar` | ✅ | ✅ | ✅ |
| Hindi | `hi` | ✅ | ✅ | ✅ |
| Ukrainian | `uk` | ✅ | ✅ | ✅ |
| Polish | `pl` | ✅ | ✅ | ✅ |
| Dutch | `nl` | ✅ | ✅ | ✅ |
| Swedish | `sv` | ✅ | ✅ | ✅ |
| Norwegian | `no` | ✅ | ✅ | ✅ |
| Danish | `da` | ✅ | ✅ | ✅ |
| Finnish | `fi` | ✅ | ✅ | ✅ |
| Turkish | `tr` | ✅ | ✅ | ✅ |
| Hebrew | `he` | ✅ | ✅ | ✅ |
| Thai | `th` | ✅ | ✅ | ✅ |
| Vietnamese | `vi` | ✅ | ✅ | ✅ |

## 🔧 Configuration Parameters

### Makefile Parameters

```bash
# Source language (optional - will auto-detect if empty)
SOURCE_LANGUAGE=ru

# Enable translation to English
TRANSLATE=true

# GPT model for translation
TRANSLATION_MODEL=gpt-4o-mini
```

### CLI Parameters

```bash
# Specify source language
--source-language ru

# Enable translation
--translate

# Translation model
--translation-model gpt-4o-mini
```

## 📖 Usage Examples

### 1. Auto-detect Language

```bash
# Let the system detect the language automatically
make full VIDEO=media/unknown_language.mp4 TRANSLATE=true
```

### 2. Russian to English

```bash
# Full workflow with Russian source
make full VIDEO=media/russian_tutorial.mp4 \
  SOURCE_LANGUAGE=ru \
  TRANSLATE=true \
  OUTPUT=out/russian_tutorial_english.mp4
```

### 3. German to English (Async)

```bash
# Faster async processing for German content
make full-async VIDEO=media/german_presentation.mp4 \
  SOURCE_LANGUAGE=de \
  TRANSLATE=true \
  MAX_CONCURRENT=8
```

### 4. French to English (High Quality)

```bash
# Use GPT-4 for better translation quality
make full VIDEO=media/french_course.mp4 \
  SOURCE_LANGUAGE=fr \
  TRANSLATE=true \
  TRANSLATION_MODEL=gpt-4o
```

### 5. Multi-language Batch Processing

```bash
# Process multiple languages
for lang in ru de fr es; do
  make full VIDEO=media/${lang}_video.mp4 \
    SOURCE_LANGUAGE=${lang} \
    TRANSLATE=true \
    OUTPUT=out/${lang}_video_english.mp4
done
```

## 🔄 Workflow Details

### 1. Language Detection
- **Automatic**: Uses Whisper API to detect language
- **Manual**: Specify `SOURCE_LANGUAGE` parameter
- **Fallback**: Defaults to English if detection fails

### 2. Transcription
- **OpenAI Whisper API**: Supports all 99+ languages
- **Local faster-whisper**: Uses multilingual `base` model
- **Language-specific**: Optimized models for better accuracy

### 3. Translation
- **GPT-powered**: Uses OpenAI GPT models for translation
- **Batch processing**: Translates multiple segments efficiently
- **Context-aware**: Maintains technical terminology and style
- **Quality control**: Low temperature for consistent results

### 4. Synchronization
- **Time-preserving**: Maintains original timing constraints
- **Segment-based**: Each segment keeps its original time bounds
- **Audio alignment**: Perfect sync between video and new audio

## 🛠️ Technical Implementation

### Language Detection
```python
def detect_language(client: OpenAI, wav_path: str) -> str | None:
    """Detect language using Whisper API"""
    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        language=None  # Auto-detect
    )
    return resp.language
```

### Translation Pipeline
```python
def translate_segments_to_english(
    client: OpenAI,
    segments: list,
    source_language: str,
    model: str = "gpt-4o-mini"
) -> list:
    """Translate segments maintaining timing"""
    # Batch processing for efficiency
    # Context-aware translation
    # Time-preserving output
```

### Multilingual STT
```python
def transcribe_local_faster_whisper(
    wav_path: str, 
    local_model: str = "base",  # Multilingual model
    language: str = None
) -> list[Segment]:
    """Transcribe with language-specific optimization"""
```

## 🎯 Best Practices

### 1. Language Selection
- **Auto-detect** for unknown languages
- **Specify explicitly** for better accuracy
- **Use full language names** for clarity

### 2. Translation Quality
- **gpt-4o-mini**: Fast and cost-effective
- **gpt-4o**: Higher quality for important content
- **Batch processing**: More efficient than individual translation

### 3. Performance Optimization
- **Async processing**: Use `full-async` for speed
- **Concurrent requests**: Increase `MAX_CONCURRENT` for parallel processing
- **Local models**: Use faster-whisper for offline processing

### 4. Quality Assurance
- **Preview segments**: Check transcription quality
- **Review translation**: Verify technical terms
- **Test sync**: Ensure audio-video alignment

## 🚨 Troubleshooting

### Common Issues

1. **Language not detected**
   ```bash
   # Specify language explicitly
   SOURCE_LANGUAGE=ru TRANSLATE=true
   ```

2. **Poor translation quality**
   ```bash
   # Use higher-quality model
   TRANSLATION_MODEL=gpt-4o
   ```

3. **Slow processing**
   ```bash
   # Use async processing
   make full-async MAX_CONCURRENT=10
   ```

4. **Audio sync issues**
   - Check original video timing
   - Verify segment boundaries
   - Use `debug-synth` for troubleshooting

### Error Messages

- **"Translation requires OpenAI client"**: Set `OPENAI_API_KEY`
- **"Language detection failed"**: Specify `SOURCE_LANGUAGE` manually
- **"Transcription returned empty text"**: Check audio quality and format

## 📊 Performance Metrics

### Processing Times (approximate)
- **Language detection**: 5-10 seconds
- **Transcription**: 1-2x video duration
- **Translation**: 0.5-1x video duration
- **TTS synthesis**: 0.3-0.5x video duration

### Cost Estimates (OpenAI API)
- **Whisper transcription**: $0.006/minute
- **GPT translation**: $0.60/1M input tokens
- **TTS synthesis**: $0.03/minute

## 🔮 Future Enhancements

- **Real-time translation**: Live streaming support
- **Voice cloning**: Preserve original speaker's voice
- **Multiple target languages**: Not just English
- **Quality scoring**: Automatic translation quality assessment
- **Custom models**: Fine-tuned translation models

## 📚 Additional Resources

- [OpenAI Whisper Documentation](https://platform.openai.com/docs/guides/speech-to-text)
- [OpenAI GPT API Reference](https://platform.openai.com/docs/api-reference)
- [faster-whisper Models](https://github.com/SYSTRAN/faster-whisper)
- [Supported Languages List](https://github.com/openai/whisper#available-models-and-languages)
