# Справочник Async API

Описание асинхронных компонентов конвейера дублирования.

## Модули

### `tts_async.py`

#### `tts_speak_openai_async(client, text, model, voice, out_path, instructions=None)`

Асинхронный синтез речи через OpenAI TTS.

**Параметры:**

- `client` (AsyncOpenAI) — клиент OpenAI
- `text` (str) — текст для синтеза
- `model` (str) — модель TTS
- `voice` (str) — идентификатор голоса
- `out_path` (str) — путь к выходному файлу
- `instructions` (str, опционально) — инструкции для голоса

**Пример:**

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

Пакетный асинхронный TTS с ограничением конкурентности.

**Параметры:**

- `client` (AsyncOpenAI)
- `texts` (List[str]) — тексты для синтеза
- `model`, `voice` — модель и голос
- `output_paths` (List[str]) — пути к выходным файлам
- `instructions` (str, опционально)
- `max_concurrent` (int) — макс. параллельных запросов

**Возвращает:** `List[str]` — пути к созданным файлам.

**Пример:**

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

Фабрика асинхронной функции пакетного синтеза для OpenAI.

**Возвращает:** вызываемый объект (async batch synth).

**Пример:**

```python
async_synth = make_synth_openai_batch_async(
    client=client,
    model="gpt-4o-mini-tts",
    voice="onyx",
    max_concurrent=5
)
results = await async_synth(texts, output_paths)
```

### `timeline_async.py`

#### `build_timeline_sentences_async(segments, sentence_groups, tmp_dir, voice_tag, synth_func_async, sample_rate=24000, cache_sig=None, no_tts=False)`

Построение таймлайна по группам предложений с асинхронным TTS.

**Параметры:**

- `segments` (List[Segment])
- `sentence_groups` (List[List[int]])
- `tmp_dir`, `voice_tag`
- `synth_func_async` — асинхронная функция синтеза
- `sample_rate`, `cache_sig`, `no_tts`

**Возвращает:** `AudioSegment`.

#### `build_timeline_wav_async(segments, tmp_dir, voice_tag, synth_func_async, sample_rate=24000, strict_timing=False, tolerance_ms=30, fit_mode="pad-or-speedup", cache_sig=None, no_tts=False)`

Построение таймлайна по сегментам с async-обработкой.

**Параметры:** как выше, плюс `strict_timing`, `tolerance_ms`, `fit_mode`.

**Возвращает:** `AudioSegment`.

### `cli_async.py`

#### `main_async()`

Точка входа async CLI.

**Использование:**

```bash
python -m dubber.cli_async --input_video video.mp4 --stage synth --async-tts
```

**Ключевые аргументы:** `--async-tts`, `--max-concurrent`, `--input_video`, `--workdir`, `--stage`.

## Обработка ошибок

- **asyncio.run() cannot be called from a running event loop** — используйте `asyncio.create_task()` вместо `asyncio.run()` внутри уже запущенного цикла.
- **'async for' requires an object with **aiter**** — с `tqdm.as_completed()` используйте обычный `for`, не `async for`.
- **httpx.ConnectError: Connection timeout** — уменьшите `max_concurrent` или проверьте сеть.

**Пример с повторными попытками:**

```python
async def robust_tts_processing(texts, outputs, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await process_tts_batch_async(...)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

## Настройка производительности

| Система / канал      | Рекомендуемый MAX_CONCURRENT |
| -------------------- | ---------------------------- |
| Медленный (<10 Mbps) | 2–3                          |
| Средний (10–50 Mbps) | 5 (по умолчанию)             |
| Быстрый (>50 Mbps)   | 8–10                         |
| Высокая квота API    | 10–15                        |

**Обработка больших объёмов батчами:**

```python
batch_size = 50
for i in range(0, len(texts), batch_size):
    await process_tts_batch_async(
        client=client,
        texts=texts[i:i+batch_size],
        output_paths=outputs[i:i+batch_size],
        max_concurrent=5
    )
```

## Интеграция

**Кастомный async-конвейер:**

```python
async def custom_async_pipeline(video_path, segments, groups):
    client = AsyncOpenAI(api_key="your-key")
    async_synth = make_synth_openai_batch_async(
        client=client,
        model="gpt-4o-mini-tts",
        voice="onyx",
        max_concurrent=5
    )
    timeline = await build_timeline_sentences_async(
        segments=segments,
        sentence_groups=groups,
        tmp_dir="tmp",
        voice_tag="main",
        synth_func_async=async_synth
    )
    return timeline
```

## Тестирование

**Unit-тест с pytest:**

```python
@pytest.mark.asyncio
async def test_async_tts():
    await tts_speak_openai_async(...)
    assert Path("test.wav").exists()
```

**Бенчмарк:**

```python
async def benchmark_async_tts(texts, max_concurrent):
    start = time.time()
    await process_tts_batch_async(..., max_concurrent=max_concurrent)
    return time.time() - start
```
