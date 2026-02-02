# Примеры асинхронной обработки

Практические примеры использования async-режима конвейера.

## Базовые примеры

### 1. Простая async-обработка

```bash
# Одно видео в async-режиме
make full-async VIDEO=media/tutorial.mp4

# Для 37 сегментов: ~15 с вместо ~73 с
```

### 2. Своя конкурентность

```bash
# Высокая нагрузка
make synth-async VIDEO=media/long_video.mp4 MAX_CONCURRENT=10

# Медленный канал
make synth-async VIDEO=media/video.mp4 MAX_CONCURRENT=3
```

### 3. Режим отладки

```bash
make debug-synth-async VIDEO=media/video.mp4 MAX_CONCURRENT=5
make show-logs WORKDIR=.work/lesson
```

## Продвинутые примеры

### 1. Пакетная обработка видео

```bash
#!/bin/bash
videos=("media/lesson1.mp4" "media/lesson2.mp4" "media/lesson3.mp4")

for video in "${videos[@]}"; do
    echo "Обработка $video..."
    make full-async VIDEO="$video" MAX_CONCURRENT=5
    echo "Готово: $video"
done
```

### 2. Тест производительности

```bash
#!/bin/bash
video="media/test_video.mp4"
concurrency_levels=(1 3 5 8 10)

for level in "${concurrency_levels[@]}"; do
    echo "Уровень конкурентности: $level"
    time make synth-async VIDEO="$video" MAX_CONCURRENT="$level"
    echo "---"
done
```

### 3. Свой конвейер

```bash
#!/bin/bash
VIDEO="media/custom_video.mp4"
WORKDIR=".work/custom"
OUTPUT="out/custom_dubbed.mp4"
MAX_CONCURRENT=8
VOICE="onyx"

make prep VIDEO="$VIDEO" WORKDIR="$WORKDIR"
make synth-async \
    VIDEO="$VIDEO" \
    WORKDIR="$WORKDIR" \
    OUTPUT="$OUTPUT" \
    MAX_CONCURRENT="$MAX_CONCURRENT" \
    VOICE="$VOICE"
make youtube VIDEO="$VIDEO" WORKDIR="$WORKDIR"
```

## Интеграция из Python

### 1. Базовый async

```python
import asyncio
from openai import AsyncOpenAI
from dubber.tts_async import process_tts_batch_async

async def basic_async_example():
    client = AsyncOpenAI(api_key="your-api-key")
    texts = [
        "Hello, welcome to our tutorial.",
        "Today we'll learn about async processing.",
        "This is much faster than sync mode."
    ]
    outputs = ["hello.wav", "welcome.wav", "tutorial.wav"]

    results = await process_tts_batch_async(
        client=client,
        texts=texts,
        model="gpt-4o-mini-tts",
        voice="onyx",
        output_paths=outputs,
        max_concurrent=5
    )
    print(f"Создано файлов: {len(results)}")
    return results

asyncio.run(basic_async_example())
```

### 2. Построение таймлайна

```python
import asyncio
from openai import AsyncOpenAI
from dubber.tts_async import make_synth_openai_batch_async
from dubber.timeline_async import build_timeline_sentences_async
from dubber.models import Segment

async def custom_timeline_example():
    client = AsyncOpenAI(api_key="your-api-key")
    async_synth = make_synth_openai_batch_async(
        client=client,
        model="gpt-4o-mini-tts",
        voice="onyx",
        max_concurrent=5
    )

    segments = [
        Segment(start=0.0, end=2.0, text="Hello, world!"),
        Segment(start=2.0, end=4.0, text="This is async processing."),
        Segment(start=4.0, end=6.0, text="Much faster than sync mode.")
    ]
    sentence_groups = [[0], [1], [2]]

    timeline = await build_timeline_sentences_async(
        segments=segments,
        sentence_groups=sentence_groups,
        tmp_dir="tmp",
        voice_tag="main",
        synth_func_async=async_synth
    )
    timeline.export("output.wav", format="wav")
    print("Таймлайн собран!")

asyncio.run(custom_timeline_example())
```

### 3. Повторы при ошибках

```python
import asyncio
from dubber.tts_async import process_tts_batch_async

async def robust_async_processing(texts, outputs, max_retries=3):
    client = AsyncOpenAI(api_key="your-api-key")

    for attempt in range(max_retries):
        try:
            print(f"Попытка {attempt + 1}/{max_retries}")
            results = await process_tts_batch_async(
                client=client,
                texts=texts,
                model="gpt-4o-mini-tts",
                voice="onyx",
                output_paths=outputs,
                max_concurrent=5
            )
            print("Успех!")
            return results
        except Exception as e:
            print(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

texts = ["Текст 1", "Текст 2", "Текст 3"]
outputs = ["output1.wav", "output2.wav", "output3.wav"]
results = asyncio.run(robust_async_processing(texts, outputs))
```

### 4. Замер производительности

```python
import asyncio
import time
from dubber.tts_async import process_tts_batch_async

async def monitor_performance(texts, outputs, max_concurrent):
    start_time = time.time()
    client = AsyncOpenAI(api_key="your-api-key")

    results = await process_tts_batch_async(
        client=client,
        texts=texts,
        model="gpt-4o-mini-tts",
        voice="onyx",
        output_paths=outputs,
        max_concurrent=max_concurrent
    )

    duration = time.time() - start_time
    print(f"Время: {duration:.2f} с")
    print(f"На сегмент: {duration/len(texts):.2f} с")
    print(f"Сегментов/с: {len(texts)/duration:.2f}")
    return results

# Бенчмарк по уровням конкурентности
async def benchmark_concurrency():
    texts = [f"Segment {i}" for i in range(20)]
    outputs = [f"output_{i}.wav" for i in range(20)]
    for level in [1, 3, 5, 8, 10]:
        print(f"\n--- Уровень {level} ---")
        await monitor_performance(texts, outputs, level)

asyncio.run(benchmark_concurrency())
```

## Интеграция с Makefile

### 1. Цель для всех видео

```makefile
.PHONY: process-all-async

process-all-async:
	@echo "Обработка всех видео в async..."
	@for video in media/*.mp4; do \
		echo "Обработка $$video..."; \
		make full-async VIDEO="$$video" MAX_CONCURRENT=5; \
		echo "Готово: $$video"; \
	done
	@echo "Все видео обработаны!"

# Использование: make process-all-async
```

### 2. Бенчмарк

```makefile
.PHONY: benchmark-async

benchmark-async:
	@echo "Бенчмарк async..."
	@for level in 1 3 5 8 10; do \
		echo "--- Уровень $$level ---"; \
		time make synth-async VIDEO=media/test.mp4 MAX_CONCURRENT=$$level; \
	done
	@echo "Бенчмарк завершён!"

# Использование: make benchmark-async
```

### 3. Свой async-конвейер

```makefile
.PHONY: custom-async-workflow

custom-async-workflow: check-env check-files
	@echo "=== Свой async-конвейер ==="
	@make prep VIDEO="$(VIDEO)" WORKDIR="$(WORKDIR)"
	@make synth-async VIDEO="$(VIDEO)" WORKDIR="$(WORKDIR)" MAX_CONCURRENT="$(MAX_CONCURRENT)" VOICE="$(VOICE)"
	@make youtube VIDEO="$(VIDEO)" WORKDIR="$(WORKDIR)"
	@echo "=== Готово ==="

# Использование: make custom-async-workflow VIDEO=media/video.mp4 MAX_CONCURRENT=8 VOICE=onyx
```

## Устранение неполадок

### 1. Отладка async

```bash
#!/bin/bash
VIDEO="media/problem_video.mp4"
WORKDIR=".work/debug"

echo "=== Отладка async ==="
if [ ! -f "$WORKDIR/subs.srt" ]; then
    make prep VIDEO="$VIDEO" WORKDIR="$WORKDIR"
fi
echo "Тест с низкой конкурентностью (2)..."
make debug-synth-async VIDEO="$VIDEO" WORKDIR="$WORKDIR" MAX_CONCURRENT=2
make show-logs WORKDIR="$WORKDIR"

if [ -f "out/lesson_dubbed.mp4" ]; then
    echo "✓ Видео создано"
else
    echo "✗ Видео не создано"
fi
```

### 2. Анализ производительности

```bash
#!/bin/bash
VIDEO="media/test_video.mp4"
LOG_FILE="performance.log"

echo "=== Анализ async ===" > "$LOG_FILE"
for level in 1 3 5 8 10; do
    echo "Уровень: $level" | tee -a "$LOG_FILE"
    start_time=$(date +%s)
    make synth-async VIDEO="$VIDEO" MAX_CONCURRENT="$level" 2>&1 | tee -a "$LOG_FILE"
    end_time=$(date +%s)
    echo "Длительность: $((end_time - start_time)) с" | tee -a "$LOG_FILE"
    echo "---" | tee -a "$LOG_FILE"
done
echo "Результаты в $LOG_FILE"
```

## Рекомендации

### 1. Начинать с малого

```bash
make synth-async MAX_CONCURRENT=3   # сначала
make synth-async MAX_CONCURRENT=5    # если стабильно
make synth-async MAX_CONCURRENT=8    # если всё ещё стабильно
```

### 2. Следить за ресурсами

```bash
htop &   # или Activity Monitor на macOS
make synth-async VIDEO=media/video.mp4 MAX_CONCURRENT=5
```

### 3. Подбирать конкурентность

```bash
# Короткие видео (<10 сегментов)
make synth-async MAX_CONCURRENT=3

# Средние (10–30 сегментов)
make synth-async MAX_CONCURRENT=5

# Длинные (>30 сегментов)
make synth-async MAX_CONCURRENT=8
```

### 4. Обработка ошибок в скриптах

```bash
if ! make synth-async VIDEO="$VIDEO" MAX_CONCURRENT=5; then
    echo "Ошибка, пробуем с меньшей конкурентностью..."
    make synth-async VIDEO="$VIDEO" MAX_CONCURRENT=3
fi
```
