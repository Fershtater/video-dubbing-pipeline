# Video Dubbing Pipeline (Dubber)

Конвейер для дублирования видео на Python: распознавание речи, полировка текста с помощью LLM, синтез речи и генерация описаний для YouTube.

---

## Для резюме / О проекте

**Pet-проект** для демонстрации работы с Python, LLM и современной разработкой.

### Стек и компетенции

| Область              | Технологии и практики                                                    |
| -------------------- | ------------------------------------------------------------------------ |
| **Python**           | 3.11+, типизация (type hints), Poetry, модульная архитектура             |
| **LLM / AI**         | OpenAI API (GPT — полировка текста, Whisper — STT, TTS), ElevenLabs TTS  |
| **Локальные модели** | faster-whisper для офлайн-транскрипции                                   |
| **Асинхронность**    | asyncio, параллельные HTTP-запросы (httpx), семафоры, 3–5× ускорение TTS |
| **Внешние API**      | REST (OpenAI, ElevenLabs), работа с ключами и квотами                    |
| **Медиа**            | ffmpeg/ffprobe (извлечение аудио, монтаж таймлайна, субтитры)            |
| **Качество кода**    | Ruff (линт/форматирование), Pyright (проверка типов), pytest             |
| **Документация**     | README, гайды по async API, примеры использования, индекс документации   |

### Что делает проект

- Извлекает аудио из видео и транскрибирует речь (локально или через API).
- Улучшает текст субтитров с помощью GPT с сохранением разбивки на сегменты.
- Синтезирует речь через OpenAI TTS или ElevenLabs с кэшированием и опциональной асинхронной обработкой.
- Собирает таймлайн, склеивает аудио и встраивает результат в видео.
- Генерирует описание, главы и теги для публикации на YouTube.

---

## Возможности

- **Извлечение аудио** — ffmpeg
- **Распознавание речи (STT)** — faster-whisper (локально) или OpenAI Whisper API
- **Полировка текста** — улучшение транскрипта с помощью GPT с сохранением сегментации
- **Синтез речи (TTS)** — OpenAI TTS или ElevenLabs с кэшированием
- **Асинхронный TTS** — параллельные запросы, ускорение в 3–5 раз
- **Таймлайн** — несколько режимов выравнивания (сегмент, предложение, блок)
- **Субтитры** — генерация и встраивание SRT
- **YouTube** — описание, главы с таймкодами, теги
- **Детекция сцен** — границы сцен для естественных речевых блоков

## Требования

- Python 3.11+
- Poetry (управление зависимостями)
- ffmpeg и ffprobe (обработка аудио/видео)
- По желанию: ключи OpenAI API, ElevenLabs API

## Быстрый старт

### 1. Установка

```bash
make setup
# или вручную:
poetry install
```

### 2. Переменные окружения

Создайте файл `.env`:

```bash
OPENAI_API_KEY=ваш_ключ_openai
ELEVENLABS_API_KEY=ваш_ключ_elevenlabs
# опционально:
ELEVENLABS_VOICE_ID=id_голоса
OPENAI_TTS_INSTRUCTIONS="Мужской голос, спокойный тон..."
```

### 3. Запуск

```bash
# Полный конвейер (синхронно)
make full VIDEO=media/ваше_видео.mp4

# Полный конвейер (асинхронно — быстрее)
make full-async VIDEO=media/ваше_видео.mp4

# По шагам:
make prep VIDEO=media/ваше_видео.mp4
make synth TTS_PROVIDER=openai VOICE=alloy
make youtube
```

## Этапы конвейера

### Подготовка (`make prep`)

- Извлечение аудио из видео
- Транскрипция (по умолчанию faster-whisper)
- Опциональная полировка текста с GPT
- Группировка предложений для синтеза
- Генерация SRT-субтитров

### Синтез (`make synth`)

- Синтез речи выбранным TTS-провайдером
- Построение таймлайна и склейка аудио
- Сведение нового аудио с видео
- Выравнивание длительности

### Асинхронный синтез (`make synth-async`)

- В 3–5 раз быстрее синхронного режима
- Параллельные TTS-запросы с настраиваемой конкурентностью
- Прогресс в реальном времени
- Оптимизация под OpenAI TTS API

### YouTube (`make youtube`)

- Описание из транскрипта
- Главы с таймкодами
- Теги и хештеги для SEO
- Опциональное улучшение описания и тегов с помощью AI

### Вжигание субтитров (`make burn`)

- Мягкие (потоковые) или жёсткие (вшитые) субтитры

## Конфигурация

### Переменные Makefile

```bash
VIDEO=media/lesson.mp4          # Входное видео
WORKDIR=.work/lesson            # Рабочая директория
OUTPUT=out/lesson_dubbed.mp4    # Выходное видео
TTS_PROVIDER=openai             # openai или elevenlabs
VOICE=alloy                     # Голос TTS
SENT_JOIN_GAP=1.2               # Порог объединения предложений
SENT_PER_CHUNK=2                # Предложений на чанк синтеза
MAX_CONCURRENT=5                # Макс. параллельных TTS-запросов (async)
YOUTUBE_SOURCE=sentences        # Источник глав YouTube
YOUTUBE_GAP_THRESH=1.5          # Порог разрыва для глав
```

### Режимы выравнивания

- **segment** — по сегментам (жёсткий тайминг)
- **sentence** — по предложениям (естественный поток)
- **block** — по сценам (плавное повествование)

### TTS-провайдеры

**OpenAI TTS** — модели `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd`; голоса: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`; инструкции через `OPENAI_TTS_INSTRUCTIONS`.

**ElevenLabs TTS** — модели `eleven_multilingual_v2`, `eleven_monolingual_v1`; голос через `ELEVENLABS_VOICE_ID`.

## Асинхронная обработка

| Режим              | Время (37 сегментов) | Ускорение | Когда использовать         |
| ------------------ | -------------------- | --------- | -------------------------- |
| Синхронный         | ~73 с                | 1×        | Отладка, короткие ролики   |
| Async (3 потока)   | ~20 с                | 3.6×      | Умеренная нагрузка         |
| Async (5 потоков)  | ~15 с                | 4.9×      | **Рекомендуется**          |
| Async (10 потоков) | ~12 с                | 6.1×      | Высокая производительность |

```bash
make synth-async VIDEO=media/видео.mp4
make full-async VIDEO=media/видео.mp4
make synth-async VIDEO=media/видео.mp4 MAX_CONCURRENT=10
```

Подробнее: [docs/ASYNC_GUIDE.md](docs/ASYNC_GUIDE.md), [docs/API_ASYNC.md](docs/API_ASYNC.md), [docs/ASYNC_EXAMPLES.md](docs/ASYNC_EXAMPLES.md).

## Файлы и структура

### Рабочая директория

```
.work/lesson/
├── extracted.wav              # Извлечённое аудио
├── segments_raw.json         # Сырая транскрипция
├── segments_polished.json    # Отполированные сегменты
├── subs.srt                  # Субтитры
├── subs_block.srt            # Субтитры по блокам
├── sentences_groups.json     # Группы предложений
├── new_audio.wav             # Синтезированное аудио
└── youtube/
    ├── description.md        # Описание для YouTube
    ├── chapters.txt          # Главы с таймкодами
    └── tags.txt              # Теги
```

## Разработка

```bash
make fmt        # Форматирование
make lint       # Линтинг
make typecheck  # Проверка типов
make test       # Тесты
make clean      # Очистка
```

## Устранение неполадок

- **«ffmpeg not found»** — установите ffmpeg (`brew install ffmpeg` или `apt install ffmpeg`).
- **Пустая транскрипция** — проверьте громкость и качество аудио, при необходимости смените модель STT.
- **Несовпадение длительности** — конвейер подстраивает длительность; смотрите логи.
- **Лимиты TTS** — используется кэш; при rate limit уменьшите `MAX_CONCURRENT`.

## Архитектура (модули)

- **io_ffmpeg.py** — работа с аудио/видео через ffmpeg
- **stt.py** — распознавание речи
- **polish.py** — полировка текста с GPT
- **srt_utils.py** — разбор SRT и нормализация текста
- **sentences.py** — группировка и разбиение на чанки
- **scenes.py** — детекция сцен и блоки
- **tts.py** / **tts_async.py** — синтез речи (sync/async)
- **timeline.py** / **timeline_async.py** — таймлайн (sync/async)
- **youtube.py** — описание и главы для YouTube
- **cli.py** / **cli_async.py** — интерфейс командной строки
- **cost.py** — оценка стоимости

## Лицензия

MIT — см. [LICENSE](LICENSE).

## Участие в разработке

Рекомендации по контрибуции — в [CONTRIBUTING.md](CONTRIBUTING.md). Изменения в коде — через fork, ветку, тесты и `make fmt lint typecheck test`, затем pull request.

## Документация

- [README.md](README.md) — этот файл
- [docs/ASYNC_GUIDE.md](docs/ASYNC_GUIDE.md) — асинхронная обработка
- [docs/API_ASYNC.md](docs/API_ASYNC.md) — API async
- [docs/ASYNC_EXAMPLES.md](docs/ASYNC_EXAMPLES.md) — примеры
- [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md) — индекс документации
