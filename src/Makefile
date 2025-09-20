# Video Dubbing Pipeline Makefile

# ---------- Config (override via env or CLI) ----------
POETRY ?= poetry
PY ?= $(POETRY) run python
ENTRY ?= -m dubber.cli

# Common I/O
VIDEO ?= media/1.mp4
WORKDIR ?= ./.work/lesson
OUTPUT ?= out/lesson_dubbed.mp4
DUBBED_OUTPUT ?= out/lesson_dubbed_temp.mp4
TTS_PROVIDER ?= openai
TTS_MODEL ?= gpt-4o-mini-tts
VOICE ?= onyx
SENT_JOIN_GAP ?= 1.2
SENT_PER_CHUNK ?= 2

# Async processing
MAX_CONCURRENT ?= 5

# YouTube generation
YOUTUBE_OUTDIR ?= $(WORKDIR)/youtube
YOUTUBE_SOURCE ?= sentences
YOUTUBE_GAP_THRESH ?= 1.5

# Multilingual support
SOURCE_LANGUAGE ?= 
TRANSLATE ?= false
TRANSLATION_MODEL ?= gpt-4o-mini

# Debug and logging
DEBUG ?= false
VERBOSE ?= --verbose

.PHONY: setup prep synth burn youtube full clean fmt lint typecheck test check-env check-files

# ---------- Setup ----------
setup:
	@echo "=== Setting up Video Dubbing Pipeline ==="
	@echo "Installing dependencies with Poetry..."
	$(POETRY) install
	@echo "Make sure ffmpeg is installed (macOS: brew install ffmpeg)"
	@echo "Configure .env with OPENAI_API_KEY / ELEVENLABS_API_KEY"
	@echo "=== Setup completed ==="

# ---------- Environment and file checks ----------
check-env:
	@echo "=== Checking environment ==="
	@if [ ! -f "../.env" ]; then \
		echo "ERROR: .env file not found in project root"; \
		echo "Expected: $(shell pwd)/../.env"; \
		exit 1; \
	fi
	@echo "✓ .env file found"
	@echo "=== Environment check passed ==="

check-files:
	@echo "=== Checking input files ==="
	@if [ ! -f "$(VIDEO)" ]; then \
		echo "ERROR: Input video not found: $(VIDEO)"; \
		echo "Available videos:"; \
		ls -la media/*.mp4 2>/dev/null || echo "No videos found in media/"; \
		exit 1; \
	fi
	@echo "✓ Input video found: $(VIDEO)"
	@echo "=== File check passed ==="

# ---------- Main workflow stages ----------
prep: check-env check-files
	@echo "=== Starting PREP stage ==="
	@echo "Video: $(VIDEO)"
	@echo "Workdir: $(WORKDIR)"
	@echo "Creating work directory..."
	@mkdir -p "$(WORKDIR)"
	@echo "Running transcription and segmentation..."
	@PYTHONUNBUFFERED=1 $(POETRY) run python -c "import dubber.cli; import sys; args = ['dubber.cli', '--input_video', '$(VIDEO)', '--workdir', '$(WORKDIR)', '--stage', 'prep', '--stt', 'local', '--skip-polish', '--align-mode', 'sentence', '--sentence-join-gap', '$(SENT_JOIN_GAP)', '--sentences-per-chunk', '$(SENT_PER_CHUNK)', '--verbose']; args.extend(['--source-language', '$(SOURCE_LANGUAGE)']) if '$(SOURCE_LANGUAGE)' != '' else None; args.extend(['--translate']) if '$(TRANSLATE)' == 'true' else None; args.extend(['--translation-model', '$(TRANSLATION_MODEL)']) if '$(TRANSLATE)' == 'true' else None; sys.argv = args; dubber.cli.main()" 2>&1 | tee "$(WORKDIR)/prep.log" || echo "Prep completed with warnings"
	@echo "Checking prep results..."
	@if [ ! -f "$(WORKDIR)/subs.srt" ]; then \
		echo "ERROR: subs.srt not created"; \
		echo "Check log: $(WORKDIR)/prep.log"; \
		exit 1; \
	fi
	@if [ ! -f "$(WORKDIR)/sentences_groups.json" ]; then \
		echo "ERROR: sentences_groups.json not created"; \
		echo "Check log: $(WORKDIR)/prep.log"; \
		exit 1; \
	fi
	@echo "✓ Prep stage completed successfully"
	@echo "Generated files:"
	@ls -la "$(WORKDIR)/"*.srt "$(WORKDIR)/"*.json 2>/dev/null || true
	@echo "=== PREP stage completed ==="

synth: check-env
	@echo "=== Starting SYNTH stage ==="
	@echo "Video: $(VIDEO)"
	@echo "Workdir: $(WORKDIR)"
	@echo "Dubbed output: $(DUBBED_OUTPUT)"
	@if [ ! -f "$(WORKDIR)/subs.srt" ]; then \
		echo "ERROR: subs.srt not found. Run 'make prep' first"; \
		exit 1; \
	fi
	@echo "Running TTS synthesis..."
	@PYTHONUNBUFFERED=1 $(POETRY) run python -c "import dubber.cli; import sys; sys.argv = ['dubber.cli', '--input_video', '$(VIDEO)', '--workdir', '$(WORKDIR)', '--stage', 'synth', '--stt', 'local', '--align-mode', 'sentence', '--sentence-join-gap', '$(SENT_JOIN_GAP)', '--sentences-per-chunk', '$(SENT_PER_CHUNK)', '--tts-provider', '$(TTS_PROVIDER)', '--tts-model', '$(TTS_MODEL)', '--voice-main', '$(VOICE)', '--output', '$(DUBBED_OUTPUT)', '$(VERBOSE)']; dubber.cli.main()" > "$(WORKDIR)/synth.log" 2>&1 || echo "Synthesis completed with warnings"
	@echo "Checking synth results..."
	@if [ ! -f "$(DUBBED_OUTPUT)" ]; then \
		echo "ERROR: Dubbed video not created: $(DUBBED_OUTPUT)"; \
		echo "Check log: $(WORKDIR)/synth.log"; \
		exit 1; \
	fi
	@echo "✓ Synth stage completed successfully"
	@echo "Dubbed video: $(DUBBED_OUTPUT) ($(shell du -h "$(DUBBED_OUTPUT)" 2>/dev/null | cut -f1))"
	@echo "=== SYNTH stage completed ==="

burn: check-env
	@echo "=== Starting BURN stage ==="
	@echo "Dubbed video: $(DUBBED_OUTPUT)"
	@echo "Workdir: $(WORKDIR)"
	@echo "Final output: $(OUTPUT)"
	@if [ ! -f "$(WORKDIR)/subs.srt" ]; then \
		echo "ERROR: subs.srt not found. Run 'make prep' first"; \
		exit 1; \
	fi
	@if [ ! -f "$(DUBBED_OUTPUT)" ]; then \
		echo "ERROR: Dubbed video not found. Run 'make synth' first"; \
		exit 1; \
	fi
	@echo "Burning subtitles to dubbed video..."
	@PYTHONUNBUFFERED=1 $(POETRY) run python -c "import dubber.cli; import sys; sys.argv = ['dubber.cli', '--only-burn-subs', '--input_video', '$(DUBBED_OUTPUT)', '--workdir', '$(WORKDIR)', '--subs-path', '$(WORKDIR)/subs.srt', '--burn-mode', 'soft', '--output', '$(OUTPUT)', '--verbose']; dubber.cli.main()" 2>&1 | tee "$(WORKDIR)/burn.log" || echo "Burn completed with warnings"
	@echo "Checking burn results..."
	@if [ ! -f "$(OUTPUT)" ]; then \
		echo "ERROR: Final output video not created: $(OUTPUT)"; \
		echo "Check log: $(WORKDIR)/burn.log"; \
		exit 1; \
	fi
	@echo "Cleaning up temporary dubbed file..."
	@rm -f "$(DUBBED_OUTPUT)"
	@echo "✓ Burn stage completed successfully"
	@echo "Final video: $(OUTPUT) ($(shell du -h "$(OUTPUT)" 2>/dev/null | cut -f1))"
	@echo "=== BURN stage completed ==="

youtube: check-env
	@echo "=== Starting YOUTUBE stage ==="
	@echo "Video: $(VIDEO)"
	@echo "Workdir: $(WORKDIR)"
	@echo "YouTube outdir: $(YOUTUBE_OUTDIR)"
	@if [ ! -f "$(WORKDIR)/subs.srt" ]; then \
		echo "ERROR: subs.srt not found. Run 'make prep' first"; \
		exit 1; \
	fi
	@echo "Creating YouTube output directory..."
	@mkdir -p "$(YOUTUBE_OUTDIR)"
	@echo "Generating YouTube assets..."
	@PYTHONUNBUFFERED=1 $(POETRY) run python -c "import dubber.cli; import sys; sys.argv = ['dubber.cli', '--input_video', '$(VIDEO)', '--workdir', '$(WORKDIR)', '--stage', 'youtube', '--youtube-outdir', '$(YOUTUBE_OUTDIR)', '--youtube-source', '$(YOUTUBE_SOURCE)', '--youtube-gap-thresh', '$(YOUTUBE_GAP_THRESH)', '--youtube-ai', '$(VERBOSE)']; dubber.cli.main()" > "$(WORKDIR)/youtube.log" 2>&1 || echo "YouTube generation completed with warnings"
	@echo "Checking YouTube results..."
	@if [ ! -f "$(YOUTUBE_OUTDIR)/chapters.txt" ]; then \
		echo "ERROR: chapters.txt not created"; \
		echo "Check log: $(WORKDIR)/youtube.log"; \
		exit 1; \
	fi
	@if [ ! -f "$(YOUTUBE_OUTDIR)/description.md" ]; then \
		echo "ERROR: description.md not created"; \
		echo "Check log: $(WORKDIR)/youtube.log"; \
		exit 1; \
	fi
	@echo "✓ YouTube stage completed successfully"
	@echo "Generated YouTube assets:"
	@ls -la "$(YOUTUBE_OUTDIR)/"
	@echo "=== YOUTUBE stage completed ==="

# ---------- Full workflows ----------
full: setup prep synth burn youtube
	@echo "=== FULL WORKFLOW COMPLETED ==="
	@echo "All stages completed successfully!"
	@echo "Output video: $(OUTPUT)"
	@echo "YouTube assets: $(YOUTUBE_OUTDIR)/"

full-ai: setup prep synth burn youtube
	@echo "=== FULL AI WORKFLOW COMPLETED ==="
	@echo "All stages completed successfully with AI enhancement!"
	@echo "Output video: $(OUTPUT)"
	@echo "YouTube assets: $(YOUTUBE_OUTDIR)/"

# ---------- Async workflows ----------
synth-async: check-env
	@echo "=== Starting ASYNC SYNTH stage ==="
	@echo "Video: $(VIDEO)"
	@echo "Workdir: $(WORKDIR)"
	@echo "Dubbed output: $(DUBBED_OUTPUT)"
	@if [ ! -f "$(WORKDIR)/subs.srt" ]; then \
		echo "ERROR: subs.srt not found. Run 'make prep' first"; \
		exit 1; \
	fi
	@echo "Running async TTS synthesis..."
	PYTHONUNBUFFERED=1 $(POETRY) run python -m dubber.cli_async --input_video "$(VIDEO)" --workdir "$(WORKDIR)" \
	  --stage synth --stt local \
	  --align-mode sentence --sentence-join-gap $(SENT_JOIN_GAP) --sentences-per-chunk $(SENT_PER_CHUNK) \
	  --tts-provider $(TTS_PROVIDER) --tts-model $(TTS_MODEL) --voice-main $(VOICE) \
	  --output "$(DUBBED_OUTPUT)" --async-tts --max-concurrent $(MAX_CONCURRENT) $(VERBOSE) 2>&1 | tee "$(WORKDIR)/synth_async.log"
	@echo "Checking async synth results..."
	@if [ ! -f "$(DUBBED_OUTPUT)" ]; then \
		echo "ERROR: Dubbed video not created: $(DUBBED_OUTPUT)"; \
		echo "Check log: $(WORKDIR)/synth_async.log"; \
		exit 1; \
	fi
	@echo "✓ Async synth stage completed successfully"
	@echo "Dubbed video: $(DUBBED_OUTPUT) ($(shell du -h "$(DUBBED_OUTPUT)" 2>/dev/null | cut -f1))"
	@echo "=== ASYNC SYNTH stage completed ==="

full-async: setup prep synth-async burn youtube
	@echo "=== FULL ASYNC WORKFLOW COMPLETED ==="
	@echo "All stages completed successfully with async TTS!"
	@echo "Output video: $(OUTPUT)"
	@echo "YouTube assets: $(YOUTUBE_OUTDIR)/"

# ---------- Development ----------
clean:
	@echo "=== Cleaning up ==="
	rm -rf .work */__pycache__ **/__pycache__ .pytest_cache dist build
	@echo "✓ Cleanup completed"

fmt:
	@echo "=== Formatting code ==="
	$(POETRY) run ruff format .
	$(POETRY) run ruff check --fix .
	@echo "✓ Code formatting completed"

lint:
	@echo "=== Linting code ==="
	$(POETRY) run ruff check .
	@echo "✓ Linting completed"

typecheck:
	@echo "=== Type checking ==="
	$(POETRY) run pyright
	@echo "✓ Type checking completed"

test:
	@echo "=== Running tests ==="
	$(POETRY) run pytest
	@echo "✓ Tests completed"

# ---------- Debug and troubleshooting ----------
debug-prep: check-env check-files
	@echo "=== DEBUG: Running prep with full output ==="
	@mkdir -p "$(WORKDIR)"
	@echo "Using direct Python call to bypass Poetry output issues..."
	PYTHONUNBUFFERED=1 $(POETRY) run python -c "import dubber.cli; import sys; sys.argv = ['dubber.cli', '--input_video', '$(VIDEO)', '--workdir', '$(WORKDIR)', '--stage', 'prep', '--stt', 'local', '--skip-polish', '--align-mode', 'sentence', '--sentence-join-gap', '$(SENT_JOIN_GAP)', '--sentences-per-chunk', '$(SENT_PER_CHUNK)', '--verbose']; dubber.cli.main()"

debug-synth: check-env
	@echo "=== DEBUG: Running synth with full output ==="
	$(PY) $(ENTRY) --input_video "$(VIDEO)" --workdir "$(WORKDIR)" \
	  --stage synth --stt local \
	  --align-mode sentence --sentence-join-gap $(SENT_JOIN_GAP) --sentences-per-chunk $(SENT_PER_CHUNK) \
	  --tts-provider $(TTS_PROVIDER) --tts-model $(TTS_MODEL) --voice-main $(VOICE) \
	  --output "$(DUBBED_OUTPUT)" --verbose

debug-synth-async: check-env
	@echo "=== DEBUG: Running async synth with full output ==="
	PYTHONUNBUFFERED=1 $(POETRY) run python -m dubber.cli_async --input_video "$(VIDEO)" --workdir "$(WORKDIR)" \
	  --stage synth --stt local \
	  --align-mode sentence --sentence-join-gap $(SENT_JOIN_GAP) --sentences-per-chunk $(SENT_PER_CHUNK) \
	  --tts-provider $(TTS_PROVIDER) --tts-model $(TTS_MODEL) --voice-main $(VOICE) \
	  --output "$(DUBBED_OUTPUT)" --async-tts --max-concurrent $(MAX_CONCURRENT) --verbose

show-logs:
	@echo "=== Recent logs ==="
	@if [ -d "$(WORKDIR)" ]; then \
		echo "Workdir: $(WORKDIR)"; \
		ls -la "$(WORKDIR)/"*.log 2>/dev/null || echo "No log files found"; \
		for log in "$(WORKDIR)/"*.log; do \
			if [ -f "$$log" ]; then \
				echo "=== $$log ==="; \
				tail -20 "$$log"; \
				echo ""; \
			fi; \
		done; \
	else \
		echo "No workdir found: $(WORKDIR)"; \
	fi

# ---------- Help ----------
help:
	@echo "Video Dubbing Pipeline - Available targets:"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Install dependencies with Poetry"
	@echo ""
	@echo "Workflow stages:"
	@echo "  prep           - Extract audio, transcribe, create segments and sentences"
	@echo "  synth          - Synthesize TTS and create dubbed video"
	@echo "  burn           - Add subtitles to video"
	@echo "  youtube        - Generate YouTube description and chapters"
	@echo ""
	@echo "Full workflows:"
	@echo "  full           - Complete workflow: setup + prep + synth + burn + youtube"
	@echo "  full-ai        - Complete workflow with AI-enhanced YouTube description"
	@echo "  full-async     - Complete workflow with async TTS processing (faster)"
	@echo ""
	@echo "Debug and troubleshooting:"
	@echo "  debug-prep        - Run prep with full debug output"
	@echo "  debug-synth       - Run synth with full debug output"
	@echo "  debug-synth-async - Run async synth with full debug output"
	@echo "  show-logs         - Show recent log files"
	@echo "  check-env      - Check environment setup"
	@echo "  check-files    - Check input files"
	@echo ""
	@echo "Development:"
	@echo "  clean          - Remove work directories and cache files"
	@echo "  fmt            - Format code with ruff"
	@echo "  lint           - Lint code with ruff"
	@echo "  typecheck      - Type check with pyright"
	@echo "  test           - Run tests"
	@echo ""
	@echo "Configuration variables (override with VAR=value):"
	@echo "  VIDEO          - Input video file (default: media/1.mp4)"
	@echo "  WORKDIR        - Working directory (default: ./.work/lesson)"
	@echo "  OUTPUT         - Final output video file (default: out/lesson_dubbed.mp4)"
	@echo "  DUBBED_OUTPUT  - Temporary dubbed video file (default: out/lesson_dubbed_temp.mp4)"
	@echo "  TTS_PROVIDER   - TTS provider: openai or elevenlabs (default: openai)"
	@echo "  VOICE          - TTS voice (default: onyx)"
	@echo "  MAX_CONCURRENT - Max concurrent TTS requests for async mode (default: 5)"
	@echo "  SOURCE_LANGUAGE- Source language code (e.g., 'ru', 'de', 'fr'). Auto-detected if empty"
	@echo "  TRANSLATE       - Enable translation to English: true/false (default: false)"
	@echo "  TRANSLATION_MODEL - GPT model for translation (default: gpt-4o-mini)"
	@echo "  YOUTUBE_SOURCE - YouTube chapters source: sentences, srt, blocks (default: sentences)"
	@echo "  VERBOSE        - Add --verbose flag (default: --verbose)"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make prep VIDEO=media/2.mp4 WORKDIR=.work/lesson2"
	@echo "  make synth TTS_PROVIDER=elevenlabs VOICE=rachel"
	@echo "  make full VIDEO=media/russian_video.mp4 SOURCE_LANGUAGE=ru TRANSLATE=true"
	@echo "  make prep VIDEO=media/german_video.mp4 SOURCE_LANGUAGE=de TRANSLATE=true"
	@echo "  make synth-async MAX_CONCURRENT=10  # Faster async processing"
	@echo "  make youtube YOUTUBE_SOURCE=srt YOUTUBE_GAP_THRESH=2.0"
	@echo "  make full VIDEO=media/tutorial.mp4 OUTPUT=out/tutorial_dubbed.mp4"
	@echo "  make full-async VIDEO=media/tutorial.mp4  # Complete async workflow"
	@echo "  make debug-synth-async VIDEO=media/2.mp4 WORKDIR=.work/lesson2"