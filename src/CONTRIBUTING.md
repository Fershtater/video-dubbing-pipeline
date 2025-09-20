# Contributing to Video Dubbing Pipeline

Thank you for your interest in contributing to the Video Dubbing Pipeline! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- ffmpeg and ffprobe
- Git

### Development Setup

1. **Fork the repository**

   ```bash
   git clone https://github.com/your-username/video-dubbing-pipeline.git
   cd video-dubbing-pipeline
   ```

2. **Install dependencies**

   ```bash
   make setup
   # or manually:
   poetry install
   ```

3. **Set up environment**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run tests**
   ```bash
   make test
   ```

## Development Workflow

### Code Style

We use the following tools for code quality:

```bash
# Format code
make fmt

# Lint code
make lint

# Type checking
make typecheck

# Run all checks
make fmt lint typecheck test
```

### Testing

```bash
# Run all tests
make test

# Run specific test file
poetry run python -m pytest tests/test_specific.py

# Run with coverage
poetry run python -m pytest --cov=dubber tests/
```

### Adding New Features

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

   - Follow existing code style
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**

   ```bash
   make fmt lint typecheck test
   ```

4. **Update documentation**

   - Update relevant `.md` files in `docs/`
   - Add examples if applicable
   - Update API reference if needed

5. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Async Processing Contributions

### Adding New Async Features

When adding async functionality:

1. **Follow async patterns**

   - Use `asyncio` for async functions
   - Use `httpx` for async HTTP requests
   - Use `tqdm[asyncio]` for progress bars

2. **Update documentation**

   - Add to `docs/API_ASYNC.md`
   - Add examples to `docs/ASYNC_EXAMPLES.md`
   - Update `docs/ASYNC_GUIDE.md` if needed

3. **Test thoroughly**
   - Test with different concurrency levels
   - Test error handling and retry logic
   - Test performance improvements

### Example Async Function

```python
import asyncio
from typing import List, Awaitable
from tqdm.asyncio import tqdm

async def your_async_function(items: List[str]) -> List[str]:
    """Your async function with progress bar."""
    results = []

    async def process_item(item: str) -> str:
        # Your async processing logic
        return processed_item

    tasks = [process_item(item) for item in items]

    for result in tqdm.as_completed(tasks, desc="Processing"):
        results.append(await result)

    return results
```

## Documentation Guidelines

### Writing Documentation

1. **Use clear, concise language**
2. **Include code examples**
3. **Update all relevant files**
4. **Test all examples**

### Documentation Structure

- `README.md` - Main project documentation
- `docs/ASYNC_GUIDE.md` - Async processing guide
- `docs/API_ASYNC.md` - API reference
- `docs/ASYNC_EXAMPLES.md` - Practical examples
- `docs/DOCS_INDEX.md` - Documentation index

### Adding Examples

When adding examples:

1. **Test the example**
2. **Include expected output**
3. **Add error handling**
4. **Update relevant documentation**

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Examples are tested
- [ ] No linting errors
- [ ] Type checking passes

### PR Description

Include:

1. **Summary** - What does this PR do?
2. **Changes** - What files were modified?
3. **Testing** - How was it tested?
4. **Documentation** - What docs were updated?
5. **Performance** - Any performance impact?

### Example PR Description

```markdown
## Summary

Adds async batch processing for TTS synthesis with 3-5x performance improvement.

## Changes

- Added `tts_async.py` with async TTS functions
- Added `timeline_async.py` for async timeline building
- Added `cli_async.py` for async CLI interface
- Updated `Makefile` with async targets

## Testing

- Tested with 37 segments: 73s → 15s (4.9x speedup)
- Tested with different concurrency levels (3, 5, 10)
- All existing tests pass

## Documentation

- Updated `docs/ASYNC_GUIDE.md`
- Added `docs/API_ASYNC.md`
- Added `docs/ASYNC_EXAMPLES.md`
- Updated `README.md` with async examples

## Performance

- 3-5x faster TTS processing
- Configurable concurrency (MAX_CONCURRENT)
- Real-time progress bars
```

## Issue Guidelines

### Reporting Bugs

Include:

1. **Environment** - OS, Python version, dependencies
2. **Steps to reproduce** - Clear reproduction steps
3. **Expected behavior** - What should happen
4. **Actual behavior** - What actually happens
5. **Logs** - Relevant error messages and logs
6. **Sample files** - If applicable

### Feature Requests

Include:

1. **Use case** - Why is this needed?
2. **Proposed solution** - How should it work?
3. **Alternatives** - Other approaches considered
4. **Implementation** - Any implementation ideas?

## Code Review Process

### Reviewers

- Check code quality and style
- Verify tests and documentation
- Test functionality
- Check performance impact

### Authors

- Address review feedback
- Update documentation if needed
- Add tests for new functionality
- Respond to questions

## Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR** - Breaking changes
- **MINOR** - New features, backward compatible
- **PATCH** - Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version is bumped
- [ ] CHANGELOG is updated
- [ ] Release notes are written

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the golden rule

### Getting Help

- Check existing issues and PRs
- Read the documentation
- Ask questions in issues
- Join discussions

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Thank You

Thank you for contributing to the Video Dubbing Pipeline! Your contributions help make video dubbing more accessible and efficient for everyone.
