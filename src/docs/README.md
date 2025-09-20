# Documentation

This directory contains comprehensive documentation for the Video Dubbing Pipeline.

## Quick Navigation

### 📖 [ASYNC_GUIDE.md](ASYNC_GUIDE.md)
**Complete async processing guide**
- Performance comparison tables
- Configuration options
- Best practices and troubleshooting
- When to use async vs sync mode

### 🔧 [API_ASYNC.md](API_ASYNC.md)
**Detailed async API reference**
- Function signatures and parameters
- Code examples and integration patterns
- Error handling and retry logic
- Performance tuning guidelines

### 💡 [ASYNC_EXAMPLES.md](ASYNC_EXAMPLES.md)
**Practical examples and use cases**
- Basic usage examples
- Advanced integration patterns
- Performance testing scripts
- Troubleshooting examples

### 📋 [DOCS_INDEX.md](DOCS_INDEX.md)
**Complete documentation index**
- Quick reference tables
- Command overview
- Configuration guide
- File structure

## Getting Started

1. **New to async processing?** Start with [ASYNC_GUIDE.md](ASYNC_GUIDE.md)
2. **Need code examples?** Check [ASYNC_EXAMPLES.md](ASYNC_EXAMPLES.md)
3. **Looking for API details?** See [API_ASYNC.md](API_ASYNC.md)
4. **Need quick reference?** Use [DOCS_INDEX.md](DOCS_INDEX.md)

## Performance Quick Reference

| Mode | Speed | Use Case |
|------|-------|----------|
| Sync | 1x | Debugging, small videos |
| Async (3x) | 3.6x | Conservative |
| Async (5x) | 4.9x | **Recommended** |
| Async (10x) | 6.1x | High-performance |

## Common Commands

```bash
# Complete async workflow
make full-async VIDEO=media/your_video.mp4

# Async synthesis only
make synth-async VIDEO=media/your_video.mp4

# Debug mode
make debug-synth-async VIDEO=media/your_video.mp4
```

## Contributing to Documentation

When adding new features or updating documentation:

1. Update the relevant guide files
2. Add examples to `ASYNC_EXAMPLES.md`
3. Update API reference in `API_ASYNC.md`
4. Update the index in `DOCS_INDEX.md`
5. Test all examples and links

## Support

For questions about async processing:

1. Check the troubleshooting section in [ASYNC_GUIDE.md](ASYNC_GUIDE.md)
2. Review examples in [ASYNC_EXAMPLES.md](ASYNC_EXAMPLES.md)
3. Check API reference in [API_ASYNC.md](API_ASYNC.md)
4. Open an issue with sample files and error messages
