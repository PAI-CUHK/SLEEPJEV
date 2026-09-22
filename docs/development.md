# Development

The project uses a `src/` layout, `pytest`, Ruff, and GitHub Actions. Install `.[test,dev]` and run:

```bash
pytest -q
ruff check src tests
python -m compileall -q src
python -m build
```

Keep runtime contracts typed and focused. Add a test when changing query semantics, masking, indexing, event-label policy, or serving behavior. Synthetic tests should remain CPU-safe and should not require private data, checkpoints, network access, or credentials.
