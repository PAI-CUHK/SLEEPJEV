# Contributing to SLEEPJEV

Contributions should improve the research software without weakening the evidence, data, or safety boundaries.

## Before opening a pull request

1. Create or update a focused test for changed behavior.
2. Run `pytest -q`.
3. Run `ruff check src tests` when touching Python code.
4. Do not commit patient text, private notes, credentials, API keys, datasets, checkpoints, or generated experiment archives.
5. Document the source, license, revision, and split for every new dataset or model dependency.
6. Keep benchmark observations separate from clinical claims.

## Pull requests

Describe the contract that changed, the tests that cover it, and whether the change affects calibration, candidate semantics, data leakage, or reproducibility. Large experiment outputs should be stored outside the source repository with a manifest and immutable commit reference.
