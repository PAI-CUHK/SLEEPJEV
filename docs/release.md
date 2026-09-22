# Release Checklist

- [x] Replace repository badges and URLs with the canonical GitHub repository.
- [ ] Replace the placeholder author, affiliation, contact, and copyright fields.
- [ ] Confirm authors, affiliation, citation, and copyright holder.
- [ ] Review the MIT license and all dataset/checkpoint licenses.
- [ ] Run `pytest -q`, `ruff check src tests`, and `python -m build`.
- [ ] Run both synthetic examples and the benchmark wrapper.
- [ ] Scan the full Git history for credentials, private paths, EDF files, and checkpoints.
- [ ] Confirm every reported result has a dataset, split, seed, device, checkpoint, and timing manifest.
- [ ] Remove stale legacy-package references and generated artifacts.
- [ ] Add a model card and data statement before distributing weights.
- [ ] Enable GitHub secret scanning, dependency alerts, issue templates, and branch protection.
