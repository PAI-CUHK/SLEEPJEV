# Repository Style References

These public repositories are design references for SLEEPJEV. They are cited for
repository organization and communication patterns, not as implementation
dependencies or evidence for SLEEPJEV's scientific claims.

| Reference | What to borrow | SLEEPJEV adaptation |
| --- | --- | --- |
| [MNE-Python](https://github.com/mne-tools/mne-python) | Scientific Python package layout, examples, contributor guidance, and explicit data conventions | Keep the `src/` package small, put runnable synthetic examples beside the package, and document data provenance separately from code |
| [Braindecode](https://github.com/braindecode/braindecode) | EEG-focused API documentation, benchmark-oriented examples, and model/data separation | Make `SleepNight` and `SleepQuery` the stable public contracts; keep dataset adapters and workload benchmarks explicit |
| [MONAI](https://github.com/Project-MONAI/MONAI) | Medical-AI safety language, modular components, CI, and reproducibility expectations | State clearly that SLEEPJEV is research software, keep patient data/checkpoints external, and require split/seed/device metadata for results |
| [PyHealth](https://github.com/sunlabuiuc/PyHealth) | Clinical ML documentation organized around data, tasks, examples, and evaluation | Separate method, data, training, evaluation, and limitations docs; describe event-label coverage instead of flattening unknowns |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Stable small APIs, estimator contracts, tests, and predictable naming | Keep query validation and cache behavior testable, deterministic, and easy to exercise on CPU |

## The chosen house style

SLEEPJEV combines these patterns into a deliberately compact research-software
repository:

1. **One public package.** Runtime code lives under `src/sleepjev/`; experimental
   scratch code and private evidence are outside the public tree.
2. **A short path to a working result.** `python -m sleepjev demo` and the two
   scripts under `examples/` run with synthetic data and no checkpoint download.
3. **Claims are separated from mechanisms.** The method docs explain what the
   implementation does; the benchmark docs state which observations are archived,
   which are reproducible, and which are still missing.
4. **Medical scope is explicit.** The package is not a diagnostic device, and
   probabilities over runtime options must not be presented as patient risk.
5. **Visuals are inspectable.** The repository keeps editable SVG diagrams and
   GPT-IMAGE prompts rather than opaque screenshots with unreviewed labels.

## What not to copy

Do not copy large framework surface area, hidden data downloads, benchmark numbers
without a protocol, or repository badges that imply a released paper, clinical
validation, or production readiness. The public repository URL and CI badge should
remain synchronized with the canonical GitHub project.
