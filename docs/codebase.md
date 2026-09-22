# Codebase Guide

| Path | Responsibility |
| --- | --- |
| `src/sleepjev/types.py` | `SleepNight`, `SleepQuery`, events, labels, and constants |
| `src/sleepjev/model.py` | hierarchical encoder, cache, sparse retrieval, option scorer, answers |
| `src/sleepjev/index.py` | time, stage, position, quality, and annotation postings |
| `src/sleepjev/events.py` | XML/EDF event normalization and unknown-label handling |
| `src/sleepjev/data.py` | Sleep-EDF/NSRR adapters and NPZ feature cache I/O |
| `src/sleepjev/baselines.py` | independent, shared multi-task, and fixed-head controls |
| `src/sleepjev/train.py` | fit/evaluate helpers and query-reuse benchmark |
| `src/sleepjev/workload.py` | query-level event retrieval metrics |
| `examples/` | data-free API demonstrations |
| `benchmarks/` | command-line benchmark wrapper and manifests |
| `tests/` | behavioral contracts and leakage guards |

The source layout deliberately contains one public package. A separate text-evidence prototype is not part of this repository. Experimental raw scripts, credentials, checkpoints, and result archives belong outside Git.
