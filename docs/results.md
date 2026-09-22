# Results Snapshot

## Evidence status

The tables below are the project's final **two-seed averages** supplied for the
current release summary. They supersede the earlier pilot snapshot, but they are
still only as reproducible as the associated experiment manifest. Before
publication, record the dataset release, subject-level split, checkpoint hash,
hardware, thread policy, exact seeds, and timing boundary.

## Effect comparison

| Method | Stage macro-F1 | Q=512 hit@5 | Q=512 PU recall@5 |
| --- | ---: | ---: | ---: |
| Independent DL | **0.5948** | 0.6666 | **0.1132** |
| Fixed Head | 0.5767 | 0.6551 | 0.1067 |
| SLEEPJEV | 0.5640 | **0.6704** | 0.1073 |
| Shared Multi-task DL | 0.5616 | 0.6658 | 0.1131 |

Interpretation: Independent DL has the best stage macro-F1 and PU recall. SLEEPJEV
has the best Q=512 hit@5. The defensible claim is competitive quality under a
high-query workload, not across-the-board superiority.

## Query-scale effect

| Q | SLEEPJEV hit@5 | Independent DL hit@5 | SLEEPJEV PU recall@5 | Independent DL PU recall@5 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5000 | 0.5000 | **0.2222** | 0.2083 |
| 8 | **0.7153** | 0.6182 | **0.1574** | 0.1393 |
| 32 | **0.6532** | 0.6374 | 0.1163 | **0.1245** |
| 128 | **0.6735** | 0.6663 | 0.1107 | **0.1189** |
| 512 | **0.6704** | 0.6666 | 0.1073 | **0.1132** |

SLEEPJEV hit@5 is higher than Independent DL for Q=8, 32, 128, and 512. PU recall
is mixed and should be reported alongside hit@5.

## Speed comparison

All values are milliseconds. SLEEPJEV first plan includes first query-plan
construction; steady serving reuses the overnight cache and compiled plan.

| Q | SLEEPJEV first plan | SLEEPJEV steady serving | Independent DL warm |
| ---: | ---: | ---: | ---: |
| 1 | 0.395 | 0.206 | 0.198 |
| 8 | 0.417 | 0.213 | 0.343 |
| 32 | 0.452 | 0.245 | 0.726 |
| 128 | 0.610 | 0.359 | 2.248 |
| 512 | 1.268 | **0.833** | 8.983 |

Relative to Independent DL warm serving, the reported SLEEPJEV speed ratios are
0.96x, 1.61x, 2.96x, 6.26x, and 10.8x for Q=1, 8, 32, 128, and 512. At Q=512,
the supplied comparison additionally reports dense JEV readout at 21.520 ms,
sparse serving at 0.833 ms, fresh overnight re-encoding at 1193 ms, and
`exact match=True` between sparse and dense event scores.

EDF reading and one-time overnight cache construction are excluded from this warm
serving table and must be reported separately.

## Quality-adjusted query throughput

The project uses the conservative composite metric:

```text
QAI = sqrt(hit@5 * PU-recall@5)
quality-retained speedup = raw speedup * min(1, QAI_SLEEPJEV / QAI_baseline)
QA-QPS = Q * QAI / serving_time_ms
```

The supplied QAI values are shown with baseline-first columns. The quality-retained
speedup is recomputed from the displayed QAI and serving-time values; the raw speedup
is included to make the timing and quality adjustment auditable.

| Q | Independent DL QAI | SLEEPJEV QAI | Raw serving speedup | Quality-retained speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3333 | 0.3227 | 0.96x | 0.93x |
| 8 | 0.3356 | 0.2934 | 1.61x | 1.41x |
| 32 | 0.2816 | 0.2757 | 2.96x | 2.90x |
| 128 | 0.2815 | 0.2731 | 6.26x | 6.08x |
| 512 | 0.2747 | 0.2683 | 10.8x | **10.53x** |

The derived quality-adjusted query throughput is:

| Q | SLEEPJEV QA-QPS | Independent DL QA-QPS | SLEEPJEV / Independent DL |
| ---: | ---: | ---: | ---: |
| 1 | 1.5665 | 1.6833 | 0.93x |
| 8 | 11.0197 | 7.8274 | 1.41x |
| 32 | 36.0098 | 12.4121 | 2.90x |
| 128 | 97.3727 | 16.0285 | 6.08x |
| 512 | **164.9095** | 15.6570 | **10.53x** |

This composite metric discounts speed when SLEEPJEV quality is lower. It supports
the main systems conclusion: high-Q serving provides the largest advantage while
quality remains in the same range.

## Claim boundary

- Independent DL remains the strongest stage-F1 baseline.
- SLEEPJEV PU recall is slightly lower at Q=32, 128, and 512.
- The speed advantage is for warm serving; it excludes EDF I/O and cache building.
- The strongest claim is **high-Q quality retention plus query-conditioned sparse
  readout efficiency**, not universal clinical or predictive superiority.
- These values do not establish clinical utility, diagnostic validity, or patient
  level generalization.
