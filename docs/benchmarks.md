# Benchmarks and Claims

## Current evidence level

The homepage reports the final **two-seed averages** supplied for the current
release summary. The complete table and claim boundary are in
[`results.md`](results.md). Dataset release, subject-level split, checkpoint hash,
hardware, thread policy, exact seeds, and timing manifest still need to be attached
before these values are used as a paper or external benchmark claim.

Older research notes contain a Sleep-EDF pilot stage accuracy of **72.57%** and
macro-F1 of **40.27%**, plus a preliminary query-reuse timing probe reporting
**0.91x, 3.84x, 6.52x, 13.11x, and 14.08x** at Q=`1, 8, 32, 128, 512`.
Those archived values are retained for provenance only and should not be compared
directly with the final release snapshot.

## Re-run the software protocol

```bash
python benchmarks/run_query_reuse.py --epochs 640 --output query-reuse.json
```

The benchmark compares a single cached encode plus sparse readout, cached dense readout, and re-encoding the night for every query. Before publishing numbers, pin hardware, device, threads, warm-up, repetitions, batch shape, model configuration, query manifest, and whether one-time encoding is included. Report median and tail latency rather than one unqualified wall-clock number.

## Required comparisons

- fixed-head and shared multi-task baselines;
- candidate permutation and query-isolation tests;
- sparse versus dense quality agreement;
- stage/event constraint leakage audit;
- positive-unlabeled event policy;
- multi-seed and subject-level split evaluation;
- failure cases and calibration scope.
