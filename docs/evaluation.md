# Evaluation

Evaluation has three separate layers:

1. **Signal quality:** sleep-stage accuracy, macro-F1, balanced accuracy, and kappa on a defined split.
2. **Runtime workload:** event query positive hit rate, positive event recall at top-k, candidate count, and query-level probability summaries.
3. **Serving efficiency:** one-time encode, sparse/dense readout, cold re-encode, memory, and quality drift.

Do not collapse these into one score. Event annotations may be positive-unlabeled, so unannotated epochs cannot be counted as true negatives. Report the unknown-label policy with every event result. Compare SLEEPJEV to a fixed-head baseline under the same feature schema, split, seed, and workload.
