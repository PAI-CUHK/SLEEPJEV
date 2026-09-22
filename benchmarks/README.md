# Benchmarks

The public benchmark entry point measures the intended serving pattern: encode one complete synthetic night once, answer a workload from the cached representation, compare with dense readout, and compare with re-encoding the night for every query.

```bash
python benchmarks/run_query_reuse.py --epochs 640 --output query-reuse.json
```

The output records one-time encoding, sparse and dense query time, cold re-encoding time, speedup ratios, and the number of selected versus dense tokens. The default benchmark is a software smoke benchmark, not a clinical performance report.

For a publishable result, pin the commit, Python/PyTorch versions, device, thread settings, feature schema, model configuration, warm-up policy, repeat count, and workload manifest. Report median and tail latency, memory, quality drift, and whether one-time encoding is included. Use the workload metrics in `sleepjev.workload` for event retrieval and keep positive-unlabeled policies visible.
