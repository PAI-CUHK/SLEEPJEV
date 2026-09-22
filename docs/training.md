# Training

The training helpers in `sleepjev.train` are intentionally small research utilities. They support stage queries, binary event tasks, event index loss, feature-schema harmonization, and fixed-head baselines. They are suitable for a controlled pilot, not a turnkey clinical training pipeline.

Typical flow:

```python
model = SleepJEV(input_dim=train_nights[0].feature_dim)
fit_sleepjev_tasks(model, train_nights, epochs=5)
metrics = evaluate_sleepjev_tasks(model, test_nights)
```

Use subject-level or recording-level splits before feature normalization. Fit normalization on training data only. Keep calibration and final test data separate. Save a manifest with the source commit, seed, split, feature schema, optimizer, device, and checkpoint hash.
