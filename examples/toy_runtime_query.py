"""Run the smallest SLEEPJEV runtime-query example without private data."""

from __future__ import annotations

import numpy as np
import torch

from sleepjev import SleepJEV, SleepNight, SleepQuery


def make_night() -> SleepNight:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(120, 6)).astype(np.float32)
    stages = np.asarray(["W", "N1", "N2", "N3", "REM"] * 24)
    return SleepNight("toy-night", features, stages)


def main() -> None:
    torch.manual_seed(7)
    night = make_night()
    model = SleepJEV(input_dim=night.feature_dim, hidden_size=32, coarse_factor=2, hour_factor=8).eval()
    cache = model.encode(torch.from_numpy(night.features))
    queries = [
        SleepQuery("sleep_stage", options=("W", "N2", "REM"), start_epoch=10, end_epoch=40),
        SleepQuery("apnea", "boolean", ("negative", "positive"), start_epoch=0, end_epoch=80),
    ]
    for result in model.answer(cache, queries):
        print(result)
    print("readout:", model.readout_stats(cache, queries))


if __name__ == "__main__":
    main()
