"""Show that many runtime queries can reuse one encoded overnight cache."""

from __future__ import annotations

import numpy as np
import torch

from sleepjev import SleepJEV, SleepNight, SleepQuery


def main() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(240, 6)).astype(np.float32)
    night = SleepNight("multi-query-demo", features, np.asarray(["W"] * 240))
    model = SleepJEV(input_dim=6, hidden_size=32, coarse_factor=4, hour_factor=16).eval()
    cache = model.encode(torch.from_numpy(night.features))
    queries = [
        SleepQuery("sleep_stage", options=("W", "N2", "REM"), start_epoch=i, end_epoch=i + 30)
        for i in range(0, 180, 30)
    ]
    answers = model.answer(cache, queries)
    print(f"encoded epochs: {cache.n_epochs}; queries: {len(answers)}")
    print("first answer:", answers[0])
    print("readout summary:", model.readout_stats(cache, queries[:2]))


if __name__ == "__main__":
    main()
