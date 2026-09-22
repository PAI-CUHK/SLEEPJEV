"""Command-line smoke demos for the public SLEEPJEV package."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .model import SleepJEV
from .train import benchmark_query_reuse
from .types import SleepNight, SleepQuery


def _synthetic_night(epochs: int = 120, feature_dim: int = 6) -> SleepNight:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(epochs, feature_dim)).astype(np.float32)
    stages = np.asarray(["W", "N1", "N2", "N3", "REM"] * (epochs // 5 + 1))[:epochs]
    return SleepNight("synthetic-demo", features, stages)


def run_demo() -> dict:
    torch.manual_seed(17)
    night = _synthetic_night()
    model = SleepJEV(input_dim=night.feature_dim, hidden_size=32, coarse_factor=2, hour_factor=8).eval()
    cache = model.encode(torch.from_numpy(night.features))
    queries = [
        SleepQuery("sleep_stage", options=("W", "N2", "REM"), start_epoch=10, end_epoch=40),
        SleepQuery("apnea", "boolean", ("negative", "positive"), start_epoch=0, end_epoch=80),
    ]
    return {"answers": model.answer(cache, queries), "readout": model.readout_stats(cache, queries)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sleepjev", description="SLEEPJEV research-prototype demos")
    parser.add_argument("command", choices=("demo", "benchmark"), nargs="?", default="demo")
    args = parser.parse_args(argv)
    night = _synthetic_night()
    torch.manual_seed(17)
    model = SleepJEV(input_dim=night.feature_dim, hidden_size=32, coarse_factor=2, hour_factor=8).eval()
    if args.command == "demo":
        print(json.dumps(run_demo(), indent=2))
    else:
        print(json.dumps(benchmark_query_reuse(model, night), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
