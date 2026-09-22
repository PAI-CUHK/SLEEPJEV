"""Benchmark cached-night, dense, and cold re-encoding query paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sleepjev import SleepJEV, SleepNight
from sleepjev.train import benchmark_query_reuse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=640)
    parser.add_argument("--feature-dim", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.feature_dim < 1:
        raise SystemExit("--epochs and --feature-dim must be positive")
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    night = SleepNight(
        "synthetic-benchmark",
        rng.normal(size=(args.epochs, args.feature_dim)).astype(np.float32),
        np.asarray(["W"] * args.epochs),
    )
    model = SleepJEV(
        input_dim=args.feature_dim,
        hidden_size=args.hidden_size,
        coarse_factor=10,
        hour_factor=120,
    )
    result = benchmark_query_reuse(model, night, device=args.device)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
