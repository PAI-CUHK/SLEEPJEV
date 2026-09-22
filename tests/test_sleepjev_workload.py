import numpy as np
import torch

from sleepjev.baselines import FixedHeadDL
from sleepjev.model import SleepJEV
from sleepjev.types import EventLabels, SleepNight, SleepQuery
from sleepjev.workload import evaluate_fixed_workloads, evaluate_sleepjev_workloads


def _night() -> SleepNight:
    return SleepNight(
        "workload",
        np.random.default_rng(17).normal(size=(12, 6)).astype(np.float32),
        np.asarray(["W", "N1", "N2", "N2", "REM", "REM", "W", "N2", "N3", "REM", "W", "N2"]),
        event_labels=EventLabels(np.asarray([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
            [-1, -1, -1], [-1, -1, 1], [-1, -1, -1], [1, -1, -1],
            [-1, -1, -1], [-1, 1, -1], [-1, -1, -1], [-1, -1, -1],
        ], dtype=np.int8)),
    )


def _workloads():
    return {
        1: [SleepQuery("apnea", "boolean", ("negative", "positive"), 0, 6, max_readout_tokens=4)],
        8: [
            SleepQuery("apnea", "boolean", ("negative", "positive"), 0, 6, max_readout_tokens=4),
            SleepQuery("hypopnea", "boolean", ("negative", "positive"), 0, 12, max_readout_tokens=4),
            SleepQuery("arousal", "boolean", ("negative", "positive"), 0, 12, max_readout_tokens=4),
        ],
    }


def _assert_metrics(result):
    assert set(result) == {"1", "8"}
    for item in result.values():
        assert item["top_k"] == 2
        assert item["unknown_event_policy"].startswith("unknown labels")
        assert item["by_event"]["apnea"]["positive_queries"] >= 1
        assert item["by_event"]["apnea"]["positive_event_recall_at_top_k"] is not None


def test_sleepjev_workload_metrics_are_query_level_and_pu_safe():
    torch.manual_seed(17)
    model = SleepJEV(6, hidden_size=16, coarse_factor=2, hour_factor=4, hierarchical_readout=True).eval()
    _assert_metrics(evaluate_sleepjev_workloads(model, [_night()], _workloads(), top_k=2))


def test_fixed_model_workload_metrics_share_the_same_contract():
    torch.manual_seed(17)
    _assert_metrics(evaluate_fixed_workloads(FixedHeadDL(6, hidden_size=16).eval(), [_night()], _workloads(), top_k=2))
