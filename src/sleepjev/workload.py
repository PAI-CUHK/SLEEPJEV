"""Query-level quality metrics for the long-night SLEEPJEV workload.

The workload contains event questions constrained by time and sleep stage.
Labels in SHHS are positive-unlabeled, so this module never interprets an
unannotated epoch as a negative answer.  It instead asks whether a model can
surface annotated positives in the top ranked locations for each question.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch

from .model import SleepJEV
from .types import EVENT_TYPES, POSITIVE_LABEL, STAGE_OPTIONS, SleepNight, SleepQuery


def _event_type(query: SleepQuery) -> str:
    target = (query.target + " " + (query.scoring_rule or "")).lower()
    if "arousal" in target:
        return "arousal"
    if "hypopnea" in target or "hypopnoea" in target:
        return "hypopnea"
    if "apnea" in target or "apnoea" in target:
        return "apnea"
    raise ValueError(f"query does not identify a supported event family: {query.render()}")


def _candidate_epochs(
    night: SleepNight,
    query: SleepQuery,
    predicted_stage_ids: np.ndarray,
    stage_names: Sequence[str],
) -> np.ndarray:
    """Return serving-time legal epochs using predicted, never gold, stages."""

    start = 0 if query.start_epoch is None else max(0, int(query.start_epoch))
    end = night.n_epochs if query.end_epoch is None else min(night.n_epochs, int(query.end_epoch))
    candidate = np.arange(start, end, dtype=np.int64)
    if not len(candidate) or not query.stage:
        return candidate
    requested = query.stage.upper()
    predicted = np.asarray(stage_names, dtype="U7")[predicted_stage_ids[candidate]]
    if requested == "NREM":
        return candidate[np.isin(predicted, ("N1", "N2", "N3"))]
    return candidate[predicted == requested]


def _truth_positives(night: SleepNight, query: SleepQuery) -> np.ndarray:
    """Return gold positives eligible for an answer under the query contract.

    The optional stage restriction is evaluated using gold stage labels only
    for scoring.  The model uses predicted stages in ``_candidate_epochs``.
    """

    if night.event_labels is None:
        return np.empty(0, dtype=np.int64)
    event_type = _event_type(query)
    start = 0 if query.start_epoch is None else max(0, int(query.start_epoch))
    end = night.n_epochs if query.end_epoch is None else min(night.n_epochs, int(query.end_epoch))
    valid = np.zeros(night.n_epochs, dtype=bool)
    valid[start:end] = True
    if query.stage:
        stage = query.stage.upper()
        if stage == "NREM":
            valid &= np.isin(night.stage_labels, ("N1", "N2", "N3"))
        else:
            valid &= night.stage_labels == stage
    return np.flatnonzero(valid & (night.event_labels.column(event_type) == POSITIVE_LABEL))


def _finalize_event(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["n_positive"] > 0]
    positives = int(sum(row["n_positive"] for row in eligible))
    retrieved = int(sum(row["retrieved_positive"] for row in eligible))
    return {
        "queries": len(rows),
        "positive_queries": len(eligible),
        "positive_query_hit_rate_at_top_k": (
            float(sum(row["retrieved_positive"] > 0 for row in eligible) / len(eligible)) if eligible else None
        ),
        "positive_event_recall_at_top_k": float(retrieved / positives) if positives else None,
        "mean_candidates_per_query": float(np.mean([row["n_candidate"] for row in rows])) if rows else 0.0,
        "mean_query_positive_probability": (
            float(np.mean([row["query_positive_probability"] for row in eligible])) if eligible else None
        ),
    }


def _summarize(rows: list[dict], top_k: int) -> dict:
    by_event: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_event[row["event_type"]].append(row)
    event_metrics = {name: _finalize_event(by_event.get(name, [])) for name in EVENT_TYPES}
    supported = [metric for metric in event_metrics.values() if metric["positive_event_recall_at_top_k"] is not None]
    return {
        "query_count": len(rows),
        "top_k": top_k,
        "unknown_event_policy": "unknown labels are excluded; only annotated positives define retrieval recall",
        "by_event": event_metrics,
        "macro_positive_query_hit_rate_at_top_k": (
            float(np.mean([item["positive_query_hit_rate_at_top_k"] for item in supported])) if supported else None
        ),
        "macro_positive_event_recall_at_top_k": (
            float(np.mean([item["positive_event_recall_at_top_k"] for item in supported])) if supported else None
        ),
    }


def _rows_for_scores(
    night: SleepNight,
    queries: Sequence[SleepQuery],
    event_scores: Mapping[str, np.ndarray],
    predicted_stage_ids: np.ndarray,
    stage_names: Sequence[str],
    query_positive_probabilities: Sequence[float],
    top_k: int,
    selected_epochs: Sequence[np.ndarray] | None = None,
) -> list[dict]:
    rows = []
    for row_index, (query, positive_probability) in enumerate(zip(queries, query_positive_probabilities)):
        event_type = _event_type(query)
        candidate = _candidate_epochs(night, query, predicted_stage_ids, stage_names)
        positives = _truth_positives(night, query)
        if selected_epochs is not None:
            selected = np.asarray(selected_epochs[row_index], dtype=np.int64)
        elif len(candidate):
            order = np.argsort(-np.asarray(event_scores[event_type])[candidate], kind="stable")
            selected = candidate[order[: min(top_k, len(order))]]
        else:
            selected = np.empty(0, dtype=np.int64)
        retrieved = int(np.isin(positives, selected).sum())
        rows.append({
            "event_type": event_type,
            "n_candidate": int(len(candidate)),
            "n_positive": int(len(positives)),
            "retrieved_positive": retrieved,
            "query_positive_probability": float(positive_probability),
        })
    return rows


@torch.no_grad()
def evaluate_sleepjev_workloads(
    model: SleepJEV,
    nights: Sequence[SleepNight],
    workloads: Mapping[int, Sequence[SleepQuery]],
    *,
    device: str | torch.device = "cpu",
    readout_mode: str = "sparse",
    top_k: int = 5,
) -> dict[str, dict]:
    """Evaluate all workload sizes using SLEEPJEV's actual runtime answers."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    device = torch.device(device)
    model.to(device).eval()
    output: dict[str, dict] = {}
    for count, queries in workloads.items():
        rows: list[dict] = []
        for night in nights:
            cache = model.encode(
                torch.from_numpy(night.features).to(device),
                event_index=night.event_index(),
                epoch_seconds=night.epoch_seconds,
                event_query_only=readout_mode in {"compiled", "compiled_dense"},
            )
            logits, option_mask, location_ids, location_mask = model.score_events_with_locations(
                cache, queries, readout_mode=readout_mode, top_k=top_k
            )
            probabilities = logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min).softmax(-1)
            event_logits = model.event_index_logits(cache)[0]
            event_scores = {
                event_type: event_logits[:, index].detach().cpu().numpy()
                for index, event_type in enumerate(EVENT_TYPES)
            }
            stage_ids = cache.runtime_stage_ids.detach().cpu().numpy()
            positive = []
            for index, query in enumerate(queries):
                try:
                    option_index = query.options.index("positive")
                except ValueError as exc:
                    raise ValueError(f"event workload query lacks a positive option: {query.render()}") from exc
                positive.append(float(probabilities[index, option_index].detach().cpu()))
            selected_epochs = [
                ids[mask].detach().cpu().numpy()
                for ids, mask in zip(location_ids, location_mask)
            ]
            rows.extend(_rows_for_scores(
                night, queries, event_scores, stage_ids, STAGE_OPTIONS, positive, top_k, selected_epochs
            ))
        output[str(count)] = _summarize(rows, top_k)
    return output


@torch.no_grad()
def evaluate_fixed_workloads(
    model: torch.nn.Module,
    nights: Sequence[SleepNight],
    workloads: Mapping[int, Sequence[SleepQuery]],
    *,
    device: str | torch.device = "cpu",
    top_k: int = 5,
) -> dict[str, dict]:
    """Evaluate fixed-head DL models under the identical query contract."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    device = torch.device(device)
    model.to(device).eval()
    output: dict[str, dict] = {}
    for count, queries in workloads.items():
        rows: list[dict] = []
        for night in nights:
            logits = model(torch.from_numpy(night.features).to(device))
            stage_ids = logits["sleep_stage"][0].argmax(-1).detach().cpu().numpy()
            event_scores = {
                event_type: logits[event_type][0].softmax(-1)[:, 1].detach().cpu().numpy()
                for event_type in EVENT_TYPES
            }
            positive = []
            for query in queries:
                candidate = _candidate_epochs(night, query, stage_ids, STAGE_OPTIONS)
                values = event_scores[_event_type(query)][candidate]
                # Noisy-or maps the baseline's independent epoch scores to the
                # same binary event-burden answer expected by the workload.
                positive.append(float(1.0 - np.prod(1.0 - values, dtype=np.float64)) if len(values) else 0.0)
            rows.extend(_rows_for_scores(
                night, queries, event_scores, stage_ids, STAGE_OPTIONS, positive, top_k
            ))
        output[str(count)] = _summarize(rows, top_k)
    return output
