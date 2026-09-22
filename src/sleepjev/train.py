"""Training and evaluation helpers for the first SleepJEV pilot."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch.nn import functional as F

from .model import SleepJEV
from .types import EVENT_TYPES, STAGE_OPTIONS, SleepNight, SleepQuery, UNKNOWN_LABEL


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, nights: Sequence[SleepNight]) -> "FeatureNormalizer":
        values = np.concatenate([x.features for x in nights], axis=0)
        mean = values.mean(0)
        std = values.std(0)
        return cls(mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32))

    def transform(self, night: SleepNight) -> SleepNight:
        return SleepNight(
            record_id=night.record_id,
            features=(night.features - self.mean) / self.std,
            stage_labels=night.stage_labels,
            feature_names=night.feature_names,
            channel_names=night.channel_names,
            sampling_rates=night.sampling_rates,
            epoch_seconds=night.epoch_seconds,
            metadata=night.metadata,
            events=night.events,
            event_labels=night.event_labels,
            position_labels=night.position_labels,
            quality_labels=night.quality_labels,
        )


def harmonize_feature_schema(
    nights: Sequence[SleepNight],
    channel_order: Sequence[str] = ("eeg", "eog", "emg", "resp", "spo2"),
) -> list[SleepNight]:
    """Make channel-missing NSRR records share one fixed feature dimension.

    SHHS/MESA/HomePAP use slightly different labels and sometimes omit a
    modality. We retain the first available channel for each canonical kind,
    zero-fill missing kinds, and keep the original provenance in metadata.
    """

    if not nights:
        return []
    per_channel = len(nights[0].feature_names) // max(1, len(nights[0].channel_names))
    if per_channel <= 0:
        per_channel = 15
    canonical_names = tuple(
        f"{kind}:{metric}"
        for kind in channel_order
        for metric in (
            "mean", "std", "min", "max", "rms", "q25", "median", "q75", "diff_abs",
            "abs_centered", "delta_power", "theta_power", "alpha_power", "sigma_power", "zero_cross",
        )[:per_channel]
    )
    result = []
    for night in nights:
        blocks = []
        channel_names = []
        rates = []
        for kind in channel_order:
            indices = [i for i, name in enumerate(night.channel_names) if name.split(":", 1)[0] == kind]
            if indices:
                channel = indices[0]
                blocks.append(night.features[:, channel * per_channel : (channel + 1) * per_channel])
                channel_names.append(night.channel_names[channel])
                rates.append(night.sampling_rates[channel] if channel < len(night.sampling_rates) else 0.0)
            else:
                blocks.append(np.zeros((night.n_epochs, per_channel), dtype=np.float32))
                channel_names.append(f"{kind}:MISSING")
                rates.append(0.0)
        result.append(
            SleepNight(
                record_id=night.record_id,
                features=np.concatenate(blocks, axis=1),
                stage_labels=night.stage_labels,
                feature_names=canonical_names,
                channel_names=tuple(channel_names),
                sampling_rates=tuple(rates),
                epoch_seconds=night.epoch_seconds,
                metadata={**night.metadata, "feature_schema_harmonized": True},
                events=night.events,
                event_labels=night.event_labels,
                position_labels=night.position_labels,
                quality_labels=night.quality_labels,
            )
        )
    return result


def epoch_stage_queries(n_epochs: int, options: Sequence[str] = STAGE_OPTIONS) -> list[SleepQuery]:
    return [SleepQuery("sleep_stage", "choice", tuple(options), i, i + 1) for i in range(n_epochs)]


def stage_targets(night: SleepNight, options: Sequence[str] = STAGE_OPTIONS) -> torch.Tensor:
    mapping = {label: i for i, label in enumerate(options)}
    return torch.tensor([mapping.get(label, mapping["UNKNOWN"]) for label in night.stage_labels], dtype=torch.long)


def event_targets(night: SleepNight, event_type: str) -> torch.Tensor:
    if night.event_labels is None:
        return torch.full((night.n_epochs,), UNKNOWN_LABEL, dtype=torch.long)
    return torch.from_numpy(night.event_labels.column(event_type).astype(np.int64))


def _binary_ece(truth: list[int], probability: list[float], bins: int = 10) -> float | None:
    if not truth or len(truth) != len(probability):
        return None
    targets = np.asarray(truth, dtype=np.float64)
    scores = np.asarray(probability, dtype=np.float64)
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(1.0 / bins, 1.0, bins)):
        keep = (scores >= lower) & ((scores < upper) if upper < 1.0 else (scores <= upper))
        if keep.any():
            ece += float(keep.mean() * abs(scores[keep].mean() - targets[keep].mean()))
    return float(ece)


def _event_metrics(
    truth: list[int],
    prediction: list[int],
    known_probability: list[float],
    all_target: list[int],
    all_probability: list[float],
) -> dict:
    """Report known-negative metrics only when they are defined, plus PU ranking."""

    classes = set(truth)
    metrics: dict[str, object] = {
        "n": len(truth),
        "metric_status": "known_binary" if len(classes) >= 2 else "positive_unlabeled",
    }
    if len(classes) >= 2 and len(known_probability) == len(truth):
        probability = np.asarray(known_probability, dtype=np.float64)
        metrics.update({
            "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(truth, probability)),
            "auprc": float(average_precision_score(truth, probability)),
            "brier": float(brier_score_loss(truth, probability)),
            "nll": float(log_loss(truth, np.column_stack([1 - probability, probability]), labels=[0, 1])),
            "ece": _binary_ece(truth, known_probability),
        })
    else:
        # A one-class known subset must not yield a misleading F1=1.0 claim.
        metrics.update({"accuracy": None, "macro_f1": None})

    targets = np.asarray(all_target, dtype=np.int64)
    scores = np.asarray(all_probability, dtype=np.float64)
    positive = targets == 1
    unknown = targets == UNKNOWN_LABEL
    negative = targets == 0
    metrics["label_coverage"] = {
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "unknown": int(unknown.sum()),
        "known_fraction": float((~unknown).mean()) if len(targets) else 0.0,
    }
    if positive.any() and len(scores):
        order = np.argsort(-scores, kind="stable")
        rank = np.empty(len(scores), dtype=np.int64)
        rank[order] = np.arange(len(scores))
        metrics["positive_unlabeled"] = {
            "positive_recall_at_top_1pct": float(positive[order[: max(1, int(np.ceil(0.01 * len(order))))]].sum() / positive.sum()),
            "positive_recall_at_top_5pct": float(positive[order[: max(1, int(np.ceil(0.05 * len(order))))]].sum() / positive.sum()),
            "positive_recall_at_top_10pct": float(positive[order[: max(1, int(np.ceil(0.10 * len(order))))]].sum() / positive.sum()),
            "mean_positive_rank_fraction": float((rank[positive] + 1).mean() / max(1, len(scores))),
            "positive_score_mean": float(scores[positive].mean()),
            "unlabeled_score_mean": float(scores[unknown].mean()) if unknown.any() else None,
        }
    return metrics


def task_queries(night: SleepNight, task: str) -> list[SleepQuery]:
    if task == "sleep_stage":
        return epoch_stage_queries(night.n_epochs)
    if task in EVENT_TYPES:
        return [
            SleepQuery(
                task,
                "boolean",
                ("negative", "positive"),
                i,
                i + 1,
                max_readout_tokens=8,
            )
            for i in range(night.n_epochs)
        ]
    raise KeyError(task)


def fit_stage_model(
    model: SleepJEV,
    nights: Sequence[SleepNight],
    *,
    epochs: int = 3,
    learning_rate: float = 2e-3,
    device: str | torch.device = "cpu",
) -> list[float]:
    device = torch.device(device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[float] = []
    for _ in range(epochs):
        losses = []
        for night in nights:
            features = torch.from_numpy(night.features).to(device)
            cache = model.encode(features, event_index=night.event_index(), epoch_seconds=night.epoch_seconds)
            queries = epoch_stage_queries(night.n_epochs)
            logits, mask = model.score(cache, queries)
            targets = stage_targets(night).to(device)
            valid = torch.from_numpy(night.stage_labels != "UNKNOWN").to(device)
            if not valid.any():
                continue
            query_loss = F.cross_entropy(logits[mask].reshape(night.n_epochs, -1)[valid], targets[valid])
            index_loss = F.cross_entropy(model.stage_index_logits(cache)[0, valid], targets[valid])
            loss = query_loss + 0.25 * index_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    return history


@torch.no_grad()
def evaluate_stage_model(model: SleepJEV, nights: Sequence[SleepNight], device: str | torch.device = "cpu") -> dict:
    device = torch.device(device)
    model.to(device).eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    nll: list[float] = []
    for night in nights:
        cache = model.encode(
            torch.from_numpy(night.features).to(device),
            event_index=night.event_index(),
            epoch_seconds=night.epoch_seconds,
        )
        queries = epoch_stage_queries(night.n_epochs)
        logits, mask = model.score(cache, queries)
        probs = logits[mask].reshape(night.n_epochs, -1).softmax(-1)
        target = stage_targets(night).to(device)
        valid = target != len(STAGE_OPTIONS) - 1
        y_true.extend(target[valid].cpu().tolist())
        y_pred.extend(probs.argmax(-1)[valid].cpu().tolist())
        nll.extend((-probs[torch.arange(len(target), device=device), target].clamp_min(1e-8).log())[valid].cpu().tolist())
    return {
        "epochs": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "nll": float(np.mean(nll)),
    }


def fit_sleepjev_tasks(
    model: SleepJEV,
    nights: Sequence[SleepNight],
    *,
    tasks: Sequence[str] = ("sleep_stage", *EVENT_TYPES),
    epochs: int = 3,
    learning_rate: float = 2e-3,
    device: str | torch.device = "cpu",
    stage_class_weights: torch.Tensor | None = None,
    stage_loss_weight: float = 1.0,
    stage_index_loss_weight: float = 0.25,
    event_index_loss_weight: float = 0.25,
    event_pu_ranking: bool = False,
    pu_max_unlabeled: int = 256,
) -> list[float]:
    """Train the shared JEV scorer on stage and event queries with masking."""

    device = torch.device(device)
    if stage_loss_weight <= 0:
        raise ValueError("stage_loss_weight must be positive")
    if stage_index_loss_weight < 0 or event_index_loss_weight < 0:
        raise ValueError("index loss weights must be non-negative")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[float] = []
    for _ in range(epochs):
        losses = []
        for night in nights:
            cache = model.encode(
                torch.from_numpy(night.features).to(device),
                event_index=night.event_index(),
                epoch_seconds=night.epoch_seconds,
            )
            night_losses = []
            stage_index_loss = None
            event_index_losses = []
            event_index_scores = model.event_index_logits(cache)[0]
            for task in tasks:
                queries = task_queries(night, task)
                logits, mask = model.score(cache, queries)
                logits = logits[mask].reshape(night.n_epochs, -1)
                if task == "sleep_stage":
                    targets = stage_targets(night).to(device)
                    valid = torch.from_numpy(night.stage_labels != "UNKNOWN").to(device)
                    if valid.any():
                        stage_index_loss = F.cross_entropy(
                            model.stage_index_logits(cache)[0, valid],
                            targets[valid],
                            weight=stage_class_weights,
                        )
                else:
                    targets = event_targets(night, task).to(device)
                    valid = targets != UNKNOWN_LABEL
                    event_column = EVENT_TYPES.index(task)
                    index_scores = event_index_scores[:, event_column]
                    positive = torch.where(targets == 1)[0]
                    unknown = torch.where(targets == UNKNOWN_LABEL)[0]
                    known_negative = torch.where(targets == 0)[0]
                    if event_pu_ranking:
                        if len(known_negative):
                            event_index_losses.append(F.softplus(index_scores[known_negative]).mean())
                        if len(positive) and len(unknown):
                            if len(unknown) > pu_max_unlabeled:
                                keep = torch.linspace(0, len(unknown) - 1, pu_max_unlabeled, device=device).long()
                                unknown = unknown[keep]
                            event_index_losses.append(
                                F.softplus(index_scores[unknown].unsqueeze(0) - index_scores[positive].unsqueeze(1)).mean()
                            )
                    elif valid.any():
                        event_index_losses.append(
                            F.binary_cross_entropy_with_logits(index_scores[valid], targets[valid].to(index_scores.dtype))
                        )
                if valid.any():
                    weight = stage_class_weights if task == "sleep_stage" else None
                    if task != "sleep_stage" and event_pu_ranking:
                        positive = torch.where(targets == 1)[0]
                        unknown = torch.where(targets == UNKNOWN_LABEL)[0]
                        known_negative = torch.where(targets == 0)[0]
                        if len(known_negative):
                            night_losses.append(F.cross_entropy(logits[known_negative], targets[known_negative]))
                        if len(positive) and len(unknown):
                            if len(unknown) > pu_max_unlabeled:
                                keep = torch.linspace(0, len(unknown) - 1, pu_max_unlabeled, device=device).long()
                                unknown = unknown[keep]
                            pos_margin = logits[positive, 1] - logits[positive, 0]
                            unlabeled_margin = logits[unknown, 1] - logits[unknown, 0]
                            # Pairwise PU ranking: annotated positives should
                            # outrank unlabeled epochs, without declaring the
                            # latter to be clinical negatives.
                            night_losses.append(F.softplus(unlabeled_margin.unsqueeze(0) - pos_margin.unsqueeze(1)).mean())
                    else:
                        task_loss = F.cross_entropy(logits[valid], targets[valid], weight=weight)
                        if task == "sleep_stage":
                            task_loss = task_loss * stage_loss_weight
                        night_losses.append(task_loss)
            if stage_index_loss is not None:
                night_losses.append(stage_index_loss_weight * stage_index_loss)
            if event_index_losses:
                # A small auxiliary loss teaches the serving indexer which
                # overnight epochs to retain. It uses the same PU policy as
                # the JEV answer scorer and never labels unknown as negative.
                night_losses.append(event_index_loss_weight * torch.stack(event_index_losses).mean())
            if not night_losses:
                continue
            loss = torch.stack(night_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    return history


@torch.no_grad()
def evaluate_sleepjev_tasks(
    model: SleepJEV,
    nights: Sequence[SleepNight],
    *,
    tasks: Sequence[str] = ("sleep_stage", *EVENT_TYPES),
    device: str | torch.device = "cpu",
    readout_mode: str = "sparse",
    use_event_index: bool = True,
    event_max_readout_tokens: int | None = None,
) -> dict:
    """Evaluate JEV tasks with explicit unknown-label masking."""

    device = torch.device(device)
    model.to(device).eval()
    result: dict[str, dict] = {}
    for task in tasks:
        truth: list[int] = []
        pred: list[int] = []
        prob: list[float] = []
        all_target: list[int] = []
        all_prob: list[float] = []
        for night in nights:
            cache = model.encode(
                torch.from_numpy(night.features).to(device),
                event_index=night.event_index() if use_event_index else None,
                epoch_seconds=night.epoch_seconds,
            )
            queries = task_queries(night, task)
            if task != "sleep_stage" and event_max_readout_tokens is not None:
                queries = [replace(query, max_readout_tokens=event_max_readout_tokens) for query in queries]
            logits, mask = model.score(cache, queries, readout_mode=readout_mode)
            logits = logits[mask].reshape(night.n_epochs, -1)
            p = logits.softmax(-1)
            target = stage_targets(night) if task == "sleep_stage" else event_targets(night, task)
            valid = (target != len(STAGE_OPTIONS) - 1) if task == "sleep_stage" else (target != UNKNOWN_LABEL)
            valid_cpu = valid.cpu() if valid.is_cuda else valid
            truth.extend(target[valid_cpu].tolist())
            pred.extend(p.argmax(-1).cpu()[valid_cpu].tolist())
            if task != "sleep_stage":
                prob.extend(p[valid_cpu, 1].cpu().tolist())
                all_target.extend(target.tolist())
                all_prob.extend(p[:, 1].cpu().tolist())
        if task == "sleep_stage":
            metrics = {
                "n": len(truth),
                "accuracy": float(accuracy_score(truth, pred)) if truth else float("nan"),
                "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)) if truth else float("nan"),
                "balanced_accuracy": float(balanced_accuracy_score(truth, pred)) if truth else float("nan"),
                "cohen_kappa": float(cohen_kappa_score(truth, pred)) if truth else float("nan"),
            }
        else:
            metrics = _event_metrics(truth, pred, prob, all_target, all_prob)
        result[task] = metrics
    return result


def fit_fixed_baseline(
    model: torch.nn.Module,
    nights: Sequence[SleepNight],
    *,
    tasks: Sequence[str] = ("sleep_stage", *EVENT_TYPES),
    epochs: int = 3,
    learning_rate: float = 2e-3,
    device: str | torch.device = "cpu",
    stage_class_weights: torch.Tensor | None = None,
    event_pu_ranking: bool = False,
    pu_max_unlabeled: int = 256,
) -> list[float]:
    """Common training loop for IndependentDL, SharedMultiTaskDL and FixedHeadDL."""

    device = torch.device(device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for _ in range(epochs):
        losses = []
        for night in nights:
            outputs = model(torch.from_numpy(night.features).to(device))
            task_losses = []
            for task in tasks:
                target = stage_targets(night).to(device) if task == "sleep_stage" else event_targets(night, task).to(device)
                valid = (target != len(STAGE_OPTIONS) - 1) if task == "sleep_stage" else (target != UNKNOWN_LABEL)
                if valid.any():
                    weight = stage_class_weights if task == "sleep_stage" else None
                    logits = outputs[task][0]
                    if task != "sleep_stage" and event_pu_ranking:
                        positive = torch.where(target == 1)[0]
                        unknown = torch.where(target == UNKNOWN_LABEL)[0]
                        known_negative = torch.where(target == 0)[0]
                        if len(known_negative):
                            task_losses.append(F.cross_entropy(logits[known_negative], target[known_negative]))
                        if len(positive) and len(unknown):
                            if len(unknown) > pu_max_unlabeled:
                                keep = torch.linspace(0, len(unknown) - 1, pu_max_unlabeled, device=device).long()
                                unknown = unknown[keep]
                            pos_margin = logits[positive, 1] - logits[positive, 0]
                            unlabeled_margin = logits[unknown, 1] - logits[unknown, 0]
                            task_losses.append(F.softplus(unlabeled_margin.unsqueeze(0) - pos_margin.unsqueeze(1)).mean())
                    else:
                        task_losses.append(F.cross_entropy(logits[valid], target[valid], weight=weight))
            if not task_losses:
                continue
            loss = torch.stack(task_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    return history


@torch.no_grad()
def evaluate_fixed_baseline(model: torch.nn.Module, nights: Sequence[SleepNight], *, tasks: Sequence[str] = ("sleep_stage", *EVENT_TYPES), device: str | torch.device = "cpu") -> dict:
    device = torch.device(device)
    model.to(device).eval()
    result = {}
    for task in tasks:
        truth: list[int] = []
        pred: list[int] = []
        prob: list[float] = []
        all_target: list[int] = []
        all_prob: list[float] = []
        for night in nights:
            outputs = model(torch.from_numpy(night.features).to(device))[task]
            if outputs.ndim == 3:
                outputs = outputs[0]
            p = outputs.softmax(-1)
            target = stage_targets(night) if task == "sleep_stage" else event_targets(night, task)
            valid = (target != len(STAGE_OPTIONS) - 1) if task == "sleep_stage" else (target != UNKNOWN_LABEL)
            valid_cpu = valid.cpu() if valid.is_cuda else valid
            truth.extend(target[valid_cpu].tolist())
            pred.extend(p.argmax(-1).cpu()[valid_cpu].tolist())
            if task != "sleep_stage":
                prob.extend(p[valid_cpu, 1].cpu().tolist())
                all_target.extend(target.tolist())
                all_prob.extend(p[:, 1].cpu().tolist())
        if task == "sleep_stage":
            metrics = {
                "n": len(truth),
                "accuracy": float(accuracy_score(truth, pred)) if truth else float("nan"),
                "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)) if truth else float("nan"),
                "balanced_accuracy": float(balanced_accuracy_score(truth, pred)) if truth else float("nan"),
                "cohen_kappa": float(cohen_kappa_score(truth, pred)) if truth else float("nan"),
            }
        else:
            metrics = _event_metrics(truth, pred, prob, all_target, all_prob)
        result[task] = metrics
    return result


@torch.no_grad()
def benchmark_query_reuse(model: SleepJEV, night: SleepNight, *, device: str | torch.device = "cpu") -> dict:
    """Compare one cached state against re-encoding the night per query."""

    device = torch.device(device)
    model.to(device).eval()
    features = torch.from_numpy(night.features).to(device)
    start = time.perf_counter()
    cache = model.encode(features)
    encode_ms = (time.perf_counter() - start) * 1000
    loads = {}
    for q_count in (1, 8, 32, 128, 512):
        q_count = min(q_count, night.n_epochs)
        queries = epoch_stage_queries(q_count)
        start = time.perf_counter()
        model.score(cache, queries, readout_mode="sparse")
        sparse_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        model.score(cache, queries, readout_mode="dense")
        dense_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        for query in queries:
            cold_cache = model.encode(features)
            model.score(cold_cache, [query], readout_mode="dense")
        cold_ms = (time.perf_counter() - start) * 1000
        loads[str(q_count)] = {
            "one_time_encode_ms": encode_ms,
            "sparse_query_ms": sparse_ms,
            "sparse_total_ms": encode_ms + sparse_ms,
            "dense_query_ms": dense_ms,
            "dense_total_ms": encode_ms + dense_ms,
            "cold_reencode_ms": cold_ms,
            "sparse_speedup": cold_ms / max(encode_ms + sparse_ms, 1e-9),
            "dense_speedup": cold_ms / max(encode_ms + dense_ms, 1e-9),
            "sparse_tokens": model.readout_stats(cache, [queries[0]])[0]["selected_tokens"],
            "dense_tokens": model.readout_stats(cache, [queries[0]], readout_mode="dense")[0]["selected_tokens"],
        }
    return {"record_id": night.record_id, "queries": loads}
