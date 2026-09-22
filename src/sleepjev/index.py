"""Temporal, stage, position and event postings used by sparse JEV readout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .events import _base_event_type
from .types import SleepEvent, SleepNight, SleepQuery


def _query_event_types(query: SleepQuery) -> tuple[str, ...]:
    text = " ".join((query.target, query.scoring_rule or "", *query.required_modalities)).lower()
    result = []
    if "arousal" in text:
        result.append("arousal")
    if "hypopnea" in text or "hypopnoea" in text:
        result.append("hypopnea")
    if "apnea" in text or "apnoea" in text:
        result.append("apnea")
    return tuple(dict.fromkeys(result))


def _postings_from_events(events: Iterable[SleepEvent], n_epochs: int, epoch_seconds: float) -> dict[str, np.ndarray]:
    buckets: dict[str, set[int]] = {}
    for event in events:
        start = max(0, int(np.floor(event.start_sec / epoch_seconds)))
        end = min(n_epochs, max(start + 1, int(np.ceil(event.end_sec / epoch_seconds))))
        base = _base_event_type(event.label)
        keys = [event.label]
        if base:
            keys.append(base)
        for key in keys:
            buckets.setdefault(key, set()).update(range(start, end))
    return {key: np.asarray(sorted(value), dtype=np.int64) for key, value in buckets.items()}


def _postings_from_states(labels: np.ndarray | None) -> dict[str, np.ndarray]:
    if labels is None:
        return {}
    labels = np.asarray(labels).astype(str)
    return {label: np.flatnonzero(labels == label).astype(np.int64) for label in np.unique(labels)}


@dataclass
class SleepEventIndex:
    """An inverted index over an overnight recording.

    Event postings are used for supervision/audit and optional oracle recall
    analysis. Runtime sparse readout does not use positive event postings by
    default, preventing label leakage; it uses query constraints and the
    learned cheap selector over cached local tokens.
    """

    n_epochs: int
    epoch_seconds: float = 30.0
    event_postings: dict[str, np.ndarray] = field(default_factory=dict)
    unknown_postings: dict[str, np.ndarray] = field(default_factory=dict)
    stage_postings: dict[str, np.ndarray] = field(default_factory=dict)
    position_postings: dict[str, np.ndarray] = field(default_factory=dict)
    quality_postings: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_runtime_states(
        cls,
        n_epochs: int,
        *,
        epoch_seconds: float = 30.0,
        stage_labels: np.ndarray | None = None,
        position_labels: np.ndarray | None = None,
        quality_labels: np.ndarray | None = None,
    ) -> "SleepEventIndex":
        """Build a label-free serving index from model-produced state labels.

        Gold annotation postings remain available through ``from_night`` for
        supervision and audit, but must never define runtime query candidates.
        """

        return cls(
            n_epochs=n_epochs,
            epoch_seconds=epoch_seconds,
            stage_postings=_postings_from_states(stage_labels),
            position_postings=_postings_from_states(position_labels),
            quality_postings=_postings_from_states(quality_labels),
        )

    @classmethod
    def from_night(cls, night: SleepNight) -> "SleepEventIndex":
        unknown: dict[str, np.ndarray] = {}
        if night.event_labels is not None:
            for event_type in night.event_labels.event_types:
                unknown[event_type] = np.flatnonzero(
                    night.event_labels.column(event_type) < 0
                ).astype(np.int64)
        return cls(
            n_epochs=night.n_epochs,
            epoch_seconds=night.epoch_seconds,
            event_postings=_postings_from_events(night.events, night.n_epochs, night.epoch_seconds),
            unknown_postings=unknown,
            stage_postings=_postings_from_states(night.stage_labels),
            position_postings=_postings_from_states(night.position_labels),
            quality_postings=_postings_from_states(night.quality_labels),
        )

    def event_epochs(self, event_type: str, *, include_unknown: bool = False) -> np.ndarray:
        positive = self.event_postings.get(event_type, np.empty(0, dtype=np.int64))
        if not include_unknown:
            return positive
        unknown = self.unknown_postings.get(event_type, np.empty(0, dtype=np.int64))
        return np.unique(np.concatenate([positive, unknown]))

    def candidate_epochs(
        self,
        query: SleepQuery,
        *,
        include_event_postings: bool = False,
    ) -> np.ndarray:
        """Return legal epoch candidates before learned top-k selection."""

        epochs = np.arange(self.n_epochs, dtype=np.int64)
        if query.start_epoch is not None:
            epochs = epochs[(epochs >= query.start_epoch) & (epochs < min(query.end_epoch or self.n_epochs, self.n_epochs))]
        if query.stage:
            if query.stage.upper() == "NREM":
                stage_epochs = np.unique(np.concatenate([
                    self.stage_postings.get(name, np.empty(0, dtype=np.int64))
                    for name in ("N1", "N2", "N3")
                ]))
            else:
                stage_epochs = self.stage_postings.get(query.stage, np.empty(0, dtype=np.int64))
            known_stage_index = any(key.upper() != "UNKNOWN" for key in self.stage_postings)
            if known_stage_index:
                epochs = np.intersect1d(epochs, stage_epochs, assume_unique=True)
        if query.position and query.position.lower() not in {"all", "any", "all positions"}:
            if query.position.lower() in {"non-supine", "nonsupine"}:
                position_epochs = np.unique(np.concatenate([
                    values for key, values in self.position_postings.items() if key.lower() != "supine"
                ])) if self.position_postings else np.empty(0, dtype=np.int64)
            else:
                position_epochs = self.position_postings.get(query.position, np.empty(0, dtype=np.int64))
            known_position_index = any(key.upper() != "UNKNOWN" for key in self.position_postings)
            if known_position_index:
                epochs = np.intersect1d(epochs, position_epochs, assume_unique=True)
        if include_event_postings:
            requested = _query_event_types(query)
            if requested:
                indexed = np.unique(np.concatenate([self.event_epochs(x) for x in requested]))
                epochs = np.intersect1d(epochs, indexed, assume_unique=True)
        return epochs

    def label_recall(self, query: SleepQuery, selected_epochs: np.ndarray) -> float:
        requested = _query_event_types(query)
        if not requested:
            return 1.0
        relevant = np.unique(np.concatenate([self.event_epochs(x) for x in requested]))
        if len(relevant) == 0:
            return 1.0
        return float(np.intersect1d(relevant, selected_epochs).size / relevant.size)
