"""Public, dataset-independent SleepJEV data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


STAGE_OPTIONS = ("W", "N1", "N2", "N3", "REM", "UNKNOWN")
EVENT_TYPES = ("apnea", "hypopnea", "arousal")
UNKNOWN_LABEL = -1
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


@dataclass(frozen=True)
class SleepQuery:
    """A runtime JEV question over a sleep-state cache.

    ``start_epoch`` and ``end_epoch`` are half-open epoch indices.  Omitting
    them asks a night-level question and makes the scorer use coarse state
    tokens.  Options are semantic strings, not fixed classifier indices.
    """

    target: str
    question_type: str = "choice"
    options: tuple[str, ...] = STAGE_OPTIONS
    start_epoch: int | None = None
    end_epoch: int | None = None
    stage: str | None = None
    position: str | None = None
    scoring_rule: str | None = None
    required_modalities: tuple[str, ...] = ()
    max_readout_tokens: int = 128

    def __post_init__(self) -> None:
        if self.question_type not in {"choice", "boolean", "score"}:
            raise ValueError("question_type must be choice, boolean, or score")
        if len(self.options) < 2 or len(set(self.options)) != len(self.options):
            raise ValueError("SleepQuery requires at least two unique options")
        if self.question_type == "boolean" and len(self.options) != 2:
            raise ValueError("boolean queries require exactly two options")
        if (self.start_epoch is None) != (self.end_epoch is None):
            raise ValueError("start_epoch and end_epoch must be supplied together")
        if self.start_epoch is not None and not 0 <= self.start_epoch < self.end_epoch:
            raise ValueError("query window must satisfy 0 <= start_epoch < end_epoch")
        if self.max_readout_tokens < 1:
            raise ValueError("max_readout_tokens must be positive")

    def render(self) -> str:
        """Stable text used by the hash-based runtime query encoder."""

        parts = [f"target={self.target}", f"type={self.question_type}"]
        if self.start_epoch is not None:
            parts.append(f"window_epoch={self.start_epoch}:{self.end_epoch}")
        if self.stage:
            parts.append(f"stage={self.stage}")
        if self.position:
            parts.append(f"position={self.position}")
        if self.scoring_rule:
            parts.append(f"rule={self.scoring_rule}")
        if self.required_modalities:
            parts.append("modalities=" + ",".join(self.required_modalities))
        return " | ".join(parts)


@dataclass(frozen=True)
class SleepEvent:
    """One normalized annotation event from an XML or EDF scorer file.

    ``label`` is a stable canonical label such as ``hypopnea``,
    ``apnea_obstructive``, ``arousal`` or ``unknown``. ``raw_label`` keeps
    the dataset-specific concept for provenance and auditability.
    """

    start_sec: float
    duration_sec: float
    label: str
    raw_label: str = ""
    source: str = ""
    signal_location: str = ""

    def __post_init__(self) -> None:
        if self.start_sec < 0 or self.duration_sec < 0:
            raise ValueError("event start and duration must be non-negative")
        if not self.label:
            raise ValueError("event label cannot be empty")

    @property
    def end_sec(self) -> float:
        return float(self.start_sec + self.duration_sec)


@dataclass
class EventLabels:
    """Epoch-aligned clinical event labels.

    Values use a three-way encoding: ``1`` positive, ``0`` observed negative,
    and ``-1`` unknown/unscored. Unknown is deliberately not converted to a
    negative label during training or evaluation.
    """

    values: np.ndarray
    event_types: tuple[str, ...] = EVENT_TYPES
    coverage: np.ndarray | None = None
    epoch_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.int8)
        if self.values.ndim != 2 or self.values.shape[1] != len(self.event_types):
            raise ValueError("event label values must have shape [epochs, event_types]")
        if not np.isin(self.values, [UNKNOWN_LABEL, NEGATIVE_LABEL, POSITIVE_LABEL]).all():
            raise ValueError("event labels must be -1, 0, or 1")
        if self.coverage is None:
            self.coverage = np.any(self.values != UNKNOWN_LABEL, axis=1)
        else:
            self.coverage = np.asarray(self.coverage, dtype=bool)
            if self.coverage.shape != (len(self.values),):
                raise ValueError("coverage must have shape [epochs]")

    @property
    def n_epochs(self) -> int:
        return int(self.values.shape[0])

    def column(self, event_type: str) -> np.ndarray:
        try:
            index = self.event_types.index(event_type)
        except ValueError as exc:
            raise KeyError(event_type) from exc
        return self.values[:, index]

    def known_mask(self, event_type: str) -> np.ndarray:
        return self.column(event_type) != UNKNOWN_LABEL

    def positives(self, event_type: str) -> np.ndarray:
        return self.column(event_type) == POSITIVE_LABEL

    def summary(self) -> dict[str, dict[str, int]]:
        return {
            event_type: {
                "positive": int(np.sum(self.column(event_type) == POSITIVE_LABEL)),
                "negative": int(np.sum(self.column(event_type) == NEGATIVE_LABEL)),
                "unknown": int(np.sum(self.column(event_type) == UNKNOWN_LABEL)),
            }
            for event_type in self.event_types
        }


@dataclass
class SleepNight:
    """A feature-level representation of one whole-night recording."""

    record_id: str
    features: np.ndarray
    stage_labels: np.ndarray
    feature_names: tuple[str, ...] = ()
    channel_names: tuple[str, ...] = ()
    sampling_rates: tuple[float, ...] = ()
    epoch_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)
    events: tuple[SleepEvent, ...] = ()
    event_labels: EventLabels | None = None
    position_labels: np.ndarray | None = None
    quality_labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        self.stage_labels = np.asarray(self.stage_labels).astype(str)
        if self.features.ndim != 2:
            raise ValueError("features must have shape [epochs, feature_dim]")
        if self.stage_labels.ndim != 1 or len(self.stage_labels) != len(self.features):
            raise ValueError("stage_labels must have one label per epoch")
        if not np.isfinite(self.features).all():
            raise ValueError("features contain NaN or infinity")
        self.events = tuple(self.events)
        if self.event_labels is not None and self.event_labels.n_epochs != len(self.features):
            raise ValueError("event_labels must have one row per epoch")
        for name in ("position_labels", "quality_labels"):
            values = getattr(self, name)
            if values is not None:
                values = np.asarray(values).astype(str)
                if values.ndim != 1 or len(values) != len(self.features):
                    raise ValueError(f"{name} must have one label per epoch")
                setattr(self, name, values)

    @property
    def n_epochs(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def event_index(self):
        """Build an auditable temporal/event index for query-conditioned readout."""

        from .index import SleepEventIndex

        return SleepEventIndex.from_night(self)
