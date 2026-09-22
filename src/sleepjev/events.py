"""Dataset-independent XML/EDF event annotation adapters for SleepJEV."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np

from .types import (
    EVENT_TYPES,
    EventLabels,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    SleepEvent,
    UNKNOWN_LABEL,
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _text(node: ET.Element, *names: str) -> str:
    wanted = {name.lower() for name in names}
    for child in list(node):
        if _local_name(child.tag) in wanted and child.text:
            return child.text.strip()
    return ""


def _number(value: str, default: float = 0.0) -> float:
    try:
        return float(value.strip())
    except (AttributeError, TypeError, ValueError):
        return default


def normalize_event_label(raw_label: str, event_type: str = "") -> str:
    """Map SHHS/MESA/HomePAP scorer vocabulary to stable event labels."""

    raw = f"{event_type} {raw_label}".strip()
    text = re.sub(r"[_|]+", " ", raw.lower())
    text = re.sub(r"[^a-z0-9% -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "unknown"
    if any(token in text for token in ("unknown", "unsure", "unscored", "cannot score", "indeterminate")):
        return "unknown"
    if "artifact" in text or "signal artifact" in text:
        return "unknown"
    if "arousal" in text:
        return "arousal"
    if "hypopnea" in text or "hypopnoea" in text:
        return "hypopnea"
    if "apnea" in text or "apnoea" in text:
        if "obstructive" in text:
            return "apnea_obstructive"
        if "central" in text:
            return "apnea_central"
        if "mixed" in text:
            return "apnea_mixed"
        return "apnea"
    if (
        "sleep stage" in text
        or "rem sleep" in text
        or re.search(r"\bwake\b", text)
        or re.search(r"\bstage\s*[0-3rw]\b", text)
    ):
        return "stage:" + normalize_stage_label(raw)
    if "position" in text or "body position" in text:
        for position in ("supine", "prone", "left", "right", "upright", "sitting"):
            if position in text:
                return f"position:{position}"
        return "position:unknown"
    if "desaturation" in text or "desat" in text:
        return "desaturation"
    if "analysis start" in text or "beginning of analysis" in text:
        return "meta:analysis_start"
    if "analysis end" in text or "end of analysis" in text:
        return "meta:analysis_end"
    if "recording start" in text:
        return "meta:recording_start"
    return "other"


def normalize_stage_label(raw_label: str) -> str:
    text = raw_label.lower()
    if re.search(r"sleep stage\s*w|\bwake\b|\bawake\b", text):
        return "W"
    if re.search(r"sleep stage\s*1|\bstage\s*1\b|\bn1\b", text):
        return "N1"
    if re.search(r"sleep stage\s*2|\bstage\s*2\b|\bn2\b", text):
        return "N2"
    if re.search(r"sleep stage\s*[34]|\bstage\s*[34]\b|\bn3\b|\bdeep\b", text):
        return "N3"
    if re.search(r"sleep stage\s*r|\bstage\s*r\b|\brem\b", text):
        return "REM"
    return "UNKNOWN"


def parse_xml_events(path: str | Path) -> tuple[SleepEvent, ...]:
    """Parse NSRR/Compumedics ``ScoredEvent`` XML files."""

    path = Path(path)
    root = ET.parse(path).getroot()
    events: list[SleepEvent] = []
    for node in root.iter():
        if _local_name(node.tag) not in {"scoredevent", "event", "annotation"}:
            continue
        start_text = _text(node, "start", "onset", "begin")
        duration_text = _text(node, "duration", "length")
        if not start_text and not duration_text:
            continue
        event_type = _text(node, "eventtype", "type")
        concept = _text(node, "eventconcept", "name", "description", "annotation")
        raw = concept or event_type or "unknown"
        events.append(
            SleepEvent(
                start_sec=max(0.0, _number(start_text)),
                duration_sec=max(0.0, _number(duration_text, 1.0)),
                label=normalize_event_label(raw, event_type),
                raw_label=raw,
                source=str(path),
                signal_location=_text(node, "signallocation", "channel"),
            )
        )
    # Profusion XML stores the hypnogram as a flat SleepStages/SleepStage
    # sequence instead of one ScoredEvent per epoch.
    stage_nodes = [node for node in root.iter() if _local_name(node.tag) == "sleepstage"]
    for index, node in enumerate(stage_nodes):
        raw = (node.text or "").strip()
        stage_map = {"0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "N3", "5": "REM", "r": "REM", "w": "W"}
        label = stage_map.get(raw.lower(), normalize_stage_label(raw))
        events.append(
            SleepEvent(
                start_sec=index * 30.0,
                duration_sec=30.0,
                label=f"stage:{label}",
                raw_label=raw,
                source=str(path),
            )
        )
    return tuple(events)


def parse_edf_events(path: str | Path) -> tuple[SleepEvent, ...]:
    """Parse EDF+ annotation channels through pyEDFlib."""

    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("EDF annotation loading requires `pyedflib`") from exc
    reader = pyedflib.EdfReader(str(path))
    try:
        onsets, durations, descriptions = reader.readAnnotations()
    finally:
        reader.close()
    events = []
    for onset, duration, raw in zip(onsets, durations, descriptions):
        description = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
        events.append(
            SleepEvent(
                start_sec=max(0.0, float(onset)),
                duration_sec=max(0.0, float(duration)),
                label=normalize_event_label(description),
                raw_label=description,
                source=str(path),
            )
        )
    return tuple(events)


def read_annotation_events(path: str | Path) -> tuple[SleepEvent, ...]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xml":
        return parse_xml_events(path)
    if suffix == ".edf":
        return parse_edf_events(path)
    raise ValueError(f"Unsupported annotation file: {path}")


def _base_event_type(label: str) -> str | None:
    if label.startswith("apnea"):
        return "apnea"
    if label in EVENT_TYPES:
        return label
    return None


def _epoch_span(event: SleepEvent, n_epochs: int, epoch_seconds: float) -> range:
    start = max(0, int(np.floor(event.start_sec / epoch_seconds + 1e-8)))
    end = int(np.ceil(max(event.end_sec, event.start_sec + 1e-6) / epoch_seconds - 1e-8))
    end = min(n_epochs, max(start + 1, end))
    return range(min(start, n_epochs), end)


def event_labels_from_events(
    events: Sequence[SleepEvent],
    n_epochs: int,
    *,
    epoch_seconds: float = 30.0,
    full_coverage: bool = False,
    event_types: tuple[str, ...] = EVENT_TYPES,
    analysis_start_sec: float | None = None,
    analysis_end_sec: float | None = None,
) -> EventLabels:
    """Align events to epochs while preserving unknown/missing labels."""

    if n_epochs < 1:
        raise ValueError("n_epochs must be positive")
    fill = NEGATIVE_LABEL if full_coverage else UNKNOWN_LABEL
    values = np.full((n_epochs, len(event_types)), fill, dtype=np.int8)
    coverage = np.full(n_epochs, bool(full_coverage), dtype=bool)
    if analysis_start_sec is not None or analysis_end_sec is not None:
        start_epoch = max(0, int(np.floor((analysis_start_sec or 0.0) / epoch_seconds)))
        end_epoch = min(n_epochs, int(np.ceil((analysis_end_sec if analysis_end_sec is not None else n_epochs * epoch_seconds) / epoch_seconds)))
        values[start_epoch:end_epoch] = NEGATIVE_LABEL
        coverage[start_epoch:end_epoch] = True
    columns = {name: i for i, name in enumerate(event_types)}
    for event in events:
        label = event.label
        span = list(_epoch_span(event, n_epochs, epoch_seconds))
        if not span:
            continue
        base = _base_event_type(label)
        if label == "unknown":
            for epoch in span:
                for column in range(len(event_types)):
                    if values[epoch, column] != POSITIVE_LABEL:
                        values[epoch, column] = UNKNOWN_LABEL
                coverage[epoch] = True
        elif base in columns:
            column = columns[base]
            values[span, column] = POSITIVE_LABEL
            coverage[span] = True
    return EventLabels(values, event_types=event_types, coverage=coverage, epoch_seconds=epoch_seconds)


def stage_and_position_labels(
    events: Sequence[SleepEvent], n_epochs: int, *, epoch_seconds: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Recover stage and body-position state labels from event annotations."""

    stages = np.full(n_epochs, "UNKNOWN", dtype="U7")
    positions = np.full(n_epochs, "UNKNOWN", dtype="U12")
    ordered = sorted(events, key=lambda event: event.start_sec)
    for event in ordered:
        span = list(_epoch_span(event, n_epochs, epoch_seconds))
        if event.label.startswith("stage:"):
            stages[span] = event.label.split(":", 1)[1]
        elif event.label.startswith("position:"):
            start = min(span) if span else min(n_epochs, int(event.start_sec / epoch_seconds))
            next_position = next(
                (other for other in ordered if other.start_sec > event.start_sec and other.label.startswith("position:")),
                None,
            )
            end = n_epochs if next_position is None else min(n_epochs, int(next_position.start_sec / epoch_seconds))
            positions[start:max(start + 1, end)] = event.label.split(":", 1)[1]
    return stages, positions


def infer_analysis_interval(events: Sequence[SleepEvent]) -> tuple[float | None, float | None]:
    """Infer an explicit scoring interval from NSRR metadata events.

    Only metadata names that explicitly identify an analysis boundary are
    accepted. Ordinary recording duration is intentionally not treated as a
    clinical scoring interval.
    """

    starts = [event.start_sec for event in events if event.label == "meta:analysis_start"]
    ends = [event.start_sec for event in events if event.label == "meta:analysis_end"]
    return (min(starts) if starts else None, max(ends) if ends else None)
