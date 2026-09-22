"""Small Sleep-EDF reader and feature cache for the first SLEEPJEV pilot."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import welch

from .events import event_labels_from_events, infer_analysis_interval, parse_xml_events, read_annotation_events, stage_and_position_labels
from .types import EventLabels, SleepEvent, SleepNight


_STAGE_PATTERNS = (
    (re.compile(r"sleep stage w", re.I), "W"),
    (re.compile(r"sleep stage 1", re.I), "N1"),
    (re.compile(r"sleep stage 2", re.I), "N2"),
    (re.compile(r"sleep stage [34]", re.I), "N3"),
    (re.compile(r"sleep stage r", re.I), "REM"),
)

_CHANNEL_ALIASES = {
    "eeg": ("eeg fpz-cz", "eeg pz-oz", "eeg c4-a1", "eeg c3-a2", "eeg"),
    "eog": ("eog horizontal", "eog left", "eog right", "eog"),
    "emg": ("emg submental", "emg chin", "emg"),
    # SHHS commonly calls the respiratory channels THOR RES/ABDO RES/NEW
    # AIR, while MESA and HomePAP use airflow or nasal-pressure names.
    "resp": (
        "resp oro-nasal", "resp airflow", "airflow", "air flow", "resp",
        "thor res", "abdo res", "thor", "abdo", "new air", "new a f", "newair",
        "nasal pressure", "nasal", "thermistor", "pressure",
    ),
    "spo2": ("sao2", "spo2", "oxygen saturation"),
}


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _choose_channels(labels: Iterable[str], max_channels: int = 5) -> list[tuple[int, str, str]]:
    labels = list(labels)
    normalized = [_norm_name(x) for x in labels]
    chosen: list[tuple[int, str, str]] = []
    used: set[int] = set()
    for kind, aliases in _CHANNEL_ALIASES.items():
        for alias in aliases:
            alias_n = _norm_name(alias)
            matches = [i for i, n in enumerate(normalized) if i not in used and alias_n in n]
            if matches:
                i = matches[0]
                chosen.append((i, labels[i], kind))
                used.add(i)
                break
        if len(chosen) >= max_channels:
            break
    if not chosen:
        raise ValueError("No supported Sleep-EDF channels found")
    return chosen


def _stage_labels(hypnogram_path: str | Path, epoch_seconds: float) -> np.ndarray:
    """Read sleep stages from either Sleep-EDF annotations or NSRR XML."""

    events = read_annotation_events(hypnogram_path)
    max_end = max((event.end_sec for event in events), default=0.0)
    n_epochs = max(1, int(np.ceil(max_end / epoch_seconds)))
    labels, _ = stage_and_position_labels(events, n_epochs, epoch_seconds=epoch_seconds)
    if not np.any(labels != "UNKNOWN"):
        raise ValueError(f"No sleep-stage annotations found in {hypnogram_path}")
    return labels


def _band_power(signal: np.ndarray, sampling_rate: float, low: float, high: float) -> float:
    if sampling_rate <= 0 or len(signal) < 8 or sampling_rate / 2 <= low:
        return 0.0
    nperseg = min(len(signal), max(16, int(round(sampling_rate * 4))))
    freqs, power = welch(signal, fs=sampling_rate, nperseg=nperseg)
    keep = (freqs >= low) & (freqs < min(high, sampling_rate / 2))
    if not keep.any():
        return 0.0
    return float(np.trapezoid(power[keep], freqs[keep]))


def _epoch_features(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal[np.isfinite(signal)]
    if len(signal) == 0:
        return np.zeros(15, dtype=np.float32)
    centered = signal - np.mean(signal)
    scale = float(np.std(signal))
    q25, median, q75 = np.percentile(signal, [25, 50, 75])
    features = [
        float(np.mean(signal)), scale, float(np.min(signal)), float(np.max(signal)),
        float(np.sqrt(np.mean(signal * signal))), float(q25), float(median), float(q75),
        float(np.mean(np.abs(np.diff(signal)))) if len(signal) > 1 else 0.0,
        float(np.mean(np.abs(centered))),
    ]
    powers = np.asarray([
        _band_power(centered, sampling_rate, 0.5, 4.0),
        _band_power(centered, sampling_rate, 4.0, 8.0),
        _band_power(centered, sampling_rate, 8.0, 12.0),
        _band_power(centered, sampling_rate, 12.0, 30.0),
    ])
    total = float(powers.sum())
    features.extend((powers / max(total, 1e-12)).tolist())
    features.append(float(np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1]))) if len(signal) > 1 else 0.0)
    return np.asarray(features, dtype=np.float32)


def _feature_names(channel_names: list[str]) -> tuple[str, ...]:
    names = ("mean", "std", "min", "max", "rms", "q25", "median", "q75", "diff_abs", "abs_centered",
             "delta_power", "theta_power", "alpha_power", "sigma_power", "zero_cross")
    return tuple(f"{channel}:{name}" for channel in channel_names for name in names)


def find_sleep_edfx_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """Find Sleep-EDF PSG/hypnogram pairs under a local PhysioNet mirror."""

    root = Path(root)
    pairs = []
    for psg in sorted(root.rglob("*-PSG.edf")):
        # Sleep-EDF uses a recording suffix for the PSG and a scorer suffix
        # for the hypnogram: SC4001E0 -> SC4001EC, ST7011J0 -> ST7011JP.
        record_stem = psg.stem.removesuffix("-PSG")
        prefix = record_stem[:-1]
        candidates = sorted(psg.parent.glob(f"{prefix}*-Hypnogram.edf"))
        if candidates:
            pairs.append((psg, candidates[0]))
    return pairs


def find_nsrr_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """Find SHHS/MESA/HomePAP EDF + NSRR XML pairs by recording id."""

    root = Path(root)
    edfs = sorted(root.rglob("*.edf"))
    xmls = sorted(root.rglob("*.xml"))
    xml_by_id: dict[str, Path] = {}
    for xml in xmls:
        key = xml.stem.removesuffix("-nsrr").removesuffix("-profusion")
        xml_by_id.setdefault(key.lower(), xml)
    pairs: list[tuple[Path, Path]] = []
    for edf in edfs:
        if "annotations" in str(edf).lower():
            continue
        key = edf.stem.lower()
        annotation = xml_by_id.get(key)
        if annotation is not None:
            pairs.append((edf, annotation))
    # HomePAP contains both a larger lab PSG and a small home subset; the lab
    # EDFs are the more consistently readable records for a first pilot.
    pairs.sort(key=lambda pair: ("-lab-" not in pair[0].name.lower(), str(pair[0])))
    return pairs


def find_nsrr_stage_annotation(root: str | Path, record_id: str) -> Path | None:
    """Find a matching Profusion/NSRR hypnogram XML when it is separate."""

    root = Path(root)
    stem = Path(record_id).stem.lower().removesuffix("-nsrr").removesuffix("-profusion")
    direct = sorted(root.rglob(f"{stem}*.xml"))
    if direct:
        candidates = []
        for xml in direct:
            try:
                events = read_annotation_events(xml)
            except (OSError, ET.ParseError):
                continue
            stage_count = sum(event.label.startswith("stage:") for event in events)
            if stage_count:
                candidates.append(("profusion" not in xml.name.lower(), -stage_count, xml))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
            return candidates[0][2]
    digits = "".join(re.findall(r"\d+", stem)[-1:])
    candidates = []
    for xml in root.rglob("*.xml"):
        name = xml.name.lower()
        if digits and digits not in name:
            continue
        try:
            events = read_annotation_events(xml)
        except (OSError, ET.ParseError):
            continue
        stage_count = sum(event.label.startswith("stage:") for event in events)
        if stage_count:
            candidates.append(("profusion" not in name, -stage_count, xml))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return candidates[0][2]


def build_nsrr_stage_index(root: str | Path, record_ids: Iterable[str] | None = None) -> dict[str, Path]:
    """Build one-pass ``record_id -> stage XML`` index for dataset audits."""

    root = Path(root)
    ranked: dict[str, tuple[bool, int, Path]] = {}
    if record_ids is None:
        xml_paths = root.rglob("*.xml")
    else:
        wanted = {Path(record_id).stem.lower() for record_id in record_ids}
        xml_paths = (
            xml for xml in root.rglob("*.xml")
            if any(xml.stem.lower().startswith(key) for key in wanted)
        )
    seen: set[Path] = set()
    for xml in xml_paths:
        if xml in seen:
            continue
        seen.add(xml)
        try:
            events = parse_xml_events(xml)
        except (OSError, ET.ParseError, ValueError):
            continue
        stage_count = sum(event.label.startswith("stage:") for event in events)
        if not stage_count:
            continue
        key = xml.stem.lower().removesuffix("-nsrr").removesuffix("-profusion")
        candidate = ("profusion" not in xml.name.lower(), -stage_count, xml)
        previous = ranked.get(key)
        if previous is None or candidate[:2] < previous[:2]:
            ranked[key] = candidate
    return {key: value[2] for key, value in ranked.items()}


def load_sleep_edfx(
    psg_path: str | Path,
    hypnogram_path: str | Path,
    *,
    epoch_seconds: float = 30.0,
    max_epochs: int | None = None,
    max_channels: int = 5,
    trim_to_scored_sleep: bool = False,
    event_annotation_path: str | Path | None = None,
    event_full_coverage: bool = False,
    allow_unknown_stages: bool = False,
) -> SleepNight:
    """Read one Sleep-EDF pair and make fixed-size epoch features.

    This loader is deliberately conservative: it uses only signal-derived
    epoch features and stage labels. It does not manufacture apnea labels,
    which Sleep-EDF does not provide.
    """

    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("Sleep-EDF loading requires `uv pip install -e '.[sleep]'`") from exc

    annotation_events = read_annotation_events(hypnogram_path)
    max_end = max((event.end_sec for event in annotation_events), default=0.0)
    annotation_epochs = max(1, int(np.ceil(max_end / epoch_seconds)))
    labels, positions = stage_and_position_labels(
        annotation_events, annotation_epochs, epoch_seconds=epoch_seconds
    )
    if not allow_unknown_stages and not np.any(labels != "UNKNOWN"):
        raise ValueError(f"No sleep-stage annotations found in {hypnogram_path}")
    sleep_epochs = np.flatnonzero(np.isin(labels, ["N1", "N2", "N3", "REM"]))
    start_epoch = 0
    if trim_to_scored_sleep and len(sleep_epochs):
        start_epoch = max(0, int(sleep_epochs[0]) - 1)
        stop_epoch = min(len(labels), int(sleep_epochs[-1]) + 2)
        labels = labels[start_epoch:stop_epoch]
        positions = positions[start_epoch:stop_epoch]
    event_events: tuple[SleepEvent, ...] = ()
    if event_annotation_path is not None:
        event_events = read_annotation_events(event_annotation_path)
    elif Path(hypnogram_path).suffix.lower() == ".xml":
        # NSRR event XML files commonly contain both stages and clinical events.
        event_events = annotation_events
    reader = pyedflib.EdfReader(str(psg_path))
    try:
        selected = _choose_channels(reader.getSignalLabels(), max_channels=max_channels)
        duration = float(reader.getFileDuration())
        available = max(0, int(np.floor(duration / epoch_seconds)) - start_epoch)
        n_epochs = min(len(labels), available)
        if max_epochs is not None:
            n_epochs = min(n_epochs, int(max_epochs))
        if n_epochs <= 0:
            raise ValueError(f"No complete epochs in {psg_path}")
        all_features: list[np.ndarray] = []
        names: list[str] = []
        rates: list[float] = []
        channels: list[str] = []
        for index, label, kind in selected:
            fs = float(reader.getSampleFrequency(index))
            read_start = int(round(start_epoch * epoch_seconds * fs))
            read_count = int(round(n_epochs * epoch_seconds * fs))
            try:
                signal = np.asarray(
                    reader.readSignal(index, start=read_start, n=read_count), dtype=np.float64
                )
            except TypeError:  # pragma: no cover - old pyedflib compatibility
                signal = np.asarray(reader.readSignal(index), dtype=np.float64)
            channels.append(f"{kind}:{label}")
            names.append(f"{kind}:{label}")
            rates.append(fs)
            per_epoch = []
            for epoch in range(n_epochs):
                start = int(round(epoch * epoch_seconds * fs))
                stop = int(round((epoch + 1) * epoch_seconds * fs))
                per_epoch.append(_epoch_features(signal[start:stop], fs))
            all_features.append(np.stack(per_epoch))
    finally:
        reader.close()
    features = np.concatenate(all_features, axis=1)
    shift_sec = start_epoch * epoch_seconds
    shifted_events: list[SleepEvent] = []
    recording_end = (start_epoch + n_epochs) * epoch_seconds
    for event in event_events:
        left = max(event.start_sec, shift_sec)
        right = min(event.end_sec, recording_end)
        if right <= left:
            continue
        shifted_events.append(
            SleepEvent(
                start_sec=left - shift_sec,
                duration_sec=right - left,
                label=event.label,
                raw_label=event.raw_label,
                source=event.source,
                signal_location=event.signal_location,
            )
        )
    event_labels = None
    if event_annotation_path is not None or Path(hypnogram_path).suffix.lower() == ".xml":
        analysis_start, analysis_end = infer_analysis_interval(event_events)
        event_labels = event_labels_from_events(
            shifted_events,
            n_epochs,
            epoch_seconds=epoch_seconds,
            full_coverage=event_full_coverage,
            analysis_start_sec=(analysis_start - shift_sec) if analysis_start is not None else None,
            analysis_end_sec=(analysis_end - shift_sec) if analysis_end is not None else None,
        )
    return SleepNight(
        record_id=Path(psg_path).stem.replace("-PSG", ""),
        features=features,
        stage_labels=labels[:n_epochs],
        feature_names=_feature_names(names),
        channel_names=tuple(channels),
        sampling_rates=tuple(rates),
        epoch_seconds=epoch_seconds,
        metadata={"psg_path": str(psg_path), "hypnogram_path": str(hypnogram_path), "start_epoch": start_epoch},
        events=tuple(shifted_events),
        event_labels=event_labels,
        position_labels=positions[:n_epochs],
    )


def load_nsrr_night(
    psg_path: str | Path,
    annotation_path: str | Path,
    *,
    stage_annotation_path: str | Path | None = None,
    epoch_seconds: float = 30.0,
    max_epochs: int | None = None,
    max_channels: int = 5,
    trim_to_scored_sleep: bool = False,
    event_full_coverage: bool = False,
) -> SleepNight:
    """Load an NSRR PSG plus its combined stage/event XML annotation file."""

    return load_sleep_edfx(
        psg_path,
        stage_annotation_path or annotation_path,
        epoch_seconds=epoch_seconds,
        max_epochs=max_epochs,
        max_channels=max_channels,
        trim_to_scored_sleep=trim_to_scored_sleep,
        event_annotation_path=annotation_path,
        event_full_coverage=event_full_coverage,
        allow_unknown_stages=stage_annotation_path is None,
    )


def save_npz_night(night: SleepNight, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        record_id=night.record_id,
        features=night.features,
        stage_labels=night.stage_labels,
        feature_names=np.asarray(night.feature_names),
        channel_names=np.asarray(night.channel_names),
        sampling_rates=np.asarray(night.sampling_rates),
        epoch_seconds=np.asarray(night.epoch_seconds),
        position_labels=np.asarray(night.position_labels if night.position_labels is not None else np.full(night.n_epochs, "UNKNOWN")),
        event_values=(night.event_labels.values if night.event_labels is not None else np.empty((0, 0), dtype=np.int8)),
        event_types=np.asarray(night.event_labels.event_types if night.event_labels is not None else ()),
        event_coverage=(night.event_labels.coverage if night.event_labels is not None else np.empty(0, dtype=bool)),
        event_start_sec=np.asarray([event.start_sec for event in night.events], dtype=np.float32),
        event_duration_sec=np.asarray([event.duration_sec for event in night.events], dtype=np.float32),
        event_labels=np.asarray([event.label for event in night.events]),
        event_raw_labels=np.asarray([event.raw_label for event in night.events]),
    )


def load_npz_night(path: str | Path) -> SleepNight:
    with np.load(path, allow_pickle=False) as data:
        return SleepNight(
            record_id=str(data["record_id"].item()),
            features=data["features"],
            stage_labels=data["stage_labels"],
            feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
            channel_names=tuple(str(x) for x in data["channel_names"].tolist()),
            sampling_rates=tuple(float(x) for x in data["sampling_rates"].tolist()),
            epoch_seconds=float(data["epoch_seconds"].item()),
            position_labels=data["position_labels"] if "position_labels" in data else None,
            events=tuple(
                SleepEvent(float(start), float(duration), str(label), str(raw))
                for start, duration, label, raw in zip(
                    data.get("event_start_sec", np.empty(0)),
                    data.get("event_duration_sec", np.empty(0)),
                    data.get("event_labels", np.empty(0)),
                    data.get("event_raw_labels", np.empty(0)),
                )
            ),
            event_labels=(
                EventLabels(
                    data["event_values"],
                    event_types=tuple(str(x) for x in data["event_types"].tolist()),
                    coverage=data["event_coverage"] if "event_coverage" in data else None,
                    epoch_seconds=float(data["epoch_seconds"].item()),
                )
                if "event_values" in data and data["event_values"].size
                else None
            ),
        )
