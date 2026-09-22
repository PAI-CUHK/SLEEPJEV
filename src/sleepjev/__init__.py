"""SLEEPJEV: runtime semantic decisions over long-horizon PSG states.

The package keeps the JEV contract explicit:

    overnight PSG state + runtime question + runtime options
        -> shared option scorer -> typed probabilities

Sleep-EDF support is intentionally small and dependency-light. NSRR adapters
can use the same :class:`SleepNight` and :class:`SleepQuery` interfaces.
"""

__version__ = "0.1.0"

from .data import (
    SleepNight,
    find_sleep_edfx_pairs,
    find_nsrr_pairs,
    find_nsrr_stage_annotation,
    build_nsrr_stage_index,
    load_nsrr_night,
    load_sleep_edfx,
    load_npz_night,
    save_npz_night,
)
from .events import (
    event_labels_from_events,
    infer_analysis_interval,
    normalize_event_label,
    parse_edf_events,
    parse_xml_events,
    read_annotation_events,
    stage_and_position_labels,
)
from .index import SleepEventIndex
from .baselines import FixedHeadDL, IndependentDL, LLMOptionBaseline, SharedMultiTaskDL
from .model import SleepCache, SleepJEV
from .types import EVENT_TYPES, STAGE_OPTIONS, EventLabels, SleepEvent, SleepQuery

__all__ = [
    "STAGE_OPTIONS",
    "EVENT_TYPES",
    "SleepQuery",
    "SleepEvent",
    "EventLabels",
    "SleepNight",
    "SleepEventIndex",
    "IndependentDL",
    "SharedMultiTaskDL",
    "FixedHeadDL",
    "LLMOptionBaseline",
    "SleepCache",
    "SleepJEV",
    "find_sleep_edfx_pairs",
    "find_nsrr_pairs",
    "find_nsrr_stage_annotation",
    "build_nsrr_stage_index",
    "load_nsrr_night",
    "load_sleep_edfx",
    "load_npz_night",
    "save_npz_night",
    "event_labels_from_events",
    "infer_analysis_interval",
    "normalize_event_label",
    "parse_edf_events",
    "parse_xml_events",
    "read_annotation_events",
    "stage_and_position_labels",
]
