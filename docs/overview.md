# SLEEPJEV Overview

SLEEPJEV is an alpha research prototype for runtime semantic decisions over long-horizon PSG representations. Its central workload is not one fixed sleep-stage label per epoch; it is a reusable overnight state that can answer many explicit time, stage, position, or event queries.

The public boundary is `sleepjev`. It accepts feature-level `SleepNight` objects, encodes them into `SleepCache`, and answers validated `SleepQuery` objects. Raw EDF/XML loading is an adapter, not a promise that every dataset has identical channels or annotation semantics.

## JEV design lens

SLEEPJEV treats runtime sleep analysis as a relationship between three explicit
parts: a shared overnight PSG state as evidence, a `SleepQuery` as the question,
and `query.options` as the candidate meanings. The scorer makes a decision over
those supplied options after signal-side filtering and sparse retrieval. This is a
JEV-inspired interface choice, not a claim that the package is an official JEV
standard or a clinical decision system.

## Terminology

- **Shared overnight state:** local epoch tokens plus coarse, hourly, and night summaries.
- **Serving index:** model-produced, label-free stage and event postings used for runtime candidate filtering.
- **JEV-style query:** a question paired with an explicit runtime option set.
- **Sparse readout:** query-conditioned selection of a bounded subset of cached tokens.
- **Positive-unlabeled event labels:** annotated positives with unannotated epochs left unknown rather than treated as negatives.

SLEEPJEV is not a diagnostic system and its probabilities should not be interpreted as disease risk.
