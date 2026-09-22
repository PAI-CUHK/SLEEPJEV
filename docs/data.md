# Data and Models

## Public data contract

`SleepNight.features` is a finite floating-point array shaped `[epochs, feature_dim]`; stage labels have one value per epoch. Optional event labels use `1` for observed positive, `0` for observed negative, and `-1` for unknown/unscored. Unknown is never silently converted to negative.

## Adapters

`load_sleep_edfx` reads a local Sleep-EDF PSG/hypnogram pair. NSRR-style EDF/XML discovery helpers are available for local mirrors, but channel names, annotation coverage, and dataset licenses remain dataset-specific. The adapter extracts compact epoch statistics and normalized band-power features; it is a baseline feature pipeline, not a validated clinical preprocessing standard.

## Reproducibility

Keep raw data outside the repository. Record dataset release, download date, subject-level split, annotation version, channel selection, epoch length, feature schema, random seed, model configuration, checkpoint hash, and evaluation manifest. Never publish identifiable signals, access credentials, or restricted checkpoints.
