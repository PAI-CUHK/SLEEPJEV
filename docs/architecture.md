# Architecture

SLEEPJEV separates signal encoding from runtime decision:

```text
SleepNight.features
       |
       v
SleepStateEncoder -> SleepCache
       |                 |
       |                 +--> runtime stage/event postings
       v
SleepQueryEncoder + legal constraints
       |
       v
query-conditioned sparse state
       |
       v
JEVOptionScorer(query.options)
       |
       v
probabilities and optional event locations
```

The core contracts are candidate-order stability, query isolation, padding/mask correctness, label-free serving indexes, and explicit unknown-event handling. These are software invariants, not evidence of clinical validity.
