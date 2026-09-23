# JEV-inspired terminology in SLEEPJEV

SLEEPJEV is an independent research implementation inspired by the idea of separating evidence formation from runtime semantic decisions. It is not the official TypeSafe AI Jev model, SDK, or standard, and it does not claim to reproduce a proprietary implementation.

## The runtime contract

SLEEPJEV makes three parts explicit:

```text
shared overnight PSG state + runtime question + candidate meanings
    -> query-conditioned sparse readout
    -> probabilities over the supplied options
```

- **Evidence:** a reusable overnight representation built from a long PSG recording.
- **Question:** a validated `SleepQuery` with a target, time window, constraints, and question type.
- **Candidate meanings:** the explicit `query.options` supplied for the current decision.
- **Decision:** probabilities returned by a shared option-conditioned scorer.

## Typed decisions

The runtime demo exposes the three decision shapes used throughout the project:

| Type | SLEEPJEV example | Output |
| --- | --- | --- |
| `Choice` | Which sleep stage best matches this window? | A probability distribution over stages such as `W`, `N2`, and `REM` |
| `Noul` | Is apnea burden positive in this window? | Probabilities over `negative` and `positive` |
| `Score` | How deep or fragmented is this sleep window? | A distribution over an ordered rubric plus an expected score |

The probabilities are normalized over the options supplied by the current query. They are research outputs over a feature representation, not disease risk, diagnosis, or clinical advice.

## Why this matters for sleep signals

The expensive operation is often forming a representation of a long recording. SLEEPJEV encodes the overnight state once, then reuses it for changing questions and time windows. Sparse temporal retrieval narrows the state before the shared scorer evaluates the candidate meanings.

This makes the JEV-inspired interface useful for long-horizon workloads:

- one overnight encoding can serve many queries;
- candidate meanings can change at runtime;
- Choice, Noul, and Score tasks can be fanned out together;
- latency, selected tokens, confidence, and K-Symmetry can be inspected alongside each answer.

## Relationship to MEDJEV

[MEDJEV](https://github.com/liuyisi123/MEDJEV) studies the same broad runtime decision pattern for clinical text and biomedical evidence. SLEEPJEV specializes the pattern for overnight PSG and physiological signals. The repositories are related but independently maintained.

## Scope boundary

SLEEPJEV is research software. It is not a medical device, diagnostic system, or clinical decision system. The public page is a replay of verified local SHHS feature-cache inference; the local Python demo performs checkpoint-backed inference.
