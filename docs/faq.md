# FAQ

**Is SLEEPJEV a diagnostic model?** No. It is research software for signal representation, runtime query contracts, and workload evaluation.

**What is dynamic?** The target, query clauses, and candidate option strings are supplied by `SleepQuery` at runtime.

**Does the runtime index use gold event annotations?** The serving path is designed to use model-produced stage/event postings. Gold postings are available for audit and scoring only.

**Why are event metrics retrieval metrics?** Many event annotations are positive-unlabeled. The current contract can establish whether annotated positives are surfaced, but not complete sensitivity without verified negatives.

**Can I use arbitrary option strings?** The API accepts unique options, but meaningful behavior for unseen semantics requires training, calibration, and an evaluation protocol. The synthetic demo is not evidence of semantic generalization.
