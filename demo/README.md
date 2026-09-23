# SLEEPJEV Runtime Query Console

This is an interactive research prototype, not a clinical application. It demonstrates the JEV-inspired runtime contract:

```text
shared overnight state + runtime question + runtime options
    -> evidence readout
    -> probabilities over the supplied options
```

## Run locally

Run this command from the parent directory of `demo/`:

```powershell
python demo/server.py
```

Then open `http://127.0.0.1:8765/`.

The local server loads the real SLEEPJEV checkpoint and the local SHHS feature cache. It performs actual PyTorch inference over the cached feature representation.

## Public demo

The public GitHub Pages build is available at:

<https://pai-cuhk.github.io/SLEEPJEV/>

GitHub Pages cannot run Python or PyTorch. The public page therefore uses a clearly labeled replay snapshot derived from verified local model runs. It does not claim live inference. The GitHub source is available at:

<https://github.com/PAI-CUHK/SLEEPJEV>

## Local runtime inputs

The local server expects:

- Checkpoint: `artifacts/formal_small_smoke/sleepjev_checkpoint.pt`
- Feature cache: `artifacts/experiment1_shhs_full/cache/shhs1-200001.npz`
- Workload: Choice, Noul, and Score queries over the same overnight representation
- Parallel serving: four time windows and twenty typed decisions in one batch
- Automatic replay: approximately 650 ms between windows and candidate sets

The checkpoint predates the label-free runtime event-index heads. The demo therefore does not claim event-posting-based selection or gold-label access. The learned encoder, query encoder, and option scorer are used by the local runtime. Noul outputs whose targets mention apnea, hypopnea, or arousal are event-like option scores over the shared representation, not calibrated clinical event probabilities; the UI marks them as unvalidated.

The `confidence` field returned by the model is the maximum probability over the supplied options (`top-option probability`). It is not a separately calibrated confidence estimate. The UI reports normalized entropy as a distribution-concentration diagnostic and labels its routing thresholds as illustrative.

## Parallel batch endpoint

In addition to `POST /api/query`, the server exposes `POST /api/multi`:

```json
{
  "views": [
    {"kind": "full", "start": 338, "end": 343, "label": "ACTIVE"},
    {"kind": "rem", "start": 420, "end": 425, "label": "REM"}
  ]
}
```

Each returned view contains one Choice task, three Noul tasks, and one Score task. The main view runs the K-Symmetry audit; the other views reuse the same audit result for the batch.

## Product contract

- `OVERNIGHT STATE`: one encoded long-horizon PSG representation and selected time window.
- `RUNTIME QUERY`: target, constraints, question type, and explicit candidate meanings.
- `JEV DECISION`: uncalibrated probabilities normalized only over the options supplied by the current query.
- `RETRIEVED EVIDENCE`: query-conditioned sparse evidence without exposing gold labels to the selector.
- `CACHE REUSED`: one overnight encoding serves many changing runtime questions.

The public replay is intentionally separated from the local live inference path because the model checkpoint and dataset-derived cache are not currently published as public artifacts.
