"""Local SLEEPJEV demo server backed by a real checkpoint and cached SHHS features."""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "SLEEPJEV-GitHub" / "src"
sys.path.insert(0, str(PACKAGE))

from sleepjev import SleepJEV  # noqa: E402
from sleepjev.data import load_npz_night  # noqa: E402
from sleepjev.types import SleepQuery  # noqa: E402


CHECKPOINT = ROOT / "artifacts" / "formal_small_smoke" / "sleepjev_checkpoint.pt"
NIGHT_FILE = ROOT / "artifacts" / "experiment1_shhs_full" / "cache" / "shhs1-200001.npz"


def load_runtime():
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = payload["method_config"]
    model = SleepJEV(
        payload["input_dim"],
        hidden_size=int(config["hidden_size"]),
        coarse_factor=10,
        hour_factor=120,
        hierarchical_readout=True,
        context_radius=int(config["context_radius"]),
    ).eval()
    # The formal smoke checkpoint predates the label-free runtime index heads.
    # We keep those heads out of the demo path: the learned encoder/query/
    # option scorer is real, while runtime index predictions are unavailable.
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    allowed_missing = {
        "stage_index_head.weight", "stage_index_head.bias",
        "event_index_head.weight", "event_index_head.bias",
    }
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint mismatch: {incompatible}")
    night = load_npz_night(NIGHT_FILE)
    mean = torch.tensor(payload["normalizer"]["mean"], dtype=torch.float32)
    std = torch.tensor(payload["normalizer"]["std"], dtype=torch.float32).clamp_min(1e-6)
    features = (torch.from_numpy(night.features) - mean) / std
    encode_start = time.perf_counter()
    cache = model.encode(features)
    cache.runtime_event_postings = None
    cache.runtime_event_range_postings = None
    cache.constraint_index = None
    encode_ms = (time.perf_counter() - encode_start) * 1000
    return payload, model, night, cache, encode_ms


PAYLOAD, MODEL, NIGHT, CACHE, ENCODE_MS = load_runtime()
QUERY_COUNT = 0


def build_workload(kind: str, start: int, end: int) -> list[SleepQuery]:
    stage_options = {
        "full": ("W", "N1", "N2", "N3", "REM"),
        "rem": ("W", "N2", "REM"),
        "n2rem": ("N2", "REM"),
        "nrem": ("N1", "N2", "N3", "REM"),
    }.get(kind, ("W", "N1", "N2", "N3", "REM"))
    # Four independent runtime questions are fanned out over one cached night.
    # These are model outputs from the shared scorer, not post-processed labels.
    return [
        SleepQuery("sleep_stage", options=stage_options, start_epoch=start, end_epoch=end),
        SleepQuery("apnea burden", "boolean", ("negative", "positive"), start_epoch=start, end_epoch=end),
        SleepQuery("hypopnea burden", "boolean", ("negative", "positive"), start_epoch=start, end_epoch=end),
        SleepQuery("arousal burden", "boolean", ("negative", "positive"), start_epoch=start, end_epoch=end),
        SleepQuery(
            "sleep depth",
            "score",
            ("light", "N1/N2", "deep", "REM-dominant", "unstable / fragmented"),
            start_epoch=start,
            end_epoch=end,
        ),
    ]


def ksymm_audit(queries: list[SleepQuery], answers: list[dict]) -> dict:
    """Measure candidate-order drift for the live scorer output."""
    drifts: list[float] = []
    for query, answer in zip(queries, answers):
        reversed_query = SleepQuery(
            query.target,
            query.question_type,
            tuple(reversed(query.options)),
            start_epoch=query.start_epoch,
            end_epoch=query.end_epoch,
            max_readout_tokens=query.max_readout_tokens,
        )
        reversed_answer = MODEL.answer(CACHE, [reversed_query])[0]
        for option in query.options:
            drifts.append(abs(float(answer["probabilities"][option]) - float(reversed_answer["probabilities"][option])))
    max_drift = max(drifts, default=0.0)
    return {
        "k": len(queries[0].options) if queries else 0,
        "max_probability_drift": max_drift,
        "status": "PASS" if max_drift < 1e-4 else "CHECK",
        "tested_tasks": len(queries),
    }


def execute_query(kind: str, start: int, end: int) -> dict:
    global QUERY_COUNT
    if not 0 <= start < end <= NIGHT.n_epochs:
        raise ValueError(f"window must satisfy 0 <= start < end <= {NIGHT.n_epochs}")
    queries = build_workload(kind, start, end)
    start_time = time.perf_counter()
    answers = MODEL.answer(CACHE, queries)
    query_ms = (time.perf_counter() - start_time) * 1000
    stats = MODEL.readout_stats(CACHE, queries)
    symmetry = ksymm_audit(queries, answers)
    QUERY_COUNT += 1
    return {
        "query_id": QUERY_COUNT,
        "window": {"start_epoch": start, "end_epoch": end},
        "tasks": [
            {
                "target": query.target,
                "primitive": "Choice" if query.question_type == "choice" else ("Score" if query.question_type == "score" else "Noul"),
                "question_type": query.question_type,
                "options": list(query.options),
                "probabilities": answer["probabilities"],
                "prediction": answer["prediction"],
                "confidence": answer["confidence"],
                "selected_tokens": stat["selected_tokens"],
                "compression": stat["compression"],
                "status": "model_output",
                "expected_score": (float(answer["expected_level"]) + 1.0) if query.question_type == "score" else None,
            }
            for query, answer, stat in zip(queries, answers, stats)
        ],
        "latency_ms": query_ms,
        "fanout": len(queries),
        "overnight_epochs": CACHE.n_epochs,
        "readout_mode": "sparse",
        "ksymm": symmetry,
    }


def execute_multi(views: list[dict]) -> dict:
    """Answer up to four live windows in one shared scorer batch."""
    global QUERY_COUNT
    clean_views: list[dict] = []
    workloads: list[list[SleepQuery]] = []
    for view in views[:4]:
        start = int(view.get("start", 0))
        end = int(view.get("end", 5))
        if not 0 <= start < end <= NIGHT.n_epochs:
            raise ValueError(f"window must satisfy 0 <= start < end <= {NIGHT.n_epochs}")
        clean_views.append({
            "kind": str(view.get("kind", "full")),
            "start": start,
            "end": end,
            "label": str(view.get("label", "live window")),
        })
        workloads.append(build_workload(clean_views[-1]["kind"], start, end))
    if not workloads:
        raise ValueError("at least one view is required")

    flat_queries = [query for workload in workloads for query in workload]
    batch_start = time.perf_counter()
    flat_answers = MODEL.answer(CACHE, flat_queries)
    flat_stats = MODEL.readout_stats(CACHE, flat_queries)
    batch_ms = (time.perf_counter() - batch_start) * 1000

    # Audit the active lane once per batch; repeating the same permutation test
    # for every monitor lane dominated the old replay latency.
    symmetry = ksymm_audit(workloads[0], flat_answers[:len(workloads[0])])
    results = []
    offset = 0
    for view, queries in zip(clean_views, workloads):
        width = len(queries)
        answers = flat_answers[offset:offset + width]
        stats = flat_stats[offset:offset + width]
        offset += width
        QUERY_COUNT += 1
        results.append({
            "query_id": QUERY_COUNT,
            "label": view["label"],
            "window": {"start_epoch": view["start"], "end_epoch": view["end"]},
            "tasks": [
                {
                    "target": query.target,
                    "primitive": "Choice" if query.question_type == "choice" else ("Score" if query.question_type == "score" else "Noul"),
                    "question_type": query.question_type,
                    "options": list(query.options),
                    "probabilities": answer["probabilities"],
                    "prediction": answer["prediction"],
                    "confidence": answer["confidence"],
                    "selected_tokens": stat["selected_tokens"],
                    "compression": stat["compression"],
                    "status": "model_output",
                    "expected_score": (float(answer["expected_level"]) + 1.0) if query.question_type == "score" else None,
                }
                for query, answer, stat in zip(queries, answers, stats)
            ],
            "latency_ms": batch_ms,
            "fanout": width,
            "overnight_epochs": CACHE.n_epochs,
            "readout_mode": "sparse_batch",
            "ksymm": symmetry if not results else {"status": "SHARED", "k": len(queries[0].options), "max_probability_drift": symmetry["max_probability_drift"], "tested_tasks": 0},
        })
    return {"views": results, "batch_latency_ms": batch_ms, "total_tasks": len(flat_queries)}


def build_channel_traces(night) -> dict[str, list[float]]:
    """Return honest epoch-feature traces for the five cached PSG channels.

    The public cache stores epoch features rather than raw EDF samples. We use
    each channel's epoch-level std feature and robustly normalize it only for
    visual display; this is not presented as a raw waveform.
    """
    names = [name.split(":", 1)[0].upper() for name in night.channel_names]
    result = {}
    for channel_index, label in enumerate(names):
        values = np.asarray(night.features[:, channel_index * 15 + 1], dtype=np.float32)
        sample_count = 240
        edges = np.linspace(0, len(values), sample_count + 1, dtype=int)
        pooled = np.asarray([values[edges[i]:edges[i + 1]].mean() for i in range(sample_count)])
        low, high = np.percentile(pooled, [2, 98])
        scaled = np.clip((pooled - low) / max(float(high - low), 1e-8), 0.02, 0.98)
        result[label] = [round(float(value), 4) for value in scaled]
    return result


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):  # noqa: N802
        if self.path == "/sleepjev-logo.svg":
            data = (Path(__file__).with_name("sleepjev-logo.svg")).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/api/state":
            self._json({
                "record_id": NIGHT.record_id,
                "epochs": NIGHT.n_epochs,
                "epoch_seconds": NIGHT.epoch_seconds,
                "feature_dim": NIGHT.feature_dim,
                "dataset": "SHHS cached feature slice",
                "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
                "encode_ms": ENCODE_MS,
                "query_count": QUERY_COUNT,
                "stage_labels": {stage: int((NIGHT.stage_labels == stage).sum()) for stage in ("W", "N1", "N2", "N3", "REM")},
                "stage_sequence": [str(NIGHT.stage_labels[index]) for index in torch.linspace(0, NIGHT.n_epochs - 1, steps=60).round().long().tolist()],
                "channel_traces": build_channel_traces(NIGHT),
                "scope_note": "Real SHHS features + real SLEEPJEV checkpoint. Five runtime questions are fanned out through the shared scorer; runtime event index heads are not claimed.",
            })
            return
        if self.path in ("/", "/index.html"):
            data = (Path(__file__).with_name("index.html")).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        if self.path not in ("/api/query", "/api/multi"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/multi":
                result = execute_multi(body.get("views", []))
            else:
                result = execute_query(str(body.get("kind", "full")), int(body.get("start", 338)), int(body.get("end", 596)))
            self._json(result)
        except Exception as exc:  # return readable local-demo errors
            self._json({"error": str(exc)}, status=400)

    def log_message(self, *_args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print(f"SLEEPJEV real demo: http://127.0.0.1:8765/  |  cache encode {ENCODE_MS:.2f} ms", flush=True)
    server.serve_forever()
