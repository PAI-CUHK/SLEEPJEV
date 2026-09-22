"""Comparable non-JEV baselines for the SleepJEV workload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

TASKS = ("sleep_stage", "apnea", "hypopnea", "arousal")
STAGE_CLASSES = ("W", "N1", "N2", "N3", "REM", "UNKNOWN")


class _EpochEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_size)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        x = self.input(features)
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)


def _heads(hidden_size: int) -> nn.ModuleDict:
    return nn.ModuleDict({
        "sleep_stage": nn.Linear(hidden_size, len(STAGE_CLASSES)),
        "apnea": nn.Linear(hidden_size, 2),
        "hypopnea": nn.Linear(hidden_size, 2),
        "arousal": nn.Linear(hidden_size, 2),
    })


class IndependentDL(nn.Module):
    """One separately parameterized encoder/head per downstream task."""

    def __init__(self, input_dim: int, hidden_size: int = 64, tasks: Sequence[str] = TASKS):
        super().__init__()
        self.tasks = tuple(tasks)
        self.encoders = nn.ModuleDict({task: _EpochEncoder(input_dim, hidden_size) for task in self.tasks})
        self.heads = nn.ModuleDict({task: _heads(hidden_size)[task] for task in self.tasks})

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        return {task: self.heads[task](self.encoders[task](features)) for task in self.tasks}


class SharedMultiTaskDL(nn.Module):
    """A conventional shared signal encoder with fixed task-specific heads."""

    def __init__(self, input_dim: int, hidden_size: int = 64, tasks: Sequence[str] = TASKS):
        super().__init__()
        self.tasks = tuple(tasks)
        self.encoder = _EpochEncoder(input_dim, hidden_size)
        self.heads = nn.ModuleDict({task: _heads(hidden_size)[task] for task in self.tasks})

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        encoded = self.encoder(features)
        return {task: self.heads[task](encoded) for task in self.tasks}


class FixedHeadDL(nn.Module):
    """A single fixed clinical head; it cannot accept unseen option sets."""

    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.encoder = _EpochEncoder(input_dim, hidden_size)
        self.head = nn.Linear(hidden_size, 11)
        self.slices = {
            "sleep_stage": slice(0, 6),
            "apnea": slice(6, 8),
            "hypopnea": slice(8, 10),
            "arousal": slice(10, 11),
        }

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        logits = self.head(self.encoder(features))
        result = {task: logits[..., span] for task, span in self.slices.items()}
        # The one-logit arousal branch is a deliberately fixed binary head.
        result["arousal"] = torch.cat([-result["arousal"], result["arousal"]], dim=-1)
        return result

    def answer(self, features: Tensor, task: str) -> Tensor:
        if task not in self.slices:
            raise KeyError(f"fixed head does not support task {task!r}")
        return self(features)[task]


@dataclass
class LLMOptionBaseline:
    """Optional local causal-LLM baseline using option log-likelihood.

    The baseline is intentionally lazy: the repository remains runnable
    without downloading a model. Pass a local HuggingFace model directory to
    ``from_pretrained`` in the comparison script to enable it.
    """

    model_name_or_path: str
    device: str = "cpu"
    max_input_tokens: int = 1024
    local_files_only: bool = True

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, local_files_only=self.local_files_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, local_files_only=self.local_files_only
        ).to(self.device).eval()

    @staticmethod
    def _summary(night) -> str:
        values = night.features.astype(np.float64)
        mean = np.nanmean(values, axis=0)[:12]
        std = np.nanstd(values, axis=0)[:12]
        return "feature_mean=" + ",".join(f"{x:.3f}" for x in mean) + "; feature_std=" + ",".join(f"{x:.3f}" for x in std)

    def answer(self, night, queries) -> list[dict]:
        results = []
        for query in queries:
            prompt = (
                "Sleep recording summary. " + self._summary(night) + "\n"
                f"Question: {query.render()}\nOptions: {', '.join(query.options)}\nAnswer:"
            )
            scores = []
            for option in query.options:
                text = prompt + " " + option
                encoded = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
                ).to(self.device)
                with torch.no_grad():
                    logits = self.model(**encoded).logits[:, :-1]
                    target = encoded.input_ids[:, 1:]
                    log_probs = logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
                scores.append(float(log_probs[:, -max(1, len(self.tokenizer(option)["input_ids"])):].sum()))
            probability = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)
            results.append({
                "target": query.target,
                "type": query.question_type,
                "probabilities": {option: float(value) for option, value in zip(query.options, probability)},
                "prediction": query.options[int(probability.argmax())],
                "confidence": float(probability.max()),
            })
        return results
