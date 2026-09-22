"""Signal-native JEV model: shared sleep state, isolated questions, options."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .types import EVENT_TYPES, STAGE_OPTIONS, SleepQuery


@dataclass
class SleepCache:
    """Reusable representation of one complete night."""

    local_tokens: Tensor
    coarse_tokens: Tensor
    night_token: Tensor
    n_epochs: int
    hour_tokens: Tensor | None = None
    local_mask: Tensor | None = None
    coarse_mask: Tensor | None = None
    hour_mask: Tensor | None = None
    epoch_seconds: float = 30.0
    event_index: object | None = None
    constraint_index: object | None = None
    # Evaluation caches these selector projections once per night. They are
    # never derived from event labels and are not retained during training.
    local_selector_keys: Tensor | None = None
    coarse_selector_keys: Tensor | None = None
    hour_selector_keys: Tensor | None = None
    runtime_stage_ids: Tensor | None = None
    # Learned, label-free event postings built once with the overnight cache.
    # They are candidate sets, not gold annotations or final predictions.
    runtime_event_postings: dict[str, Tensor] | None = None
    runtime_event_scores: Tensor | None = None
    # A stage-aware segment-tree posting cache.  Each node stores the top
    # ``event_range_top_k`` predicted event epochs for its time interval.  It
    # makes a time/stage constrained top-k event query exact with respect to a
    # dense scan of ``runtime_event_scores`` while reading O(log T) postings.
    runtime_event_range_postings: Tensor | None = None
    runtime_event_range_leaf_count: int = 0
    # Query plans are derived exclusively from public query clauses and the
    # fixed night length. Reusing them avoids Python-side interval planning in
    # a warm multi-query serving loop.
    runtime_range_query_plans: dict[tuple[SleepQuery, ...], tuple[Tensor, Tensor]] = field(default_factory=dict)
    # Fully materialized event plans reuse the range decomposition, posting
    # expansion, legality mask, and static learned scores. They are safe to
    # cache because an evaluation cache is immutable after overnight encoding.
    # This is the serving analogue of a compiled query plan: repeated high-Q
    # batches do not redo index_select/gather work for the same clinical list.
    runtime_compiled_event_plans: dict[tuple[SleepQuery, ...], tuple[Tensor, Tensor, Tensor]] = field(default_factory=dict)


def _hash_indices(text: str, buckets: int) -> list[int]:
    text = re.sub(r"\s+", " ", text.lower().strip())
    pieces = text.split()
    grams = pieces[:]
    grams.extend(text[i : i + 3] for i in range(max(0, len(text) - 2)))
    if not grams:
        grams = ["<empty>"]
    result = []
    for piece in grams:
        digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
        result.append(int.from_bytes(digest, "little") % buckets)
    return result


class HashTextEncoder(nn.Module):
    """Small runtime-safe text encoder for dynamic query and option strings."""

    def __init__(self, hidden_size: int, buckets: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(buckets, hidden_size)
        self.buckets = buckets

    def forward(self, texts: Sequence[str], device: torch.device) -> Tensor:
        token_lists = [_hash_indices(text, self.buckets) for text in texts]
        if not token_lists:
            return torch.empty((0, self.embedding.embedding_dim), device=device)
        width = max(len(tokens) for tokens in token_lists)
        indices = np.zeros((len(token_lists), width), dtype=np.int64)
        lengths = np.empty(len(token_lists), dtype=np.int64)
        for row, tokens in enumerate(token_lists):
            indices[row, : len(tokens)] = tokens
            lengths[row] = len(tokens)
        token_ids = torch.from_numpy(indices).to(device=device, dtype=torch.long)
        token_lengths = torch.from_numpy(lengths).to(device=device, dtype=torch.long)
        mask = torch.arange(width, device=device).unsqueeze(0) < token_lengths.unsqueeze(1)
        vectors = self.embedding(token_ids)
        return (vectors * mask.unsqueeze(-1)).sum(dim=1) / token_lengths.unsqueeze(-1).to(vectors.dtype)


class SleepStateEncoder(nn.Module):
    """Efficient hierarchical encoder whose output is cached per night."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        coarse_factor: int = 10,
        hour_factor: int = 120,
        contextualize_hierarchy: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.coarse_factor = coarse_factor
        self.hour_factor = hour_factor
        self.contextualize_hierarchy = bool(contextualize_hierarchy)
        self.input = nn.Linear(input_dim, hidden_size)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2, groups=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, 5, padding=4, dilation=2, groups=1),
            nn.GELU(),
        )
        self.temporal_norm = nn.LayerNorm(hidden_size)
        if self.contextualize_hierarchy:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=max(1, min(4, hidden_size // 16)),
                dim_feedforward=4 * hidden_size, batch_first=True, norm_first=True,
            )
            self.coarse_encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            hour_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=max(1, min(4, hidden_size // 16)),
                dim_feedforward=4 * hidden_size, batch_first=True, norm_first=True,
            )
            self.hour_encoder = nn.TransformerEncoder(hour_layer, num_layers=1, enable_nested_tensor=False)
        else:
            self.coarse_encoder = None
            self.hour_encoder = None

    @staticmethod
    def _pool(x: Tensor, mask: Tensor, factor: int) -> tuple[Tensor, Tensor]:
        b, t, d = x.shape
        groups = (t + factor - 1) // factor
        pad = groups * factor - t
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)
        x = x.reshape(b, groups, factor, d)
        m = mask.reshape(b, groups, factor)
        denom = m.sum(-1, keepdim=True).clamp_min(1).to(x.dtype)
        pooled = (x * m.unsqueeze(-1)).sum(2) / denom
        return pooled, m.any(-1)

    def forward(
        self,
        features: Tensor,
        mask: Tensor | None = None,
        *,
        skip_hierarchy: bool = False,
    ) -> SleepCache:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError("features must have shape [epochs, dim] or [batch, epochs, dim]")
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if mask.shape != features.shape[:2]:
            raise ValueError("mask must have shape [batch, epochs]")
        x = self.input(features)
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = self.temporal_norm(x)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        coarse, coarse_mask = self._pool(x, mask, self.coarse_factor)
        if self.coarse_encoder is not None and not skip_hierarchy:
            coarse = self.coarse_encoder(coarse, src_key_padding_mask=~coarse_mask)
        hour, hour_mask = self._pool(x, mask, self.hour_factor)
        if self.hour_encoder is not None and not skip_hierarchy:
            hour = self.hour_encoder(hour, src_key_padding_mask=~hour_mask)
        night = (hour * hour_mask.unsqueeze(-1)).sum(1) / hour_mask.sum(1, keepdim=True).clamp_min(1)
        return SleepCache(
            local_tokens=x,
            coarse_tokens=coarse,
            night_token=night,
            n_epochs=int(features.shape[1]),
            hour_tokens=hour,
            local_mask=mask,
            coarse_mask=coarse_mask,
            hour_mask=hour_mask,
        )


class SleepQueryEncoder(nn.Module):
    def __init__(self, hidden_size: int, buckets: int = 2048):
        super().__init__()
        self.text = HashTextEncoder(hidden_size, buckets=buckets)
        self.window = nn.Sequential(nn.Linear(4, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.output = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU())

    def forward(self, queries: Sequence[SleepQuery], n_epochs: int, device: torch.device) -> Tensor:
        text = self.text([q.render() for q in queries], device)
        numeric = []
        for q in queries:
            start = 0 if q.start_epoch is None else q.start_epoch / max(n_epochs, 1)
            end = 1 if q.end_epoch is None else q.end_epoch / max(n_epochs, 1)
            numeric.append([start, end, end - start, float(q.start_epoch is None)])
        window = torch.tensor(numeric, device=device, dtype=text.dtype)
        return self.output(text + self.window(window))


class JEVOptionScorer(nn.Module):
    """JEV-style option-conditioned attention and shared pointer scoring."""

    def __init__(self, hidden_size: int, rank: int | None = None, dropout: float = 0.05):
        super().__init__()
        rank = rank or hidden_size
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.state_key = nn.Linear(hidden_size, rank, bias=False)
        self.state_value = nn.Linear(hidden_size, rank, bias=False)
        self.option = nn.Linear(hidden_size, rank, bias=False)
        self.branch = nn.Sequential(nn.LayerNorm(rank * 2), nn.Linear(rank * 2, rank), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.scale = rank ** -0.5

    def forward(
        self,
        state_tokens: Tensor,
        state_mask: Tensor,
        question: Tensor,
        options: Tensor,
        option_mask: Tensor,
    ) -> Tensor:
        q = self.query(question).unsqueeze(1)
        k = self.state_key(state_tokens)
        v = self.state_value(state_tokens)
        attention = torch.matmul(q, k.transpose(1, 2)).squeeze(1) * self.scale
        attention = attention.masked_fill(~state_mask, torch.finfo(attention.dtype).min)
        context = torch.bmm(attention.softmax(-1).unsqueeze(1), v).squeeze(1)
        branch = self.branch(torch.cat([context, self.query(question)], dim=-1))
        option = self.option(options)
        # Symmetric set context gives candidates access to the offered set but
        # cannot encode candidate position, preserving permutation equivariance.
        set_context = (option * option_mask.unsqueeze(-1)).sum(1) / option_mask.sum(1, keepdim=True).clamp_min(1)
        logits = torch.bmm(self.dropout(option + set_context.unsqueeze(1)), branch.unsqueeze(-1)).squeeze(-1) * self.scale
        return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)


class SleepJEV(nn.Module):
    """Complete SleepJEV model with cached-night and dynamic-query APIs."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        coarse_factor: int = 10,
        hour_factor: int = 120,
        hierarchical_readout: bool = False,
        context_radius: int = 2,
        event_index_tokens: int = 128,
        event_range_top_k: int = 8,
        fast_hierarchy: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hierarchical_readout = bool(hierarchical_readout)
        self.context_radius = max(0, int(context_radius))
        self.fast_hierarchy = bool(fast_hierarchy)
        self.state_encoder = SleepStateEncoder(
            input_dim,
            hidden_size,
            coarse_factor=coarse_factor,
            hour_factor=hour_factor,
            contextualize_hierarchy=not self.fast_hierarchy,
        )
        self.query_encoder = SleepQueryEncoder(hidden_size)
        self.option_encoder = HashTextEncoder(hidden_size)
        self.scorer = JEVOptionScorer(hidden_size)
        # A cheap auxiliary head produces the stage postings used by runtime
        # query constraints. It replaces any scorer annotations in the index.
        self.stage_index_head = nn.Linear(hidden_size, len(STAGE_OPTIONS))
        # This is the event-side JEV indexer.  It predicts a short candidate
        # posting list for each event family once per night, so serving a
        # multi-query worklist need not scan all 960 overnight tokens again.
        self.event_index_head = nn.Linear(hidden_size, len(EVENT_TYPES))
        self.event_index_tokens = max(1, int(event_index_tokens))
        self.event_range_top_k = max(1, int(event_range_top_k))

    def stage_index_logits(self, cache: SleepCache) -> Tensor:
        """Return per-epoch logits for the label-free serving state index."""

        return self.stage_index_head(cache.local_tokens)

    def event_index_logits(self, cache: SleepCache) -> Tensor:
        """Return one label-free event-index score per epoch and event family."""

        return self.event_index_head(cache.local_tokens)

    @staticmethod
    def _range_leaf_count(n_epochs: int) -> int:
        """Return the power-of-two leaf count of the immutable night index."""

        return 1 << max(0, int(n_epochs - 1)).bit_length()

    def _build_event_range_postings(self, event_scores: Tensor, stage_ids: Tensor) -> tuple[Tensor, int]:
        """Build predicted-event top-k postings for canonical time intervals.

        The extra stage group is the unconstrained view.  All remaining groups
        correspond to the *predicted* stage index, so this cache never reads
        scorer annotations.  A range is decomposed into disjoint tree nodes at
        serving time; the union of their local top-k lists contains the exact
        global top-k for the requested event/time/stage region.
        """

        n_epochs, event_count = event_scores.shape
        leaves = self._range_leaf_count(n_epochs)
        stage_count = len(STAGE_OPTIONS) + 1
        k = min(self.event_range_top_k, max(1, n_epochs))
        postings = torch.full(
            (event_count, stage_count, 2 * leaves, k),
            -1,
            device=event_scores.device,
            dtype=torch.long,
        )
        epoch_ids = torch.arange(leaves, device=event_scores.device, dtype=torch.long)
        in_night = epoch_ids < n_epochs
        stage_groups = torch.arange(stage_count, device=event_scores.device, dtype=torch.long)
        # The final group means all predicted stages.  The leaf tensor is
        # [event, stage-group, epoch], ready to be merged up the tree.
        stage_match = torch.zeros((stage_count, leaves), device=event_scores.device, dtype=torch.bool)
        stage_match[:-1, :n_epochs] = stage_ids.unsqueeze(0) == stage_groups[:-1].unsqueeze(1)
        stage_match[-1] = in_night
        leaf_ids = torch.full(
            (event_count, stage_count, leaves, k), -1, device=event_scores.device, dtype=torch.long
        )
        leaf_ids[..., 0] = epoch_ids.view(1, 1, leaves).expand(event_count, stage_count, leaves)
        leaf_ids = leaf_ids.masked_fill(~stage_match.view(1, stage_count, leaves, 1), -1)
        postings[:, :, leaves : 2 * leaves] = leaf_ids

        # Vectorized bottom-up merges avoid a Python loop over intervals.  The
        # index is tiny for an 8-hour PSG (1024 leaves) but supports arbitrary
        # recording lengths and stays on the model device.
        width = leaves
        while width > 1:
            parent_count = width // 2
            children = postings[:, :, width : 2 * width].reshape(event_count, stage_count, parent_count, 2 * k)
            safe_ids = children.clamp_min(0)
            score_source = event_scores.transpose(0, 1).unsqueeze(1).expand(-1, stage_count, -1)
            values = score_source.gather(
                2,
                safe_ids.reshape(event_count, stage_count, -1),
            ).reshape(event_count, stage_count, parent_count, 2 * k)
            values = values.masked_fill(children < 0, torch.finfo(values.dtype).min)
            choices = torch.topk(values, k=k, dim=-1, largest=True, sorted=True).indices
            postings[:, :, width // 2 : width] = children.gather(-1, choices)
            width //= 2
        return postings, leaves

    @staticmethod
    def _event_type_for_query(query: SleepQuery) -> str | None:
        text = (query.target + " " + (query.scoring_rule or "")).lower()
        if "arousal" in text:
            return "arousal"
        if "hypopnea" in text or "hypopnoea" in text:
            return "hypopnea"
        if "apnea" in text or "apnoea" in text:
            return "apnea"
        return None

    def encode(
        self,
        features: Tensor,
        mask: Tensor | None = None,
        *,
        event_index: object | None = None,
        epoch_seconds: float = 30.0,
        event_query_only: bool = False,
        build_event_range_index: bool = True,
    ) -> SleepCache:
        """Encode a night, optionally materializing only the event cache view.

        Compiled binary event worklists do not read hierarchy tokens or the
        generic option-attention selector.  Skipping these cache views leaves
        local signal states, predicted stages, event scores, and compiled
        answers unchanged while removing cold/fresh work that no query uses.
        """

        cache = self.state_encoder(features, mask, skip_hierarchy=event_query_only)
        cache.event_index = event_index
        cache.epoch_seconds = epoch_seconds
        # Query constraints must be formed from predictions available at
        # serving time, not from gold stage labels held in ``event_index``.
        from .index import SleepEventIndex

        with torch.no_grad():
            runtime_stage_ids = self.stage_index_logits(cache).argmax(-1)[0].detach()
            if not self.training:
                event_scores = self.event_index_logits(cache)[0]
                cache.runtime_event_scores = event_scores.detach()
                if not event_query_only:
                    posting_size = min(self.event_index_tokens, cache.n_epochs)
                    cache.runtime_event_postings = {
                        event_type: torch.topk(event_scores[:, column], k=posting_size, largest=True, sorted=True).indices.detach()
                        for column, event_type in enumerate(EVENT_TYPES)
                    }
                if build_event_range_index:
                    range_postings, range_leaves = self._build_event_range_postings(event_scores, runtime_stage_ids)
                    cache.runtime_event_range_postings = range_postings.detach()
                    cache.runtime_event_range_leaf_count = range_leaves
        cache.runtime_stage_ids = runtime_stage_ids
        if not event_query_only:
            stage_ids = runtime_stage_ids.cpu().numpy()
            cache.constraint_index = SleepEventIndex.from_runtime_states(
                cache.n_epochs,
                epoch_seconds=epoch_seconds,
                stage_labels=np.asarray(STAGE_OPTIONS, dtype="U7")[stage_ids],
            )
        if not self.training and not event_query_only:
            # Sparse readout otherwise repeats state_key over the same long
            # overnight sequence for every query. Keeping this label-free
            # projection in the cache is the intended multi-query mechanism.
            with torch.no_grad():
                cache.local_selector_keys = self.scorer.state_key(cache.local_tokens[0]).detach()
                cache.coarse_selector_keys = self.scorer.state_key(cache.coarse_tokens[0]).detach()
                if cache.hour_tokens is not None:
                    cache.hour_selector_keys = self.scorer.state_key(cache.hour_tokens[0]).detach()
        return cache

    def _sparse_candidate_tokens(self, cache: SleepCache, query: SleepQuery) -> tuple[Tensor, Tensor | None]:
        """Construct label-free sparse candidates before query-specific top-k."""

        max_tokens = query.max_readout_tokens
        index = cache.constraint_index or cache.event_index
        if index is not None and hasattr(index, "candidate_epochs"):
            candidate = index.candidate_epochs(query, include_event_postings=False)
        else:
            start = 0 if query.start_epoch is None else query.start_epoch
            end = cache.n_epochs if query.end_epoch is None else min(query.end_epoch, cache.n_epochs)
            candidate = torch.arange(start, end, device=cache.local_tokens.device).cpu().numpy()
        event_type = self._event_type_for_query(query)
        if not self.training and event_type and cache.runtime_event_postings and event_type in cache.runtime_event_postings:
            # These postings are produced by ``event_index_head`` from signal
            # tokens, never by annotation events. Intersecting them with the
            # legal time/stage candidate set is therefore safe at serving time.
            predicted = cache.runtime_event_postings[event_type].detach().cpu().numpy()
            candidate = np.intersect1d(candidate, predicted, assume_unique=False)
        if len(candidate) == 0:
            # A known-but-empty clinical constraint is not a reason to read
            # the whole night. Keep a single coarse fallback token so the
            # answer remains well-defined while preserving sparse semantics.
            if index is not None and (query.stage is not None or query.position is not None):
                return cache.coarse_tokens[0, :1], (
                    cache.coarse_selector_keys[:1] if cache.coarse_selector_keys is not None else None
                )
            candidate = torch.arange(cache.n_epochs, device=cache.local_tokens.device).cpu().numpy()

        event_query = event_type is not None
        constrained = query.start_epoch is not None or query.stage is not None or query.position is not None
        use_hour = (
            query.start_epoch is None and not constrained and not event_query
            and cache.hour_tokens is not None and cache.hour_tokens.shape[1] <= max_tokens
        )
        if use_hour:
            return cache.hour_tokens[0], (
                cache.hour_selector_keys if cache.hour_selector_keys is not None else None
            )

        # The legacy selector reads only local tokens.  The enhanced selector
        # adds a small temporal neighborhood and aligned coarse/hour tokens so
        # a 30-second clinical query can use sleep continuity without
        # re-encoding the overnight signal.  It never consults positive event
        # postings, so this context is label-free at runtime. The optional
        # runtime event postings above are model predictions, not annotations.
        local_candidate = np.asarray(candidate, dtype=np.int64)
        local_ids = torch.as_tensor(local_candidate, device=cache.local_tokens.device)
        local_tokens = cache.local_tokens[0, local_ids]
        local_keys = (
            cache.local_selector_keys[local_ids]
            if cache.local_selector_keys is not None
            else None
        )
        if self.hierarchical_readout:
            if query.start_epoch is not None and self.context_radius > 0:
                start = max(0, int(query.start_epoch) - self.context_radius)
                end = min(cache.n_epochs, int(query.end_epoch) + self.context_radius)
                local_candidate = np.unique(np.concatenate([local_candidate, np.arange(start, end, dtype=np.int64)]))
                local_ids = torch.as_tensor(local_candidate, device=cache.local_tokens.device)
                local_tokens = cache.local_tokens[0, local_ids]
                local_keys = (
                    cache.local_selector_keys[local_ids]
                    if cache.local_selector_keys is not None
                    else None
                )
            pieces = [local_tokens]
            key_pieces = [local_keys] if local_keys is not None else []
            if cache.coarse_tokens is not None and cache.coarse_tokens.shape[1] > 0:
                coarse_ids = np.unique(local_candidate // max(1, self.state_encoder.coarse_factor))
                coarse_ids = coarse_ids[coarse_ids < cache.coarse_tokens.shape[1]]
                if len(coarse_ids):
                    coarse_tensor = torch.as_tensor(coarse_ids, device=cache.local_tokens.device)
                    pieces.append(cache.coarse_tokens[0, coarse_tensor])
                    if cache.coarse_selector_keys is not None:
                        key_pieces.append(cache.coarse_selector_keys[coarse_tensor])
            if cache.hour_tokens is not None and cache.hour_tokens.shape[1] > 0:
                hour_ids = np.unique(local_candidate // max(1, self.state_encoder.hour_factor))
                hour_ids = hour_ids[hour_ids < cache.hour_tokens.shape[1]]
                if len(hour_ids):
                    hour_tensor = torch.as_tensor(hour_ids, device=cache.local_tokens.device)
                    pieces.append(cache.hour_tokens[0, hour_tensor])
                    if cache.hour_selector_keys is not None:
                        key_pieces.append(cache.hour_selector_keys[hour_tensor])
            local = torch.cat(pieces, dim=0)
            selector_keys = torch.cat(key_pieces, dim=0) if len(key_pieces) == len(pieces) else None
        else:
            local = local_tokens
            selector_keys = local_keys
        return local, selector_keys

    def _select_tokens(
        self,
        cache: SleepCache,
        query: SleepQuery,
        question: Tensor,
        *,
        readout_mode: str = "sparse",
    ) -> Tensor:
        """Select a small query-conditioned token set from the overnight cache.

        The index only applies legal temporal/stage/position constraints. The
        event postings are not consulted at runtime, so clinical labels cannot
        leak into the sparse readout. A cheap dot-product selector ranks local
        tokens and the JEV scorer reads only the top-k set.
        """

        if readout_mode not in {"sparse", "dense"}:
            raise ValueError("readout_mode must be sparse or dense")
        if readout_mode == "dense":
            start = 0 if query.start_epoch is None else query.start_epoch
            end = cache.n_epochs if query.end_epoch is None else min(query.end_epoch, cache.n_epochs)
            selected = cache.local_tokens[0, start:end]
            return selected if selected.shape[0] else cache.local_tokens[0]
        local, selector_keys = self._sparse_candidate_tokens(cache, query)
        max_tokens = query.max_readout_tokens
        if local.shape[0] <= max_tokens:
            return local
        q = self.scorer.query(question).squeeze(0)
        relevance = torch.matmul(selector_keys if selector_keys is not None else self.scorer.state_key(local), q)
        top = torch.topk(relevance, k=max_tokens, largest=True, sorted=False).indices
        return local[top]

    def _select_indexed_event_tokens(
        self,
        cache: SleepCache,
        queries: Sequence[SleepQuery],
        question: Tensor,
        *,
        return_locations: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """Serve an all-event worklist from cached predicted postings only.

        This is the fast JEV path. ``event_index_head`` has already reduced an
        overnight sequence to a short label-free candidate list per event
        family at cache-build time. Query serving only filters those postings
        by time/stage and applies a small query-conditioned top-k, avoiding a
        Q-by-whole-night selector scan.
        """

        if cache.runtime_event_postings is None or cache.local_selector_keys is None:
            return None
        event_types = [self._event_type_for_query(query) for query in queries]
        if any(event_type is None for event_type in event_types):
            return None
        if any(
            query.position is not None and query.position.lower() not in {"all", "any", "all positions"}
            for query in queries
        ):
            return None
        postings = [cache.runtime_event_postings[event_type] for event_type in event_types]
        if not postings or len({posting.shape[0] for posting in postings}) != 1:
            return None

        device = cache.local_tokens.device
        ids = torch.stack(postings, dim=0)
        candidate_count = ids.shape[1]
        starts = torch.as_tensor(
            [0 if query.start_epoch is None else query.start_epoch for query in queries],
            device=device,
            dtype=torch.long,
        ).clamp(0, cache.n_epochs)
        ends = torch.as_tensor(
            [cache.n_epochs if query.end_epoch is None else query.end_epoch for query in queries],
            device=device,
            dtype=torch.long,
        ).clamp(0, cache.n_epochs)
        candidate_mask = (ids >= starts.unsqueeze(1)) & (ids < ends.unsqueeze(1))
        requested_stage = [query.stage.upper() if query.stage else "" for query in queries]
        if cache.runtime_stage_ids is not None and any(requested_stage):
            stage_codes = {name: index for index, name in enumerate(STAGE_OPTIONS)}
            stage_ids = cache.runtime_stage_ids[ids]
            for row, stage in enumerate(requested_stage):
                if stage == "NREM":
                    allowed = torch.zeros(candidate_count, dtype=torch.bool, device=device)
                    for name in ("N1", "N2", "N3"):
                        allowed |= stage_ids[row] == stage_codes[name]
                    candidate_mask[row] &= allowed
                elif stage:
                    candidate_mask[row] &= stage_ids[row] == stage_codes.get(stage, -1)

        state = cache.local_tokens[0, ids]
        selector_keys = cache.local_selector_keys[ids]
        max_output = min(max(query.max_readout_tokens for query in queries), candidate_count)
        query_keys = self.scorer.query(question)
        relevance = torch.bmm(selector_keys, query_keys.unsqueeze(-1)).squeeze(-1)
        relevance = relevance.masked_fill(~candidate_mask, torch.finfo(relevance.dtype).min)
        selected = torch.topk(relevance, k=max_output, largest=True, sorted=True).indices
        selected_ids = ids.gather(1, selected)
        state = state.gather(1, selected.unsqueeze(-1).expand(-1, -1, self.hidden_size))
        state_mask = candidate_mask.gather(1, selected)
        budgets = torch.as_tensor([query.max_readout_tokens for query in queries], device=device, dtype=torch.long)
        state_mask &= torch.arange(max_output, device=device).unsqueeze(0) < budgets.unsqueeze(1)

        # Preserve the defined answer behavior when a predicted event posting
        # has no entry in a requested legal time/stage region.
        fallback = ~state_mask.any(dim=1)
        if bool(fallback.any()):
            state[fallback] = 0
            state[fallback, 0] = cache.coarse_tokens[0, 0]
            state_mask[fallback] = False
            state_mask[fallback, 0] = True
        if return_locations:
            # A coarse fallback makes the answer defined but is not a real
            # event location and must never receive retrieval credit.
            location_mask = candidate_mask.gather(1, selected)
            location_mask[fallback] = False
            return state, state_mask, selected_ids, location_mask
        return state, state_mask

    @torch.no_grad()
    def localize_events(
        self,
        cache: SleepCache,
        queries: Sequence[SleepQuery],
        *,
        top_k: int = 5,
    ) -> tuple[Tensor, Tensor]:
        """Return query-conditioned event epoch IDs from runtime postings.

        This uses the same label-free event postings, predicted-stage filters
        and query relevance selector as the sparse event answer path.  Gold
        event annotations are never available to this method.
        """

        if top_k < 1:
            raise ValueError("top_k must be positive")
        if cache.runtime_event_postings is None or cache.local_selector_keys is None:
            raise RuntimeError("event localization requires an evaluation cache")
        if not queries:
            raise ValueError("at least one event query is required")
        event_types = [self._event_type_for_query(query) for query in queries]
        if any(event_type is None for event_type in event_types):
            raise ValueError("event localization requires event-family queries")
        if any(
            query.position is not None and query.position.lower() not in {"all", "any", "all positions"}
            for query in queries
        ):
            raise ValueError("event localization does not support unaudited position constraints")

        device = cache.local_tokens.device
        ids = torch.stack([cache.runtime_event_postings[event_type] for event_type in event_types], dim=0)
        candidate_count = ids.shape[1]
        starts = torch.as_tensor(
            [0 if query.start_epoch is None else query.start_epoch for query in queries],
            device=device,
            dtype=torch.long,
        ).clamp(0, cache.n_epochs)
        ends = torch.as_tensor(
            [cache.n_epochs if query.end_epoch is None else query.end_epoch for query in queries],
            device=device,
            dtype=torch.long,
        ).clamp(0, cache.n_epochs)
        valid = (ids >= starts.unsqueeze(1)) & (ids < ends.unsqueeze(1))
        requested_stage = [query.stage.upper() if query.stage else "" for query in queries]
        if cache.runtime_stage_ids is not None and any(requested_stage):
            stage_codes = {name: index for index, name in enumerate(STAGE_OPTIONS)}
            stage_ids = cache.runtime_stage_ids[ids]
            for row, stage in enumerate(requested_stage):
                if stage == "NREM":
                    allowed = torch.zeros(candidate_count, dtype=torch.bool, device=device)
                    for name in ("N1", "N2", "N3"):
                        allowed |= stage_ids[row] == stage_codes[name]
                    valid[row] &= allowed
                elif stage:
                    valid[row] &= stage_ids[row] == stage_codes.get(stage, -1)

        question = self.query_encoder(queries, cache.n_epochs, device).to(cache.local_tokens.dtype)
        relevance = torch.bmm(
            cache.local_selector_keys[ids], self.scorer.query(question).unsqueeze(-1)
        ).squeeze(-1)
        relevance = relevance.masked_fill(~valid, torch.finfo(relevance.dtype).min)
        count = min(int(top_k), candidate_count)
        choices = torch.topk(relevance, k=count, largest=True, sorted=True).indices
        return ids.gather(1, choices), valid.gather(1, choices)

    def _select_tokens_batched(
        self,
        cache: SleepCache,
        queries: Sequence[SleepQuery],
        question: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Batch sparse selection without materialising one token tensor per query.

        The earlier implementation built ``[candidate_tokens, hidden]`` on the
        host for every clinical query, then copied each candidate set back to
        the GPU.  That overhead can dominate a 512-query worklist even though
        each JEV readout uses only a few tokens.  Here the overnight cache is
        flattened once; query-specific legal constraints become a compact
        boolean mask, and one GPU matrix product ranks all candidates.

        The masks retain the exact serving-time semantics: stage and position
        constraints come from ``constraint_index`` (predicted states), event
        postings are never consulted, and empty constrained sets use the same
        coarse-token fallback as ``_sparse_candidate_tokens``.
        """

        if cache.local_selector_keys is None or cache.coarse_selector_keys is None:
            raise RuntimeError("batched sparse selection requires evaluation selector keys")
        indexed = self._select_indexed_event_tokens(cache, queries, question)
        if indexed is not None:
            return indexed
        device = cache.local_tokens.device
        local_count = cache.local_tokens.shape[1]
        coarse_count = cache.coarse_tokens.shape[1]
        token_pieces = [cache.local_tokens[0], cache.coarse_tokens[0]]
        key_pieces = [cache.local_selector_keys, cache.coarse_selector_keys]
        hour_count = 0
        if cache.hour_tokens is not None and cache.hour_selector_keys is not None:
            token_pieces.append(cache.hour_tokens[0])
            key_pieces.append(cache.hour_selector_keys)
            hour_count = cache.hour_tokens.shape[1]
        token_pool = torch.cat(token_pieces, dim=0)
        key_pool = torch.cat(key_pieces, dim=0)
        pool_count = token_pool.shape[0]

        # The primary workload uses time windows and stage constraints. Build
        # their Q x T legal mask entirely on the GPU rather than shipping a
        # host-generated mask per query. Position constraints remain harmless
        # here because SHHS position codes are not enabled until audited.
        q_count = len(queries)
        starts = torch.as_tensor(
            [0 if query.start_epoch is None else query.start_epoch for query in queries],
            device=device,
            dtype=torch.long,
        ).clamp(0, local_count)
        ends = torch.as_tensor(
            [local_count if query.end_epoch is None else query.end_epoch for query in queries],
            device=device,
            dtype=torch.long,
        ).clamp(0, local_count)
        has_window = torch.as_tensor(
            [query.start_epoch is not None for query in queries], device=device, dtype=torch.bool
        )
        epoch_ids = torch.arange(local_count, device=device).unsqueeze(0)
        local_mask = (epoch_ids >= starts.unsqueeze(1)) & (epoch_ids < ends.unsqueeze(1))

        # Mixed query batches (for example a stage question plus two event
        # questions in a unit test) retain the same predicted-event candidate
        # restriction as the scalar path. Pure event worklists use the faster
        # direct-posting path above and never allocate this Q x T mask.
        event_types = [self._event_type_for_query(query) for query in queries]
        if cache.runtime_event_postings is not None:
            for row, event_type in enumerate(event_types):
                if event_type and event_type in cache.runtime_event_postings:
                    postings = cache.runtime_event_postings[event_type]
                    legal = local_mask[row, postings].clone()
                    local_mask[row] = False
                    local_mask[row, postings] = legal

        requested_stage = [query.stage.upper() if query.stage else "" for query in queries]
        has_stage = torch.as_tensor([bool(stage) for stage in requested_stage], device=device, dtype=torch.bool)
        if cache.runtime_stage_ids is not None:
            stage_codes = {name: index for index, name in enumerate(STAGE_OPTIONS)}
            stage_mask = torch.ones((q_count, local_count), device=device, dtype=torch.bool)
            known_stage = bool((cache.runtime_stage_ids != stage_codes["UNKNOWN"]).any())
            if known_stage and bool(has_stage.any()):
                target_codes = torch.as_tensor(
                    [stage_codes.get(stage, -1) for stage in requested_stage], device=device, dtype=torch.long
                )
                stage_mask = cache.runtime_stage_ids.unsqueeze(0) == target_codes.unsqueeze(1)
                nrem_rows = torch.as_tensor(
                    [stage == "NREM" for stage in requested_stage], device=device, dtype=torch.bool
                )
                if bool(nrem_rows.any()):
                    nrem = torch.zeros(local_count, device=device, dtype=torch.bool)
                    for stage in ("N1", "N2", "N3"):
                        nrem |= cache.runtime_stage_ids == stage_codes[stage]
                    stage_mask[nrem_rows] = nrem
                local_mask &= torch.where(has_stage.unsqueeze(1), stage_mask, torch.ones_like(stage_mask))

        has_position = torch.as_tensor(
            [query.position is not None and query.position.lower() not in {"all", "any", "all positions"} for query in queries],
            device=device,
            dtype=torch.bool,
        )
        constrained = has_window | has_stage | has_position
        empty = ~local_mask.any(dim=1)
        fallback = empty & constrained
        active = ~fallback

        if self.hierarchical_readout and self.context_radius > 0 and bool(has_window.any()):
            context_starts = (starts - self.context_radius).clamp_min(0)
            context_ends = (ends + self.context_radius).clamp_max(local_count)
            context_mask = (epoch_ids >= context_starts.unsqueeze(1)) & (epoch_ids < context_ends.unsqueeze(1))
            local_mask |= context_mask & has_window.unsqueeze(1) & active.unsqueeze(1)

        coarse_mask = torch.zeros((q_count, coarse_count), device=device, dtype=torch.bool)
        hour_mask = torch.zeros((q_count, hour_count), device=device, dtype=torch.bool)
        if self.hierarchical_readout:
            coarse_pad = coarse_count * self.state_encoder.coarse_factor - local_count
            coarse_source = F.pad(local_mask, (0, coarse_pad), value=False) if coarse_pad else local_mask
            coarse_mask = coarse_source.reshape(q_count, coarse_count, self.state_encoder.coarse_factor).any(dim=-1)
            if hour_count:
                hour_pad = hour_count * self.state_encoder.hour_factor - local_count
                hour_source = F.pad(local_mask, (0, hour_pad), value=False) if hour_pad else local_mask
                hour_mask = hour_source.reshape(q_count, hour_count, self.state_encoder.hour_factor).any(dim=-1)

        event_query = torch.as_tensor([event_type is not None for event_type in event_types], device=device, dtype=torch.bool)
        use_hour = ~constrained & ~event_query & (hour_count > 0)
        if hour_count:
            budgets = torch.as_tensor([query.max_readout_tokens for query in queries], device=device)
            use_hour &= hour_count <= budgets
            if bool(use_hour.any()):
                local_mask[use_hour] = False
                coarse_mask[use_hour] = False
                hour_mask[use_hour] = True
        if bool(fallback.any()):
            local_mask[fallback] = False
            coarse_mask[fallback] = False
            coarse_mask[fallback, 0] = True

        candidate_mask = torch.cat((local_mask, coarse_mask, hour_mask), dim=1)
        max_output = min(max(query.max_readout_tokens for query in queries), pool_count)
        query_keys = self.scorer.query(question)
        relevance = torch.matmul(query_keys, key_pool.transpose(0, 1))
        relevance = relevance.masked_fill(~candidate_mask, torch.finfo(relevance.dtype).min)
        # ``max_output`` can exceed an individual query's token budget.  Keep
        # the selected rows ordered by relevance before masking each row down
        # to its own budget; an unordered top-k would otherwise retain an
        # arbitrary subset for smaller-budget queries.
        selected_indices = torch.topk(relevance, k=max_output, largest=True, sorted=True).indices
        state = token_pool[selected_indices]
        state_mask = candidate_mask.gather(1, selected_indices)
        budgets = torch.as_tensor(
            [query.max_readout_tokens for query in queries], device=device, dtype=torch.long
        )
        state_mask &= torch.arange(max_output, device=device).unsqueeze(0) < budgets.unsqueeze(1)
        return state, state_mask

    def _score_selected_state(
        self,
        state: Tensor,
        state_mask: Tensor,
        question: Tensor,
        queries: Sequence[SleepQuery],
    ) -> tuple[Tensor, Tensor]:
        """Apply the shared option scorer after a query readout is selected."""

        device = state.device
        max_options = max(len(query.options) for query in queries)
        options = state.new_zeros((len(queries), max_options, self.hidden_size))
        option_mask = torch.zeros((len(queries), max_options), dtype=torch.bool, device=device)
        option_texts: list[str] = []
        spans = []
        for query in queries:
            start = len(option_texts)
            option_texts.extend(query.options)
            spans.append((start, len(option_texts)))
        encoded_options = self.option_encoder(option_texts, device).to(options.dtype)
        for index, (start, end) in enumerate(spans):
            options[index, : end - start] = encoded_options[start:end]
            option_mask[index, : end - start] = True
        return self.scorer(state, state_mask, question.to(state.dtype), options, option_mask), option_mask

    @torch.no_grad()
    def score_events_with_locations(
        self,
        cache: SleepCache,
        queries: Sequence[SleepQuery],
        *,
        top_k: int = 5,
        readout_mode: str = "sparse",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Answer event queries and localize them in one JEV sparse readout."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not queries:
            raise ValueError("at least one query is required")
        queries = list(queries)
        device = cache.local_tokens.device
        if readout_mode == "compiled":
            return self._score_compiled_event_queries(cache, queries, top_k=top_k)
        if readout_mode == "compiled_dense":
            return self._score_compiled_dense_event_queries(cache, queries, top_k=top_k)
        if readout_mode not in {"sparse", "dense"}:
            raise ValueError("readout_mode must be sparse, dense, compiled, or compiled_dense")
        if readout_mode == "sparse":
            question = self.query_encoder(queries, cache.n_epochs, device).to(cache.local_tokens.dtype)
            indexed = self._select_indexed_event_tokens(
                cache, queries, question, return_locations=True
            )
            if indexed is not None:
                state, state_mask, location_ids, location_mask = indexed
                logits, option_mask = self._score_selected_state(state, state_mask, question, queries)
                return logits, option_mask, location_ids[:, :top_k], location_mask[:, :top_k]
        logits, option_mask = self.score(cache, queries, readout_mode=readout_mode)
        location_ids, location_mask = self.localize_events(cache, queries, top_k=top_k)
        return logits, option_mask, location_ids, location_mask

    @staticmethod
    def _range_node_cover(start: int, end: int, leaves: int) -> list[int]:
        """Return canonical segment-tree nodes that exactly cover ``[start,end)``."""

        left = max(0, min(int(start), leaves)) + leaves
        right = max(0, min(int(end), leaves)) + leaves
        nodes: list[int] = []
        while left < right:
            if left & 1:
                nodes.append(left)
                left += 1
            if right & 1:
                right -= 1
                nodes.append(right)
            left //= 2
            right //= 2
        return nodes

    @staticmethod
    def _query_stage_groups(query: SleepQuery) -> list[int]:
        """Map a runtime stage clause to prediction-only cache groups."""

        all_stages = len(STAGE_OPTIONS)
        if not query.stage:
            return [all_stages]
        requested = query.stage.upper()
        if requested == "NREM":
            return [STAGE_OPTIONS.index(stage) for stage in ("N1", "N2", "N3")]
        try:
            return [STAGE_OPTIONS.index(requested)]
        except ValueError:
            return []

    def _score_range_indexed_event_queries(
        self,
        cache: SleepCache,
        queries: Sequence[SleepQuery],
        event_types: Sequence[str],
        *,
        top_k: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Read exact event top-k evidence from the segmented overnight cache.

        The range cover contains disjoint intervals.  Since each interval
        stores its local top-k, their union necessarily contains the dense
        top-k over the requested time and predicted-stage constraints.  This
        makes the sparse path quality-preserving relative to the same learned
        event-index head, independent of the total number of queries.
        """

        if cache.runtime_event_range_postings is None or cache.runtime_event_scores is None:
            raise RuntimeError("range-indexed event queries require an evaluation cache")
        device = cache.local_tokens.device
        leaves = cache.runtime_event_range_leaf_count
        if leaves < cache.n_epochs:
            raise RuntimeError("invalid range-index leaf count")
        plan_key = tuple(queries)
        plan = cache.runtime_range_query_plans.get(plan_key)
        if plan is None:
            node_lists = [
                self._range_node_cover(
                    0 if query.start_epoch is None else query.start_epoch,
                    cache.n_epochs if query.end_epoch is None else query.end_epoch,
                    leaves,
                )
                for query in queries
            ]
            group_lists = [self._query_stage_groups(query) for query in queries]
            max_nodes = max(map(len, node_lists))
            max_groups = max(map(len, group_lists))
            if max_nodes == 0 or max_groups == 0:
                raise RuntimeError("compiled event query has no valid temporal or stage range")
            node_rows = torch.full((len(queries), max_nodes), -1, dtype=torch.long, device=device)
            group_rows = torch.full((len(queries), max_groups), -1, dtype=torch.long, device=device)
            for row, (nodes, groups) in enumerate(zip(node_lists, group_lists)):
                if nodes:
                    node_rows[row, : len(nodes)] = torch.as_tensor(nodes, device=device)
                if groups:
                    group_rows[row, : len(groups)] = torch.as_tensor(groups, device=device)
            cache.runtime_range_query_plans[plan_key] = (node_rows, group_rows)
        else:
            node_rows, group_rows = plan

        compiled_plan = cache.runtime_compiled_event_plans.get(plan_key)
        if compiled_plan is None:
            columns = torch.as_tensor([EVENT_TYPES.index(event_type) for event_type in event_types], device=device)
            safe_nodes = node_rows.clamp_min(0)
            safe_groups = group_rows.clamp_min(0)
            _, stage_count, node_count, posting_width = cache.runtime_event_range_postings.shape
            flat_postings = cache.runtime_event_range_postings.reshape(-1, posting_width)
            flat_ids = (
                columns[:, None, None] * (stage_count * node_count)
                + safe_groups[:, :, None] * node_count
                + safe_nodes[:, None, :]
            )
            postings = flat_postings.index_select(0, flat_ids.reshape(-1)).reshape(
                len(queries), group_rows.shape[1], node_rows.shape[1], posting_width
            )
            valid = (node_rows[:, None, :, None] >= 0) & (group_rows[:, :, None, None] >= 0) & (postings >= 0)
            candidates = postings.reshape(len(queries), -1)
            valid = valid.reshape(len(queries), -1)
            safe_candidates = candidates.clamp_min(0)
            all_event_scores = cache.runtime_event_scores.transpose(0, 1).index_select(0, columns)
            evidence = all_event_scores.gather(1, safe_candidates).masked_fill(
                ~valid, torch.finfo(cache.runtime_event_scores.dtype).min
            )
            # The plan contains only model-predicted evidence. Gold event
            # labels never enter this cache, and the evaluation cache is
            # immutable after encode().
            compiled_plan = (candidates, valid, evidence)
            cache.runtime_compiled_event_plans[plan_key] = compiled_plan
        candidates, valid, evidence = compiled_plan
        count = min(int(top_k), evidence.shape[1])
        choices = torch.topk(evidence, k=count, dim=1, largest=True, sorted=True).indices
        locations = candidates.gather(1, choices)
        location_mask = valid.gather(1, choices)
        selected_evidence = evidence.gather(1, choices)
        burden = selected_evidence.masked_fill(~location_mask, torch.finfo(evidence.dtype).min).max(dim=1).values
        burden = torch.where(torch.isfinite(burden), burden, torch.zeros_like(burden))
        logits = cache.local_tokens.new_zeros((len(queries), 2))
        option_mask = torch.ones_like(logits, dtype=torch.bool)
        rows = torch.arange(len(queries), device=device)
        positive = torch.as_tensor([query.options.index("positive") for query in queries], device=device)
        negative = torch.as_tensor([query.options.index("negative") for query in queries], device=device)
        logits[rows, positive] = burden
        logits[rows, negative] = -burden
        return logits, option_mask, locations, location_mask

    @torch.no_grad()
    def _score_compiled_event_queries(
        self,
        cache: SleepCache,
        queries: Sequence[SleepQuery],
        *,
        top_k: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compile a binary event workload to exact sparse range-index reads."""

        if cache.runtime_event_range_postings is None or cache.runtime_event_scores is None:
            return self._score_compiled_dense_event_queries(cache, queries, top_k=top_k)
        if any(set(query.options) != {"negative", "positive"} for query in queries):
            return self.score_events_with_locations(cache, queries, top_k=top_k, readout_mode="sparse")
        event_types = [self._event_type_for_query(query) for query in queries]
        if any(event_type is None for event_type in event_types):
            return self.score_events_with_locations(cache, queries, top_k=top_k, readout_mode="sparse")
        if any(
            query.position is not None and query.position.lower() not in {"all", "any", "all positions"}
            for query in queries
        ):
            raise ValueError("compiled event queries do not support unaudited position constraints")
        if top_k > self.event_range_top_k:
            return self._score_compiled_dense_event_queries(cache, queries, top_k=top_k)
        return self._score_range_indexed_event_queries(cache, queries, event_types, top_k=top_k)

    @torch.no_grad()
    def _score_compiled_dense_event_queries(
        self,
        cache: SleepCache,
        queries: Sequence[SleepQuery],
        *,
        top_k: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Dense control for compiled binary event questions.

        This scans every cached overnight event score under the same predicted
        time/stage constraints.  It is the quality oracle for the segmented
        sparse path, not an externally supervised oracle.
        """

        if cache.runtime_event_scores is None:
            raise RuntimeError("compiled event queries require an evaluation cache")
        if any(set(query.options) != {"negative", "positive"} for query in queries):
            return self.score_events_with_locations(cache, queries, top_k=top_k, readout_mode="sparse")
        event_types = [self._event_type_for_query(query) for query in queries]
        if any(event_type is None for event_type in event_types):
            return self.score_events_with_locations(cache, queries, top_k=top_k, readout_mode="sparse")
        if any(
            query.position is not None and query.position.lower() not in {"all", "any", "all positions"}
            for query in queries
        ):
            raise ValueError("compiled event queries do not support unaudited position constraints")

        device = cache.local_tokens.device
        ids = torch.arange(cache.n_epochs, device=device, dtype=torch.long).unsqueeze(0).expand(len(queries), -1)
        candidate_count = ids.shape[1]
        starts = torch.as_tensor(
            [0 if query.start_epoch is None else query.start_epoch for query in queries],
            device=device,
            dtype=torch.long,
        ).clamp(0, cache.n_epochs)
        ends = torch.as_tensor(
            [cache.n_epochs if query.end_epoch is None else query.end_epoch for query in queries],
            device=device,
            dtype=torch.long,
        ).clamp(0, cache.n_epochs)
        valid = (ids >= starts.unsqueeze(1)) & (ids < ends.unsqueeze(1))
        requested_stage = [query.stage.upper() if query.stage else "" for query in queries]
        if cache.runtime_stage_ids is not None and any(requested_stage):
            stage_codes = {name: index for index, name in enumerate(STAGE_OPTIONS)}
            stage_ids = cache.runtime_stage_ids[ids]
            for row, stage in enumerate(requested_stage):
                if stage == "NREM":
                    allowed = torch.zeros(candidate_count, dtype=torch.bool, device=device)
                    for name in ("N1", "N2", "N3"):
                        allowed |= stage_ids[row] == stage_codes[name]
                    valid[row] &= allowed
                elif stage:
                    valid[row] &= stage_ids[row] == stage_codes.get(stage, -1)

        columns = torch.as_tensor([EVENT_TYPES.index(event_type) for event_type in event_types], device=device)
        evidence = cache.runtime_event_scores[ids, columns.unsqueeze(1)]
        evidence = evidence.masked_fill(~valid, torch.finfo(evidence.dtype).min)
        count = min(int(top_k), candidate_count)
        choices = torch.topk(evidence, k=count, dim=1, largest=True, sorted=True).indices
        locations = ids.gather(1, choices)
        location_mask = valid.gather(1, choices)
        selected_evidence = evidence.gather(1, choices)
        burden = selected_evidence.masked_fill(~location_mask, torch.finfo(evidence.dtype).min).max(dim=1).values
        burden = torch.where(torch.isfinite(burden), burden, torch.zeros_like(burden))
        logits = cache.local_tokens.new_zeros((len(queries), 2))
        option_mask = torch.ones_like(logits, dtype=torch.bool)
        for row, query in enumerate(queries):
            positive = query.options.index("positive")
            negative = query.options.index("negative")
            logits[row, positive] = burden[row]
            logits[row, negative] = -burden[row]
        return logits, option_mask, locations, location_mask

    def score(
        self,
        cache: SleepCache,
        queries: Sequence[SleepQuery],
        *,
        readout_mode: str = "sparse",
        _deduplicate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Return padded logits and option mask for many isolated questions."""

        if not queries:
            raise ValueError("at least one query is required")
        device = cache.local_tokens.device
        queries = list(queries)
        # Repeated questions in an operational worklist can share exact JEV
        # computation while preserving caller order. Q-scaling experiments use
        # distinct questions and therefore do not receive this optimization.
        if _deduplicate and not self.training and not torch.is_grad_enabled() and len(queries) > 1:
            unique_queries: list[SleepQuery] = []
            inverse: list[int] = []
            seen: dict[SleepQuery, int] = {}
            for query in queries:
                index = seen.get(query)
                if index is None:
                    index = len(unique_queries)
                    seen[query] = index
                    unique_queries.append(query)
                inverse.append(index)
            if len(unique_queries) < len(queries):
                logits, option_mask = self.score(
                    cache,
                    unique_queries,
                    readout_mode=readout_mode,
                    _deduplicate=False,
                )
                restore = torch.as_tensor(inverse, device=device, dtype=torch.long)
                return logits[restore], option_mask[restore]
        question = self.query_encoder(queries, cache.n_epochs, device).to(cache.local_tokens.dtype)
        if readout_mode == "sparse" and cache.local_selector_keys is not None:
            state, state_mask = self._select_tokens_batched(cache, queries, question)
        else:
            state_list = [
                self._select_tokens(cache, q, question[i : i + 1], readout_mode=readout_mode)
                for i, q in enumerate(queries)
            ]
            max_state = max(x.shape[0] for x in state_list)
            state = cache.local_tokens.new_zeros((len(queries), max_state, self.hidden_size))
            state_mask = torch.zeros((len(queries), max_state), dtype=torch.bool, device=device)
            for i, x in enumerate(state_list):
                state[i, : x.shape[0]] = x
                state_mask[i, : x.shape[0]] = True
        return self._score_selected_state(state, state_mask, question, queries)

    @torch.no_grad()
    def readout_stats(
        self,
        cache: SleepCache,
        queries: Sequence[SleepQuery],
        *,
        readout_mode: str = "sparse",
    ) -> list[dict]:
        """Report how many overnight tokens each query actually reads."""

        device = cache.local_tokens.device
        query_vectors = self.query_encoder(list(queries), cache.n_epochs, device)
        stats = []
        for query, vector in zip(queries, query_vectors):
            selected = self._select_tokens(cache, query, vector.unsqueeze(0), readout_mode=readout_mode)
            stats.append({
                "target": query.target,
                "selected_tokens": int(selected.shape[0]),
                "overnight_epochs": int(cache.n_epochs),
                "compression": float(cache.n_epochs / max(1, selected.shape[0])),
            })
        return stats

    @torch.no_grad()
    def answer(self, cache: SleepCache, queries: Sequence[SleepQuery]) -> list[dict]:
        logits, option_mask = self.score(cache, queries)
        results = []
        for i, query in enumerate(queries):
            p = logits[i, option_mask[i]].float().softmax(-1)
            probs = {option: float(value) for option, value in zip(query.options, p.cpu())}
            item = {
                "target": query.target,
                "type": query.question_type,
                "probabilities": probs,
                "prediction": query.options[int(p.argmax())],
                "confidence": float(p.max()),
            }
            if query.question_type == "boolean":
                item["probability_true"] = probs.get("true", float(p[-1]))
            if query.question_type == "score":
                item["expected_level"] = float((p * torch.arange(len(p), device=p.device, dtype=p.dtype)).sum())
            results.append(item)
        return results
