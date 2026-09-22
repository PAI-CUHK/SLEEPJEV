import numpy as np
import torch

from sleepjev.baselines import FixedHeadDL, IndependentDL, SharedMultiTaskDL
from sleepjev.events import event_labels_from_events, infer_analysis_interval, normalize_event_label
from sleepjev.index import SleepEventIndex
from sleepjev.model import SleepJEV
from sleepjev.train import evaluate_fixed_baseline, evaluate_sleepjev_tasks, fit_sleepjev_tasks
from sleepjev.types import STAGE_OPTIONS, EventLabels, SleepEvent, SleepNight, SleepQuery


def _model():
    torch.manual_seed(17)
    return SleepJEV(input_dim=6, hidden_size=32, coarse_factor=2).eval()


def test_sleepjev_dynamic_candidates_and_permutation_equivariance():
    model = _model()
    features = torch.randn(20, 6)
    cache = model.encode(features)
    q = SleepQuery("respiratory_event", options=("none", "obstructive", "hypopnea"), start_epoch=4, end_epoch=5)
    p = model.answer(cache, [q])[0]["probabilities"]
    q2 = SleepQuery("respiratory_event", options=("hypopnea", "none", "obstructive"), start_epoch=4, end_epoch=5)
    p2 = model.answer(cache, [q2])[0]["probabilities"]
    assert set(p) == set(p2)
    for key in p:
        assert abs(p[key] - p2[key]) < 1e-6


def test_query_render_keeps_time_windows_distinct_but_not_option_order():
    first = SleepQuery("hypopnea", "boolean", ("negative", "positive"), start_epoch=0, end_epoch=60)
    second = SleepQuery("hypopnea", "boolean", ("positive", "negative"), start_epoch=60, end_epoch=120)
    same_window_reordered = SleepQuery("hypopnea", "boolean", ("positive", "negative"), start_epoch=0, end_epoch=60)
    assert first.render() != second.render()
    assert first.render() == same_window_reordered.render()


def test_sleepjev_answers_many_isolated_queries_with_dynamic_k():
    model = _model()
    cache = model.encode(torch.randn(32, 6))
    queries = [
        SleepQuery("sleep_stage", options=("W", "N1", "N2", "N3", "REM", "UNKNOWN"), start_epoch=i, end_epoch=i + 1)
        for i in range(8)
    ]
    queries.append(SleepQuery("sleep_stage", options=("W", "N2"), start_epoch=8, end_epoch=9))
    result = model.answer(cache, queries)
    assert len(result) == 9
    assert set(result[-1]["probabilities"]) == {"W", "N2"}


def test_batched_sparse_selection_matches_sequential_eval_path():
    model = _model()
    cache = model.encode(torch.randn(40, 6))
    queries = [
        SleepQuery("sleep_stage", options=("W", "N1", "N2", "N3", "REM", "UNKNOWN"), start_epoch=2, end_epoch=3),
        SleepQuery("apnea", "boolean", ("negative", "positive"), max_readout_tokens=8),
        SleepQuery("hypopnea", "boolean", ("negative", "positive"), max_readout_tokens=4),
    ]
    batched_logits, batched_mask = model.score(cache, queries, readout_mode="sparse")
    cache.local_selector_keys = None
    sequential_logits, sequential_mask = model.score(cache, queries, readout_mode="sparse")
    assert torch.equal(batched_mask, sequential_mask)
    assert torch.allclose(batched_logits, sequential_logits, atol=1e-5)


def test_event_answer_with_locations_reuses_the_sparse_jev_selection():
    model = _model()
    cache = model.encode(torch.randn(40, 6))
    queries = [
        SleepQuery("apnea", "boolean", ("negative", "positive"), 0, 20, max_readout_tokens=8),
        SleepQuery("arousal", "boolean", ("negative", "positive"), 10, 40, max_readout_tokens=8),
    ]
    logits, option_mask = model.score(cache, queries, readout_mode="sparse")
    combined_logits, combined_mask, locations, location_mask = model.score_events_with_locations(
        cache, queries, readout_mode="sparse", top_k=3
    )
    assert torch.equal(option_mask, combined_mask)
    assert torch.allclose(logits, combined_logits, atol=1e-6)
    assert locations.shape == (2, 3)
    assert location_mask.shape == (2, 3)
    compiled_logits, compiled_mask, compiled_locations, compiled_location_mask = model.score_events_with_locations(
        cache, queries, readout_mode="compiled", top_k=3
    )
    assert compiled_logits.shape == logits.shape
    assert torch.equal(compiled_mask, option_mask)
    assert compiled_locations.shape == locations.shape
    assert compiled_location_mask.shape == location_mask.shape


def test_segmented_event_index_matches_dense_event_score_oracle_for_time_and_stage_queries():
    model = _model()
    cache = model.encode(torch.randn(73, 6))
    predicted_stage = STAGE_OPTIONS[int(cache.runtime_stage_ids[9])]
    queries = [
        SleepQuery("apnea", "boolean", ("negative", "positive"), 3, 67),
        SleepQuery("hypopnea", "boolean", ("negative", "positive"), 9, 41, stage=predicted_stage),
        SleepQuery("arousal", "boolean", ("negative", "positive"), 0, 73),
    ]
    sparse = model.score_events_with_locations(cache, queries, readout_mode="compiled", top_k=5)
    dense = model.score_events_with_locations(cache, queries, readout_mode="compiled_dense", top_k=5)
    sparse_logits, sparse_options, sparse_locations, sparse_valid = sparse
    dense_logits, dense_options, dense_locations, dense_valid = dense
    assert torch.equal(sparse_logits, dense_logits)
    assert torch.equal(sparse_options, dense_options)
    assert torch.equal(sparse_valid, dense_valid)
    assert torch.equal(sparse_locations[sparse_valid], dense_locations[dense_valid])


def test_compiled_event_plan_reuses_materialized_candidates_without_changing_answers():
    model = _model()
    cache = model.encode(torch.randn(73, 6))
    queries = [
        SleepQuery("apnea", "boolean", ("negative", "positive"), 3, 67),
        SleepQuery("hypopnea", "boolean", ("negative", "positive"), 9, 41),
    ]
    first = model.score_events_with_locations(cache, queries, readout_mode="compiled", top_k=5)
    assert tuple(queries) in cache.runtime_compiled_event_plans
    planned = cache.runtime_compiled_event_plans[tuple(queries)]
    second = model.score_events_with_locations(cache, queries, readout_mode="compiled", top_k=5)
    assert cache.runtime_compiled_event_plans[tuple(queries)] is planned
    for first_value, second_value in zip(first, second):
        assert torch.equal(first_value, second_value)


def test_lazy_event_cache_keeps_compiled_answers_exact_and_can_omit_range_index_for_fresh_query():
    model = _model()
    features = torch.randn(73, 6)
    queries = [
        SleepQuery("apnea", "boolean", ("negative", "positive"), 3, 67),
        SleepQuery("hypopnea", "boolean", ("negative", "positive"), 9, 41),
        SleepQuery("arousal", "boolean", ("negative", "positive"), 0, 73),
    ]
    full = model.encode(features)
    lazy = model.encode(features, event_query_only=True)
    fresh = model.encode(features, event_query_only=True, build_event_range_index=False)
    assert torch.equal(full.runtime_stage_ids, lazy.runtime_stage_ids)
    assert torch.equal(full.runtime_event_scores, lazy.runtime_event_scores)
    compiled_full = model.score_events_with_locations(full, queries, readout_mode="compiled", top_k=5)
    compiled_lazy = model.score_events_with_locations(lazy, queries, readout_mode="compiled", top_k=5)
    dense_fresh = model.score_events_with_locations(fresh, queries, readout_mode="compiled", top_k=5)
    for full_value, lazy_value, fresh_value in zip(compiled_full, compiled_lazy, dense_fresh):
        if full_value.dtype == torch.bool:
            assert torch.equal(full_value, lazy_value)
            assert torch.equal(full_value, fresh_value)
        else:
            assert torch.equal(full_value, lazy_value)
    full_locations, full_valid = compiled_full[2:]
    assert torch.equal(compiled_full[0], dense_fresh[0])
    assert torch.equal(compiled_full[1], dense_fresh[1])
    fresh_locations, fresh_valid = dense_fresh[2:]
    assert torch.equal(full_valid, fresh_valid)
    assert torch.equal(full_locations[full_valid], fresh_locations[fresh_valid])


def test_sleepjev_cache_shape_and_night_query():
    model = _model()
    cache = model.encode(torch.randn(40, 6))
    assert cache.local_tokens.shape == (1, 40, 32)
    assert cache.coarse_tokens.shape[0] == 1
    result = model.answer(cache, [SleepQuery("night_summary", options=("sleep", "wake"))])
    assert abs(sum(result[0]["probabilities"].values()) - 1.0) < 1e-6
    assert cache.hour_tokens is not None
    assert cache.local_selector_keys is not None
    assert cache.coarse_selector_keys is not None
    assert cache.hour_selector_keys is not None


def test_fast_hierarchy_keeps_the_same_multiresolution_cache_contract():
    model = SleepJEV(input_dim=6, hidden_size=16, coarse_factor=2, hour_factor=4, fast_hierarchy=True).eval()
    cache = model.encode(torch.randn(20, 6))
    assert cache.local_tokens.shape == (1, 20, 16)
    assert cache.coarse_tokens.shape == (1, 10, 16)
    assert cache.hour_tokens.shape == (1, 5, 16)


def test_unknown_events_are_masked_not_negative():
    events = (
        SleepEvent(0.0, 10.0, "hypopnea", "Hypopnea"),
        SleepEvent(30.0, 5.0, "unknown", "Unsure"),
    )
    labels = event_labels_from_events(events, 3)
    assert labels.column("hypopnea").tolist() == [1, -1, -1]
    assert labels.column("apnea").tolist() == [-1, -1, -1]
    assert normalize_event_label("Obstructive Apnea") == "apnea_obstructive"
    assert normalize_event_label("Signal artifact") == "unknown"
    assert infer_analysis_interval((SleepEvent(10, 0, "meta:analysis_start"),)) == (10.0, None)


def test_event_index_applies_query_constraints_without_label_posting_leakage():
    events = (SleepEvent(60.0, 30.0, "arousal", "Arousal"),)
    night = SleepNight(
        "x", np.zeros((8, 2), dtype=np.float32), np.asarray(["W", "W", "REM", "REM", "W", "W", "W", "W"]),
        events=events,
        event_labels=EventLabels(np.zeros((8, 3), dtype=np.int8)),
        position_labels=np.asarray(["supine"] * 4 + ["left"] * 4),
    )
    index = SleepEventIndex.from_night(night)
    query = SleepQuery("arousal", options=("negative", "positive"), stage="REM", position="supine")
    assert index.candidate_epochs(query).tolist() == [2, 3]
    assert index.candidate_epochs(query, include_event_postings=True).tolist() == [2]


def test_runtime_stage_constraint_index_does_not_read_gold_stage_labels():
    model = _model()
    features = torch.randn(12, 6)
    first = SleepNight("first", features.numpy(), np.asarray(["N2"] * 12))
    second = SleepNight("second", features.numpy(), np.asarray(["REM"] * 12))
    first_cache = model.encode(features, event_index=first.event_index())
    second_cache = model.encode(features, event_index=second.event_index())
    query = SleepQuery("hypopnea", "boolean", ("negative", "positive"), stage="N2")
    first_candidates = first_cache.constraint_index.candidate_epochs(query)
    second_candidates = second_cache.constraint_index.candidate_epochs(query)
    assert np.array_equal(first_candidates, second_candidates)


def test_fixed_baseline_shapes_and_dynamic_jev_event_query():
    features = torch.randn(1, 12, 6)
    for model in (IndependentDL(6), SharedMultiTaskDL(6), FixedHeadDL(6)):
        outputs = model(features)
        assert outputs["sleep_stage"].shape == (1, 12, 6)
        assert outputs["apnea"].shape == (1, 12, 2)
        assert outputs["hypopnea"].shape == (1, 12, 2)
        assert outputs["arousal"].shape == (1, 12, 2)
    model = _model()
    cache = model.encode(torch.randn(32, 6))
    result = model.answer(
        cache,
        [SleepQuery("hypopnea", "boolean", ("negative", "positive"), max_readout_tokens=4)],
    )[0]
    assert set(result["probabilities"]) == {"negative", "positive"}
    stats = model.readout_stats(cache, [SleepQuery("hypopnea", options=("negative", "positive"), max_readout_tokens=4)])[0]
    assert stats["selected_tokens"] <= 4
    dense_stats = model.readout_stats(
        cache,
        [SleepQuery("hypopnea", options=("negative", "positive"))],
        readout_mode="dense",
    )[0]
    assert dense_stats["selected_tokens"] == 32


def test_positive_unlabeled_events_do_not_emit_spurious_f1():
    night = SleepNight(
        "pu",
        np.zeros((8, 6), dtype=np.float32),
        np.asarray(["W", "N1", "N2", "N3", "REM", "W", "N2", "REM"]),
        event_labels=EventLabels(np.asarray([
            [-1, -1, -1], [1, -1, -1], [-1, -1, -1], [-1, -1, -1],
            [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1],
        ], dtype=np.int8)),
    )
    jev_metrics = evaluate_sleepjev_tasks(_model(), [night], tasks=("apnea",))
    fixed_metrics = evaluate_fixed_baseline(FixedHeadDL(6, hidden_size=32), [night], tasks=("apnea",))
    for metrics in (jev_metrics["apnea"], fixed_metrics["apnea"]):
        assert metrics["metric_status"] == "positive_unlabeled"
        assert metrics["macro_f1"] is None
        assert metrics["label_coverage"]["positive"] == 1


def test_index_loss_weights_are_configurable_and_reject_negative_values():
    night = SleepNight(
        "tiny", np.zeros((4, 6), dtype=np.float32), np.asarray(["W", "N1", "N2", "REM"]),
        event_labels=EventLabels(np.full((4, 3), -1, dtype=np.int8)),
    )
    with __import__("pytest").raises(ValueError):
        fit_sleepjev_tasks(_model(), [night], epochs=1, stage_index_loss_weight=-0.1)
