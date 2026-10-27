"""Tests for the ranking metrics added for the evaluation report (AP / mAP / PR).

Pure functions, hand-verified — no retrieval or models involved.
"""

from __future__ import annotations

import math

from rag_ucdp.evaluation import (
    average_precision,
    average_precision_at_k,
    interpolated_pr_curve,
    mean_average_precision,
    mean_average_precision_at_k,
    mean_pr_curve,
)


def _close(a, b, tol=1e-9):
    return math.isclose(a, b, abs_tol=tol)


# --- average_precision -------------------------------------------------------


def test_ap_perfect_ranking_is_one():
    assert _close(average_precision(["A", "B", "C", "X", "Y"], ["A", "B", "C"]), 1.0)


def test_ap_interleaved_relevant():
    # hits at ranks 1,3,5 -> (1/1 + 2/3 + 3/5) / 3
    ap = average_precision(["A", "X", "B", "X", "C"], ["A", "B", "C"])
    assert _close(ap, (1.0 + 2 / 3 + 3 / 5) / 3)


def test_ap_all_relevant_at_end():
    # hits at ranks 3,4,5 -> (1/3 + 2/4 + 3/5) / 3
    ap = average_precision(["X", "Y", "A", "B", "C"], ["A", "B", "C"])
    assert _close(ap, (1 / 3 + 2 / 4 + 3 / 5) / 3)


def test_ap_empty_gold_is_zero():
    assert average_precision(["A", "B"], []) == 0.0


def test_ap_no_hits_is_zero():
    assert average_precision(["X", "Y"], ["A"]) == 0.0


def test_ap_normalises_by_gold_size_not_hits():
    # only 1 of 2 gold retrieved, at rank 1: precision_sum=1.0, /|gold|=2 -> 0.5
    assert _close(average_precision(["A", "X", "Y"], ["A", "B"]), 0.5)


def test_ap_id_types_normalised():
    assert _close(average_precision([1, 2, 3], ["1", "2", "3"]), 1.0)


# --- mean_average_precision --------------------------------------------------


def test_map_averages_over_queries():
    pred = {"q1": ["A", "B"], "q2": ["X", "A"]}
    gold = {"q1": ["A", "B"], "q2": ["A"]}  # AP q1=1.0, AP q2=0.5
    assert _close(mean_average_precision(pred, gold), 0.75)


def test_map_skips_queries_without_gold():
    pred = {"q1": ["A"], "q2": ["X"]}
    gold = {"q1": ["A"], "q2": []}  # q2 skipped -> mAP = AP(q1) = 1.0
    assert _close(mean_average_precision(pred, gold), 1.0)


def test_map_missing_prediction_scores_zero():
    pred = {"q1": ["A"]}  # q2 has no prediction
    gold = {"q1": ["A"], "q2": ["B"]}  # AP q1=1.0, q2=0.0 -> 0.5
    assert _close(mean_average_precision(pred, gold), 0.5)


# --- average_precision_at_k (MAP@k, min(k,R) normalisation) -------------------


def test_ap_at_k_all_within_k_matches_full_ap():
    # all relevant inside top-10 -> same as full AP
    ap = average_precision_at_k(["A", "X", "B", "X", "C"], ["A", "B", "C"], 10)
    assert _close(ap, (1.0 + 2 / 3 + 3 / 5) / 3)


def test_ap_at_k_truncates_and_normalises_by_min_k_r():
    # k=2: only A (rank1) inside top-2; denom = min(2,3) = 2 -> 1.0/2
    assert _close(average_precision_at_k(["A", "X", "B", "X", "C"], ["A", "B", "C"], 2), 0.5)


def test_ap_at_k_perfect_topk_is_one_even_when_R_exceeds_k():
    # 5 relevant, k=3, perfect top-3 -> min(3,5)=3, sum=3 -> 1.0
    ap = average_precision_at_k(["A", "B", "C", "X", "Y"], ["A", "B", "C", "D", "E"], 3)
    assert _close(ap, 1.0)


def test_ap_at_k_short_list():
    # only 1 of 2 relevant retrieved (rank1); denom = min(10,2)=2 -> 0.5
    assert _close(average_precision_at_k(["A"], ["A", "B"], 10), 0.5)


def test_ap_at_k_empty_gold_zero():
    assert average_precision_at_k(["A"], [], 10) == 0.0


def test_map_at_k_averages_over_queries():
    pred = {"q1": ["A", "B"], "q2": ["X", "A"]}
    gold = {"q1": ["A", "B"], "q2": ["A"]}  # AP@10 q1=1.0, q2=0.5
    assert _close(mean_average_precision_at_k(pred, gold, 10), 0.75)


# --- interpolated_pr_curve ---------------------------------------------------


def test_pr_curve_length_matches_levels():
    curve = interpolated_pr_curve(["A"], ["A"])
    assert len(curve) == 11  # 0.0 .. 1.0


def test_pr_curve_interleaved_values():
    curve = interpolated_pr_curve(["A", "X", "B", "X", "C"], ["A", "B", "C"])
    # recall/precision points: (1/3,1.0),(2/3,2/3),(1.0,3/5)
    expected = [1.0, 1.0, 1.0, 1.0, 2 / 3, 2 / 3, 2 / 3, 3 / 5, 3 / 5, 3 / 5, 3 / 5]
    assert all(_close(a, b) for a, b in zip(curve, expected, strict=True))


def test_pr_curve_perfect_is_flat_one():
    curve = interpolated_pr_curve(["A", "B", "C"], ["A", "B", "C"])
    assert all(_close(p, 1.0) for p in curve)


def test_pr_curve_empty_gold_all_zero():
    assert interpolated_pr_curve(["A"], []) == [0.0] * 11


def test_mean_pr_curve_averages_elementwise():
    pred = {"q1": ["A", "B"], "q2": ["X", "A"]}
    gold = {"q1": ["A", "B"], "q2": ["A"]}
    c1 = interpolated_pr_curve(["A", "B"], ["A", "B"])  # all 1.0
    c2 = interpolated_pr_curve(["X", "A"], ["A"])  # precision 0.5 at recall 1.0
    mean = mean_pr_curve(pred, gold)
    expected = [(a + b) / 2 for a, b in zip(c1, c2, strict=True)]
    assert all(_close(a, b) for a, b in zip(mean, expected, strict=True))
