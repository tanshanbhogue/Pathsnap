"""Evaluation metrics. Reference labels are passed in explicitly; this module never loads them."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from pathsnap.hmm import RoadGraph


def observation_metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    """Top-1 accuracy over ALL observations (missing candidates count as wrong) and macro F1."""
    ref = np.asarray(ref)
    pred = np.asarray(pred)
    if len(ref) == 0:
        return {"top1_accuracy": float("nan"), "macro_f1": float("nan"), "observations": 0}
    return {
        "top1_accuracy": float((ref == pred).mean()),
        "macro_f1": float(f1_score(ref, pred, average="macro", zero_division=0)),
        "observations": int(len(ref)),
    }


def collapse_route(edges) -> list[int]:
    """Drop NONE (-1) states and merge consecutive repeats into a route sequence."""
    out: list[int] = []
    for e in edges:
        e = int(e)
        if e < 0:
            continue
        if not out or out[-1] != e:
            out.append(e)
    return out


def _lcs_length(a: list[int], b: list[int]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def route_continuity(route: list[int], graph: RoadGraph) -> float:
    """Share of consecutive route arcs that are the same or directly connected."""
    if len(route) < 2:
        return float("nan")
    ok = sum(graph.hop_bucket(a, b) <= 1 for a, b in zip(route[:-1], route[1:]))
    return ok / (len(route) - 1)


def route_metrics(pred_edges, ref_route, graph: RoadGraph) -> dict:
    """Sequence-level metrics comparing the predicted route with the reference route."""
    pred_route = collapse_route(pred_edges)
    ref_list = [int(x) for x in ref_route]
    ref_collapsed = collapse_route(ref_list)

    p_set, r_set = set(pred_route), set(ref_list)
    inter = len(p_set & r_set)
    precision = inter / len(p_set) if p_set else 0.0
    recall = inter / len(r_set) if r_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    lcs = _lcs_length(pred_route, ref_collapsed)
    return {
        "route_edge_precision": precision,
        "route_edge_recall": recall,
        "route_edge_f1": f1,
        "route_lcs_precision": lcs / len(pred_route) if pred_route else 0.0,
        "route_lcs_recall": lcs / len(ref_collapsed) if ref_collapsed else 0.0,
        "exact_sequence_match": float(pred_route == ref_collapsed),
        "predicted_route_length": len(pred_route),
        "continuity": route_continuity(pred_route, graph),
    }


def mean_of_dicts(dicts: list[dict]) -> dict:
    """Average numeric values across trips, ignoring NaN."""
    if not dicts:
        return {}
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if d.get(k) is not None and not np.isnan(d[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
