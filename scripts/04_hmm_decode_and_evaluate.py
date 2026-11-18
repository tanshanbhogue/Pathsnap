"""Step 4: learn HMM transitions, tune decoder parameters, compare classifier-only vs HMM/Viterbi.

* Transition table: counted from consecutive REFERENCE arcs of the 80 TRAINING trips only.
* Decoder parameters (emission scale, transition weight, NONE penalties) are chosen by the
  pooled observation Top-1 accuracy on out-of-fold (OOF) training scores. The student-test
  partition is used only for the final, single evaluation of the selected parameters.

Writes: artifacts/transition_table.json, artifacts/hmm_params.json,
        reports/hmm_grid_train_oof.csv, reports/classifier_vs_hmm_student.csv,
        reports/hmm_results.json
"""
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from pathsnap.config import ARTIFACT_DIR, PROCESSED_DIR, REPORT_DIR, STUDENT_DIR, TRAIN_DIR
from pathsnap.evaluation import mean_of_dicts, observation_metrics, route_metrics
from pathsnap.geo import list_trip_dirs, load_trip, reference_edges
from pathsnap.hmm import (DISTANCE_EDGES_M, GAP_EDGES_S, HmmParams, RoadGraph, build_trip_cache,
                          classifier_only, counts_to_log_table, learn_transition_counts,
                          viterbi_decode, N_KEYS, N_HOP_BUCKETS)


def load_partition(folder, partition):
    """Raw trips with reference labels (labels used only for transitions and evaluation)."""
    out = []
    for d in list_trip_dirs(folder):
        trip = load_trip(d, partition, load_route=True)
        out.append((trip, RoadGraph(trip.arcs), reference_edges(trip)))
    return out


def build_caches(partition_trips, scores: pd.DataFrame):
    caches = []
    grouped = {tid: g for tid, g in scores.groupby("trip_id", sort=False)}
    for trip, graph, ref in partition_trips:
        g = grouped.get(trip.trip_id, pd.DataFrame(columns=["obs_idx", "edge_id", "prob"]))
        caches.append(build_trip_cache(trip, g, graph, ref, trip.route))
    return caches


def evaluate(caches, decode, *, with_route=True):
    """Pool observation-level metrics over trips; average route-level metrics per trip."""
    refs, preds, route_rows, t_decode = [], [], [], 0.0
    for c in caches:
        t0 = time.perf_counter()
        pred = decode(c)
        t_decode += time.perf_counter() - t0
        refs.append(c.ref)
        preds.append(pred)
        if with_route:
            route_rows.append(route_metrics(pred, c.route, c.graph))
    out = observation_metrics(np.concatenate(refs), np.concatenate(preds))
    if with_route:
        out.update(mean_of_dicts(route_rows))
    out["decode_seconds"] = t_decode
    out["decode_ms_per_observation"] = 1000 * t_decode / max(sum(len(r) for r in refs), 1)
    return out


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw training and student trips...", flush=True)
    train_trips = load_partition(TRAIN_DIR, "train")
    stud_trips = load_partition(STUDENT_DIR, "student")

    # ---- Transition table from training reference routes only ----
    counts = np.zeros((N_KEYS, N_HOP_BUCKETS))
    for trip, graph, ref in train_trips:
        counts += learn_transition_counts(ref, trip, graph)
    log_table = counts_to_log_table(counts, alpha=1.0)
    key_definition = {
        "distance_edges_m": DISTANCE_EDGES_M.tolist(),
        "gap_edges_s": GAP_EDGES_S.tolist(),
        "key": "distance_bin * (len(gap_edges_s)+1) + gap_bin",
        "hop_buckets": {"0": "same arc", "1": "directly connected", "2": "two arcs",
                        "3": "three arcs", "4": "more than three arcs or unreachable"},
    }
    (ARTIFACT_DIR / "transition_table.json").write_text(json.dumps({
        "definition": key_definition,
        "training_transition_counts": counts.astype(int).tolist(),
        "log_probabilities": log_table.tolist(),
    }, indent=2), encoding="utf-8")
    print("Transition counts learned from", len(train_trips), "training trips;",
          int(counts.sum()), "transitions", flush=True)

    # ---- Caches (classifier scores: OOF for train, final model for student) ----
    oof = pd.read_csv(PROCESSED_DIR / "oof_train_scores.csv.gz", dtype={"trip_id": str})
    stu = pd.read_csv(PROCESSED_DIR / "student_scores.csv.gz", dtype={"trip_id": str})
    train_caches = build_caches(train_trips, oof)
    stud_caches = build_caches(stud_trips, stu)

    tt = log_table

    # ---- Classifier-only on OOF train (reference point) ----
    clf_train = evaluate(train_caches, classifier_only, with_route=False)

    # ---- Grid search on OOF training scores only ----
    grid = []
    for scale, w, none_e, none_t in itertools.product(
            [0.5, 1.0, 2.0], [0.5, 1.0, 2.0],
            [np.log(0.01), np.log(0.001)], [np.log(0.1), np.log(0.01)]):
        params = HmmParams(emission_scale=scale, transition_weight=w,
                           none_log_emission=float(none_e), none_log_transition=float(none_t))
        m = evaluate(train_caches, lambda c, p=params: viterbi_decode(c, p, tt), with_route=False)
        grid.append({**params.to_dict(), "train_oof_top1": m["top1_accuracy"],
                     "train_oof_macro_f1": m["macro_f1"], "decode_seconds": m["decode_seconds"]})
        print(f"scale={scale} w={w} none_e={np.exp(none_e):.3f} none_t={np.exp(none_t):.2f} "
              f"-> OOF top1 {m['top1_accuracy']:.4%}", flush=True)
    grid_df = pd.DataFrame(grid).sort_values("train_oof_top1", ascending=False)
    grid_df.to_csv(REPORT_DIR / "hmm_grid_train_oof.csv", index=False)
    best = grid_df.iloc[0]
    params = HmmParams(emission_scale=float(best["emission_scale"]),
                       transition_weight=float(best["transition_weight"]),
                       none_log_emission=float(best["none_log_emission"]),
                       none_log_transition=float(best["none_log_transition"]))
    (ARTIFACT_DIR / "hmm_params.json").write_text(json.dumps(params.to_dict(), indent=2), encoding="utf-8")
    print("Selected HMM params (by training OOF):", params.to_dict(), flush=True)

    # ---- Single evaluation on student-test trips ----
    clf_student = evaluate(stud_caches, classifier_only, with_route=True)
    hmm_student = evaluate(stud_caches, lambda c: viterbi_decode(c, params, tt), with_route=True)
    # Reference: route metrics of the ground truth itself (sanity check of the metric code).
    ref_continuity = mean_of_dicts([route_metrics(c.ref, c.route, c.graph) for c in stud_caches])

    rows = []
    for name, m in [("XGBoost classifier only", clf_student), ("XGBoost + HMM/Viterbi", hmm_student)]:
        rows.append({"method": name, **m})
    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "classifier_vs_hmm_student.csv", index=False)

    results = {
        "selected_hmm_params": params.to_dict(),
        "train_oof": {"classifier_only": clf_train, "hmm_selected": {
            "top1_accuracy": float(best["train_oof_top1"]), "macro_f1": float(best["train_oof_macro_f1"])}},
        "student_test": {"classifier_only": clf_student, "hmm_viterbi": hmm_student},
        "reference_route_sanity": ref_continuity,
    }
    (REPORT_DIR / "hmm_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("\nSTUDENT TEST: classifier-only vs HMM")
    print(table.set_index("method").T.round(4).to_string())


if __name__ == "__main__":
    main()
