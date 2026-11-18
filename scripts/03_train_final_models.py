"""Step 3: cross-validate and refit XGBoost; evaluate on the student-test partition.

Compares, on the same trip-grouped folds:
  A. untuned XGBoost, 9 base features (five-model Colab configuration)
  B. untuned XGBoost, 10 features (base + heading_diff_deg)  -> validates the heading feature
  C. tuned XGBoost (Optuna best), 10 features                 -> the model served by the backend

Writes:
  artifacts/xgb_tuned.json, artifacts/model_config.json      (served model)
  reports/model_comparison.csv, reports/model_comparison.json
  data/processed/oof_train_scores.csv.gz   out-of-fold candidate scores for training trips (decoder tuning)
  data/processed/student_scores.csv.gz     candidate scores for student-test trips (tuned model)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from pathsnap.config import ARTIFACT_DIR, MAX_CANDIDATES, PROCESSED_DIR, RADIUS_M, RANDOM_SEED, REPORT_DIR
from pathsnap.cv import classifier_predictions, cross_validate, fit_and_score
from pathsnap.evaluation import observation_metrics
from pathsnap.geo import BASE_FEATURES, FEATURES, FEATURES_V1
from pathsnap.model import UNTUNED_XGB_PARAMS, choose_device


def load(partition_name):
    cand = pd.read_csv(PROCESSED_DIR / "candidates.csv.gz", dtype={"trip_id": str})
    obs = pd.read_csv(PROCESSED_DIR / "observations.csv.gz", dtype={"trip_id": str})
    return (cand[cand["partition"] == partition_name].reset_index(drop=True),
            obs[obs["partition"] == partition_name].reset_index(drop=True))


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    best = json.loads((ARTIFACT_DIR / "xgb_best_params.json").read_text(encoding="utf-8"))
    tuned_params = best["xgb_params"]
    tuned_spw = best["spw_multiplier"]

    train_c, train_o = load("train")
    stud_c, stud_o = load("student")

    configs = {
        "A_untuned_9feat": (UNTUNED_XGB_PARAMS, BASE_FEATURES, 1.0),
        "B_untuned_10feat_v1": (UNTUNED_XGB_PARAMS, FEATURES_V1, 1.0),
        "C_tuned_12feat_v2": (tuned_params, FEATURES, tuned_spw),
    }

    rows = []
    oof_tuned = None
    student_result = {}
    for name, (params, feats, spw_mult) in configs.items():
        print(f"\n=== {name} ===", flush=True)
        keep = name.startswith("C_")  # keep the served model's out-of-fold scores
        t0 = time.time()
        folds, oof = cross_validate(params, feats, train_c, train_o, device, spw_mult, keep_oof=keep)
        if keep:
            oof_tuned = oof
        print(folds[["fold", "top1_accuracy", "macro_f1", "candidate_logloss", "fit_seconds"]].round(4).to_string(index=False), flush=True)

        # Refit on all 80 training trips and score the 10 student-test trips.
        scored, fit_s, infer_s, model = fit_and_score(params, feats, train_c, stud_c, device, spw_mult)
        pred = classifier_predictions(scored, stud_o)
        sm = observation_metrics(pred["ref_edge_id"].to_numpy(), pred["pred_edge_id"].to_numpy())
        student = {
            "student_top1_accuracy": sm["top1_accuracy"],
            "student_macro_f1": sm["macro_f1"],
            "student_candidate_logloss": logloss(scored["label"].to_numpy(), scored["prob"].to_numpy()),
            "student_fit_seconds": fit_s,
            "student_inference_seconds": infer_s,
            "student_inference_ms_per_observation": 1000 * infer_s / max(len(stud_o), 1),
        }
        row = {
            "model": name,
            "cv_top1_accuracy_mean": folds["top1_accuracy"].mean(),
            "cv_top1_accuracy_std": folds["top1_accuracy"].std(),
            "cv_macro_f1_mean": folds["macro_f1"].mean(),
            "cv_candidate_logloss_mean": folds["candidate_logloss"].mean(),
            "cv_fit_seconds_mean": folds["fit_seconds"].mean(),
            "cv_total_seconds": time.time() - t0,
            **student,
        }
        rows.append(row)
        student_result[name] = {"scored": scored, "model": model, "features": feats,
                                "params": params, "spw_mult": spw_mult, "folds": folds}
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, indent=2), flush=True)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "model_comparison.json").write_text(comparison.to_json(orient="records", indent=2), encoding="utf-8")
    for name, res in student_result.items():
        res["folds"].to_csv(REPORT_DIR / f"cv_folds_{name}.csv", index=False)

    # ---- Save the served (tuned) model and its configuration ----
    tuned = student_result["C_tuned_12feat_v2"]
    tuned["model"].save_model(str(ARTIFACT_DIR / "xgb_tuned.json"))
    model_config = {
        "model_file": "xgb_tuned.json",
        "features": FEATURES,
        "xgb_params": tuned_params,
        "scale_pos_weight_multiplier": tuned_spw,
        "radius_m": RADIUS_M,
        "max_candidates": MAX_CANDIDATES,
        "seed": RANDOM_SEED,
        "device_used_for_training": device,
        "training_partition": "train (80 trips)",
        "xgboost_version": __import__("xgboost").__version__,
        "sklearn_version": __import__("sklearn").__version__,
    }
    (ARTIFACT_DIR / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    # ---- Save scores used by the HMM decoder ----
    keep_cols = ["trip_id", "obs_idx", "edge_id", "prob", "label"]
    oof_tuned[keep_cols].to_csv(PROCESSED_DIR / "oof_train_scores.csv.gz", index=False, compression="gzip")
    tuned["scored"][keep_cols].to_csv(PROCESSED_DIR / "student_scores.csv.gz", index=False, compression="gzip")
    print("\nSaved model:", ARTIFACT_DIR / "xgb_tuned.json")
    print(comparison.round(4).T.to_string())


if __name__ == "__main__":
    main()
