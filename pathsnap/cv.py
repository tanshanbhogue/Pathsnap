"""Trip-grouped cross-validation shared by the optimisation and final training scripts."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pathsnap.config import N_CV_FOLDS, RANDOM_SEED
from pathsnap.evaluation import observation_metrics
from pathsnap.geo import FEATURES
from pathsnap.model import make_xgb, rows_with_truth, scale_pos_weight_for


def trip_folds(trip_ids, n_splits: int = N_CV_FOLDS, seed: int = RANDOM_SEED):
    """Split whole trips into folds. No trip's observations ever appear in both fit and validation."""
    trips = np.array(sorted(set(trip_ids)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(set(trips[a]), set(trips[b])) for a, b in kf.split(trips)]


def classifier_predictions(scored: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Top-1 arc per observation from candidate probabilities. Missing candidates -> -1."""
    best = (scored.sort_values("prob", ascending=False, kind="stable")
            .drop_duplicates(["trip_id", "obs_idx"])[["trip_id", "obs_idx", "edge_id"]])
    out = obs[["trip_id", "obs_idx", "ref_edge_id"]].merge(best, on=["trip_id", "obs_idx"], how="left")
    out["pred_edge_id"] = out["edge_id"].fillna(-1).astype(np.int64)
    return out


def fit_and_score(params: dict, features: list[str], fit_cand: pd.DataFrame, val_cand: pd.DataFrame,
                  device: str, spw_multiplier: float = 1.0) -> tuple[pd.DataFrame, float, float]:
    """Fit on ``fit_cand`` (observations with a positive) and score every row of ``val_cand``."""
    fit = rows_with_truth(fit_cand)
    spw = scale_pos_weight_for(fit["label"]) * spw_multiplier
    model = make_xgb(params, spw, device)
    t0 = time.perf_counter()
    model.fit(fit[features], fit["label"])
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    prob = model.predict_proba(val_cand[features])[:, 1] if len(val_cand) else np.empty(0)
    infer_s = time.perf_counter() - t1
    scored = val_cand[["trip_id", "obs_idx", "edge_id", "label"]].assign(prob=prob)
    return scored, fit_s, infer_s, model


def cross_validate(params: dict, features: list[str], cand: pd.DataFrame, obs: pd.DataFrame,
                   device: str, spw_multiplier: float = 1.0, keep_oof: bool = False):
    """Trip-grouped CV on the training partition. Returns per-fold metrics (and OOF scores)."""
    folds = trip_folds(obs["trip_id"].unique())
    rows, oof = [], []
    for k, (fit_trips, val_trips) in enumerate(folds, 1):
        fit_c = cand[cand["trip_id"].isin(fit_trips)]
        val_c = cand[cand["trip_id"].isin(val_trips)]
        val_o = obs[obs["trip_id"].isin(val_trips)]
        scored, fit_s, infer_s, _ = fit_and_score(params, features, fit_c, val_c, device, spw_multiplier)
        pred = classifier_predictions(scored, val_o)
        m = observation_metrics(pred["ref_edge_id"].to_numpy(), pred["pred_edge_id"].to_numpy())
        ll_p = np.clip(scored["prob"].to_numpy(), 1e-7, 1 - 1e-7)
        y = scored["label"].to_numpy()
        logloss = float(-np.mean(y * np.log(ll_p) + (1 - y) * np.log(1 - ll_p))) if len(y) else float("nan")
        rows.append({"fold": k, **m, "candidate_logloss": logloss,
                     "fit_seconds": fit_s, "inference_seconds": infer_s,
                     "val_trips": len(val_trips)})
        if keep_oof:
            oof.append(scored)
    result = pd.DataFrame(rows)
    if keep_oof:
        return result, pd.concat(oof, ignore_index=True)
    return result, None
