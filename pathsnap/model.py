"""XGBoost candidate classifier helpers shared by training scripts and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from pathsnap.config import RANDOM_SEED

# Untuned configuration taken from the five-model Colab benchmark (PathSnap_Five_Model_Comparison).
UNTUNED_XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.035, subsample=0.85,
    colsample_bytree=0.9, min_child_weight=2, reg_lambda=2.0, reg_alpha=0.05,
)


def choose_device() -> str:
    """Use the GPU if XGBoost can run on it, otherwise fall back to CPU."""
    try:
        x = np.random.default_rng(0).random((200, 3))
        y = np.tile([0, 1], 100)
        XGBClassifier(n_estimators=2, device="cuda", tree_method="hist").fit(x, y)
        return "cuda"
    except Exception:
        return "cpu"


def scale_pos_weight_for(labels) -> float:
    labels = np.asarray(labels)
    pos = max(int((labels == 1).sum()), 1)
    neg = max(int((labels == 0).sum()), 1)
    return neg / pos


def make_xgb(params: dict, scale_pos_weight: float, device: str, seed: int = RANDOM_SEED) -> XGBClassifier:
    return XGBClassifier(
        **params,
        scale_pos_weight=float(scale_pos_weight),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=-1,
    )


def rows_with_truth(cand: pd.DataFrame) -> pd.DataFrame:
    """Keep only observations whose correct arc is among their candidates (for fitting)."""
    has_pos = cand.groupby(["trip_id", "obs_idx"])["label"].transform("max").eq(1)
    return cand[has_pos]
