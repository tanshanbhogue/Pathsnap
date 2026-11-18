"""Step 5: feature ablation on the TRAINING partition only (trip-grouped CV, untuned XGBoost).

Compares the feature sets on identical folds. Student-test data is not used here.
Output: reports/ablation_features.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from pathsnap.config import PROCESSED_DIR, REPORT_DIR
from pathsnap.cv import cross_validate
from pathsnap.geo import FEATURES, FEATURES_V1
from pathsnap.model import UNTUNED_XGB_PARAMS, choose_device


def main() -> None:
    cand = pd.read_csv(PROCESSED_DIR / "candidates.csv.gz", dtype={"trip_id": str})
    obs = pd.read_csv(PROCESSED_DIR / "observations.csv.gz", dtype={"trip_id": str})
    cand = cand[cand["partition"] == "train"].reset_index(drop=True)
    obs = obs[obs["partition"] == "train"].reset_index(drop=True)
    device = choose_device()

    sets = {
        "v1_10feat (heading)": FEATURES_V1,
        "v2_12feat (+arc length, along-track)": FEATURES,
    }
    rows = []
    for name, feats in sets.items():
        folds, _ = cross_validate(UNTUNED_XGB_PARAMS, feats, cand, obs, device)
        row = {"feature_set": name, "n_features": len(feats),
               "cv_top1_mean": folds["top1_accuracy"].mean(),
               "cv_top1_std": folds["top1_accuracy"].std(),
               "cv_macro_f1_mean": folds["macro_f1"].mean(),
               "cv_candidate_logloss_mean": folds["candidate_logloss"].mean()}
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "ablation_features.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
