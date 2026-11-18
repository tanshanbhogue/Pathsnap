"""Step 2: Bayesian hyperparameter optimisation (Optuna TPE) for XGBoost.

Objective: mean observation-level Top-1 accuracy over the 3 trip-grouped CV folds on the
80 training trips. The student-test partition is NOT used here.
Results: artifacts/optuna_study.db (resumable), reports/optuna_trials.csv, artifacts/xgb_best_params.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optuna
import pandas as pd

from pathsnap.config import ARTIFACT_DIR, PROCESSED_DIR, RANDOM_SEED, REPORT_DIR
from pathsnap.cv import cross_validate
from pathsnap.geo import FEATURES
from pathsnap.model import choose_device

N_TRIALS = int(__import__("os").environ.get("PATHSNAP_OPTUNA_TRIALS", 40))
TIMEOUT_S = int(__import__("os").environ.get("PATHSNAP_OPTUNA_TIMEOUT", 3600))


def load_train():
    cand = pd.read_csv(PROCESSED_DIR / "candidates.csv.gz", dtype={"trip_id": str})
    obs = pd.read_csv(PROCESSED_DIR / "observations.csv.gz", dtype={"trip_id": str})
    cand = cand[cand["partition"] == "train"].reset_index(drop=True)
    obs = obs[obs["partition"] == "train"].reset_index(drop=True)
    return cand, obs


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cand, obs = load_train()
    device = choose_device()
    print(f"Device: {device}; train candidates {len(cand)}, observations {len(obs)}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 800, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        )
        spw_mult = trial.suggest_float("spw_multiplier", 0.5, 2.0, log=True)
        t0 = time.time()
        folds, _ = cross_validate(params, FEATURES, cand, obs, device, spw_mult)
        trial.set_user_attr("cv_macro_f1", float(folds["macro_f1"].mean()))
        trial.set_user_attr("cv_candidate_logloss", float(folds["candidate_logloss"].mean()))
        trial.set_user_attr("cv_top1_per_fold", folds["top1_accuracy"].round(5).tolist())
        trial.set_user_attr("trial_seconds", round(time.time() - t0, 1))
        top1 = float(folds["top1_accuracy"].mean())
        print(f"trial {trial.number:03d}: CV top-1 {top1:.4%}  macro-F1 {folds['macro_f1'].mean():.4f}"
              f"  ({time.time() - t0:.0f}s)  {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()})}"
              f" spw_mult={spw_mult:.3f}", flush=True)
        return top1

    storage = f"sqlite:///{(ARTIFACT_DIR / 'optuna_study.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(study_name="xgb_top1", direction="maximize", sampler=sampler,
                                storage=storage, load_if_exists=True)
    done = len(study.trials)
    remaining = max(N_TRIALS - done, 0)
    print(f"Existing trials: {done}; running {remaining} more", flush=True)
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=TIMEOUT_S)

    best = study.best_trial
    best_params = {k: v for k, v in best.params.items() if k != "spw_multiplier"}
    payload = {
        "best_trial_number": best.number,
        "cv_top1_accuracy_mean": best.value,
        "xgb_params": best_params,
        "spw_multiplier": best.params["spw_multiplier"],
        "n_trials_total": len(study.trials),
        "device": device,
        "seed": RANDOM_SEED,
    }
    (ARTIFACT_DIR / "xgb_best_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    trials = study.trials_dataframe()
    trials.to_csv(REPORT_DIR / "optuna_trials.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
