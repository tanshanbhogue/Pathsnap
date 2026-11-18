"""Central configuration.

Paths can be overridden with environment variables so that no machine-specific
path is hard-coded. Defaults point at ``data/raw`` inside the project, which
should be a copy of (or a junction to) the supplied ``Student folder``.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset root containing ``train/`` and ``student_test/`` (and optionally an evaluator folder).
DATASET_ROOT = Path(os.environ.get("PATHSNAP_DATASET", PROJECT_ROOT / "data" / "raw"))
TRAIN_DIR = DATASET_ROOT / "train"
STUDENT_DIR = DATASET_ROOT / "student_test"

# Evaluator-testing trips are supplied separately at evaluation time. Leave unset until they are provided.
_evaluator = os.environ.get("PATHSNAP_EVALUATOR_DATASET")
_evaluator_default = PROJECT_ROOT / "data" / "evaluator" / "evaluator_test"
EVALUATOR_DIR = Path(_evaluator) if _evaluator else (_evaluator_default if _evaluator_default.exists() else None)

ARTIFACT_DIR = Path(os.environ.get("PATHSNAP_ARTIFACTS", PROJECT_ROOT / "artifacts"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42
RADIUS_M = float(os.environ.get("PATHSNAP_RADIUS_M", 50.0))        # candidate search radius (metres); 50 m as in the Colab notebooks
MAX_CANDIDATES = int(os.environ.get("PATHSNAP_MAX_CANDIDATES", 8))  # nearest candidate edges kept per observation
N_CV_FOLDS = 3           # trip-grouped cross-validation folds on the 80 training trips
