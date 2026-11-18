"""Load saved artifacts once and run the full prediction pipeline for a trip.

The reference route is never read here. It is loaded separately by the caller
for evaluation after predictions are produced.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from pathsnap.geo import FEATURES, add_features, generate_candidates
from pathsnap.hmm import HmmParams, RoadGraph, build_trip_cache, classifier_only, viterbi_decode

log = logging.getLogger("pathsnap.inference")


class PathSnapModel:
    def __init__(self, artifact_dir: Path | str):
        self.dir = Path(artifact_dir)
        cfg_path = self.dir / "model_config.json"
        hmm_path = self.dir / "hmm_params.json"
        tt_path = self.dir / "transition_table.json"
        model_path = self.dir / "xgb_tuned.json"
        for p in (cfg_path, hmm_path, tt_path, model_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing artifact: {p}. Run the training scripts first.")

        self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        if list(self.config["features"]) != FEATURES:
            raise ValueError("Saved feature list does not match pathsnap.geo.FEATURES")
        self.hmm_params = HmmParams.from_dict(json.loads(hmm_path.read_text(encoding="utf-8")))
        self.transition_table = np.asarray(
            json.loads(tt_path.read_text(encoding="utf-8"))["log_probabilities"], dtype=float
        )
        self.classifier = XGBClassifier()
        self.classifier.load_model(str(model_path))
        log.info("Loaded PathSnap artifacts from %s", self.dir)

    def predict(self, trip) -> dict:
        """Run candidate generation, scoring, HMM decoding. Returns arrays and timings."""
        t0 = time.perf_counter()
        cand = add_features(trip, generate_candidates(
            trip, self.config["radius_m"], self.config["max_candidates"]))
        t_cand = time.perf_counter()

        if len(cand):
            cand["prob"] = self.classifier.predict_proba(cand[FEATURES])[:, 1]
        else:
            cand["prob"] = pd.Series(dtype=float)
        t_clf = time.perf_counter()

        graph = RoadGraph(trip.arcs)
        cache = build_trip_cache(trip, cand[["obs_idx", "edge_id", "prob"]], graph,
                                 ref=np.empty(0, np.int64), route=np.empty(0, np.int64))
        clf_edges = classifier_only(cache)
        hmm_edges = viterbi_decode(cache, self.hmm_params, self.transition_table)
        t_end = time.perf_counter()

        return {
            "candidates": cand,
            "classifier_edges": clf_edges,
            "hmm_edges": hmm_edges,
            "graph": graph,
            "timings_ms": {
                "candidate_generation": round((t_cand - t0) * 1000, 1),
                "classifier_inference": round((t_clf - t_cand) * 1000, 1),
                "hmm_viterbi_decoding": round((t_end - t_clf) * 1000, 1),
                "total_model_time": round((t_end - t0) * 1000, 1),
            },
        }
