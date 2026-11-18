"""Geometry, feature and end-to-end API tests. Dataset-dependent tests are skipped when data is absent."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from pathsnap.config import ARTIFACT_DIR, PROCESSED_DIR, STUDENT_DIR
from pathsnap.geo import FEATURES, Trip, add_features, generate_candidates, load_trip, reference_edges

HAS_DATA = STUDENT_DIR.exists() and any(STUDENT_DIR.iterdir())
HAS_ARTIFACTS = (ARTIFACT_DIR / "xgb_tuned.json").exists()


def _synthetic_trip():
    # One straight northbound road of nodes 0->1->2 (arcs 0 and 1) plus its reverse (arcs 2, 3).
    nodes_xy = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    arcs = np.array([[0, 1], [1, 2], [2, 1], [1, 0]])
    xy = np.array([[0.5, 10.0], [0.5, 60.0], [0.5, 110.0]])  # moving north
    return Trip(trip_id="syn", partition="test",
                lonlat=np.zeros((3, 2)), timestamps=np.array([1.0, 2.0, 3.0]),
                node_lonlat=np.zeros((3, 2)), arcs=arcs, epsg=0,
                xy=xy, node_xy=nodes_xy, route=None)


def test_heading_separates_opposite_directions_on_same_road():
    trip = _synthetic_trip()
    cand = add_features(trip, generate_candidates(trip, radius_m=5.0, max_candidates=8))
    # Both directions of the road are candidates at the same distance; heading must separate them.
    first = cand[cand["obs_idx"] == 0].set_index("edge_id")
    assert {0, 3} <= set(first.index)
    assert first.loc[0, "heading_diff_deg"] < 10          # north-bound arc agrees with motion
    assert first.loc[3, "heading_diff_deg"] > 170         # south-bound arc is opposite


def test_candidate_features_have_expected_columns():
    trip = _synthetic_trip()
    cand = add_features(trip, generate_candidates(trip, radius_m=50.0))
    assert set(FEATURES) <= set(cand.columns)
    assert cand["distance_rank"].min() == 1


def test_empty_candidate_set_is_handled():
    trip = _synthetic_trip()
    trip.xy = trip.xy + 10_000.0  # far away from every arc
    cand = add_features(trip, generate_candidates(trip))
    assert cand.empty


@pytest.mark.skipif(not HAS_DATA, reason="dataset not available")
def test_inference_features_match_training_table():
    """Training and inference must produce identical candidate features for the same trip."""
    trip = load_trip(STUDENT_DIR / "00000000", "student", load_route=False)
    fresh = add_features(trip, generate_candidates(trip)).sort_values(["obs_idx", "edge_id"]).reset_index(drop=True)
    stored = pd.read_csv(PROCESSED_DIR / "candidates.csv.gz", dtype={"trip_id": str})
    stored = stored[stored["trip_id"] == "00000000"].sort_values(["obs_idx", "edge_id"]).reset_index(drop=True)
    assert len(fresh) == len(stored)
    np.testing.assert_allclose(fresh[FEATURES].to_numpy(float), stored[FEATURES].to_numpy(float), equal_nan=True, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not (HAS_DATA and HAS_ARTIFACTS), reason="dataset or model artifacts missing")
def test_api_end_to_end_and_no_label_leak():
    from fastapi.testclient import TestClient
    from backend.app import app

    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["model_loaded"] is True
        trips = client.get("/api/trips").json()["trips"]
        assert len(trips) >= 10
        resp = client.post("/api/predict", json={"trip_id": "00000000"})
        assert resp.status_code == 200, resp.text
        body = resp.text
        # Reference labels and the reference route must never appear in a response.
        assert "ref_edge" not in body and "reference" not in body.lower()
        data = resp.json()
        assert data["metrics"]["hmm_viterbi"]["observations"] == len(data["observations"])
        assert len(data["hmm_viterbi"]["per_observation_arc_id"]) == len(data["observations"])
        bad = client.post("/api/predict", json={"trip_id": "../etc"})
        assert bad.status_code == 422
        missing = client.post("/api/predict", json={"trip_id": "99999999"})
        assert missing.status_code == 404
