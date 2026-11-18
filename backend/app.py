"""PathSnap FastAPI backend (runs on Laptop 1).

Run:  uvicorn backend.app:app --host 0.0.0.0 --port 8000

Ground-truth (.route) files are read only AFTER the prediction for a trip has been
produced, and only to compute the evaluation metrics returned to the caller. Reference
arcs and routes are never included in any response.
"""
from __future__ import annotations

import logging
import re
import sys
import time
import traceback
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pathsnap.config import ARTIFACT_DIR, EVALUATOR_DIR, PROJECT_ROOT, STUDENT_DIR
from pathsnap.evaluation import collapse_route, observation_metrics, route_metrics
from pathsnap.geo import TripDataError, generate_candidates, list_trip_dirs, load_trip, reference_edges
from pathsnap.inference import PathSnapModel

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("pathsnap.api")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(_fmt)
_file = RotatingFileHandler(LOG_DIR / "api.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
_file.setFormatter(_fmt)
logger.addHandler(_console)
logger.addHandler(_file)
logger.propagate = False

TRIP_ID_PATTERN = re.compile(r"^[0-9]{8}$")

app = FastAPI(title="PathSnap API", version="1.0.0")


# ----------------------------------------------------------------- state
class State:
    model: PathSnapModel | None = None
    model_error: str | None = None
    registry: dict[str, tuple[str, Path]] = {}


def discover_trips() -> dict[str, tuple[str, Path]]:
    """Map trip_id -> (partition, folder) for every evaluation trip that is configured."""
    partitions = [("student_test", STUDENT_DIR)]
    if EVALUATOR_DIR is not None and EVALUATOR_DIR.exists():
        partitions.append(("evaluator", EVALUATOR_DIR))
    registry: dict[str, tuple[str, Path]] = {}
    for name, folder in partitions:
        for d in list_trip_dirs(folder):
            if not TRIP_ID_PATTERN.match(d.name):
                continue
            if d.name in registry:
                raise RuntimeError(f"Duplicate trip id {d.name} across partitions")
            registry[d.name] = (name, d)
    return registry


@app.on_event("startup")
def startup() -> None:
    logger.info("Starting PathSnap API; artifacts=%s", ARTIFACT_DIR)
    try:
        State.model = PathSnapModel(ARTIFACT_DIR)
        logger.info("Model artifacts loaded (features=%d)", len(State.model.config["features"]))
    except Exception as exc:  # keep the server up so /api/health can report the problem
        State.model_error = str(exc)
        logger.error("Model failed to load: %s", exc)
    try:
        State.registry = discover_trips()
        logger.info("Registered %d evaluation trips: %s", len(State.registry),
                    {p: sum(1 for v in State.registry.values() if v[0] == p)
                     for p in {v[0] for v in State.registry.values()}})
    except Exception as exc:
        logger.error("Trip discovery failed: %s", exc)
        State.registry = {}


# ----------------------------------------------------------------- middleware / errors
@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    logger.info("[%s] REQUEST %s %s from %s", request_id, request.method, request.url.path,
                request.client.host if request.client else "?")
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("[%s] RESPONSE %s %s -> %d in %.0f ms", request_id, request.method,
                request.url.path, response.status_code, ms)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "?")
    logger.error("[%s] UNHANDLED ERROR: %s\n%s", rid, exc, traceback.format_exc())
    return JSONResponse(status_code=500, content={
        "error": "internal_error", "detail": "The server could not complete the request.",
        "request_id": rid})


# ----------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    trip_id: str = Field(..., pattern=r"^[0-9]{8}$", description="Eight-digit evaluation trip id")


# ----------------------------------------------------------------- helpers
def _require_model() -> PathSnapModel:
    if State.model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {State.model_error}")
    return State.model


def _lookup(trip_id: str) -> tuple[str, Path]:
    if not TRIP_ID_PATTERN.match(trip_id) or trip_id not in State.registry:
        raise HTTPException(status_code=404, detail=f"Unknown trip id '{trip_id}'")
    return State.registry[trip_id]


def _segments(trip, edge_ids) -> list[list[list[float]]]:
    """[[lat, lon], [lat, lon]] per arc, for map drawing."""
    out = []
    for e in edge_ids:
        u, v = trip.arcs[int(e)]
        out.append([[float(trip.node_lonlat[u, 1]), float(trip.node_lonlat[u, 0])],
                    [float(trip.node_lonlat[v, 1]), float(trip.node_lonlat[v, 0])]])
    return out


def _json_number(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _clean(d: dict) -> dict:
    return {k: (_json_number(v) if isinstance(v, (float, np.floating)) else
                int(v) if isinstance(v, (np.integer,)) else v) for k, v in d.items()}


# ----------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {
        "status": "ok" if State.model is not None else "degraded",
        "model_loaded": State.model is not None,
        "model_error": State.model_error,
        "trips_registered": len(State.registry),
        "partitions": sorted({v[0] for v in State.registry.values()}),
        "evaluator_partition_configured": EVALUATOR_DIR is not None,
    }


@app.get("/api/trips")
def list_trips():
    trips = [{"trip_id": tid, "partition": part} for tid, (part, _) in sorted(State.registry.items())]
    logger.info("Trip list requested: %d trips", len(trips))
    return {"count": len(trips), "trips": trips}


@app.get("/api/trips/{trip_id}/network")
def trip_network(trip_id: str):
    """Road arcs that are candidates for this trip's GPS points (geometry only, no labels)."""
    partition, folder = _lookup(trip_id)
    trip = load_trip(folder, partition, load_route=False)
    cand = generate_candidates(trip)
    edge_ids = sorted(cand["edge_id"].unique().tolist()) if len(cand) else []
    logger.info("Network requested for trip %s: %d candidate arcs", trip_id, len(edge_ids))
    return {
        "trip_id": trip_id,
        "candidate_arcs": [{"edge_id": int(e), "segment": seg}
                           for e, seg in zip(edge_ids, _segments(trip, edge_ids))],
    }


@app.post("/api/predict")
def predict(body: PredictRequest, request: Request):
    rid = getattr(request.state, "request_id", "?")
    partition, folder = _lookup(body.trip_id)
    model = _require_model()
    logger.info("[%s] PREDICT trip=%s partition=%s: loading trip", rid, body.trip_id, partition)

    try:
        # 1) Inference uses GPS observations only (load_route=False guarantees no reference access).
        trip = load_trip(folder, partition, load_route=False)
        logger.info("[%s] Loaded %d GPS observations", rid, trip.n_obs)
        result = model.predict(trip)
        t = result["timings_ms"]
        logger.info("[%s] Candidates generated (%.0f ms); classifier scoring (%.0f ms); "
                    "HMM/Viterbi decoding (%.0f ms)", rid, t["candidate_generation"],
                    t["classifier_inference"], t["hmm_viterbi_decoding"])

        # 2) Evaluation uses the internal reference route AFTER predictions exist.
        logger.info("[%s] Computing evaluation metrics from internal reference data", rid)
        ref_trip = load_trip(folder, partition, load_route=True)
        ref = reference_edges(ref_trip)
        graph = result["graph"]
        clf_edges = result["classifier_edges"]
        hmm_edges = result["hmm_edges"]
        cand = result["candidates"]
        covered_mask = np.zeros(trip.n_obs, bool)
        if len(cand):
            hit = cand["edge_id"].to_numpy() == ref[cand["obs_idx"].to_numpy()]
            covered_mask[cand.loc[hit, "obs_idx"].to_numpy()] = True
        metrics = {
            "classifier_only": _clean({**observation_metrics(ref, clf_edges),
                                       **route_metrics(clf_edges, ref_trip.route, graph)}),
            "hmm_viterbi": _clean({**observation_metrics(ref, hmm_edges),
                                   **route_metrics(hmm_edges, ref_trip.route, graph)}),
            "candidate_coverage": float(covered_mask.mean()) if trip.n_obs else None,
            "observations": int(trip.n_obs),
        }
        logger.info("[%s] Metrics: classifier top1=%.4f, HMM top1=%.4f", rid,
                    metrics["classifier_only"]["top1_accuracy"], metrics["hmm_viterbi"]["top1_accuracy"])
    except TripDataError as exc:
        logger.error("[%s] Trip data error for %s: %s", rid, body.trip_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))

    def route_payload(edges):
        route = collapse_route(edges)
        return {
            "per_observation_arc_id": [int(x) for x in edges],
            "route_arc_ids": route,
            "route_segments": _segments(trip, route),
        }

    logger.info("[%s] PREDICT complete for trip %s", rid, body.trip_id)
    return {
        "request_id": rid,
        "trip": {"trip_id": body.trip_id, "partition": partition, "observations": int(trip.n_obs),
                 "projection_epsg": trip.epsg},
        "observations": [{"obs_idx": i, "lat": float(lat), "lon": float(lon), "timestamp": float(ts)}
                         for i, ((lon, lat), ts) in enumerate(zip(trip.lonlat, trip.timestamps))],
        "classifier_only": route_payload(clf_edges),
        "hmm_viterbi": route_payload(hmm_edges),
        "metrics": metrics,
        "timings_ms": t,
        "model": {"model_file": "xgb_tuned.json", "features": model.config["features"],
                  "xgboost_version": model.config.get("xgboost_version")},
    }
