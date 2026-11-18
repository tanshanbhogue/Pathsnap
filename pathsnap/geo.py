"""Trip loading, road-network geometry, candidate generation and features.

Everything here works from the raw ``.track``, ``.nodes`` and ``.arcs`` files.
The reference route (``.route``) is loaded only when ``load_route=True`` and is
used for training labels and post-inference evaluation. Prediction never uses it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import STRtree

from pathsnap.config import MAX_CANDIDATES, RADIUS_M

# Nine candidate features carried over from the Colab notebooks.
BASE_FEATURES = [
    "distance_m", "distance_rank", "candidate_count", "min_distance", "max_distance",
    "distance_from_best", "distance_range", "distance_ratio_to_best", "rank_fraction",
]
# Added in this project: directed angle between the candidate arc and the vehicle's
# movement direction (from neighbouring GPS points). Helps separate parallel roads.
FEATURES_V1 = BASE_FEATURES + ["heading_diff_deg"]
# Added in the improvement pass: arc geometry. arc_length_m separates short junction arcs from
# long road arcs; along_track_fraction (0 = arc start, 1 = arc end) shows whether the GPS point
# lies on the arc or beyond its end, which is typical at junctions.
FEATURES = FEATURES_V1
CANDIDATE_COLUMNS = ["obs_idx", "edge_id", "distance_m", "distance_rank"]


class TripDataError(ValueError):
    """Raised when a trip's raw files are missing or internally inconsistent."""


@dataclass
class Trip:
    trip_id: str
    partition: str
    lonlat: np.ndarray        # (n, 2) GPS longitude/latitude
    timestamps: np.ndarray    # (n,) seconds
    node_lonlat: np.ndarray   # (N, 2) road-node longitude/latitude
    arcs: np.ndarray          # (m, 2) from_node, to_node; row index is arc_id
    epsg: int                 # local UTM projection used for metric geometry
    xy: np.ndarray            # (n, 2) GPS in metres
    node_xy: np.ndarray       # (N, 2) road nodes in metres
    route: np.ndarray | None = None   # reference arc ids (evaluation/training only)
    _edge_geoms: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_obs(self) -> int:
        return len(self.lonlat)


def _read_table(path: Path, dtype=float, columns: int | None = None) -> np.ndarray:
    if not path.exists():
        raise TripDataError(f"Missing file: {path}")
    try:
        arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    except ValueError as exc:
        raise TripDataError(f"Could not parse {path}: {exc}") from exc
    if arr.size == 0:
        raise TripDataError(f"Empty file: {path}")
    if columns is not None and arr.shape[1] != columns:
        raise TripDataError(f"{path} must have {columns} columns, found {arr.shape[1]}")
    return arr


def utm_epsg(lon0: float, lat0: float) -> int:
    zone = int((lon0 + 180) // 6) + 1
    return (32600 if lat0 >= 0 else 32700) + zone


def load_trip(trip_dir: Path | str, partition: str, load_route: bool = False) -> Trip:
    """Load one trip folder. ``load_route`` must only be True for labels or evaluation."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    base = trip_dir / tid

    track = _read_table(Path(f"{base}.track"), columns=3)
    nodes = _read_table(Path(f"{base}.nodes"), columns=2)
    arcs = _read_table(Path(f"{base}.arcs"), dtype=np.int64, columns=2)

    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise TripDataError(f"Arc node index out of range in trip {tid}")

    lon0, lat0 = np.median(track[:, :2], axis=0)
    epsg = utm_epsg(lon0, lat0)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    node_x, node_y = transformer.transform(nodes[:, 0], nodes[:, 1])
    gps_x, gps_y = transformer.transform(track[:, 0], track[:, 1])

    route = None
    if load_route:
        route = _read_table(Path(f"{base}.route"), dtype=np.int64).reshape(-1)
        if route.min() < 0 or route.max() >= len(arcs):
            raise TripDataError(f"Route arc index out of range in trip {tid}")

    return Trip(
        trip_id=tid,
        partition=partition,
        lonlat=track[:, :2].copy(),
        timestamps=track[:, 2].copy(),
        node_lonlat=nodes.copy(),
        arcs=arcs,
        epsg=epsg,
        xy=np.column_stack([gps_x, gps_y]),
        node_xy=np.column_stack([node_x, node_y]),
        route=route,
    )


def edge_geometries(trip: Trip) -> np.ndarray:
    """Directed LineString per arc (row index = arc_id), built in the local metric projection."""
    if trip._edge_geoms is None:
        a = trip.arcs
        coords = np.stack([trip.node_xy[a[:, 0]], trip.node_xy[a[:, 1]]], axis=1)  # (m, 2, 2)
        trip._edge_geoms = shapely.linestrings(coords)
    return trip._edge_geoms


def generate_candidates(trip: Trip, radius_m: float = RADIUS_M,
                        max_candidates: int = MAX_CANDIDATES) -> pd.DataFrame:
    """Nearest road arcs within ``radius_m`` of each GPS point (no ground truth used)."""
    edges = edge_geometries(trip)
    points = shapely.points(trip.xy[:, 0], trip.xy[:, 1])
    tree = STRtree(edges)
    obs_idx, edge_idx = tree.query(points, predicate="dwithin", distance=radius_m)
    if len(obs_idx) == 0:
        return pd.DataFrame({c: pd.Series(dtype=float if c == "distance_m" else np.int64)
                             for c in CANDIDATE_COLUMNS})

    dist = shapely.distance(points[obs_idx], edges[edge_idx])
    cand = pd.DataFrame({
        "obs_idx": obs_idx.astype(np.int64),
        "edge_id": edge_idx.astype(np.int64),
        "distance_m": dist,
    })
    cand = cand.sort_values(["obs_idx", "distance_m", "edge_id"], kind="stable").reset_index(drop=True)
    cand["distance_rank"] = cand.groupby("obs_idx").cumcount() + 1
    cand = cand[cand["distance_rank"] <= max_candidates].reset_index(drop=True)
    return cand[CANDIDATE_COLUMNS]


def _movement_bearing(xy: np.ndarray) -> np.ndarray:
    """Compass bearing (degrees) of the vehicle at each GPS point, from its neighbours."""
    n = len(xy)
    if n < 2:
        return np.full(n, np.nan)
    prev = np.clip(np.arange(n) - 1, 0, n - 1)
    nxt = np.clip(np.arange(n) + 1, 0, n - 1)
    dx = xy[nxt, 0] - xy[prev, 0]
    dy = xy[nxt, 1] - xy[prev, 1]
    moved = np.hypot(dx, dy) >= 1.0          # ignore stationary/jitter points
    bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
    return np.where(moved, bearing, np.nan)


def _heading_difference(trip: Trip, cand: pd.DataFrame) -> np.ndarray:
    """Absolute angle (0-180 deg) between the directed arc and the movement direction."""
    move = _movement_bearing(trip.xy)[cand["obs_idx"].to_numpy()]
    a = trip.arcs[cand["edge_id"].to_numpy()]
    dxe = trip.node_xy[a[:, 1], 0] - trip.node_xy[a[:, 0], 0]
    dye = trip.node_xy[a[:, 1], 1] - trip.node_xy[a[:, 0], 1]
    arc_bearing = np.degrees(np.arctan2(dxe, dye)) % 360.0
    arc_bearing = np.where(np.hypot(dxe, dye) < 1e-6, np.nan, arc_bearing)
    return np.abs((arc_bearing - move + 180.0) % 360.0 - 180.0)


def _arc_geometry(trip: Trip, cand: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Arc length, and the GPS point's projected position along the arc (clipped to 0..1)."""
    a = trip.arcs[cand["edge_id"].to_numpy()]
    start = trip.node_xy[a[:, 0]]
    end = trip.node_xy[a[:, 1]]
    d = end - start
    length = np.hypot(d[:, 0], d[:, 1])
    p = trip.xy[cand["obs_idx"].to_numpy()]
    denom = np.where(length ** 2 < 1e-9, np.nan, length ** 2)
    t = np.einsum("ij,ij->i", p - start, d) / denom
    return length, np.clip(t, 0.0, 1.0)


def add_features(trip: Trip, cand: pd.DataFrame) -> pd.DataFrame:
    """Compute the feature matrix. Used identically for training and inference."""
    out = cand.copy()
    if out.empty:
        for col in FEATURES:
            out[col] = pd.Series(dtype=float)
        return out

    g = out.groupby("obs_idx")["distance_m"]
    out["candidate_count"] = g.transform("size").astype(float)
    out["min_distance"] = g.transform("min")
    out["max_distance"] = g.transform("max")
    out["distance_from_best"] = out["distance_m"] - out["min_distance"]
    out["distance_range"] = out["max_distance"] - out["min_distance"]
    out["distance_ratio_to_best"] = out["distance_m"] / out["min_distance"].replace(0, np.nan)
    out["rank_fraction"] = (out["distance_rank"] - 1) / (out["candidate_count"] - 1).replace(0, np.nan)
    out["heading_diff_deg"] = _heading_difference(trip, out)
    out["arc_length_m"], out["along_track_fraction"] = _arc_geometry(trip, out)
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)
    return out


def reference_edges(trip: Trip) -> np.ndarray:
    """Training/evaluation label only: nearest arc of the reference route for each GPS point."""
    if trip.route is None:
        raise TripDataError("reference_edges requires a trip loaded with load_route=True")
    points = shapely.points(trip.xy[:, 0], trip.xy[:, 1])
    route_geoms = edge_geometries(trip)[trip.route]
    pos = STRtree(route_geoms).nearest(points)
    return trip.route[pos]


def list_trip_dirs(folder: Path) -> list[Path]:
    if not folder.exists():
        raise TripDataError(f"Dataset folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_dir())
