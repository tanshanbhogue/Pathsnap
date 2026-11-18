"""Hidden Markov Model with Viterbi decoding over candidate road segments.

Hidden states at each GPS observation t are the candidate arcs for that
observation, plus one extra ``NONE`` state meaning "no road matched". The NONE
state keeps the chain alive when an observation has no candidates or when the
true road is missing from the candidate list, so decoding never fails.

Emission  : log-probability from the XGBoost candidate scores, normalised within
            the observation's candidate set and scaled by ``emission_scale``.
Transition: a log-probability table P(hop-bucket | displacement bin, time-gap bin)
            learned from the TRAINING reference routes only. The hop bucket is the
            number of road arcs needed to get from candidate a to candidate b on the
            directed road network (0 = same arc, 1 = directly connected, ...,
            4 = more than three arcs away or unreachable). Transitions that contradict
            the road network are therefore penalised, not invented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Binning of GPS displacement (metres) and time gap (seconds) for transitions.
DISTANCE_EDGES_M = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
GAP_EDGES_S = np.array([2.0, 10.0, 30.0])
N_GAP_BINS = len(GAP_EDGES_S) + 1
N_KEYS = (len(DISTANCE_EDGES_M) + 1) * N_GAP_BINS
N_HOP_BUCKETS = 5           # buckets 0..4
NONE_BUCKET = 5             # used internally when either state is NONE


def transition_keys(displacement_m: np.ndarray, gap_s: np.ndarray) -> np.ndarray:
    disp = np.nan_to_num(np.asarray(displacement_m, float), nan=1e9)
    gap = np.nan_to_num(np.asarray(gap_s, float), nan=1e9)
    d = np.searchsorted(DISTANCE_EDGES_M, disp, side="right")
    g = np.searchsorted(GAP_EDGES_S, gap, side="right")
    return (d * N_GAP_BINS + g).astype(np.int64)


class RoadGraph:
    """Directed road network built from the arcs file. Caches short-range reachability."""

    def __init__(self, arcs: np.ndarray):
        self.tail = arcs[:, 0].tolist()
        self.head = arcs[:, 1].tolist()
        self._out: dict[int, list[int]] = {}
        for u, v in zip(self.tail, self.head):
            self._out.setdefault(u, []).append(v)
        self._reach: dict[int, dict[int, int]] = {}

    def _reach_from(self, node: int) -> dict[int, int]:
        """Nodes reachable within two arcs of ``node``, with the arc count."""
        cached = self._reach.get(node)
        if cached is None:
            cached = {node: 0}
            frontier = [node]
            for depth in (1, 2):
                nxt = []
                for u in frontier:
                    for v in self._out.get(u, ()):
                        if v not in cached:
                            cached[v] = depth
                            nxt.append(v)
                frontier = nxt
            self._reach[node] = cached
        return cached

    def hop_bucket(self, a: int, b: int) -> int:
        """0 same arc; 1..3 = arcs traversed from a to b inclusive; 4 = farther or unreachable."""
        if a == b:
            return 0
        d = self._reach_from(self.head[a]).get(self.tail[b])
        if d is None:
            return 4
        return d + 1


@dataclass
class HmmParams:
    emission_scale: float = 1.0
    transition_weight: float = 1.0
    none_log_emission: float = float(np.log(0.01))
    none_log_transition: float = float(np.log(0.1))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HmmParams":
        return cls(**{k: float(v) for k, v in d.items()})


@dataclass
class TripCache:
    """Everything the decoder needs for one trip; independent of the decoder parameters."""
    trip_id: str
    n_obs: int
    edges: list          # per observation: candidate arc ids (int64 array)
    probs: list          # per observation: classifier probabilities for those candidates
    keys: np.ndarray     # transition key for step t (length n_obs-1); step t uses keys[t-1]
    buckets: list        # hop-bucket matrices, index t-1 for step t; shape (k_prev+1, k_cur+1)
    graph: RoadGraph
    ref: np.ndarray      # reference arc per observation (evaluation only)
    route: np.ndarray    # reference route (evaluation only)


def build_trip_cache(trip, cand_scored, graph: RoadGraph, ref: np.ndarray,
                     route: np.ndarray) -> TripCache:
    """``cand_scored`` has columns obs_idx, edge_id, prob for one trip."""
    n = trip.n_obs
    edges = [np.empty(0, np.int64) for _ in range(n)]
    probs = [np.empty(0, float) for _ in range(n)]
    for obs_i, grp in cand_scored.groupby("obs_idx", sort=False):
        edges[int(obs_i)] = grp["edge_id"].to_numpy(np.int64)
        probs[int(obs_i)] = grp["prob"].to_numpy(float)

    if n >= 2:
        step = np.diff(trip.xy, axis=0)
        displacement = np.hypot(step[:, 0], step[:, 1])
        keys = transition_keys(displacement, np.diff(trip.timestamps))
    else:
        keys = np.empty(0, np.int64)

    buckets = []
    for t in range(1, n):
        prev, cur = edges[t - 1], edges[t]
        mat = np.full((len(prev) + 1, len(cur) + 1), NONE_BUCKET, dtype=np.int64)
        for i, a in enumerate(prev):
            for j, b in enumerate(cur):
                mat[i, j] = graph.hop_bucket(int(a), int(b))
        buckets.append(mat)

    return TripCache(trip.trip_id, n, edges, probs, keys, buckets, graph, ref, route)


def learn_transition_counts(ref: np.ndarray, trip, graph: RoadGraph) -> np.ndarray:
    """Count (key, hop-bucket) pairs from consecutive reference arcs. Training trips only."""
    counts = np.zeros((N_KEYS, N_HOP_BUCKETS), dtype=float)
    n = len(ref)
    if n < 2:
        return counts
    step = np.diff(trip.xy, axis=0)
    keys = transition_keys(np.hypot(step[:, 0], step[:, 1]), np.diff(trip.timestamps))
    for t in range(1, n):
        bucket = graph.hop_bucket(int(ref[t - 1]), int(ref[t]))
        counts[keys[t - 1], bucket] += 1
    return counts


def counts_to_log_table(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    smoothed = counts + alpha
    return np.log(smoothed / smoothed.sum(axis=1, keepdims=True))


def _emission(probs: np.ndarray, params: HmmParams) -> np.ndarray:
    if len(probs) == 0:
        return np.array([params.none_log_emission])
    p = np.clip(probs, 1e-9, None)
    q = p / p.sum()
    return np.append(params.emission_scale * np.log(q), params.none_log_emission)


def viterbi_decode(cache: TripCache, params: HmmParams, transition_table: np.ndarray) -> np.ndarray:
    """Most likely hidden-state path. Returns an arc id per observation, or -1 for NONE."""
    n = cache.n_obs
    if n == 0:
        return np.empty(0, np.int64)

    # Table with an extra column for transitions that involve the NONE state.
    tt = np.empty((N_KEYS, N_HOP_BUCKETS + 1))
    tt[:, :N_HOP_BUCKETS] = transition_table
    tt[:, NONE_BUCKET] = params.none_log_transition

    V = _emission(cache.probs[0], params)
    back = []
    for t in range(1, n):
        E = _emission(cache.probs[t], params)
        log_trans = tt[cache.keys[t - 1], cache.buckets[t - 1]]
        M = V[:, None] + params.transition_weight * log_trans   # (k_prev+1, k_cur+1)
        arg = M.argmax(axis=0)
        V = M[arg, np.arange(M.shape[1])] + E
        back.append(arg)

    path = np.empty(n, np.int64)
    path[n - 1] = int(V.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t - 1][path[t]]

    edges = np.full(n, -1, np.int64)
    for t in range(n):
        if path[t] < len(cache.edges[t]):
            edges[t] = cache.edges[t][path[t]]
    return edges


def classifier_only(cache: TripCache) -> np.ndarray:
    """Baseline: per observation, the highest-probability candidate (or -1 if none)."""
    out = np.full(cache.n_obs, -1, np.int64)
    for t in range(cache.n_obs):
        if len(cache.probs[t]):
            out[t] = cache.edges[t][int(np.argmax(cache.probs[t]))]
    return out
