"""Unit tests for the road graph, transition keys and Viterbi decoder (synthetic data, no dataset needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pathsnap.hmm import (N_HOP_BUCKETS, N_KEYS, HmmParams, RoadGraph, TripCache, classifier_only,
                          transition_keys, viterbi_decode)


class _FakeTrip:
    def __init__(self, xy, ts):
        self.xy = np.asarray(xy, float)
        self.timestamps = np.asarray(ts, float)
        self.n_obs = len(xy)


# Arcs (row index = arc id): a straight road 0->1->2->3, a disconnected road 4->5, and a branch 1->6.
ARCS = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [1, 6]])


def _cache(cands, probs, xy=None, ts=None):
    n = len(cands)
    xy = np.column_stack([np.arange(n) * 10.0, np.zeros(n)]) if xy is None else xy
    ts = np.arange(n, dtype=float) if ts is None else ts
    trip = _FakeTrip(xy, ts)
    graph = RoadGraph(ARCS)
    edges = [np.asarray(c, np.int64) for c in cands]
    ps = [np.asarray(p, float) for p in probs]
    steps = np.diff(xy, axis=0)
    keys = transition_keys(np.hypot(steps[:, 0], steps[:, 1]), np.diff(ts)) if n > 1 else np.empty(0, np.int64)
    buckets = []
    for t in range(1, n):
        m = np.full((len(edges[t - 1]) + 1, len(edges[t]) + 1), 5, dtype=np.int64)
        for i, a in enumerate(edges[t - 1]):
            for j, b in enumerate(edges[t]):
                m[i, j] = graph.hop_bucket(int(a), int(b))
        buckets.append(m)
    return TripCache("test", n, edges, ps, keys, buckets, graph,
                     ref=np.full(n, -1), route=np.empty(0, np.int64))


def _uniform_table(value=-1.0):
    return np.full((N_KEYS, N_HOP_BUCKETS), value)


def test_hop_buckets_follow_directed_network():
    g = RoadGraph(ARCS)
    assert g.hop_bucket(0, 0) == 0
    assert g.hop_bucket(0, 1) == 1          # 0 ends where 1 starts
    assert g.hop_bucket(0, 2) == 2          # two arcs
    assert g.hop_bucket(0, 4) == 1          # arc 4 (1->6) also starts at node 1
    assert g.hop_bucket(0, 3) == 4          # arc 3 (4->5) is a disconnected road
    assert g.hop_bucket(1, 0) == 4          # arcs are directed: cannot go backwards


def test_transition_keys_handle_large_gaps_and_nan():
    keys = transition_keys(np.array([0.0, 3.0, 500.0, np.nan]), np.array([1.0, 5.0, 120.0, np.nan]))
    assert keys.min() >= 0 and keys.max() < N_KEYS


def test_viterbi_prefers_connected_path_over_disconnected_high_score():
    # Step 0: arc 0 (weak) or arc 3 (disconnected, slightly stronger). Step 1: arc 1 (connected to 0 only).
    cache = _cache(cands=[[0, 3], [1]], probs=[[0.4, 0.6], [1.0]])
    # Transition table strongly favours "directly connected" (bucket 1) over "far" (bucket 4).
    table = _uniform_table()
    table[:, 1] = np.log(0.9)
    table[:, 4] = np.log(0.01)
    params = HmmParams(emission_scale=1.0, transition_weight=3.0)
    path = viterbi_decode(cache, params, table)
    assert path.tolist() == [0, 1]
    # Classifier-only would have chosen the disconnected arc 4 at step 0.
    assert classifier_only(cache).tolist() == [3, 1]


def test_missing_candidates_do_not_crash_and_return_none():
    cache = _cache(cands=[[0], [], [1]], probs=[[0.9], [], [0.8]])
    path = viterbi_decode(cache, HmmParams(), _uniform_table())
    assert path.shape == (3,)
    assert path[1] == -1
    assert path[0] == 0 and path[2] == 1


def test_all_observations_without_candidates():
    cache = _cache(cands=[[], [], []], probs=[[], [], []])
    assert viterbi_decode(cache, HmmParams(), _uniform_table()).tolist() == [-1, -1, -1]


def test_empty_and_single_observation_trips():
    empty = _cache(cands=[], probs=[])
    assert viterbi_decode(empty, HmmParams(), _uniform_table()).size == 0
    single = _cache(cands=[[2, 3]], probs=[[0.2, 0.8]])
    assert viterbi_decode(single, HmmParams(), _uniform_table()).tolist() == [3]


def test_output_preserves_chronological_length_and_ids():
    rng = np.random.default_rng(0)
    cands = [rng.choice(5, size=3, replace=False).tolist() for _ in range(30)]
    probs = [rng.random(3).tolist() for _ in range(30)]
    cache = _cache(cands=cands, probs=probs)
    path = viterbi_decode(cache, HmmParams(), _uniform_table())
    assert len(path) == 30
    for t, e in enumerate(path):
        assert e == -1 or e in cands[t]
