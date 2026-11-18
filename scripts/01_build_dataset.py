"""Step 1: build observation and candidate tables for the train (80) and student-test (10) partitions.

Outputs (data/processed/):
  observations.csv.gz  one row per GPS point, with reference arc (evaluation/training label only)
  candidates.csv.gz    one row per (observation, candidate arc), with features and label
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from pathsnap.config import MAX_CANDIDATES, PROCESSED_DIR, RADIUS_M, REPORT_DIR, STUDENT_DIR, TRAIN_DIR
from pathsnap.geo import add_features, generate_candidates, list_trip_dirs, load_trip, reference_edges


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    obs_frames, cand_frames = [], []
    t_start = time.time()
    for partition, folder in [("train", TRAIN_DIR), ("student", STUDENT_DIR)]:
        dirs = list_trip_dirs(folder)
        print(f"{partition}: {len(dirs)} trips", flush=True)
        for i, trip_dir in enumerate(dirs, 1):
            trip = load_trip(trip_dir, partition, load_route=True)
            ref = reference_edges(trip)
            cand = add_features(trip, generate_candidates(trip, RADIUS_M, MAX_CANDIDATES))

            obs_frames.append(pd.DataFrame({
                "trip_id": trip.trip_id,
                "partition": partition,
                "obs_idx": np.arange(trip.n_obs),
                "timestamp": trip.timestamps,
                "lon": trip.lonlat[:, 0],
                "lat": trip.lonlat[:, 1],
                "x": trip.xy[:, 0],
                "y": trip.xy[:, 1],
                "ref_edge_id": ref,
            }))
            cand.insert(0, "trip_id", trip.trip_id)
            cand.insert(1, "partition", partition)
            cand["label"] = (cand["edge_id"].to_numpy() == ref[cand["obs_idx"].to_numpy()]).astype(int)
            cand_frames.append(cand)
            if i % 10 == 0 or i == len(dirs):
                print(f"  {partition}: {i}/{len(dirs)}  ({time.time() - t_start:.0f}s)", flush=True)

    obs = pd.concat(obs_frames, ignore_index=True)
    cand = pd.concat(cand_frames, ignore_index=True)

    if obs.duplicated(["trip_id", "obs_idx"]).any():
        raise RuntimeError("Duplicate observation keys generated")
    if cand.duplicated(["trip_id", "obs_idx", "edge_id"]).any():
        raise RuntimeError("Duplicate candidate rows generated")

    covered = cand.groupby(["trip_id", "obs_idx"])["label"].max().rename("covered").reset_index()
    obs = obs.merge(covered, on=["trip_id", "obs_idx"], how="left")
    obs["covered"] = obs["covered"].fillna(0).astype(int)

    summary = {}
    for part, g in obs.groupby("partition"):
        summary[part] = {
            "trips": int(g["trip_id"].nunique()),
            "observations": int(len(g)),
            "candidate_rows": int((cand["partition"] == part).sum()),
            "candidate_coverage": float(g["covered"].mean()),
            "candidate_miss_rate": float(1 - g["covered"].mean()),
        }
    summary["radius_m"] = RADIUS_M
    summary["max_candidates"] = MAX_CANDIDATES
    (REPORT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    obs.to_csv(PROCESSED_DIR / "observations.csv.gz", index=False, compression="gzip")
    cand.to_csv(PROCESSED_DIR / "candidates.csv.gz", index=False, compression="gzip")
    print(f"Wrote {len(obs)} observations and {len(cand)} candidate rows in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
