#!/usr/bin/env python3
"""
Prints (1) a dataset inventory and (2) per-trip summary statistics
for the GPS trajectory + road network dataset.

Usage:
    python gps_summary.py "path/to/Student folder"

Expected layout:
    <root>/<split>/<trip_id>/<trip_id>.track|.route|.nodes|.arcs
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXTS = ["track", "route", "nodes", "arcs"]


def haversine_steps(lon, lat):
    """Distance in metres between consecutive GPS points."""
    R = 6371000.0
    la, lo = np.radians(lat), np.radians(lon)
    a = (np.sin(np.diff(la) / 2) ** 2
         + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def load_trip(folder: Path):
    tid = folder.name
    track = pd.read_csv(folder / f"{tid}.track", sep="\t", header=None,
                        names=["lon", "lat", "t"])
    nodes = pd.read_csv(folder / f"{tid}.nodes", sep="\t", header=None,
                        names=["lon", "lat"])
    arcs = pd.read_csv(folder / f"{tid}.arcs", sep="\t", header=None,
                       names=["u", "v"])
    route = pd.read_csv(folder / f"{tid}.route", header=None, names=["arc"])
    return track, nodes, arcs, route


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")

    rows = []
    # Find trips wherever they are: any folder holding <name>/<name>.track
    track_files = sorted(root.rglob("*.track"))
    if not track_files:
        sys.exit(f"No .track files found under {root}. Check the folder path.")
    for tf in track_files:
        trip = tf.parent
        split_name = trip.parent.name
        if not all((trip / f"{trip.name}.{e}").exists() for e in EXTS):
            print(f"WARNING: incomplete trip skipped: {trip}")
            continue
        track, nodes, arcs, route = load_trip(trip)
        dt = np.diff(track["t"].values)
        span = track["t"].iloc[-1] - track["t"].iloc[0] + 1
        steps = haversine_steps(track["lon"].values, track["lat"].values)
        rows.append(dict(
            split=split_name,
            trip=trip.name,
            files=len(EXTS),
            points=len(track),
            nodes=len(nodes),
            arcs=len(arcs),
            route_arcs=len(route),
            length_km=steps.sum() / 1000,
            median_step_m=float(np.median(steps)),
            median_dt_s=float(np.median(dt)),
            gaps_gt1s=int((dt > 1).sum()),
            gaps_gt60s=int((dt > 60).sum()),
            max_gap_s=float(dt.max()),
            missing_pct=100 * (1 - len(track) / span),
            nan_cells=int(track.isna().sum().sum()),
        ))

    if not rows:
        sys.exit("No trips found. Check the folder path.")
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    # ---------- Table 1: inventory ----------
    inv = (df.groupby("split")
             .agg(trips=("trip", "count"), files=("files", "sum"),
                  gps_points=("points", "sum"), network_nodes=("nodes", "sum"),
                  network_arcs=("arcs", "sum"), route_arcs=("route_arcs", "sum")))
    inv.loc["TOTAL"] = inv.sum()
    inv = inv.astype(int)
    print("\n=== TABLE 1: DATASET INVENTORY ===")
    print(inv.to_string(float_format=lambda x: f"{x:,.0f}",
                        formatters={c: "{:,}".format for c in inv.columns}))

    # ---------- Table 2: per-trip summary statistics ----------
    measures = [
        ("GPS points per trip", "points"),
        ("Trajectory length (km)", "length_km"),
        ("Median step between points (m)", "median_step_m"),
        ("Network nodes", "nodes"),
        ("Network arcs", "arcs"),
        ("Route length (arcs)", "route_arcs"),
        ("Missing time steps (%)", "missing_pct"),
    ]
    stats = pd.DataFrame({
        label: [df[col].mean(), df[col].median(), df[col].min(), df[col].max()]
        for label, col in measures
    }, index=["Mean", "Median", "Min", "Max"]).T
    print("\n=== TABLE 2: SUMMARY STATISTICS (per trip) ===")
    print(stats.round(1).to_string(float_format=lambda x: f"{x:,.1f}"))

    print("\nSampling interval and missing values:")
    print(f"  Median sampling interval (s): min {df.median_dt_s.min():.0f}, "
          f"max {df.median_dt_s.max():.0f}")
    print(f"  Gaps > 1 s : {df.gaps_gt1s.sum():,} in {(df.gaps_gt1s > 0).sum()} trips")
    print(f"  Gaps > 60 s: {df.gaps_gt60s.sum():,} in {(df.gaps_gt60s > 0).sum()} trips")
    print(f"  Longest gap: {df.max_gap_s.max():,.0f} s")
    print(f"  Trips with >10% missing time steps: {(df.missing_pct > 10).sum()}")
    print(f"  NaN / empty cells in track files: {df.nan_cells.sum()}")

    out = Path("trip_summary.csv")
    df.to_csv(out, index=False)
    print(f"\nPer-trip details saved to {out.resolve()}")


if __name__ == "__main__":
    main()
