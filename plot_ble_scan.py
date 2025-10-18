#!/usr/bin/env python3
"""Quick matplotlib visualizer for LASER lines stored in ble_parsed.log."""

from __future__ import annotations

import argparse
import datetime
import math
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import shutil

LASER_RE = re.compile(r"^LASER ")
KEY_RE = re.compile(r"([a-zA-Z_]+)=([+-]?[0-9]*\.?[0-9]+)")


def parse_laser_line(line: str) -> Tuple[float, float] | None:
    if not LASER_RE.match(line):
        return None
    values = {match.group(1): float(match.group(2)) for match in KEY_RE.finditer(line)}
    if "angle" not in values:
        return None
    distance = values.get("distance_mm") or values.get("distance")
    if distance is None:
        return None
    angle_deg = values["angle"] % 360.0
    distance_m = max(0.0, distance) / 1000.0
    return angle_deg, distance_m


def load_all_scans(
    path: Path, samples_per_scan: int
) -> Tuple[List[float], List[float], List[int]]:
    all_angles: List[float] = []
    all_ranges: List[float] = []
    scan_ids: List[int] = []
    current_angles: List[float] = []
    current_ranges: List[float] = []
    scan_idx = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_laser_line(line)
            if parsed is None:
                continue
            angle_deg, distance_m = parsed
            current_angles.append(angle_deg)
            current_ranges.append(distance_m)

            if len(current_angles) == samples_per_scan:
                all_angles.extend(current_angles)
                all_ranges.extend(current_ranges)
                scan_ids.extend([scan_idx] * samples_per_scan)
                current_angles = []
                current_ranges = []
                scan_idx += 1

    return all_angles, all_ranges, scan_ids



def build_cartesian(angles_deg: List[float], ranges_m: List[float]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for angle_deg, distance_m in zip(angles_deg, ranges_m):
        theta = math.radians(angle_deg)
        xs.append(distance_m * math.cos(theta))
        ys.append(distance_m * math.sin(theta))
    return xs, ys


def snapshot_source(source: Path, enabled: bool) -> Path:
    if not enabled or not source.exists():
        return source
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = source.with_name(f"{source.stem}_{timestamp}{source.suffix}.bak")
    shutil.copy2(source, snapshot)
    print(f"[snapshot] Copied {source} -> {snapshot}")
    return snapshot


def visualize(
    path: Path,
    samples_per_scan: int,
    min_distance: float,
    max_distance: float | None,
    create_snapshot: bool,
) -> None:
    source = snapshot_source(path, create_snapshot)
    angles_deg, ranges_m, scan_ids = load_all_scans(source, samples_per_scan)
    if not angles_deg:
        raise SystemExit("No complete laser scans found in the source file.")

    filtered_angles: List[float] = []
    filtered_ranges: List[float] = []
    filtered_ids: List[int] = []
    for angle_deg, range_m, scan_id in zip(angles_deg, ranges_m, scan_ids):
        if range_m < min_distance:
            continue
        if max_distance is not None and range_m > max_distance:
            continue
        filtered_angles.append(angle_deg)
        filtered_ranges.append(range_m)
        filtered_ids.append(scan_id)

    if not filtered_angles:
        raise SystemExit("No laser samples found that satisfy the filtering criteria.")

    xs, ys = build_cartesian(filtered_angles, filtered_ranges)
    max_radius = max(filtered_ranges) if max_distance is None else max_distance

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        xs,
        ys,
        s=8,
        c=filtered_ids,
        cmap="turbo",
        alpha=0.6,
        edgecolors="none",
    )
    ax.scatter([0.0], [0.0], c="red", s=40, label="Robot")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Overlayed Laser Scans from {source.name}")
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Scan index")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize BLE laser data in matplotlib.")
    parser.add_argument("source", nargs="?", default="ble_parsed.log", help="Log file containing LASER lines.")
    parser.add_argument(
        "--samples-per-scan",
        type=int,
        default=250,
        help="Number of LASER entries expected per full scan.",
    )
    parser.add_argument("--min-distance", type=float, default=0.05, help="Ignore points closer than this many meters.")
    parser.add_argument("--max-distance", type=float, default=None, help="Ignore points farther than this many meters.")
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Do not create a timestamped backup of the source file before plotting.",
    )
    args = parser.parse_args()

    visualize(
        Path(args.source),
        args.samples_per_scan,
        args.min_distance,
        args.max_distance,
        create_snapshot=not args.no_snapshot,
    )


if __name__ == "__main__":
    main()
