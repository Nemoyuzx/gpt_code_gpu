#!/usr/bin/env python3
"""Utilities to bridge real BLE robot data into the navigation stack."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from bluetooth_connection import (
    BleBackgroundListener,
    LaserSample,
    MpuStatus,
    get_laser_samples,
    get_mpu_status,
    start_background_listener,
)


@dataclass
class ScanData:
    """Container carrying a normalized scan."""

    distances: List[float]
    clean: List[float]
    angles_deg: List[float]


class LaserDataAdapter:
    """Convert raw laser samples cached by bluetooth_connection into scans."""

    def __init__(
        self,
        *,
        samples_per_scan: int = 250,
        angle_resolution: float = 360.0 / 250.0,
        distance_scale: float = 0.001,
        max_range: float = 12.0,
        min_fill_ratio: float = 0.8,
        poll_interval: float = 0.02,
    ) -> None:
        self._samples_per_scan = samples_per_scan
        self._angle_resolution = angle_resolution
        self._distance_scale = distance_scale
        self._max_range = max_range
        self._min_fill_ratio = max(0.1, min(1.0, min_fill_ratio))
        self._poll_interval = max(0.001, poll_interval)

    def get_latest_scan(self, timeout: float = 0.5) -> Optional[ScanData]:
        """Return the latest scan if enough samples have been received."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            scan = self._build_scan_snapshot()
            if scan is not None:
                return scan
            if time.monotonic() >= deadline:
                return None
            time.sleep(self._poll_interval)

    def _build_scan_snapshot(self) -> Optional[ScanData]:
        samples = get_laser_samples(self._samples_per_scan * 2)
        if not samples:
            return None

        angle_map: dict[int, LaserSample] = {}
        for sample in reversed(samples):
            angle_key = int(round(sample.angle * 100.0))
            if angle_key not in angle_map:
                angle_map[angle_key] = sample
            if len(angle_map) >= self._samples_per_scan:
                break

        if not angle_map:
            return None

        sorted_samples = sorted(angle_map.values(), key=lambda s: s.angle)
        if len(sorted_samples) > self._samples_per_scan:
            sorted_samples = sorted_samples[-self._samples_per_scan:]

        filled = len(sorted_samples)
        if filled / self._samples_per_scan < self._min_fill_ratio:
            return None

        distances: List[float] = [self._max_range] * self._samples_per_scan
        clean: List[float] = [self._max_range] * self._samples_per_scan
        angles: List[float] = [float("nan")] * self._samples_per_scan

        for idx, sample in enumerate(sorted_samples):
            distance_mm = sample.distance
            if distance_mm <= 0:
                dist = self._max_range
            else:
                dist = distance_mm * self._distance_scale
                if not math.isfinite(dist) or dist <= 0:
                    dist = self._max_range
                dist = min(dist, self._max_range)
            distances[idx] = dist
            clean[idx] = dist
            angles[idx] = sample.angle

        return ScanData(distances=distances, clean=clean, angles_deg=angles)


class MotionDataAdapter:
    """Estimate motion deltas from cached MPU/encoder frames."""

    def __init__(
        self,
        *,
        wheel_track: float,
        ticks_per_meter: float,
        poll_interval: float = 0.01,
        timeout: float = 0.25,
        encoder_modulus: Optional[int] = 2 ** 32,
        invert_left: bool = False,
        invert_right: bool = False,
    ) -> None:
        if ticks_per_meter <= 0:
            raise ValueError("ticks_per_meter must be positive")
        if wheel_track <= 0:
            raise ValueError("wheel_track must be positive")

        self._wheel_track = wheel_track
        self._ticks_per_meter = ticks_per_meter
        self._poll_interval = max(0.001, poll_interval)
        self._timeout = max(0.0, timeout)
        self._encoder_modulus = encoder_modulus
        self._encoder_half_range = encoder_modulus / 2 if encoder_modulus else None
        self._invert_left = -1 if invert_left else 1
        self._invert_right = -1 if invert_right else 1

        self._prev_counts: Optional[Tuple[int, int]] = None
        self._prev_time: Optional[float] = None
        # Cache last valid motion to avoid blocking on timeout
        self._last_valid_motion: Tuple[float, float, Optional[Tuple[float, float]]] = (0.0, 0.0, None)

    def reset(self) -> None:
        self._prev_counts = None
        self._prev_time = None
        self._last_valid_motion = (0.0, 0.0, None)

    def poll_motion(self) -> Tuple[float, float, Optional[Tuple[float, float]]]:
        deadline = time.monotonic() + self._timeout
        start_time = time.monotonic()
        while True:
            status = get_mpu_status()
            if status is not None:
                result = self._process_status(status)
                if result is not None:
                    self._last_valid_motion = result  # Cache valid data
                    return result
            if time.monotonic() >= deadline:
                # Return cached motion instead of (0,0,None) to maintain continuity
                elapsed_ms = (time.monotonic() - start_time) * 1000.0
                if elapsed_ms > 50.0:  # Warn if blocking > 50ms
                    print(f"[WARN] poll_motion timeout after {elapsed_ms:.1f}ms, using cached data")
                return self._last_valid_motion
            time.sleep(self._poll_interval)

    def _process_status(
        self, status: MpuStatus
    ) -> Optional[Tuple[float, float, Optional[Tuple[float, float]]]]:
        counts = (int(status.count_run1), int(status.count_run2))

        if self._prev_counts is None:
            self._prev_counts = counts
            self._prev_time = time.monotonic()
            return 0.0, 0.0, None

        if counts == self._prev_counts:
            return None

        now = time.monotonic()
        dt = max(1e-3, now - (self._prev_time or now))
        delta_left = self._unwrap_delta(counts[0] - self._prev_counts[0]) * self._invert_left
        delta_right = self._unwrap_delta(counts[1] - self._prev_counts[1]) * self._invert_right
        self._prev_counts = counts
        self._prev_time = now

        meters_left = delta_left / self._ticks_per_meter
        meters_right = delta_right / self._ticks_per_meter
        d_trans = 0.5 * (meters_left + meters_right)
        d_rot = (meters_right - meters_left) / self._wheel_track
        velocity = (d_trans / dt, d_rot / dt)
        return d_trans, d_rot, velocity

    def _unwrap_delta(self, delta: int) -> int:
        if self._encoder_modulus is None or self._encoder_half_range is None:
            return delta
        if delta > self._encoder_half_range:
            delta -= self._encoder_modulus
        elif delta < -self._encoder_half_range:
            delta += self._encoder_modulus
        return delta


class BleRobotBridge:
    """Convenience wrapper bundling BLE listener, laser, and motion adapters."""

    def __init__(
        self,
        address: str,
        notify_char: str,
        *,
        adapter: Optional[str] = None,
        connect_timeout: float = 8.0,
        delimiter: str = "\n",
        decode_errors: str = "replace",
        show_raw: bool = False,
        reconnect_delay: float = 3.0,
        laser_params: Optional[dict] = None,
        motion_params: Optional[dict] = None,
    ) -> None:
        self.listener: BleBackgroundListener = start_background_listener(
            address,
            notify_char,
            adapter=adapter,
            connect_timeout=connect_timeout,
            delimiter=delimiter,
            decode_errors=decode_errors,
            show_raw=show_raw,
            reconnect_delay=reconnect_delay,
        )
        self.laser = LaserDataAdapter(**(laser_params or {}))
        if motion_params is None:
            raise ValueError("motion_params must be provided to configure wheel geometry")
        self.motion = MotionDataAdapter(**motion_params)

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self.listener.ready_event.wait(timeout)

    def stop(self) -> None:
        self.listener.stop()