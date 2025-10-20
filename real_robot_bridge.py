#!/usr/bin/env python3
"""Utilities to bridge real BLE robot data into the navigation stack."""

from __future__ import annotations

import math
import threading
import time
import os
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


class MotorController:
    """控制器：将速度命令转换为编码器速度并发送到小车"""

    def __init__(
        self,
        wheel_track: float,
        ticks_per_meter: float,
        min_encoder_speed: int = 20,
        speed_scale: float = 1.0,
        deadband_threshold: float = 0.01,
    ) -> None:
        """
        Args:
            wheel_track: 两轮中心距(m)
            ticks_per_meter: 编码器脉冲数/米
            min_encoder_speed: 最小编码器速度(防止电压不足)
            speed_scale: 速度缩放因子（用于调整整体速度）
            deadband_threshold: 速度死区阈值(m/s)，低于此值视为停止
        """
        self._wheel_track = wheel_track
        self._ticks_per_meter = ticks_per_meter
        self._min_encoder_speed = min_encoder_speed
        self._speed_scale = speed_scale
        self._deadband_threshold = deadband_threshold
        self._command_queue: List[str] = []
        self._lock = threading.Lock()

    def velocity_to_encoder_speeds(
        self, v: float, w: float, dt: float
    ) -> Tuple[int, int]:
        """
        将线速度v(m/s)和角速度w(rad/s)转换为左右轮编码器速度
        改进：转弯时保持速度差的比例，避免最小速度保护破坏差速效果
        
        Args:
            v: 线速度 (m/s)
            w: 角速度 (rad/s)
            dt: 时间步长 (s)
            
        Returns:
            (left_speed, right_speed): 左右轮编码器速度
        """
        # 计算左右轮线速度
        v_left = v - (w * self._wheel_track / 2.0)
        v_right = v + (w * self._wheel_track / 2.0)
        
        # 死区处理：速度太小时直接归零
        if abs(v_left) < self._deadband_threshold:
            v_left = 0.0
        if abs(v_right) < self._deadband_threshold:
            v_right = 0.0
        
        # 转换为编码器速度 (ticks/s) 并应用缩放因子
        encoder_left = v_left * self._ticks_per_meter * self._speed_scale
        encoder_right = v_right * self._ticks_per_meter * self._speed_scale
        
        # 改进的最小速度保护逻辑：
        # 1. 如果两个轮子速度都非常小，直接停止
        if abs(encoder_left) < 1.0 and abs(encoder_right) < 1.0:
            encoder_left = 0
            encoder_right = 0
        # 2. 如果是转弯（左右轮速度差异明显），使用比例缩放而非固定最小值
        elif abs(encoder_left - encoder_right) > 5.0:
            # 这是转弯指令，保持速度差的比例
            max_wheel = max(abs(encoder_left), abs(encoder_right))
            if max_wheel < self._min_encoder_speed:
                # 两个轮子都低于最小速度，按比例放大
                scale_factor = self._min_encoder_speed / max_wheel
                encoder_left *= scale_factor
                encoder_right *= scale_factor
            # 如果只有一个轮子低于最小速度，不强制提升（保持差速）
        # 3. 直行或速度差很小时，应用标准最小速度保护
        else:
            if 0 < abs(encoder_left) < self._min_encoder_speed:
                encoder_left = self._min_encoder_speed if encoder_left > 0 else -self._min_encoder_speed
            if 0 < abs(encoder_right) < self._min_encoder_speed:
                encoder_right = self._min_encoder_speed if encoder_right > 0 else -self._min_encoder_speed
        
        return int(round(encoder_left)), int(round(encoder_right))

    def send_command(self, left_speed: int, right_speed: int) -> str:
        """
        生成控制命令字符串
        
        Args:
            left_speed: 左轮编码器速度
            right_speed: 右轮编码器速度
            
        Returns:
            命令字符串，格式: "CMD-SET a b"
        """
        command = f"CMD-SET {left_speed} {right_speed}"
        with self._lock:
            self._command_queue.append(command)
        return command

    def get_pending_commands(self) -> List[str]:
        """获取待发送的命令列表"""
        with self._lock:
            commands = list(self._command_queue)
            self._command_queue.clear()
        return commands


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
        motor_control_params: Optional[dict] = None,
        write_char: Optional[str] = None,
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
            write_char=write_char,
        )
        self.laser = LaserDataAdapter(**(laser_params or {}))
        if motion_params is None:
            raise ValueError("motion_params must be provided to configure wheel geometry")
        self.motion = MotionDataAdapter(**motion_params)
        
        # 电机控制器 - 从 motion_params 获取几何参数，从 motor_control_params 获取控制参数
        wheel_track = motion_params.get("wheel_track", 0.168)
        ticks_per_meter = motion_params.get("ticks_per_meter", 2000.0)
        
        motor_ctrl_params = motor_control_params or {}
        min_encoder_speed = motor_ctrl_params.get("min_encoder_speed", 20)
        speed_scale = motor_ctrl_params.get("speed_scale", 1.0)  # 保持1:1输出，缩放交由规划器控制
        deadband_threshold = motor_ctrl_params.get("deadband_threshold", 0.01)
        
        self.motor_controller = MotorController(
            wheel_track=wheel_track,
            ticks_per_meter=ticks_per_meter,
            min_encoder_speed=min_encoder_speed,
            speed_scale=speed_scale,
            deadband_threshold=deadband_threshold,
        )
        
        # 写入特征UUID（用于发送控制命令）
        self._write_char = write_char
        self._client = None  # BLE客户端引用（需要从listener获取）
        # 全局速度缩放比例，仅作用于实际下发的指令，不影响规划/可视化逻辑
        overall_scale = motor_ctrl_params.get("overall_speed_scale")
        if overall_scale is None:
            overall_scale = os.getenv("BLE_MOTOR_SPEED_SCALE", "1.0")
        try:
            self._motor_speed_scale = float(overall_scale)
        except (TypeError, ValueError):
            self._motor_speed_scale = 1.0

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self.listener.ready_event.wait(timeout)

    def send_motor_command(self, v: float, w: float, dt: float = 0.1) -> bool:
        """
        发送电机控制命令
        
        Args:
            v: 线速度 (m/s)
            w: 角速度 (rad/s)
            dt: 时间步长 (s)
            
        Returns:
            是否成功发送命令
        """
        try:
            left_speed, right_speed = self.motor_controller.velocity_to_encoder_speeds(v, w, dt)
            scale = self._motor_speed_scale
            scaled_left = int(round(left_speed * scale))
            scaled_right = int(round(right_speed * scale))
            command = self.motor_controller.send_command(scaled_left, scaled_right)
            
            # 通过蓝牙发送命令
            if self._write_char:
                self.listener.write_command(command)
                print(
                    "[MOTOR] %s (requested v=%.3f w=%.3f | scale=%.3f)"
                    % (command, v, w, scale)
                )
            else:
                print(f"[WARN] No write_char configured, command not sent: {command}")
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send motor command: {e}")
            return False

    def stop(self) -> None:
        # 停止前发送停止命令
        try:
            self.motor_controller.send_command(0, 0)
        except Exception:
            pass
        self.listener.stop()