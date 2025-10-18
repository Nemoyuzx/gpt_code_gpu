import math
import threading
from queue import Queue, Empty

import numpy as np
from typing import Optional


class Robot:
    """机器人运动模型与控制器。支持简单前进/旋转以及速度积分接口(DWA使用)。"""

    def __init__(self, start_pose, odom_noise=(0.0, 0.0)):
        """
        start_pose: (x, y, theta)
        odom_noise: (trans_noise, rot_noise) 里程计噪声标准差
        """
        self.x, self.y, self.theta = start_pose
        # 轨迹记录（用于最终路径导出/可视化）
        self.trajectory = [(self.x, self.y)]
        # 里程计读数（初始化为真值）
        self.odom_x, self.odom_y, self.odom_theta = self.x, self.y, self.theta
        # 噪声标准差
        self.trans_noise, self.rot_noise = odom_noise
        # 当前速度（供DWA状态使用）
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        # 降噪滤波器引用
        self.noise_filter = None
        # 线程控制
        self._thread_enabled = False
        self._cmd_queue: Optional[Queue] = None
        self._result_queue: Optional[Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_stop = threading.Event()
        self._thread_timeout = 1.0

    # ---------------- 直线 / 旋转 基础接口 ----------------
    def move(self, distance: float):
        """沿当前朝向前进 distance 米。"""
        self.x += distance * math.cos(self.theta)
        self.y += distance * math.sin(self.theta)
        # 里程计（加噪声）
        if self.trans_noise > 0:
            distance_odom = distance + np.random.normal(0, self.trans_noise)
        else:
            distance_odom = distance
        self.odom_x += distance_odom * math.cos(self.odom_theta)
        self.odom_y += distance_odom * math.sin(self.odom_theta)
        self.trajectory.append((self.x, self.y))
        self.linear_vel = 0.0  # 该接口不维护瞬时速度，主要用于离散跳转
        self.angular_vel = 0.0

    def rotate(self, angle: float):
        """原地旋转 angle (rad)。"""
        self.theta = math.atan2(math.sin(self.theta + angle), math.cos(self.theta + angle))
        if self.rot_noise > 0:
            angle_odom = angle + np.random.normal(0, self.rot_noise)
        else:
            angle_odom = angle
        self.odom_theta = math.atan2(math.sin(self.odom_theta + angle_odom), math.cos(self.odom_theta + angle_odom))
        self.linear_vel = 0.0
        self.angular_vel = 0.0

    def move_to(self, target_x: float, target_y: float):
        dx = target_x - self.x
        dy = target_y - self.y
        desired_theta = math.atan2(dy, dx)
        angle_diff = math.atan2(math.sin(desired_theta - self.theta), math.cos(desired_theta - self.theta))
        if abs(angle_diff) > 0.1:
            print(f"警告：朝向({self.theta:.2f})与目标方向({desired_theta:.2f})偏差较大")
        distance = math.hypot(dx, dy)
        self.move(distance)
        return distance

    # ---------------- 查询接口 ----------------
    def get_pose(self):
        return (self.x, self.y, self.theta)

    def get_odom_pose(self):
        if self.noise_filter is not None:
            fx, fy, fth = self.noise_filter.filter_odometry_data(self.odom_x, self.odom_y, self.odom_theta)
            return (fx, fy, fth)
        return (self.odom_x, self.odom_y, self.odom_theta)

    def set_noise_filter(self, noise_filter):
        self.noise_filter = noise_filter

    # ---------------- 速度积分接口 (DWA 使用) ----------------
    def velocity_step(self, v_cmd: float, w_cmd: float, dt: float):
        """按 (v,w) 指令积分 dt，更新真实与里程计位姿。
        返回 (平移距离, 旋转角度) 供 SLAM 使用。"""
        if self._thread_enabled:
            if self._cmd_queue is None or self._result_queue is None:
                raise RuntimeError("Robot threaded mode is misconfigured: queues are missing")
            try:
                self._cmd_queue.put((float(v_cmd), float(w_cmd), float(dt)))
                result = self._result_queue.get(timeout=self._thread_timeout)
                return result
            except Empty as exc:
                raise RuntimeError("Robot thread did not respond in time") from exc
        return self._integrate_velocity(v_cmd, w_cmd, dt)

    def start_threaded(self, *, thread_name: str = "RobotThread", command_queue_size: int = 32,
                       timeout: float = 1.0) -> None:
        """启动后台线程，在队列中消费速度指令。"""
        if self._thread_enabled:
            return
        self._cmd_queue = Queue(maxsize=command_queue_size)
        self._result_queue = Queue()
        self._thread_timeout = max(timeout, 0.1)
        self._thread_stop.clear()
        self._thread = threading.Thread(target=self._thread_loop, name=thread_name, daemon=True)
        self._thread_enabled = True
        self._thread.start()

    def stop_threaded(self) -> None:
        """停止后台线程。"""
        if not self._thread_enabled:
            return
        self._thread_enabled = False
        if self._cmd_queue is not None:
            try:
                self._cmd_queue.put_nowait(None)
            except Exception:
                pass
        self._thread_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._cmd_queue = None
        self._result_queue = None
        self._thread_stop.clear()

        def apply_motion(self, distance: float, rotation: float, *, linear_vel: Optional[float] = None,
                         angular_vel: Optional[float] = None) -> None:
            """Integrate an externally measured motion increment into the robot state."""
            theta0 = self.theta
            if abs(rotation) < 1e-8:
                dx = distance * math.cos(theta0)
                dy = distance * math.sin(theta0)
            else:
                theta1 = theta0 + rotation
                radius = distance / rotation if abs(rotation) > 1e-8 else 0.0
                dx = radius * (math.sin(theta1) - math.sin(theta0))
                dy = -radius * (math.cos(theta1) - math.cos(theta0))
            self.x += dx
            self.y += dy
            self.theta = math.atan2(math.sin(theta0 + rotation), math.cos(theta0 + rotation))
            self.trajectory.append((self.x, self.y))
            self.odom_x = self.x
            self.odom_y = self.y
            self.odom_theta = self.theta
            if linear_vel is not None:
                self.linear_vel = linear_vel
            if angular_vel is not None:
                self.angular_vel = angular_vel

    def _thread_loop(self) -> None:
        if self._cmd_queue is None or self._result_queue is None:
            return
        while not self._thread_stop.is_set():
            try:
                item = self._cmd_queue.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                break
            v_cmd, w_cmd, dt = item
            result = self._integrate_velocity(v_cmd, w_cmd, dt)
            self._result_queue.put(result)

    def _integrate_velocity(self, v_cmd: float, w_cmd: float, dt: float):
        """执行一次速度积分并返回 (平移距离, 旋转角度)。"""
        theta0 = self.theta
        if abs(w_cmd) < 1e-8:
            dx = v_cmd * dt * math.cos(theta0)
            dy = v_cmd * dt * math.sin(theta0)
            dtheta = 0.0
        else:
            dtheta = w_cmd * dt
            theta1 = theta0 + dtheta
            R = v_cmd / w_cmd if abs(w_cmd) > 1e-8 else 0.0
            dx = R * (math.sin(theta1) - math.sin(theta0))
            dy = -R * (math.cos(theta1) - math.cos(theta0))
        # 真实位姿
        self.x += dx
        self.y += dy
        self.theta = math.atan2(math.sin(self.theta + dtheta), math.cos(self.theta + dtheta))
        self.linear_vel = v_cmd
        self.angular_vel = w_cmd
        self.trajectory.append((self.x, self.y))
        # 噪声里程计（保持平移符号）：用弧长 v*dt 作为带符号的平移增量
        distance_signed = v_cmd * dt
        rot = dtheta
        if self.trans_noise > 0:
            distance_odom = distance_signed + np.random.normal(0, self.trans_noise)
        else:
            distance_odom = distance_signed
        if self.rot_noise > 0:
            rot_odom = rot + np.random.normal(0, self.rot_noise)
        else:
            rot_odom = rot
        if abs(rot_odom) < 1e-8:
            dx_o = distance_odom * math.cos(self.odom_theta)
            dy_o = distance_odom * math.sin(self.odom_theta)
        else:
            if dt > 0:
                v_odom = distance_odom / dt  # 可能为负，保持符号
                R_o = v_odom / (rot_odom / dt) if abs(rot_odom/dt) > 1e-8 else 0.0
            else:
                R_o = 0.0
            theta_new_o = self.odom_theta + rot_odom
            dx_o = R_o * (math.sin(theta_new_o) - math.sin(self.odom_theta))
            dy_o = -R_o * (math.cos(theta_new_o) - math.cos(self.odom_theta))
            self.odom_theta = theta_new_o
        self.odom_theta = math.atan2(math.sin(self.odom_theta), math.cos(self.odom_theta))
        self.odom_x += dx_o
        self.odom_y += dy_o
        # 返回带符号的平移增量，供 SLAM 正确区分前进/倒车
        return distance_signed, rot
