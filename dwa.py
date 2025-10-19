import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


@dataclass
class LegacyDWAConfig:
    """DWA 参数总览（手动调参指南 - 精简版）

         # 叠加路径贴合方向作为"小角度"判据（A*仅作方向提示，不改变终点）
        self._path_align_diff_for_dw = math.inf
        # 预处理路径提示：避免在采样循环中重复计算
        self._path_hint_cache = None
        if path_hint is not None and len(path_hint) >= 2:
            # 以当前姿态评估与路径切向的夹角
            cur_end_state = np.array([state[0], state[1], state[2], state[3], state[4]], dtype=float)
            pa, _, _ = self._path_hint_components(cur_end_state, np.asarray(path_hint, dtype=float))
            self._path_align_diff_for_dw = pa
            # 预计算路径段向量，减少循环中的重复计算
            self._path_hint_cache = self._precompute_path_segments(np.asarray(path_hint, dtype=float))    - 速度单位 m/s，角速度 rad/s，角度 rad，时间 s，距离 m。
    - 机器人半径 robot_radius 与安全间隙 safety_clearance 一起决定“膨胀半径”。
    - 若出现“减速不及时/打滑感”，优先调整 max_accel、brake_*、smoothing_alpha。
    
    建议调参顺序：max_speed → max_accel → robot_radius/safety_clearance → obstacle/clearance 代价 →
    rotation/turn_* → progress/speed 代价 → reverse 系列 → brake_* → 细节开关。
    """
    max_speed: float = 0.5  # 降低最大线速度，配合低角速度提升SLAM稳定性
    min_speed: float = -0.7  # 默认禁倒车（如需倒车可设为负）。
    max_yaw_rate: float = 110.0 * math.pi / 180.0  # 提升最大角速度以增强转弯响应
    max_accel: float = 0.6  # 降低加速度，让运动更平滑(m/s^2)。直接影响刹车距离：d≈v^2/(2a)。过小会显得“刹不住”。
    max_delta_yaw_rate: float = 220.0 * math.pi / 180.0  # 角速度变化率上限，加大w窗口对急转更友好。
    v_resolution: float = 0.1  # 速度采样步长。越小越细但更慢；调整0.05->0.06降低采样数。
    yaw_rate_resolution: float = 5.0 * math.pi / 180.0  # 角速度采样步长。调整0.5°->8°降低采样数。
    dt: float = 0.1  # 控制周期(s)。与 SLAM/仿真一致；越小越灵敏也越耗时。
    predict_time: float = 1.4  # 预测时域(s)。短：更激进近视；长：更保守远视。1.0~2.0 常见。
    to_goal_cost_gain: float = 0.8  # 目标朝向代价权重。大→更快对准目标方向。
    to_goal_dist_cost_gain: float = 0.6  # 目标距离代价权重。大→更偏好缩短终点距离。
    speed_cost_gain: float = 0.40  # 降低速度奖励，避免“速度至上”。
    obstacle_cost_gain: float = 0.8  # 障碍代价权重。配合 obstacle_cost_divisor/cap 共同决定力度。
    rotation_cost_gain: float = 0.18  # 降低旋转代价，允许更积极的转向。
    progress_cost_gain: float = 1.0  # 更注重向目标推进。
    change_yaw_cost_gain: float = 0.5  # 角速度变化代价。大→更平滑，不易“抖动”。
    smoothing_alpha: float = 0.6  # 输出平滑系数(EMA)。小→更跟随历史，响应慢；大→更跟随当前，响应快。
    small_angle: float = 10.0 * math.pi / 180.0  # 认为“已较好对齐”的角度阈值，用于若干条件。
    small_angle_rot_scale: float = 1.8  # 小角度时增加旋转代价的比例，鼓励直行。
    robot_radius: float = 0.25  # 机器人半径(m)。与地图分辨率/真实底盘匹配。
    stuck_vel: float = 0.01  # 判定“卡住”的速度阈值。
    safety_clearance: float = 0  # 额外安全间隙(m)。膨胀半径 = robot_radius + safety_clearance。
    clearance_cost_gain: float = 1.0  # 接近膨胀半径时的代价权重。大→更远离墙。
    spin_penalty_gain: float = 0.22  # 进一步降低自旋惩罚，提升原地旋转意愿。
    min_forward_ratio: float = 0.08  # 小角度时最低前进速度占比。
    near_wall_threshold: float = 0.1  # 判定“靠墙”的gap阈值(m)。
    near_wall_rot_boost: float = 0.3  # 靠墙时加大旋转代价比例，避免贴墙小幅摆动。
    near_wall_forward_bias_gain: float = 0.3  # 靠墙且前进速度不足时的附加惩罚增益。
    align_deadband: float = 8.0 * math.pi/180.0  # 对齐死区(rad)。小角度下过滤无意义大角速。
    forward_bias_min_disp: float = 0.01  # 预测末端位移阈值。位移很小却大旋转→惩罚。
    forward_bias_cost_gain: float = 1.2  # 上述惩罚权重。
    debug: bool = True  # 打印内部组件代价与状态。
    # 动态窗口打印
    dw_debug: bool = True           # 是否定期打印动态窗口范围
    dw_log_interval: int = 10         # 打印间隔步数
    dwell_penalty_gain: float = 0.3  # 长时间低速/停滞惩罚增益。
    dwell_speed_threshold: float = 0.05  # 低于该速度计入“滞留”。
    low_forward_cost_gain: float = 2.5  # 小角度时前进不足惩罚增益。
    obstacle_cost_divisor: float = 5.0  # 降低障碍代价绝对量级的缩放因子。
    obstacle_cost_cap: float = 3.0  # 缩放后障碍代价上限；≤0 表示不封顶。
    progress_dist_gain: float = 0.25  # 基于终距缩短的奖励增益。
    forward_disp_reward_gain: float = 0.8  # 位移大小奖励权重。大→更鼓励加速前进。
    adaptive_dwell_threshold: int = 12  # 滞留步数超过该阈值启用额外前进奖励。
    adaptive_progress_extra_gain: float = 0.6  # 上述额外奖励权重。
    # 早期加速提升：在速度很低阶段暂时放宽动态窗口上限，加快脱离爬行
    accel_boost_speed: float = 0.3  # 低速阶段触发临时放宽线速度上界的门限。
    accel_boost_factor: float = 3.0  # 触发时上界放宽倍数。
    # 角度偏转减速：当朝向与目标方向存在较大偏差时降低允许前进最大速度
    turn_slow_angle: float = 32.0 * math.pi / 180.0  # 略提前减速触发点，配合降低最小速度实现更紧凑转向。
    turn_min_speed_scale: float = 0.18  # 在最大朝向偏差(≈pi)时进一步降低可用前进速度。
    turn_debug: bool = False  # 打印转向减速信息。
    # ---- 反复前后抖动抑制相关配置 ----
    allow_reverse: bool = True  # 是否允许倒车（全局开关）。
    reverse_heading_threshold: float = 90.0 * math.pi/180.0  # 与目标方向夹角大于该值时才考虑倒车。
    reverse_clearance_threshold: float = 0.35  # 前向清距极小时才考虑倒车（米）。
    oscillation_window_steps: int = 20  # 振荡检测窗口长度（步）。
    oscillation_disp_epsilon: float = 0.18  # 振荡判定位移阈值。
    oscillation_min_switches: int = 4  # 振荡判定的最小方向切换次数。
    oscillation_block_reverse_cycles: int = 60  # 检出振荡后禁倒车的持续步数。
    # ---- 前向清距配置 ----
    front_clear_cone_deg: float = 50.0  # 前向清距的视场角度(度)。
    # ---- 倒车转向优化 ----
    reverse_rot_cost_scale: float = 0.85  # 倒车时旋转代价缩放(<1 更易大角度转弯)。
    reverse_min_speed_scale: float = 0.7  # 倒车允许的最小速度过滤比例缩放。
    reverse_spin_penalty_scale: float = 0.85  # 倒车时对“打转”惩罚的缩放。
    reverse_turn_bonus_gain: float = 0.12  # 倒车+较大角速度的奖励(降低总cost)。
    # ---- 直接倒车支持 ----
    direct_reverse_enabled: bool = True  # 默认开启直接倒车。
    direct_reverse_gap_threshold: float = 0.15  # 直接倒车的gap阈值。
    direct_reverse_reward_gain: float = 0.3  # 直接倒车奖励权重（降低）。

    disable_fallback: bool = True  # 禁用 fallback；失败时改为放宽过滤重采样。
    reverse_no_heading_gate: bool = False  # 倒车仍需满足朝向阈值。
    # ---- 方向切换锐化 ----
    direction_switch_skip_smoothing: bool = True  # 线速度正负切换时跳过平滑，立即响应。
    reverse_initial_speed: float = 0.05  # 首次倒车的最小速度幅度。
    reverse_sign_change_boost_factor: float = 1.2  # 前进→倒车时的负向加速度放大量。
    direction_switch_cost_gain: float = 0.9  # 方向切换惩罚。
    # ---- 倒车对称化与灵活性增强 ----
    reverse_equal_speed: bool = False  # 默认不与前进对称。
    reverse_accel_factor: float = 1.0  # 倒车加速度放大倍数(×max_accel)。
    reverse_turn_rate_factor: float = 1.2  # 倒车阶段角速度倍率（已禁用，见_calc_dynamic_window）。
    reverse_rot_cost_scale_extra: float = 1.0  # 倒车时额外的旋转代价缩放(与已有乘积)，改为1.0使倒车和前进转向代价相同。
    reverse_allow_low_speed_small_angle: bool = False  # 小角度下不鼓励低速倒车。
    # 倒车微幅死区：抑制 |v| 很小的“试探性倒车”（非直接倒车场景）
    reverse_deadband: float = 0.22  # 低于该幅度的负速度将被过滤或钳制
    reverse_deadband_turn_angle_deg: float = 50.0  # 朝向误差超过该角度时放宽倒车死区
    reverse_deadband_turn_w: float = 0.25  # 角速度超过该阈值时放宽倒车死区
    # ---- 倒车->前进 制动/切换优化 ----
    reverse_brake_boost_factor: float = 2.0  # 倒车→前进时允许更大正向加速度以快速刹停。
    reverse_continue_penalty_gain: float = 2.2  # 已对齐仍倒车的惩罚。
    reverse_reward_angle_gate_deg: float = 6.0  # 角度误差阈值(度)，小于此不再奖励倒车。
    reverse_small_heading_gap: float = 0.22  # 小角度面向目标时允许倒车的最大前向净空(m)。
    # ---- 前进优先 / 启动阶段策略 ----
    initial_no_reverse_steps: int = 0  # 启动阶段不额外禁倒车（已整体禁倒车）。
    forward_pref_angle_deg: float = 40.0  # 角度小于该值时偏好前进而非倒车。
    forward_pref_gap_thresh: float = 0.50  # gap 大且角度小则抑制倒车的阈值。
    forward_pref_cost_gain: float = 0.45  # 违反前进偏好(仍倒车)的惩罚增益。
    forward_pref_initial_gain: float = 2.0  # 启动阶段的附加惩罚倍增。
    # ---- 墙距奖励（越远离墙奖励越大；靠墙奖励越低/甚至无） ----
    wall_reward_gain: float = 0.9           # 墙距奖励增益（加入为负成本，数值越大越鼓励离墙）
    wall_reward_max_gap: float = 0.6        # 超过该净空(gap)视为满奖励，上限封顶（米）
    wall_reward_power: float = 1.0          # 奖励幂次（>1使靠墙时奖励增长更慢，<1更快）
    # ---- 原地旋转策略（在大偏角或前向净空较小时，允许 v≈0 进行就地转向） ----
    enable_inplace_rotation: bool = True    # 打开原地旋转
    inplace_angle_deg: float = 15.0         # 当与目标方向夹角超过该值时触发
    inplace_gap_thresh: float = 0.5        # 或当前前向净空(gap)小于该阈值时触发
    inplace_rot_cost_scale: float = 0.5     # 触发时降低旋转代价（<1）
    inplace_spin_penalty_scale: float = 0.6 # 触发时降低打转惩罚（<1）
    # ---- 预测制动（提升减速及时性） ----
    brake_enable: bool = True                        # 开启基于前向净空的速度上界裁剪
    brake_react_time: float = 0.1                    # 反应时间(s)，v*treact
    brake_margin_m: float = 0.3                    # 额外安全裕度(m)
    brake_margin_min: float = 0.05                  # 动态裁剪时保证的最小裕度（避免完全清零gap）。
    brake_margin_ratio: float = 0.45                # 裕度随gap缩放比，gap越小越少扣减。
    brake_decel_factor: float = 1.3                  # 相对 max_accel 的制动放大倍数
    brake_skip_smoothing: bool = True                # 制动时跳过线速度平滑
    brake_drop_threshold: float = 0.15               # 需要降速超过该阈值则跳过平滑
    brake_debug: bool = False                        # 打印制动信息
    # ---- 全局路径贴合（仅作方向提示，不改变终点） ----
    path_align_gain: float = 0.5      # 和路径切向对齐的角度代价权重
    path_deviation_gain: float = 0.8  # 相对路径的横向偏差（米）代价权重
    path_progress_gain: float = 0   # 可选：沿路径前进的奖励（默认关闭）
    # ---- 实现中用到的通用阈值（统一收口，消除魔法数） ----
    # 倒车判定/采样与惩罚相关的小阈值
    reverse_sample_eps: float = 0.01     # 采样/判定倒车使用的速度阈值(|v|>eps 才视作倒车)
    reverse_plan_eps: float = 0.05       # 用于奖励/惩罚倒车的最小幅度(|v|>eps)
    # 小角度下大角速过滤阈值
    align_yaw_rate_mult: float = 5.0     # 与 yaw_rate_resolution 的倍乘系数
    align_small_speed_frac: float = 0.05 # 小角度场景下认为“速度很小”的比例阈值(×max_speed)
    # 前进偏置/转弯奖励阈值
    forward_bias_w_thresh: float = 0.1   # 位移很小却大旋转的角速度阈值(rad/s)
    reverse_turn_bonus_w: float = 0.3    # 倒车转弯奖励的角速度阈值(rad/s)
    reverse_turn_bonus_v: float = 0.05   # 倒车转弯奖励的速度幅度阈值(|v|>此值)
    # 自旋惩罚中的速度偏置，避免除零
    spin_penalty_v_eps: float = 0.05
    # 制动判定中的“前向净空很小”下限，和 safety_clearance 取较大者
    brake_small_gap_min: float = 0.25
    # （已删除脱困相关参数）
    # 振荡检测中将近零速度当作0的阈值
    oscillation_sign_eps: float = 0.01
    # ---- 内存与性能优化：障碍物评估参数 ----
    obstacle_eval_local_radius: float = 4.0  # 仅评估轨迹附近该半径(米)内的障碍（降低5.0->4.0）
    obstacle_eval_step_stride: int = 3       # 轨迹评估步长（每隔多少个时间步采样一次，2->3）
    obstacle_eval_max_points: int = 1500     # 参与评估的障碍点最大数量上限（2000->1500）
    visual_eval_max_paths: int = 60          # 可视化时最多展示的采样候选轨迹数（120->60）


class LegacyDWAPlanner:
    def __init__(self, config: LegacyDWAConfig):
        self.cfg = config
        # 使用 tuple[float, float]，运行时可容纳 numpy.float64
        self._last_u = (0.0, 0.0)
        self._dwell_count = 0  # 连续选择极低线速度计数
        # 振荡检测历史（必须在 __init__ 内）
        self._pos_hist = deque(maxlen=config.oscillation_window_steps)
        self._sign_hist = deque(maxlen=config.oscillation_window_steps)
        self._reverse_block_count = 0  # 剩余禁用倒车周期
        # 动态窗口调试缓存
        self.last_dw = None  # [v_min, v_max, w_min, w_max]
        self.last_dw_detail = None  # 记录Vs/Vd/制动上限等细节
        self._last_brake_v_cap = None
        self.last_timing = {}
        self._obs_local_cache = None  # 缓存当前周期的局部障碍点
        self._path_hint_cache = None  # 缓存预计算的路径段信息
        self.last_eval_paths = None    # 用于可视化的采样轨迹集合

    def plan(self, state: np.ndarray, goal: Tuple[float, float], obstacles: np.ndarray, path_hint: np.ndarray | None = None):
        """核心规划：返回平滑后的控制 (v, w) 及最佳轨迹。"""
        start_time = time.perf_counter()
        timing = {}
        if not hasattr(self, '_global_step'):
            self._global_step = 0
        # --- 1. 前向清距 & 动态窗口 ---
        precalc_start = start_time
        self._front_clearance_cache = self._front_clearance(state, obstacles)
        self._front_gap_cache = self._front_clearance_cache - self.cfg.robot_radius
        gdx = goal[0] - state[0]; gdy = goal[1] - state[1]
        heading_to_goal = math.atan2(gdy, gdx)
        heading_diff = abs(math.atan2(math.sin(heading_to_goal - state[2]), math.cos(heading_to_goal - state[2])))
        # 缓存供动态窗口阶段使用：小角度+前方净空充分时直接在DW内禁用倒车
        self._heading_diff_for_dw = heading_diff
        # 叠加路径贴合方向作为“小角度”判据（A*仅作方向提示，不改变终点）
        self._path_align_diff_for_dw = math.inf
        self._path_hint_cache = None
        if path_hint is not None:
            try:
                path_hint_arr = np.asarray(path_hint, dtype=float)
            except Exception:
                path_hint_arr = None
            if path_hint_arr is not None and path_hint_arr.ndim == 2 and len(path_hint_arr) >= 2:
                # 以当前姿态评估与路径切向的夹角
                cur_end_state = np.array([state[0], state[1], state[2], state[3], state[4]], dtype=float)
                pa, _, _ = self._path_hint_components(cur_end_state, path_hint_arr)
                self._path_align_diff_for_dw = pa
                self._path_hint_cache = self._precompute_path_segments(path_hint_arr)
        # 直接倒车区域判定（前向gap不足）
        direct_reverse_zone = (self.cfg.direct_reverse_enabled and self._front_gap_cache < self.cfg.direct_reverse_gap_threshold)
        timing['pre_calc'] = (time.perf_counter() - precalc_start) * 1000.0
        dw_start = time.perf_counter()
        dw = self._calc_dynamic_window(state)
        timing['dynamic_window'] = (time.perf_counter() - dw_start) * 1000.0
        self._prepare_local_obstacles(state, obstacles)

        best_cost = float('inf')
        best_u = (0.0, 0.0)
        best_traj = None
        best_components = None
        eval_paths = []

        # 目标方向参数
        start_x, start_y = state[0], state[1]
        gdx = goal[0] - start_x
        gdy = goal[1] - start_y
        gdist = math.hypot(gdx, gdy) + 1e-9
        gdir = (gdx / gdist, gdy / gdist)
        forward_pref_angle_rad = math.radians(self.cfg.forward_pref_angle_deg)
        reverse_reward_angle_rad = math.radians(self.cfg.reverse_reward_angle_gate_deg)
        reverse_deadband_angle_rad = math.radians(self.cfg.reverse_deadband_turn_angle_deg)
        reverse_deadband_w = self.cfg.reverse_deadband_turn_w
        align_yaw_rate_thresh = self.cfg.yaw_rate_resolution * self.cfg.align_yaw_rate_mult
        align_small_speed = self.cfg.align_small_speed_frac * self.cfg.max_speed
        min_forward_speed = self.cfg.min_forward_ratio * self.cfg.max_speed
        reverse_min_speed = min_forward_speed * self.cfg.reverse_min_speed_scale
        small_angle = self.cfg.small_angle
        small_angle_loose = small_angle * 0.7
        reverse_sample_eps = self.cfg.reverse_sample_eps
        inplace_angle_rad = math.radians(self.cfg.inplace_angle_deg)
        reverse_small_heading_gap = getattr(self.cfg, 'reverse_small_heading_gap', None)
        # 预计算循环内常量
        inflated_r = self.cfg.robot_radius + self.cfg.safety_clearance
        robot_radius = self.cfg.robot_radius

        # --- 2. 转向减速 ---
        if self.cfg.turn_slow_angle > 0:
            goal_heading = math.atan2(gdy, gdx)
            heading_diff = abs(math.atan2(math.sin(goal_heading - state[2]), math.cos(goal_heading - state[2])))
            if heading_diff > self.cfg.turn_slow_angle:
                ratio = min(1.0, (heading_diff - self.cfg.turn_slow_angle) / (math.pi - self.cfg.turn_slow_angle))
                turn_scale = 1.0 - (1.0 - self.cfg.turn_min_speed_scale) * ratio
                orig = dw[1]
                dw[1] = max(dw[0], min(dw[1], self.cfg.max_speed * turn_scale))
                if self.cfg.turn_debug and self.cfg.debug:
                    print(f"[TURN-SLOW] diff={heading_diff*180/math.pi:.1f}° scale={turn_scale:.2f} {orig:.2f}->{dw[1]:.2f}")

        any_candidate = False
        dynamic_allow_reverse = self.cfg.allow_reverse and self._reverse_block_count <= 0
        sample_main_start = time.perf_counter()
        v_samples = np.arange(dw[0], dw[1] + 1e-9, self.cfg.v_resolution)
        w_samples = np.arange(dw[2], dw[3] + 1e-9, self.cfg.yaw_rate_resolution)
        if v_samples.size == 0:
            v_samples = np.array([dw[0]])
        if w_samples.size == 0:
            w_samples = np.array([dw[2]])
        
        # 性能分析计数器
        timing['sample_traj_pred'] = 0.0
        timing['sample_goal_cost'] = 0.0
        timing['sample_obs_cost'] = 0.0
        timing['sample_other_costs'] = 0.0
        timing['sample_filter'] = 0.0
        sample_count = 0

        # --- 3. 采样评估 ---
        for v in v_samples:
            for w in w_samples:
                sample_count += 1
                t_traj = time.perf_counter()
                traj = self._predict_trajectory(state, v, w)
                timing['sample_traj_pred'] += (time.perf_counter() - t_traj) * 1000.0
                
                t_goal = time.perf_counter()
                ang_c, dist_c = self._goal_cost(traj, goal)
                timing['sample_goal_cost'] += (time.perf_counter() - t_goal) * 1000.0
                
                t_obs = time.perf_counter()
                obs_min_dist, obs_raw_cost = self._obstacle_cost_components(traj, obstacles)
                timing['sample_obs_cost'] += (time.perf_counter() - t_obs) * 1000.0
                
                # 早期碰撞检测：提前退出
                t_filter = time.perf_counter()
                if obs_min_dist <= robot_radius:
                    timing['sample_filter'] += (time.perf_counter() - t_filter) * 1000.0
                    continue
                # 倒车约束
                if v < 0:
                    if not dynamic_allow_reverse:
                        timing['sample_filter'] += (time.perf_counter() - t_filter) * 1000.0
                        continue
                    # 倒车死区：非直接倒车区域内，过滤微幅倒车，避免 v≈-0.05~-0.10 抖动
                    allow_small_reverse = (
                        direct_reverse_zone or
                        abs(w) >= reverse_deadband_w or
                        ang_c >= reverse_deadband_angle_rad
                    )
                    if (abs(v) < self.cfg.reverse_deadband) and not allow_small_reverse:
                        timing['sample_filter'] += (time.perf_counter() - t_filter) * 1000.0
                        continue
                    # 小角度且前向净空充足时，采样阶段也直接忽略倒车（双重保护）
                    small_heading = (heading_diff < forward_pref_angle_rad)
                    small_path_align = (getattr(self, '_path_align_diff_for_dw', math.inf) < forward_pref_angle_rad)
                    if (reverse_small_heading_gap is not None and reverse_small_heading_gap > 0):
                        gap_now = getattr(self, '_front_gap_cache', float('inf'))
                        if ((small_heading or small_path_align) and gap_now > reverse_small_heading_gap and not direct_reverse_zone):
                            timing['sample_filter'] += (time.perf_counter() - t_filter) * 1000.0
                            continue
                    if ((small_heading or small_path_align) and 
                        self._front_gap_cache > self.cfg.forward_pref_gap_thresh):
                        timing['sample_filter'] += (time.perf_counter() - t_filter) * 1000.0
                        continue
                    if (not self.cfg.reverse_no_heading_gate and 
                        (ang_c < self.cfg.reverse_heading_threshold and obs_min_dist > (self.cfg.robot_radius + self.cfg.reverse_clearance_threshold))):
                        timing['sample_filter'] += (time.perf_counter() - t_filter) * 1000.0
                        continue
                # 小角度大转速过滤
                if (
                    ang_c < self.cfg.align_deadband
                    and abs(w) > align_yaw_rate_thresh
                    and v > -reverse_sample_eps
                    and abs(v) < align_small_speed
                ):
                    close_obstacle_ahead = math.isfinite(obs_min_dist) and obs_min_dist <= (inflated_r + 0.12)
                    if not close_obstacle_ahead:
                        timing['sample_filter'] += (time.perf_counter() - t_filter) * 1000.0
                        continue
                if ang_c < small_angle_loose:
                    base_need = 0.5 * min_forward_speed
                    if v >= 0:
                        if abs(v) < base_need:
                            timing['sample_filter'] += (time.perf_counter() - t_filter) * 1000.0
                            continue
                    else:
                        if not self.cfg.reverse_allow_low_speed_small_angle:
                            need_rev = 0.5 * reverse_min_speed
                            if abs(v) < need_rev:
                                timing['sample_filter'] += (time.perf_counter() - t_filter) * 1000.0
                                continue
                timing['sample_filter'] += (time.perf_counter() - t_filter) * 1000.0
                
                t_other = time.perf_counter()
                any_candidate = True

                try:
                    traj_xy = traj[:, :2]
                    stride = max(1, int(len(traj_xy) / 6))
                    sampled = traj_xy[::stride]
                    if sampled.shape[0] == 0:
                        sampled = traj_xy[-1:, :]
                    elif not np.array_equal(sampled[-1], traj_xy[-1]):
                        sampled = np.vstack((sampled, traj_xy[-1]))
                    eval_paths.append(np.asarray(sampled, dtype=np.float32))
                except Exception:
                    pass

                to_goal_c = (self.cfg.to_goal_cost_gain * ang_c + self.cfg.to_goal_dist_cost_gain * dist_c)
                speed_c = self.cfg.speed_cost_gain * (self.cfg.max_speed - abs(traj[-1, 3]))
                if v < -reverse_sample_eps:
                    give_reward = True
                    # 若与目标方向角度已很小则不再奖励倒车
                    if ang_c < reverse_reward_angle_rad:
                        give_reward = False
                    if direct_reverse_zone and give_reward:
                        speed_c -= self.cfg.direct_reverse_reward_gain * (-v)
                # 前进偏好代价：小角度且gap充足仍倒车
                forward_pref_c = 0.0
                if v < -reverse_sample_eps:
                    if (ang_c < forward_pref_angle_rad and 
                        self._front_gap_cache > self.cfg.forward_pref_gap_thresh):
                        gain = self.cfg.forward_pref_cost_gain
                        if self._global_step < self.cfg.initial_no_reverse_steps:
                            gain *= self.cfg.forward_pref_initial_gain
                        forward_pref_c = gain * (-v)
                scaled_obs_raw = obs_raw_cost / max(1e-6, self.cfg.obstacle_cost_divisor)
                if self.cfg.obstacle_cost_cap > 0:
                    scaled_obs_raw = min(scaled_obs_raw, self.cfg.obstacle_cost_cap)
                obs_c = self.cfg.obstacle_cost_gain * scaled_obs_raw
                if obs_min_dist < inflated_r and obs_min_dist > robot_radius:
                    clearance_c = self.cfg.clearance_cost_gain * ((inflated_r - obs_min_dist) / inflated_r)
                else:
                    clearance_c = 0.0
                rot_scale = 1.0
                if ang_c < self.cfg.small_angle:
                    rot_scale += self.cfg.small_angle_rot_scale * (1 - ang_c / self.cfg.small_angle)
                rot_c = self.cfg.rotation_cost_gain * abs(w) * rot_scale
                # 倒车时进一步降低旋转代价，促使倒车结合较大角速度（减小转弯半径）
                if v < -reverse_sample_eps:
                    rot_c *= self.cfg.reverse_rot_cost_scale * self.cfg.reverse_rot_cost_scale_extra
                # （已移除脱困后的强化转向阶段）
                disp_x = traj[-1, 0] - start_x
                disp_y = traj[-1, 1] - start_y
                proj = max(0.0, disp_x * gdir[0] + disp_y * gdir[1])
                progress_c = - self.cfg.progress_cost_gain * proj
                net_disp = math.hypot(disp_x, disp_y)
                forward_bias_c = 0.0
                if net_disp < self.cfg.forward_bias_min_disp and abs(w) > self.cfg.forward_bias_w_thresh:
                    forward_bias_c = self.cfg.forward_bias_cost_gain * (self.cfg.forward_bias_min_disp - net_disp)
                disp_rew = - self.cfg.forward_disp_reward_gain * net_disp
                delta_w_state = w - state[4]
                delta_w_last = w - self._last_u[1]
                change_w_c = self.cfg.change_yaw_cost_gain * (abs(delta_w_state) + 0.5 * abs(delta_w_last))
                spin_c = self.cfg.spin_penalty_gain * (abs(w) / (abs(v) + self.cfg.spin_penalty_v_eps))
                if v < -reverse_sample_eps:
                    spin_c *= self.cfg.reverse_spin_penalty_scale
                min_fwd = min_forward_speed if ang_c < small_angle else 0.0
                low_forward_c = 0.0 if abs(v) >= min_fwd else (min_fwd - abs(v)) * self.cfg.low_forward_cost_gain
                dir_switch_c = self.cfg.direction_switch_cost_gain if self._last_u[0] * v < -1e-4 else 0.0
                wall_gap = obs_min_dist - robot_radius
                if wall_gap < self.cfg.near_wall_threshold:
                    nn = (self.cfg.near_wall_threshold - max(wall_gap, 0.0)) / self.cfg.near_wall_threshold
                    rot_c *= (1.0 + nn * (self.cfg.near_wall_rot_boost - 1.0))
                    desired_min = self.cfg.min_forward_ratio * self.cfg.max_speed
                    if abs(v) < desired_min:
                        low_forward_c += self.cfg.near_wall_forward_bias_gain * nn * (desired_min - abs(v))
                dwell_c = self.cfg.dwell_penalty_gain * (self._dwell_count + 1) if abs(v) < self.cfg.dwell_speed_threshold else 0.0
                dist_progress_c = - self.cfg.progress_dist_gain * (gdist - dist_c) if self.cfg.progress_dist_gain > 0 else 0.0
                extra_prog = 0.0
                if self._dwell_count > self.cfg.adaptive_dwell_threshold:
                    factor = min(1.0, (self._dwell_count - self.cfg.adaptive_dwell_threshold) / max(1.0, self.cfg.adaptive_dwell_threshold))
                    extra_prog = - self.cfg.adaptive_progress_extra_gain * factor * proj
                # 倒车转弯奖励：鼓励同时具有负速度和显著角速度
                reverse_turn_bonus = 0.0
                if v < -self.cfg.reverse_turn_bonus_v and abs(w) > self.cfg.reverse_turn_bonus_w:
                    reverse_turn_bonus = - self.cfg.reverse_turn_bonus_gain * abs(w)
                # 持续倒车惩罚：已较好对齐仍倒车
                reverse_continue_penalty = 0.0
                if v < -self.cfg.reverse_plan_eps and ang_c < reverse_reward_angle_rad:
                    reverse_continue_penalty = self.cfg.reverse_continue_penalty_gain * (-v)
                # 原地旋转策略：当角度偏差大或gap较小时，鼓励 v≈0 的就地转向
                if self.cfg.enable_inplace_rotation:
                    cond_angle = (ang_c > inplace_angle_rad)
                    gap_now = getattr(self, '_front_gap_cache', float('inf'))
                    cond_gap = (gap_now < self.cfg.inplace_gap_thresh)
                    if cond_angle or cond_gap:
                        rot_c *= self.cfg.inplace_rot_cost_scale
                        spin_c *= self.cfg.inplace_spin_penalty_scale
                        # 放宽对低前进速度与小位移的惩罚，避免阻碍原地旋转
                        if abs(v) < 0.05 * self.cfg.max_speed:
                            low_forward_c *= 0.2
                            forward_bias_c *= 0.2
                # 墙距奖励：gap 越大奖励越大（以负成本形式加入）；靠墙(gap小)奖励越低
                wall_reward = 0.0
                if math.isfinite(obs_min_dist):
                    gap = max(0.0, obs_min_dist - robot_radius)
                    cap = max(1e-6, self.cfg.wall_reward_max_gap)
                    norm = min(1.0, gap / cap)
                    # 奖励取负成本：-gain * norm^power
                    wall_reward = - self.cfg.wall_reward_gain * (norm ** max(0.0, self.cfg.wall_reward_power))
                # 路径贴合（若提供path_hint）
                path_align_c = 0.0
                path_dev_c = 0.0
                path_prog_c = 0.0
                if self._path_hint_cache is not None:
                    pa, pd, prog = self._path_hint_components_fast(traj[-1, 0], traj[-1, 1], traj[-1, 2])
                    path_align_c = self.cfg.path_align_gain * pa
                    path_dev_c = self.cfg.path_deviation_gain * abs(pd)
                    path_prog_c = - self.cfg.path_progress_gain * max(0.0, prog)

                cost = (to_goal_c + speed_c + obs_c + clearance_c + rot_c + progress_c + dist_progress_c + extra_prog +
                        change_w_c + spin_c + low_forward_c + forward_bias_c + dwell_c + disp_rew + dir_switch_c +
                        wall_reward + path_align_c + path_dev_c + path_prog_c +
                        reverse_turn_bonus + reverse_continue_penalty + forward_pref_c)
                timing['sample_other_costs'] += (time.perf_counter() - t_other) * 1000.0
                
                if math.isinf(cost):
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_u = (v, w)
                    best_traj = traj
                    best_components = {
                        'to_goal': to_goal_c, 'speed': speed_c, 'obs': obs_c, 'clear': clearance_c,
                        'rot': rot_c, 'progress': progress_c, 'dist_prog': dist_progress_c, 'extra_prog': extra_prog,
                        'spin': spin_c, 'low_fwd': low_forward_c, 'change_w': change_w_c, 'fwd_bias': forward_bias_c,
                        'dwell': dwell_c, 'disp_rew': disp_rew, 'dir_switch': dir_switch_c, 'min_dist': obs_min_dist,
                        'rev_turn_bonus': reverse_turn_bonus, 'rev_keep_pen': reverse_continue_penalty,
                        'fwd_pref': forward_pref_c, 'wall_reward': wall_reward,
                        'path_align': path_align_c, 'path_dev': path_dev_c, 'path_prog': -path_prog_c
                    }
        timing['sample_main'] = (time.perf_counter() - sample_main_start) * 1000.0
        timing['sample_count'] = sample_count
        timing['sample_relax'] = 0.0

        # --- 4. 二次放宽采样或回退 ---
        if not any_candidate and best_traj is None:
            fallback_start = time.perf_counter()
            if self.cfg.disable_fallback:
                if self.cfg.debug:
                    print("[RELAX] 首次采样无候选，放宽过滤重新采样 (禁用fallback)")
                for v in v_samples:
                    for w in w_samples:
                        traj = self._predict_trajectory(state, v, w)
                        ang_c, dist_c = self._goal_cost(traj, goal)
                        obs_min_dist, obs_raw_cost = self._obstacle_cost_components(traj, obstacles)
                        if obs_min_dist <= self.cfg.robot_radius:
                            continue
                        try:
                            traj_xy = traj[:, :2]
                            stride = max(1, int(len(traj_xy) / 6))
                            sampled = traj_xy[::stride]
                            if sampled.shape[0] == 0:
                                sampled = traj_xy[-1:, :]
                            elif not np.array_equal(sampled[-1], traj_xy[-1]):
                                sampled = np.vstack((sampled, traj_xy[-1]))
                            eval_paths.append(np.asarray(sampled, dtype=np.float32))
                        except Exception:
                            pass
                        to_goal_c = (self.cfg.to_goal_cost_gain * ang_c + self.cfg.to_goal_dist_cost_gain * dist_c)
                        speed_c = self.cfg.speed_cost_gain * (self.cfg.max_speed - abs(traj[-1, 3]))
                        if v < -self.cfg.reverse_sample_eps and direct_reverse_zone:
                            speed_c -= self.cfg.direct_reverse_reward_gain * (-v)
                        scaled_obs_raw = obs_raw_cost / max(1e-6, self.cfg.obstacle_cost_divisor)
                        if self.cfg.obstacle_cost_cap > 0:
                            scaled_obs_raw = min(scaled_obs_raw, self.cfg.obstacle_cost_cap)
                        obs_c = self.cfg.obstacle_cost_gain * scaled_obs_raw
                        cost = to_goal_c + speed_c + obs_c
                        if math.isinf(cost):
                            continue
                        if cost < best_cost:
                            best_cost = cost
                            best_u = (v, w)
                            best_traj = traj
                            best_components = {'relax': True}
            else:
                for v in v_samples:
                    for w in w_samples:
                        traj = self._predict_trajectory(state, v, w)
                        ang_c, dist_c = self._goal_cost(traj, goal)
                        try:
                            traj_xy = traj[:, :2]
                            stride = max(1, int(len(traj_xy) / 6))
                            sampled = traj_xy[::stride]
                            if sampled.shape[0] == 0:
                                sampled = traj_xy[-1:, :]
                            elif not np.array_equal(sampled[-1], traj_xy[-1]):
                                sampled = np.vstack((sampled, traj_xy[-1]))
                            eval_paths.append(np.asarray(sampled, dtype=np.float32))
                        except Exception:
                            pass
                        to_goal_c = (self.cfg.to_goal_cost_gain * ang_c + self.cfg.to_goal_dist_cost_gain * dist_c)
                        speed_c = self.cfg.speed_cost_gain * (self.cfg.max_speed - abs(traj[-1, 3]))
                        if v < -self.cfg.reverse_sample_eps and direct_reverse_zone:
                            speed_c -= self.cfg.direct_reverse_reward_gain * (-v)
                        if math.isinf(to_goal_c):
                            continue
                        cost = to_goal_c + speed_c
                        if cost < best_cost:
                            best_cost = cost
                            best_u = (v, w)
                            best_traj = traj
                            best_components = {'fallback': True}
            timing['sample_relax'] = (time.perf_counter() - fallback_start) * 1000.0

        # --- 5. stuck补偿与平滑 ---
        smooth_start = time.perf_counter()
        # 卡住检测：只有在运行了至少10步后才启用（避免起步误触发）
        if (abs(best_u[0]) < self.cfg.stuck_vel and abs(state[3]) < self.cfg.stuck_vel and 
            getattr(self, '_global_step', 0) > 10):
            best_u = (0.0, self.cfg.max_delta_yaw_rate * 0.5)
        # 全局倒车死区：若选择了微幅倒车且不在直接倒车区域，改为不倒车（消除微幅来回）
        final_heading_err = float('inf')
        if best_traj is not None:
            final_heading_err, _ = self._goal_cost(best_traj, goal)
        allow_small_reverse_final = (
            (self.cfg.direct_reverse_enabled and self._front_gap_cache < self.cfg.direct_reverse_gap_threshold) or
            abs(best_u[1]) >= reverse_deadband_w or
            final_heading_err >= reverse_deadband_angle_rad
        )
        if best_u[0] < 0 and abs(best_u[0]) < self.cfg.reverse_deadband and not allow_small_reverse_final:
            best_u = (0.0, best_u[1])
        # 若配置禁用倒车，硬钳制不允许负速度
        if not self.cfg.allow_reverse and best_u[0] < 0:
            best_u = (0.0, best_u[1])
        # 方向符号切换锐化：若线速度符号改变并启用跳过平滑
        if self.cfg.direction_switch_skip_smoothing and (best_u[0] * self._last_u[0] < -1e-4):
            # 倒车初始速度最小幅度
            if best_u[0] < 0 and abs(best_u[0]) < self.cfg.reverse_initial_speed:
                best_u = (-self.cfg.reverse_initial_speed, best_u[1])
            sm_v, sm_w = best_u
        else:
            # 制动场景：显著降速或前向净空过小，加快响应（跳过/收紧线速度平滑）
            need_drop = (self._last_u[0] - best_u[0]) > self.cfg.brake_drop_threshold
            small_gap = getattr(self, '_front_gap_cache', float('inf')) < max(self.cfg.brake_small_gap_min, self.cfg.safety_clearance)
            if self.cfg.brake_skip_smoothing and (need_drop or small_gap):
                sm_v = best_u[0]
                sm_w = self.cfg.smoothing_alpha * best_u[1] + (1 - self.cfg.smoothing_alpha) * self._last_u[1]
            else:
                sm_v = self.cfg.smoothing_alpha * best_u[0] + (1 - self.cfg.smoothing_alpha) * self._last_u[0]
                sm_w = self.cfg.smoothing_alpha * best_u[1] + (1 - self.cfg.smoothing_alpha) * self._last_u[1]
        # 最终一重保护：禁倒车时确保线速度非负
        if not self.cfg.allow_reverse and sm_v < 0:
            sm_v = 0.0
        self._last_u = (sm_v, sm_w)
        self.last_cost_components = best_components
        # 让可视化的预测轨迹与最终(平滑后的)控制一致，避免显示与执行不符
        out_traj = best_traj
        if best_traj is not None and (abs(sm_v - best_u[0]) > 1e-9 or abs(sm_w - best_u[1]) > 1e-9):
            out_traj = self._predict_trajectory(state, sm_v, sm_w)
        timing['smoothing'] = (time.perf_counter() - smooth_start) * 1000.0

        # --- 6. dwell计数 ---
        post_start = time.perf_counter()
        if abs(best_u[0]) < self.cfg.dwell_speed_threshold:
            self._dwell_count += 1
        else:
            self._dwell_count = 0
        
        # --- 7. 振荡检测 ---
        self._pos_hist.append((state[0], state[1]))
        self._sign_hist.append(1 if best_u[0] > self.cfg.oscillation_sign_eps else (-1 if best_u[0] < -self.cfg.oscillation_sign_eps else 0))
        if len(self._pos_hist) == self._pos_hist.maxlen:
            xs = [p[0] for p in self._pos_hist]; ys = [p[1] for p in self._pos_hist]
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            switches = sum(1 for i in range(1, len(self._sign_hist)) if self._sign_hist[i] * self._sign_hist[i-1] < 0)
            if span < self.cfg.oscillation_disp_epsilon and switches >= self.cfg.oscillation_min_switches and self._reverse_block_count <= 0:
                self._reverse_block_count = self.cfg.oscillation_block_reverse_cycles
                if self.cfg.debug:
                    print(f"[OSC] span={span:.3f} switches={switches} block_rev={self._reverse_block_count}")
        if self._reverse_block_count > 0:
            self._reverse_block_count -= 1

        # --- 8. Debug ---
        if self.cfg.debug and isinstance(best_components, dict):
            print(f"[DWA] v={best_u[0]:.2f} w={best_u[1]:.2f} cost={best_cost:.3f} comps={best_components}")
        # 全局步计数（用于启动阶段前进优先策略）
        self._global_step += 1
        timing['post_update'] = (time.perf_counter() - post_start) * 1000.0
        timing['total'] = (time.perf_counter() - start_time) * 1000.0
        self.last_timing = timing
        # 采样轨迹可视化缓存（裁剪数量以避免绘制过载）
        if eval_paths:
            max_paths = max(0, int(getattr(self.cfg, 'visual_eval_max_paths', 0)))
            if max_paths > 0 and len(eval_paths) > max_paths:
                step = int(math.ceil(len(eval_paths) / max_paths))
                eval_paths = eval_paths[::step]
            self.last_eval_paths = [p for p in eval_paths if isinstance(p, np.ndarray) and p.shape[0] >= 2]
        else:
            self.last_eval_paths = None
        return self._last_u, out_traj

    def _prepare_local_obstacles(self, state: np.ndarray, obstacles: np.ndarray | None):
        """预先筛选当前周期关心的障碍点，供采样阶段重复使用。"""
        if obstacles is None or len(obstacles) == 0:
            self._obs_local_cache = None
            return
        cfg = self.cfg
        obs = np.asarray(obstacles, dtype=np.float32)
        x0 = float(state[0]); y0 = float(state[1])
        max_disp = cfg.max_speed * cfg.predict_time
        local_r = cfg.obstacle_eval_local_radius + max_disp + cfg.robot_radius + cfg.safety_clearance
        dx = obs[:, 0] - x0
        dy = obs[:, 1] - y0
        mask = (dx * dx + dy * dy) <= (local_r * local_r)
        if not np.any(mask):
            self._obs_local_cache = None
            return
        selected = obs[mask]
        if selected.shape[0] > cfg.obstacle_eval_max_points:
            step = int(np.ceil(selected.shape[0] / cfg.obstacle_eval_max_points))
            selected = selected[::step]
        self._obs_local_cache = selected

    def _precompute_path_segments(self, path: np.ndarray):
        """预计算路径段信息以加速查询。
        返回: (p0s, p1s, vs, vv_inv, tangents) 所有为 numpy 数组
        """
        n = len(path)
        if n < 2:
            return None
        p0s = path[:-1]  # [N-1, 2]
        p1s = path[1:]   # [N-1, 2]
        vs = p1s - p0s   # [N-1, 2]
        vv = np.sum(vs * vs, axis=1)  # [N-1]
        valid = vv > 1e-9
        if not np.any(valid):
            return None
        # 仅保留有效段
        p0s = p0s[valid]
        p1s = p1s[valid]
        vs = vs[valid]
        vv = vv[valid]
        vv_inv = 1.0 / vv
        tangents = np.arctan2(vs[:, 1], vs[:, 0])
        return (p0s, p1s, vs, vv_inv, tangents)
    
    def _path_hint_components_fast(self, ex: float, ey: float, etheta: float):
        """使用预计算的路径段快速查询（避免循环）。"""
        if self._path_hint_cache is None:
            return 0.0, 0.0, 0.0
        p0s, p1s, vs, vv_inv, tangents = self._path_hint_cache
        # 向量化计算到所有线段的距离
        # t = ((ex - p0) · v) / vv，钳制到 [0, 1]
        dx = ex - p0s[:, 0]  # [N]
        dy = ey - p0s[:, 1]  # [N]
        t = (dx * vs[:, 0] + dy * vs[:, 1]) * vv_inv  # [N]
        t = np.clip(t, 0.0, 1.0)
        # 投影点: proj = p0 + t*v
        proj_x = p0s[:, 0] + t * vs[:, 0]
        proj_y = p0s[:, 1] + t * vs[:, 1]
        # 距离
        dist = np.hypot(ex - proj_x, ey - proj_y)
        idx = np.argmin(dist)
        min_dist = dist[idx]
        # 最近段信息
        tangent = tangents[idx]
        angle_diff = abs(math.atan2(math.sin(tangent - etheta), math.cos(tangent - etheta)))
        # 横向偏差（叉积符号）
        cross = (ex - proj_x[idx]) * vs[idx, 1] - (ey - proj_y[idx]) * vs[idx, 0]
        cross_track = math.copysign(min_dist, cross)
        # 沿线段进度
        progress = ((ex - proj_x[idx]) * math.cos(tangent) + (ey - proj_y[idx]) * math.sin(tangent))
        return angle_diff, cross_track, progress

    def _path_hint_components(self, end_state: np.ndarray, path_hint: np.ndarray):
        """
        基于全局路径的方向提示：
        - 角度项：末端朝向与路径切向的夹角（[0,pi]）
        - 偏差项：末端位置到最近路径线段的横向距离（米，取绝对值前）
        - 进度项：沿最近线段方向的投影位移（米，>=0 视作正向前进）
        path_hint: N×2 世界坐标折线
        返回: (angle_diff, cross_track, progress_along)
        """
        ex, ey, etheta = float(end_state[0]), float(end_state[1]), float(end_state[2])
        pts = np.asarray(path_hint, dtype=float)
        # 找到离末端最近的线段
        best = None
        min_dist = float('inf')
        for i in range(len(pts) - 1):
            p0 = pts[i]; p1 = pts[i+1]
            v = p1 - p0
            vv = float(v[0]**2 + v[1]**2)
            if vv < 1e-9:
                continue
            t = ((ex - p0[0]) * v[0] + (ey - p0[1]) * v[1]) / vv
            t = max(0.0, min(1.0, t))
            proj = p0 + t * v
            dx = ex - proj[0]
            dy = ey - proj[1]
            d = math.hypot(dx, dy)
            if d < min_dist:
                min_dist = d
                best = (p0, p1, proj, v)
        if best is None:
            return 0.0, 0.0, 0.0
        p0, p1, proj, v = best
        # 切向方向角
        tangent = math.atan2(v[1], v[0])
        angle_diff = abs(math.atan2(math.sin(tangent - etheta), math.cos(tangent - etheta)))
        # 横向偏差：带符号（根据左/右）
        # 叉积符号：sign = sign((ex-proj) × v)
        cross = (ex - proj[0]) * v[1] - (ey - proj[1]) * v[0]
        cross_track = math.copysign(min_dist, cross)
        # 沿线段方向的进度（投影长度）
        progress = ((ex - proj[0]) * math.cos(tangent) + (ey - proj[1]) * math.sin(tangent))
        return angle_diff, cross_track, progress

    def _calc_dynamic_window(self, state):
        cfg = self.cfg
        # 基本速度/角速度边界
        max_yaw = cfg.max_yaw_rate
        step = getattr(self, '_global_step', 0)
        if step < 3:
            max_yaw = min(max_yaw, math.radians(30.0))
        elif step < 10:
            max_yaw = min(max_yaw, math.radians(45.0))

        Vs = [cfg.min_speed, cfg.max_speed, -max_yaw, max_yaw]
        # 倒车速度对称化
        if cfg.reverse_equal_speed:
            Vs[0] = -cfg.max_speed
        # 全局不允许倒车：钳制下界 >= 0
        if not cfg.allow_reverse:
            Vs[0] = max(0.0, Vs[0])
        # 启动阶段禁止倒车
        if getattr(self, '_global_step', 0) < cfg.initial_no_reverse_steps:
            Vs[0] = max(0.0, Vs[0])
        # 小角度且前向净空充足时，直接在动态窗口阶段禁止倒车
        try:
            hd = getattr(self, '_heading_diff_for_dw', math.inf)
            pa = getattr(self, '_path_align_diff_for_dw', math.inf)
            if ((hd < math.radians(cfg.forward_pref_angle_deg) or pa < math.radians(cfg.forward_pref_angle_deg)) and
                getattr(self, '_front_gap_cache', -math.inf) > cfg.forward_pref_gap_thresh):
                Vs[0] = max(0.0, Vs[0])
        except Exception:
            pass
        # 移除倒车时角速度放大逻辑，使倒车和前进使用相同的角速度上限
        # (原代码：倒车时角速度可达前进的1.4倍，现统一使用 max_yaw_rate)
        # if cfg.reverse_equal_speed and (state[3] < 0 or Vs[0] < 0):
        #     max_yaw_rev = max_yaw * cfg.reverse_turn_rate_factor
        #     Vs[2] = -max_yaw_rev
        #     Vs[3] = max_yaw_rev

        Vd = [state[3] - cfg.max_accel * cfg.dt,
              state[3] + cfg.max_accel * cfg.dt,
              state[4] - cfg.max_delta_yaw_rate * cfg.dt,
              state[4] + cfg.max_delta_yaw_rate * cfg.dt]
        Vd_raw = Vd.copy()  # 记录未调制前的动态窗口
        # 倒车加速度放大
        if state[3] <= 0 and cfg.reverse_equal_speed:
            Vd[0] = state[3] - cfg.max_accel * cfg.dt * cfg.reverse_accel_factor
        # 倒车->前进 制动：允许更大的正向加速度（快速减小负速度幅度）
        if state[3] < -0.02:
            Vd[1] = state[3] + cfg.max_accel * cfg.dt * cfg.reverse_brake_boost_factor
        # 提升初期加速度窗口：低速阶段临时放宽线速度上界，加快摆脱0.05平台
        if state[3] < cfg.accel_boost_speed:
            Vd[1] = state[3] + cfg.max_accel * cfg.dt * cfg.accel_boost_factor
        # 符号切换下加大向负方向的加速度允许
        if hasattr(self, '_front_clearance_cache'):
            pass
        # 方向符号切换强化：若上一周期为正向且本周期考虑负向区域，进一步放宽
        if self._last_u[0] > 0.02:
            Vd[0] = min(Vd[0], state[3] - cfg.max_accel * cfg.dt * cfg.reverse_sign_change_boost_factor)
        # 记录制动前的窗口（含上述各项调制）
        Vd_mod_pre_brake = Vd.copy()
        # 预测制动：根据前向净空(gap)限制线速度上界，避免靠近障碍时减速不及时
        if self.cfg.brake_enable and hasattr(self, '_front_gap_cache'):
            gap = max(0.0, float(self._front_gap_cache))  # gap = 前向清距 - 机器人半径
            a_brake = max(1e-6, cfg.max_accel * self.cfg.brake_decel_factor)
            margin = self.cfg.brake_margin_m
            if gap > 1e-6:
                adaptive = gap * max(0.0, self.cfg.brake_margin_ratio)
                margin = min(margin, max(self.cfg.brake_margin_min, adaptive))
                margin = min(margin, gap * 0.95)
            eff_gap = max(0.0, gap - margin)
            t = max(0.0, self.cfg.brake_react_time)
            # v_max 解: v^2/(2a) + v*t <= eff_gap  => v = -a*t + sqrt((a*t)^2 + 2*a*eff_gap)
            disc = (a_brake * t) ** 2 + 2.0 * a_brake * eff_gap
            v_cap = max(0.0, -a_brake * t + math.sqrt(disc))
            # 仅裁剪前进方向的上界
            Vd[1] = min(Vd[1], v_cap)
            if self.cfg.brake_debug and (getattr(self, '_global_step', 0) % 10 == 0):
                print(f"[BRAKE] gap={gap:.2f} eff={eff_gap:.2f} v_cap={v_cap:.2f} -> Vmax={Vd[1]:.2f}")
            self._last_brake_v_cap = v_cap
            last_gap = gap
        else:
            self._last_brake_v_cap = None
            last_gap = None

        # 合成立即可用的动态窗口
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        # 若禁用倒车，确保下界非负
        if not cfg.allow_reverse and dw[0] < 0:
            dw[0] = 0.0

        # 缓存与可选打印
        self.last_dw = dw
        self.last_dw_detail = {
            'Vs': Vs.copy(),
            'Vd_raw': Vd_raw,
            'Vd_mod_pre_brake': Vd_mod_pre_brake,
            'dw': dw.copy(),
            'brake_v_cap': self._last_brake_v_cap,
            'gap': last_gap,
            'front_clear': getattr(self, '_front_clearance_cache', None)
        }
        if getattr(self.cfg, 'dw_debug', False):
            step = getattr(self, '_global_step', 0)
            interval = max(1, int(getattr(self.cfg, 'dw_log_interval', 10)))
            if step % interval == 0:
                extra = ""
                if self._last_brake_v_cap is not None:
                    extra = f" cap={self._last_brake_v_cap:.2f} gap={last_gap:.2f}"
                print(f"[DW] v:[{dw[0]:.2f},{dw[1]:.2f}] w:[{dw[2]:.2f},{dw[3]:.2f}]" + extra)
        return dw

    def _front_clearance(self, state, obstacles):
        """估算当前朝向前方锥形区内的最近障碍距离。"""
        if obstacles is None or len(obstacles) == 0:
            return float('inf')
        cfg = self.cfg
        heading = state[2]
        # 向量差
        dx = obstacles[:,0] - state[0]
        dy = obstacles[:,1] - state[1]
        dist = np.hypot(dx, dy)
        # 排除自身附近的点
        mask = dist > 1e-3
        if not np.any(mask):
            return float('inf')
        dx = dx[mask]; dy = dy[mask]; dist = dist[mask]
        ang = np.arctan2(dy, dx)
        dtheta = np.abs(np.arctan2(np.sin(ang - heading), np.cos(ang - heading)))
        cone = math.radians(cfg.front_clear_cone_deg)
        m2 = dtheta <= cone * 0.5
        if not np.any(m2):
            return float('inf')
        return float(np.min(dist[m2]))

    def _predict_trajectory(self, state, v, w):
        cfg = self.cfg
        dt = cfg.dt
        steps = max(1, int(math.ceil(cfg.predict_time / dt)))
        ts = dt * np.arange(1, steps + 1, dtype=np.float32)
        traj = np.empty((steps + 1, 5), dtype=np.float32)
        traj[0, 0] = state[0]
        traj[0, 1] = state[1]
        traj[0, 2] = state[2]
        traj[0, 3] = state[3]
        traj[0, 4] = state[4]
        x0 = float(state[0]); y0 = float(state[1]); theta0 = float(state[2])
        v_float = float(v); w_float = float(w)

        if abs(w_float) < 1e-8:
            disp = v_float * ts
            cos_t = math.cos(theta0)
            sin_t = math.sin(theta0)
            traj[1:, 0] = x0 + disp * cos_t
            traj[1:, 1] = y0 + disp * sin_t
            traj[1:, 2] = theta0
        else:
            theta = theta0 + w_float * ts
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            R = v_float / w_float
            sin0 = math.sin(theta0)
            cos0 = math.cos(theta0)
            traj[1:, 0] = x0 + R * (sin_theta - sin0)
            traj[1:, 1] = y0 - R * (cos_theta - cos0)
            traj[1:, 2] = np.arctan2(sin_theta, cos_theta)

        traj[1:, 3] = v_float
        traj[1:, 4] = w_float
        return traj

    def _goal_cost(self, traj, goal):
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        heading = math.atan2(dy, dx)
        diff = heading - traj[-1, 2]
        ang_cost = abs(math.atan2(math.sin(diff), math.cos(diff)))
        dist = math.hypot(dx, dy)
        return ang_cost, dist

    def _obstacle_cost_components(self, traj, obstacles):
        """返回轨迹的最小障碍距离与平滑后的障碍代价（GPU加速版）。"""
        cfg = self.cfg
        cache = self._obs_local_cache
        if cache is None:
            if obstacles is None or len(obstacles) == 0:
                return float('inf'), 0.0
            cache = np.asarray(obstacles, dtype=np.float32)
        if cache.size == 0:
            return float('inf'), 0.0
        obs = cache.astype(np.float32, copy=False)
        if obs.shape[0] > cfg.obstacle_eval_max_points:
            step = int(np.ceil(obs.shape[0] / cfg.obstacle_eval_max_points))
            obs = obs[::step]
        step_stride = max(1, int(cfg.obstacle_eval_step_stride))
        pts = traj[::step_stride, :2].astype(np.float32, copy=False)
        if pts.size == 0:
            pts = traj[-1:, :2].astype(np.float32, copy=False)
        
        # 批量计算所有轨迹点到所有障碍点的距离
        diff = pts[:, None, :] - obs[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        min_dist_sq = float(np.min(dist_sq))
        
        if min_dist_sq <= 0.0:
            return 0.0, float('inf')
        min_dist = math.sqrt(min_dist_sq)
        if min_dist <= cfg.robot_radius:
            return min_dist, float('inf')
        rel = max(1e-3, min_dist - cfg.robot_radius)
        smoothed = 1.0 / (rel + 0.05)
        return min_dist, smoothed


def occupancy_to_obstacles(occupancy, maze_bounds, resolution, stride=1):
    min_x, min_y, _, _ = maze_bounds
    ys, xs = np.where(occupancy == 1)
    if len(xs) == 0:
        return np.empty((0, 2))
    xs = xs[::stride]
    ys = ys[::stride]
    world_x = min_x + (xs + 0.5) * resolution
    world_y = min_y + (ys + 0.5) * resolution
    return np.vstack((world_x, world_y)).T


# ---- Simplified DWA implementation based on dynamic_window_approach.py ----


class RobotType(Enum):
    circle = 0
    rectangle = 1


@dataclass
class DWAConfig:
    max_speed: float = 1.0  # 最大线速度上限
    min_speed: float = -1.0  # 最小线速度（允许倒车则为负）
    max_yaw_rate: float = 90.0 * math.pi / 180.0  # 最大角速度（提高以支持大角度原地调头）
    max_accel: float = 0.6  # 线速度加速度上限
    max_delta_yaw_rate: float = 200.0 * math.pi / 180.0  # 角速度变化率上限（提升瞬时转向能力）
    v_resolution: float = 0.05  # 线速度采样步长
    yaw_rate_resolution: float = 1.0 * math.pi / 180.0  # 角速度采样步长
    dt: float = 0.1  # 控制周期
    predict_time: float = 1.6  # 预测时间窗口（缩短以避免远期误差导致停摆）
    to_goal_cost_gain: float = 1.5  # 朝向目标的角度代价权重（提高以增强转向意愿）
    to_goal_dist_cost_gain: float = 1.2  # 终点距离代价权重（提高以更积极接近目标）
    speed_cost_gain: float = 0.4  # 速度偏差代价权重
    obstacle_cost_gain: float = 0.15  # 障碍物代价权重（降低以避免过度避障）
    clearance_cost_gain: float = 0.08  # 安全间隙代价权重（降低以配合降低的obstacle_cost_gain）
    path_align_gain: float = 0.05  # 与参考路径切向对齐权重
    path_deviation_gain: float = 0.15  # 路径横向偏差权重
    smoothing_alpha: float = 0.4  # 输出平滑系数
    smoothness_gain: float = 0.4  # 与上一周期控制的变化代价（调低，让微转向时仍能前进）
    robot_radius: float = 0.107  # 机器人圆形半径（默认匹配 main 中的 ROBOT_COLLISION_RADIUS）
    safety_clearance: float = 0.05  # 附加安全余量（默认与主程序安全裕度一致）
    robot_type: RobotType = RobotType.circle  # 碰撞模型类型
    robot_width: float = 0.45  # 矩形模型宽度
    robot_length: float = 0.6  # 矩形模型长度
    stuck_vel: float = 0.02  # 判定卡住的速度阈值
    allow_reverse: bool = True  # 是否允许倒车指令
    debug: bool = True  # 是否打印调试信息

    def __post_init__(self) -> None:
        if self.min_speed > self.max_speed:
            raise ValueError("min_speed must be <= max_speed")
        if self.v_resolution <= 0 or self.yaw_rate_resolution <= 0:
            raise ValueError("sampling resolutions must be positive")
        if self.predict_time <= 0 or self.dt <= 0:
            raise ValueError("predict_time and dt must be positive")


class DWAPlanner:
    """Lightweight DWA planner closely following the reference implementation."""

    def __init__(self, config: DWAConfig):
        self.cfg = config
        self._last_u: Tuple[float, float] = (0.0, 0.0)
        self.last_dw: Optional[Tuple[float, float, float, float]] = None
        self.last_dw_detail: Optional[dict] = None
        self.last_timing: dict = {}
        self.last_cost_components: Optional[dict] = None
        # compatibility fields expected by higher-level planner code
        self._dwell_count: int = 0
        self._reverse_block_count: int = 0

    def plan(
        self,
        state: np.ndarray,
        goal: Tuple[float, float],
        obstacles: Optional[np.ndarray],
        path_hint: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[float, float], np.ndarray]:
        cfg = self.cfg
        start_time = time.perf_counter()
        timing: dict = {}

        dw_start = time.perf_counter()
        dw, detail = self._calc_dynamic_window(state)
        timing["dynamic_window"] = (time.perf_counter() - dw_start) * 1000.0
        self.last_dw = tuple(dw)
        self.last_dw_detail = detail

        v_samples = self._sample_range(dw[0], dw[1], cfg.v_resolution)
        w_samples = self._sample_range(dw[2], dw[3], cfg.yaw_rate_resolution)
        path_hint_arr = self._prepare_path_hint(path_hint)

        best_cost = float("inf")
        best_u: Tuple[float, float] = (0.0, 0.0)
        best_traj: Optional[np.ndarray] = None
        best_components: Optional[dict] = None

        obs_array = None if obstacles is None else np.asarray(obstacles, dtype=float)
        sample_start = time.perf_counter()

        for v in v_samples:
            if not cfg.allow_reverse and v < 0.0:
                continue
            for w in w_samples:
                traj = self._predict_trajectory(state, float(v), float(w))
                heading_cost = self._heading_cost(traj, goal)
                dist_cost = math.hypot(goal[0] - traj[-1, 0], goal[1] - traj[-1, 1])

                obstacle_cost, min_dist = self._obstacle_cost(traj, obs_array)
                if math.isinf(obstacle_cost):
                    continue

                clearance_cost = 0.0
                inflated = cfg.robot_radius + cfg.safety_clearance
                if np.isfinite(min_dist):
                    gap = max(0.0, min_dist - inflated)
                    if gap <= 0.0:
                        continue
                    if cfg.clearance_cost_gain > 0.0:
                        clearance_cost = cfg.clearance_cost_gain * (1.0 / max(gap, 1e-6))

                speed_cost = cfg.speed_cost_gain * (cfg.max_speed - abs(traj[-1, 3]))
                smooth_cost = cfg.smoothness_gain * (
                    abs(v - self._last_u[0]) + abs(w - self._last_u[1])
                )

                path_cost = 0.0
                if path_hint_arr is not None:
                    align_diff, deviation = self._path_hint_cost(traj[-1], path_hint_arr)
                    path_cost += cfg.path_align_gain * align_diff
                    path_cost += cfg.path_deviation_gain * abs(deviation)

                total_cost = (
                    cfg.to_goal_cost_gain * heading_cost
                    + cfg.to_goal_dist_cost_gain * dist_cost
                    + cfg.obstacle_cost_gain * obstacle_cost
                    + clearance_cost
                    + speed_cost
                    + smooth_cost
                    + path_cost
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_u = (float(v), float(w))
                    best_traj = traj
                    best_components = {
                        "goal_angle": cfg.to_goal_cost_gain * heading_cost,
                        "goal_dist": cfg.to_goal_dist_cost_gain * dist_cost,
                        "obstacle": cfg.obstacle_cost_gain * obstacle_cost,
                        "clearance": clearance_cost,
                        "speed": speed_cost,
                        "smooth": smooth_cost,
                        "path": path_cost,
                        "min_obs_dist": float(min_dist),
                    }

        timing["sampling"] = (time.perf_counter() - sample_start) * 1000.0

        if best_traj is None:
            best_u = (0.0, 0.0)
            best_traj = self._predict_trajectory(state, 0.0, 0.0)
            best_components = None

        if abs(best_u[0]) < cfg.stuck_vel and abs(state[3]) < cfg.stuck_vel:
            spin = cfg.yaw_rate_resolution * 2.0
            best_u = (0.0, math.copysign(spin, best_u[1] if abs(best_u[1]) > 1e-6 else 1.0))
            best_traj = self._predict_trajectory(state, *best_u)

        alpha = cfg.smoothing_alpha
        sm_v = alpha * best_u[0] + (1.0 - alpha) * self._last_u[0]
        sm_w = alpha * best_u[1] + (1.0 - alpha) * self._last_u[1]
        sm_v = float(np.clip(sm_v, dw[0], dw[1]))
        sm_w = float(np.clip(sm_w, dw[2], dw[3]))
        if not cfg.allow_reverse and sm_v < 0.0:
            sm_v = 0.0
        smoothed = (sm_v, sm_w)

        if abs(sm_v - best_u[0]) > 1e-6 or abs(sm_w - best_u[1]) > 1e-6:
            out_traj = self._predict_trajectory(state, sm_v, sm_w)
        else:
            out_traj = best_traj

        self._last_u = smoothed
        self.last_cost_components = best_components
        timing["total"] = (time.perf_counter() - start_time) * 1000.0
        self.last_timing = timing

        if cfg.debug:
            self._print_debug(best_u, best_cost, best_components)

        return smoothed, out_traj

    def _calc_dynamic_window(self, state: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.cfg
        Vs = np.array([
            cfg.min_speed,
            cfg.max_speed,
            -cfg.max_yaw_rate,
            cfg.max_yaw_rate,
        ], dtype=float)
        Vd = np.array([
            state[3] - cfg.max_accel * cfg.dt,
            state[3] + cfg.max_accel * cfg.dt,
            state[4] - cfg.max_delta_yaw_rate * cfg.dt,
            state[4] + cfg.max_delta_yaw_rate * cfg.dt,
        ], dtype=float)
        dw = np.array([
            max(Vs[0], Vd[0]),
            min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]),
            min(Vs[3], Vd[3]),
        ], dtype=float)
        if dw[0] > dw[1]:
            center = 0.5 * (dw[0] + dw[1])
            dw[0] = dw[1] = center
        if dw[2] > dw[3]:
            center = 0.5 * (dw[2] + dw[3])
            dw[2] = dw[3] = center
        detail = {"Vs": Vs.tolist(), "Vd": Vd.tolist(), "dw": dw.tolist()}
        return dw, detail

    def _predict_trajectory(self, state: np.ndarray, v: float, w: float) -> np.ndarray:
        cfg = self.cfg
        x = np.array(state, dtype=float)
        traj = [x.copy()]
        time_acc = 0.0
        while time_acc < cfg.predict_time:
            x = self._motion(x, v, w, cfg.dt)
            traj.append(x.copy())
            time_acc += cfg.dt
        return np.asarray(traj)

    def _obstacle_cost(
        self,
        trajectory: np.ndarray,
        obstacles: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        cfg = self.cfg
        if obstacles is None or obstacles.size == 0:
            return 0.0, float("inf")
        traj_xy = trajectory[:, :2]
        diff = traj_xy[:, None, :] - obstacles[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        min_dist = float(np.min(dists))
        if cfg.robot_type == RobotType.rectangle:
            if self._rectangle_collision(trajectory, obstacles):
                return float("inf"), min_dist
        else:
            inflated = cfg.robot_radius + cfg.safety_clearance
            if min_dist <= inflated:
                return float("inf"), min_dist
        adjusted = max(1e-6, min_dist - (cfg.robot_radius + cfg.safety_clearance))
        return 1.0 / adjusted, min_dist

    def _rectangle_collision(self, trajectory: np.ndarray, obstacles: np.ndarray) -> bool:
        cfg = self.cfg
        yaw = trajectory[:, 2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot = np.stack(
            [
                np.stack([cos_yaw, -sin_yaw], axis=1),
                np.stack([sin_yaw, cos_yaw], axis=1),
            ],
            axis=1,
        )
        local = obstacles[None, :, :] - trajectory[:, None, :2]
        local = np.einsum("nij,nkj->nki", rot, local)
        half_l = cfg.robot_length / 2.0
        half_w = cfg.robot_width / 2.0
        inside = (
            (local[:, :, 0] <= half_l)
            & (local[:, :, 0] >= -half_l)
            & (local[:, :, 1] <= half_w)
            & (local[:, :, 1] >= -half_w)
        )
        return bool(np.any(inside))

    def _path_hint_cost(self, end_state: np.ndarray, path: np.ndarray) -> Tuple[float, float]:
        ex, ey, etheta = float(end_state[0]), float(end_state[1]), float(end_state[2])
        best_idx = -1
        best_dist = float("inf")
        proj_point = None
        for i in range(len(path) - 1):
            p0 = path[i]
            p1 = path[i + 1]
            v = p1 - p0
            length_sq = float(np.dot(v, v))
            if length_sq < 1e-9:
                continue
            t = ((ex - p0[0]) * v[0] + (ey - p0[1]) * v[1]) / length_sq
            t = max(0.0, min(1.0, t))
            proj = p0 + t * v
            dist = math.hypot(ex - proj[0], ey - proj[1])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                proj_point = proj
        if proj_point is None or best_idx < 0:
            return 0.0, 0.0
        segment = path[best_idx + 1] - path[best_idx]
        tangent = math.atan2(segment[1], segment[0])
        angle_diff = abs(math.atan2(math.sin(tangent - etheta), math.cos(tangent - etheta)))
        cross = (ex - proj_point[0]) * segment[1] - (ey - proj_point[1]) * segment[0]
        deviation = math.copysign(best_dist, cross)
        return angle_diff, deviation

    def _prepare_path_hint(self, path_hint: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if path_hint is None:
            return None
        try:
            arr = np.asarray(path_hint, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        return arr

    def _sample_range(self, r_min: float, r_max: float, step: float) -> np.ndarray:
        if r_min > r_max:
            return np.array([float(r_min)], dtype=float)
        values = np.arange(r_min, r_max + step * 0.5, step, dtype=float)
        if values.size == 0:
            values = np.array([float(r_min)], dtype=float)
        return values

    def _heading_cost(self, trajectory: np.ndarray, goal: Tuple[float, float]) -> float:
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        angle_to_goal = math.atan2(dy, dx)
        diff = angle_to_goal - trajectory[-1, 2]
        return abs(math.atan2(math.sin(diff), math.cos(diff)))

    @staticmethod
    def _motion(state: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
        x = np.array(state, dtype=float)
        x[2] += w * dt
        x[0] += v * math.cos(x[2]) * dt
        x[1] += v * math.sin(x[2]) * dt
        x[3] = v
        x[4] = w
        return x

    def _print_debug(self, command: Tuple[float, float], cost: float, components: Optional[dict]) -> None:
        parts = [f"[DWA] v={command[0]:.2f} w={command[1]:.2f} cost={cost:.3f}"]
        if components:
            detail = ", ".join(
                f"{k}:{v:.3f}" for k, v in components.items() if isinstance(v, (int, float))
            )
            parts.append(f"[{detail}]")
        print(" ".join(parts))


