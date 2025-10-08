import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from collections import deque


@dataclass
class DWAConfig:
    """DWA 参数总览（手动调参指南 - 精简版）

    约定：
    - 速度单位 m/s，角速度 rad/s，角度 rad，时间 s，距离 m。
    - 机器人半径 robot_radius 与安全间隙 safety_clearance 一起决定“膨胀半径”。
    - 若出现“减速不及时/打滑感”，优先调整 max_accel、brake_*、smoothing_alpha。
    
    建议调参顺序：max_speed → max_accel → robot_radius/safety_clearance → obstacle/clearance 代价 →
    rotation/turn_* → progress/speed 代价 → reverse 系列 → brake_* → 细节开关。
    """
    max_speed: float = 1.3  # 最大线速度上限。路径较直、环境宽阔可调大；窄通道建议 ≤1.0。
    min_speed: float = -1.3 # 默认禁倒车（如需倒车可设为负）。
    max_yaw_rate: float = 230.0 * math.pi / 180.0  # 最大角速度上限，适当提高以便小半径转弯。
    max_accel: float = 1.7  # 最大线加速度(m/s^2)。直接影响刹车距离：d≈v^2/(2a)。过小会显得“刹不住”。
    max_delta_yaw_rate: float = 230.0 * math.pi / 180.0  # 角速度变化率上限(配合更大的角速)。
    v_resolution: float = 0.05  # 速度采样步长。越小越细但更慢；常取 0.03~0.06。
    yaw_rate_resolution: float = 0.5 * math.pi / 180.0  # 角速度采样步长。更细的 1° 提升转向精度。
    dt: float = 0.1  # 控制周期(s)。与 SLAM/仿真一致；越小越灵敏也越耗时。
    predict_time: float = 1.1  # 预测时域(s)。短：更激进近视；长：更保守远视。1.0~2.0 常见。
    to_goal_cost_gain: float = 0.6  # 目标朝向代价权重。大→更快对准目标方向。
    to_goal_dist_cost_gain: float = 0.25  # 目标距离代价权重。大→更偏好缩短终点距离。
    speed_cost_gain: float = 0.50  # 降低速度奖励，避免“速度至上”。
    obstacle_cost_gain: float = 0.8  # 障碍代价权重。配合 obstacle_cost_divisor/cap 共同决定力度。
    rotation_cost_gain: float = 0.25  # 更鼓励转向（配合小半径转弯）。
    progress_cost_gain: float = 2.5  # 更注重向目标推进。
    change_yaw_cost_gain: float = 0.4  # 角速度变化代价。大→更平滑，不易“抖动”。
    smoothing_alpha: float = 0.5  # 输出平滑系数(EMA)。小→更跟随历史，响应慢；大→更跟随当前，响应快。
    small_angle: float = 10.0 * math.pi / 180.0  # 认为“已较好对齐”的角度阈值，用于若干条件。
    small_angle_rot_scale: float = 3.0  # 小角度时增加旋转代价的比例，鼓励直行。
    robot_radius: float = 0.3  # 机器人半径(m)。与地图分辨率/真实底盘匹配。
    stuck_vel: float = 0.01  # 判定“卡住”的速度阈值。
    safety_clearance: float = 0.25  # 额外安全间隙(m)。膨胀半径 = robot_radius + safety_clearance。
    clearance_cost_gain: float = 3.0  # 接近膨胀半径时的代价权重。大→更远离墙。
    spin_penalty_gain: float = 0.3  # 适度降低自旋惩罚，结合转向更灵活。
    min_forward_ratio: float = 0.15  # 小角度时最低前进速度占比。
    near_wall_threshold: float = 0.3  # 判定“靠墙”的gap阈值(m)。
    near_wall_rot_boost: float = 2.5  # 靠墙时加大旋转代价比例，避免贴墙小幅摆动。
    near_wall_forward_bias_gain: float = 1.0  # 靠墙且前进速度不足时的附加惩罚增益。
    align_deadband: float = 3.0 * math.pi/180.0  # 对齐死区(rad)。小角度下过滤无意义大角速。
    forward_bias_min_disp: float = 0.01  # 预测末端位移阈值。位移很小却大旋转→惩罚。
    forward_bias_cost_gain: float = 1.2  # 上述惩罚权重。
    debug: bool = False  # 打印内部组件代价与状态。
    # 动态窗口打印
    dw_debug: bool = True            # 是否定期打印动态窗口范围
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
    turn_slow_angle: float = 15.0 * math.pi / 180.0  # 超过该角度开始对前进速度降额。
    turn_min_speed_scale: float = 0.08  # 在最大朝向偏差(≈pi)时的最大速度比例。
    turn_debug: bool = True  # 打印转向减速信息。
    # ---- 反复前后抖动抑制相关配置 ----
    allow_reverse: bool = True  # 是否允许倒车（全局开关）。
    reverse_heading_threshold: float = 50.0 * math.pi/180.0  # 与目标方向夹角大于该值时才考虑倒车。
    reverse_clearance_threshold: float = 0.28  # 前向清距不足时更倾向倒车（米）。
    oscillation_window_steps: int = 14  # 振荡检测窗口长度（步）。
    oscillation_disp_epsilon: float = 0.18  # 振荡判定位移阈值。
    oscillation_min_switches: int = 4  # 振荡判定的最小方向切换次数。
    oscillation_block_reverse_cycles: int = 60  # 检出振荡后禁倒车的持续步数。
    # ---- 前向清距配置 ----
    front_clear_cone_deg: float = 50.0  # 前向清距的视场角度(度)。
    # ---- 倒车转向优化 ----
    reverse_rot_cost_scale: float = 0.6  # 倒车时旋转代价缩放(<1 更易大角度转向)。
    reverse_min_speed_scale: float = 0.4  # 倒车允许的最小速度过滤比例缩放。
    reverse_spin_penalty_scale: float = 0.6  # 倒车时对“打转”惩罚的缩放。
    reverse_turn_bonus_gain: float = 0.15  # 倒车+较大角速度的奖励(降低总cost)。
    # ---- 直接倒车支持 ----
    direct_reverse_enabled: bool = True  # 默认开启直接倒车。
    direct_reverse_gap_threshold: float = 0.38  # 直接倒车的gap阈值。
    direct_reverse_reward_gain: float = 0.3  # 直接倒车奖励权重（降低）。

    disable_fallback: bool = True  # 禁用 fallback；失败时改为放宽过滤重采样。
    reverse_no_heading_gate: bool = True  # 允许倒车不受朝向阈值限制。
    # ---- 方向切换锐化 ----
    direction_switch_skip_smoothing: bool = True  # 线速度正负切换时跳过平滑，立即响应。
    reverse_initial_speed: float = 0.25  # 首次倒车的最小速度幅度。
    reverse_sign_change_boost_factor: float = 5.0  # 前进→倒车时的负向加速度放大量。
    direction_switch_cost_gain: float = 0.25  # 方向切换惩罚。
    # ---- 倒车对称化与灵活性增强 ----
    reverse_equal_speed: bool = False  # 默认不与前进对称。
    reverse_accel_factor: float = 2.0  # 倒车加速度放大倍数(×max_accel)。
    reverse_turn_rate_factor: float = 1.4  # 倒车阶段角速度倍率。
    reverse_rot_cost_scale_extra: float = 0.75  # 倒车时额外的旋转代价缩放(与已有乘积)。
    reverse_allow_low_speed_small_angle: bool = False  # 小角度下不鼓励低速倒车。
    # 倒车微幅死区：抑制 |v| 很小的“试探性倒车”（非直接倒车场景）
    reverse_deadband: float = 0.12  # 低于该幅度的负速度将被过滤或钳制
    # ---- 倒车->前进 制动/切换优化 ----
    reverse_brake_boost_factor: float = 4.0  # 倒车→前进时允许更大正向加速度以快速刹停。
    reverse_continue_penalty_gain: float = 1.2  # 已对齐仍倒车的惩罚。
    reverse_reward_angle_gate_deg: float = 6.0  # 角度误差阈值(度)，小于此不再奖励倒车。
    # ---- 前进优先 / 启动阶段策略 ----
    initial_no_reverse_steps: int = 0  # 启动阶段不额外禁倒车（已整体禁倒车）。
    forward_pref_angle_deg: float = 40.0  # 角度小于该值时偏好前进而非倒车。
    forward_pref_gap_thresh: float = 0.35  # gap 大且角度小则抑制倒车的阈值。
    forward_pref_cost_gain: float = 0.0  # 违反前进偏好(仍倒车)的惩罚增益（禁用）。
    forward_pref_initial_gain: float = 2.0  # 启动阶段的附加惩罚倍增。
    # ---- 墙距奖励（越远离墙奖励越大；靠墙奖励越低/甚至无） ----
    wall_reward_gain: float = 0.6           # 墙距奖励增益（加入为负成本，数值越大越鼓励离墙）
    wall_reward_max_gap: float = 0.6        # 超过该净空(gap)视为满奖励，上限封顶（米）
    wall_reward_power: float = 1.0          # 奖励幂次（>1使靠墙时奖励增长更慢，<1更快）
    # ---- 原地旋转策略（在大偏角或前向净空较小时，允许 v≈0 进行就地转向） ----
    enable_inplace_rotation: bool = True    # 打开原地旋转
    inplace_angle_deg: float = 25.0         # 当与目标方向夹角超过该值时触发
    inplace_gap_thresh: float = 0.35        # 或当前前向净空(gap)小于该阈值时触发
    inplace_rot_cost_scale: float = 0.7     # 触发时降低旋转代价（<1）
    inplace_spin_penalty_scale: float = 0.6 # 触发时降低打转惩罚（<1）
    # ---- 预测制动（提升减速及时性） ----
    brake_enable: bool = True                        # 开启基于前向净空的速度上界裁剪
    brake_react_time: float = 0.1                    # 反应时间(s)，v*treact
    brake_margin_m: float = 0.12                     # 额外安全裕度(m)
    brake_decel_factor: float = 1.2                  # 相对 max_accel 的制动放大倍数
    brake_skip_smoothing: bool = True                # 制动时跳过线速度平滑
    brake_drop_threshold: float = 0.15               # 需要降速超过该阈值则跳过平滑
    brake_debug: bool = False                        # 打印制动信息
    # ---- 全局路径贴合（仅作方向提示，不改变终点） ----
    path_align_gain: float = 0.6      # 和路径切向对齐的角度代价权重
    path_deviation_gain: float = 0.8  # 相对路径的横向偏差（米）代价权重
    path_progress_gain: float = 0.2   # 可选：沿路径前进的奖励（默认关闭）
    # ---- 实现中用到的通用阈值（统一收口，消除魔法数） ----
    # 倒车判定/采样与惩罚相关的小阈值
    reverse_sample_eps: float = 0.01     # 采样/判定倒车使用的速度阈值(|v|>eps 才视作倒车)
    reverse_plan_eps: float = 0.05       # 用于奖励/惩罚倒车的最小幅度(|v|>eps)
    # 小角度下大角速过滤阈值
    align_yaw_rate_mult: float = 1.5     # 与 yaw_rate_resolution 的倍乘系数
    align_small_speed_frac: float = 0.05 # 小角度场景下认为“速度很小”的比例阈值(×max_speed)
    # 前进偏置/转弯奖励阈值
    forward_bias_w_thresh: float = 0.2   # 位移很小却大旋转的角速度阈值(rad/s)
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
    obstacle_eval_local_radius: float = 6.0  # 仅评估轨迹附近该半径(米)内的障碍
    obstacle_eval_step_stride: int = 2       # 轨迹评估步长（每隔多少个时间步采样一次）
    obstacle_eval_max_points: int = 2000     # 参与评估的障碍点最大数量上限


class DWAPlanner:
    def __init__(self, config: DWAConfig):
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
        if path_hint is not None and len(path_hint) >= 2:
            # 以当前姿态评估与路径切向的夹角
            cur_end_state = np.array([state[0], state[1], state[2], state[3], state[4]], dtype=float)
            pa, _, _ = self._path_hint_components(cur_end_state, np.asarray(path_hint, dtype=float))
            self._path_align_diff_for_dw = pa
        # 直接倒车区域判定（前向gap不足）
        direct_reverse_zone = (self.cfg.direct_reverse_enabled and self._front_gap_cache < self.cfg.direct_reverse_gap_threshold)
        timing['pre_calc'] = (time.perf_counter() - precalc_start) * 1000.0
        dw_start = time.perf_counter()
        dw = self._calc_dynamic_window(state)
        timing['dynamic_window'] = (time.perf_counter() - dw_start) * 1000.0

        best_cost = float('inf')
        best_u = (0.0, 0.0)
        best_traj = None
        best_components = None

        # 目标方向参数
        start_x, start_y = state[0], state[1]
        gdx = goal[0] - start_x
        gdy = goal[1] - start_y
        gdist = math.hypot(gdx, gdy) + 1e-9
        gdir = (gdx / gdist, gdy / gdist)

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

        # --- 3. 采样评估 ---
        for v in np.arange(dw[0], dw[1] + 1e-9, self.cfg.v_resolution):
            for w in np.arange(dw[2], dw[3] + 1e-9, self.cfg.yaw_rate_resolution):
                traj = self._predict_trajectory(state, v, w)
                ang_c, dist_c = self._goal_cost(traj, goal)
                obs_min_dist, obs_raw_cost = self._obstacle_cost_components(traj, obstacles)
                # 倒车约束
                if v < 0:
                    if not dynamic_allow_reverse:
                        continue
                    # 倒车死区：非直接倒车区域内，过滤微幅倒车，避免 v≈-0.05~-0.10 抖动
                    if (abs(v) < self.cfg.reverse_deadband) and not direct_reverse_zone:
                        continue
                    # 小角度且前向净空充足时，采样阶段也直接忽略倒车（双重保护）
                    small_heading = (heading_diff < math.radians(self.cfg.forward_pref_angle_deg))
                    small_path_align = (getattr(self, '_path_align_diff_for_dw', math.inf) < math.radians(self.cfg.forward_pref_angle_deg))
                    if ((small_heading or small_path_align) and 
                        self._front_gap_cache > self.cfg.forward_pref_gap_thresh):
                        continue
                    if (not self.cfg.reverse_no_heading_gate and 
                        (ang_c < self.cfg.reverse_heading_threshold and obs_min_dist > (self.cfg.robot_radius + self.cfg.reverse_clearance_threshold))):
                        continue
                # 小角度大转速过滤
                if (
                    ang_c < self.cfg.align_deadband
                    and abs(w) > self.cfg.yaw_rate_resolution * self.cfg.align_yaw_rate_mult
                    and v > -self.cfg.reverse_sample_eps
                    and abs(v) < self.cfg.align_small_speed_frac * self.cfg.max_speed
                ):
                    continue
                if ang_c < self.cfg.small_angle * 0.7:
                    base_need = 0.5 * self.cfg.min_forward_ratio * self.cfg.max_speed
                    if v >= 0:
                        if abs(v) < base_need:
                            continue
                    else:
                        if not self.cfg.reverse_allow_low_speed_small_angle:
                            need_rev = base_need * self.cfg.reverse_min_speed_scale
                            if abs(v) < need_rev:
                                continue
                any_candidate = True

                to_goal_c = (self.cfg.to_goal_cost_gain * ang_c + self.cfg.to_goal_dist_cost_gain * dist_c)
                speed_c = self.cfg.speed_cost_gain * (self.cfg.max_speed - abs(traj[-1, 3]))
                if v < -self.cfg.reverse_sample_eps:
                    give_reward = True
                    # 若与目标方向角度已很小则不再奖励倒车
                    if ang_c < math.radians(self.cfg.reverse_reward_angle_gate_deg):
                        give_reward = False
                    if direct_reverse_zone and give_reward:
                        speed_c -= self.cfg.direct_reverse_reward_gain * (-v)
                # 前进偏好代价：小角度且gap充足仍倒车
                forward_pref_c = 0.0
                if v < -self.cfg.reverse_sample_eps:
                    if (ang_c < math.radians(self.cfg.forward_pref_angle_deg) and 
                        self._front_gap_cache > self.cfg.forward_pref_gap_thresh):
                        gain = self.cfg.forward_pref_cost_gain
                        if self._global_step < self.cfg.initial_no_reverse_steps:
                            gain *= self.cfg.forward_pref_initial_gain
                        forward_pref_c = gain * (-v)
                scaled_obs_raw = obs_raw_cost / max(1e-6, self.cfg.obstacle_cost_divisor)
                if self.cfg.obstacle_cost_cap > 0:
                    scaled_obs_raw = min(scaled_obs_raw, self.cfg.obstacle_cost_cap)
                obs_c = self.cfg.obstacle_cost_gain * scaled_obs_raw
                inflated_r = self.cfg.robot_radius + self.cfg.safety_clearance
                if obs_min_dist < inflated_r and obs_min_dist > self.cfg.robot_radius:
                    clearance_c = self.cfg.clearance_cost_gain * ((inflated_r - obs_min_dist) / inflated_r)
                elif obs_min_dist <= self.cfg.robot_radius:
                    clearance_c = float('inf')
                else:
                    clearance_c = 0.0
                rot_scale = 1.0
                if ang_c < self.cfg.small_angle:
                    rot_scale += self.cfg.small_angle_rot_scale * (1 - ang_c / self.cfg.small_angle)
                rot_c = self.cfg.rotation_cost_gain * abs(w) * rot_scale
                # 倒车时进一步降低旋转代价，促使倒车结合较大角速度（减小转弯半径）
                if v < -self.cfg.reverse_sample_eps:
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
                if v < -self.cfg.reverse_sample_eps:
                    spin_c *= self.cfg.reverse_spin_penalty_scale
                min_fwd = self.cfg.min_forward_ratio * self.cfg.max_speed if ang_c < self.cfg.small_angle else 0.0
                low_forward_c = 0.0 if abs(v) >= min_fwd else (min_fwd - abs(v)) * self.cfg.low_forward_cost_gain
                dir_switch_c = self.cfg.direction_switch_cost_gain if self._last_u[0] * v < -1e-4 else 0.0
                wall_gap = obs_min_dist - self.cfg.robot_radius
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
                if v < -self.cfg.reverse_plan_eps and ang_c < math.radians(self.cfg.reverse_reward_angle_gate_deg):
                    reverse_continue_penalty = self.cfg.reverse_continue_penalty_gain * (-v)
                # 原地旋转策略：当角度偏差大或gap较小时，鼓励 v≈0 的就地转向
                if self.cfg.enable_inplace_rotation:
                    cond_angle = (ang_c > math.radians(self.cfg.inplace_angle_deg))
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
                    gap = max(0.0, obs_min_dist - self.cfg.robot_radius)
                    cap = max(1e-6, self.cfg.wall_reward_max_gap)
                    norm = min(1.0, gap / cap)
                    # 奖励取负成本：-gain * norm^power
                    wall_reward = - self.cfg.wall_reward_gain * (norm ** max(0.0, self.cfg.wall_reward_power))
                # 路径贴合（若提供path_hint）
                path_align_c = 0.0
                path_dev_c = 0.0
                path_prog_c = 0.0
                if path_hint is not None and len(path_hint) >= 2:
                    pa, pd, prog = self._path_hint_components(traj[-1, :], path_hint)
                    path_align_c = self.cfg.path_align_gain * pa
                    path_dev_c = self.cfg.path_deviation_gain * abs(pd)
                    path_prog_c = - self.cfg.path_progress_gain * max(0.0, prog)

                cost = (to_goal_c + speed_c + obs_c + clearance_c + rot_c + progress_c + dist_progress_c + extra_prog +
                        change_w_c + spin_c + low_forward_c + forward_bias_c + dwell_c + disp_rew + dir_switch_c +
                        wall_reward + path_align_c + path_dev_c + path_prog_c)
                cost += reverse_turn_bonus + reverse_continue_penalty
                cost += forward_pref_c
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
        timing['sample_relax'] = 0.0

        # --- 4. 二次放宽采样或回退 ---
        if not any_candidate and best_traj is None:
            fallback_start = time.perf_counter()
            if self.cfg.disable_fallback:
                if self.cfg.debug:
                    print("[RELAX] 首次采样无候选，放宽过滤重新采样 (禁用fallback)")
                for v in np.arange(dw[0], dw[1] + 1e-9, self.cfg.v_resolution):
                    for w in np.arange(dw[2], dw[3] + 1e-9, self.cfg.yaw_rate_resolution):
                        traj = self._predict_trajectory(state, v, w)
                        ang_c, dist_c = self._goal_cost(traj, goal)
                        obs_min_dist, obs_raw_cost = self._obstacle_cost_components(traj, obstacles)
                        if obs_min_dist <= self.cfg.robot_radius:
                            continue
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
                for v in np.arange(dw[0], dw[1] + 1e-9, self.cfg.v_resolution):
                    for w in np.arange(dw[2], dw[3] + 1e-9, self.cfg.yaw_rate_resolution):
                        traj = self._predict_trajectory(state, v, w)
                        ang_c, dist_c = self._goal_cost(traj, goal)
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
        if abs(best_u[0]) < self.cfg.stuck_vel and abs(state[3]) < self.cfg.stuck_vel:
            best_u = (0.0, self.cfg.max_delta_yaw_rate * 0.5)
        # 全局倒车死区：若选择了微幅倒车且不在直接倒车区域，改为不倒车（消除微幅来回）
        if best_u[0] < 0 and abs(best_u[0]) < self.cfg.reverse_deadband and not (self.cfg.direct_reverse_enabled and self._front_gap_cache < self.cfg.direct_reverse_gap_threshold):
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
        return self._last_u, out_traj

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
        # 基于当前是否在倒车规划阶段（上一次或速度窗口下界<0）提高角速度上限
        if cfg.reverse_equal_speed and (state[3] < 0 or Vs[0] < 0):
            max_yaw_rev = max_yaw * cfg.reverse_turn_rate_factor
            Vs[2] = -max_yaw_rev
            Vs[3] = max_yaw_rev

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
            eff_gap = max(0.0, gap - self.cfg.brake_margin_m)
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
        x = np.array(state, dtype=float)
        traj = [x.copy()]
        t = 0.0
        while t < cfg.predict_time:
            theta0 = x[2]
            if abs(w) < 1e-8:
                dx = v * cfg.dt * math.cos(theta0)
                dy = v * cfg.dt * math.sin(theta0)
                dtheta = 0.0
                theta1 = theta0
            else:
                dtheta = w * cfg.dt
                theta1 = theta0 + dtheta
                R = v / w
                dx = R * (math.sin(theta1) - math.sin(theta0))
                dy = -R * (math.cos(theta1) - math.cos(theta0))
            # 更新状态（与 Robot.velocity_step 一致）
            x[0] += dx
            x[1] += dy
            x[2] = math.atan2(math.sin(theta1), math.cos(theta1))
            x[3] = v
            x[4] = w
            traj.append(x.copy())
            t += cfg.dt
        # 使用 float32 降低内存占用
        return np.array(traj, dtype=np.float32)

    def _goal_cost(self, traj, goal):
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        heading = math.atan2(dy, dx)
        diff = heading - traj[-1, 2]
        ang_cost = abs(math.atan2(math.sin(diff), math.cos(diff)))
        dist = math.hypot(dx, dy)
        return ang_cost, dist

    def _obstacle_cost(self, traj, obstacles):
        if obstacles is None or len(obstacles) == 0:
            return 0.0
        ox = obstacles[:, 0]
        oy = obstacles[:, 1]
        dx = traj[:, 0][:, None] - ox[None, :]
        dy = traj[:, 1][:, None] - oy[None, :]
        dist = np.hypot(dx, dy)
        min_dist = np.min(dist)
        if min_dist <= self.cfg.robot_radius:
            return float('inf')
        return 1.0 / min_dist

    # 复用：返回最小距离和原生障碍代价
    def _obstacle_cost_components(self, traj, obstacles):
        """内存友好版本：
        - 仅在轨迹附近的局部半径内选取障碍点
        - 按步长对轨迹点子采样
        - 避免构造 (T×K) 的大矩阵，改为逐步 1D 计算并取最小值
        """
        if obstacles is None or len(obstacles) == 0:
            return float('inf'), 0.0
        cfg = self.cfg
        # 确保 float32，减半内存占用
        obs = obstacles.astype(np.float32, copy=False)
        ox = obs[:, 0]
        oy = obs[:, 1]
        # 基于轨迹起点与最大位移确定局部搜索半径
        x0 = float(traj[0, 0]); y0 = float(traj[0, 1])
        dx_all = traj[:, 0] - traj[0, 0]
        dy_all = traj[:, 1] - traj[0, 1]
        max_disp = float(np.max(np.hypot(dx_all, dy_all)))
        local_r = cfg.obstacle_eval_local_radius + max_disp
        # 先用起点近似筛选局部障碍（快速，避免先构造大矩阵）
        dx0 = ox - x0
        dy0 = oy - y0
        mask = (dx0 * dx0 + dy0 * dy0) <= (local_r * local_r)
        if not np.any(mask):
            return float('inf'), 0.0
        ox_local = ox[mask]
        oy_local = oy[mask]
        # 限制参与评估的障碍点总数
        if ox_local.shape[0] > cfg.obstacle_eval_max_points:
            step = max(1, int(np.ceil(ox_local.shape[0] / cfg.obstacle_eval_max_points)))
            ox_local = ox_local[::step]
            oy_local = oy_local[::step]
        # 沿轨迹按步长评估最小距离（逐步 1D 向量，不建 2D 矩阵）
        step_stride = max(1, int(cfg.obstacle_eval_step_stride))
        min_dist = float('inf')
        for p in traj[::step_stride]:
            dx = ox_local - float(p[0])
            dy = oy_local - float(p[1])
            dmin = float(np.min(np.hypot(dx, dy)))
            if dmin < min_dist:
                min_dist = dmin
                # 早停：已穿入碰撞半径，无需继续
                if min_dist <= cfg.robot_radius:
                    break
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


# 已移除 make_main_dwa_config：请直接实例化 DWAConfig()
