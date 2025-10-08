import time
import math
import os
import matplotlib.pyplot as plt
from maze_loader import MazeLoader
from robot import Robot
from lidar import Lidar
from icp_slam import ICPSlam    
from frontier_explorer import FrontierExplorer
from dwa import DWAPlanner, DWAConfig, occupancy_to_obstacles
from collections import deque
from visualizer import Visualizer
from noise_filter import NoiseFilter
import numpy as np

# ==================== 系统参数配置 ====================
# 路径规划参数
SAFETY_DISTANCE_FACTOR = 0.7  # 路径截断百分比，表示只执行路径的前70%
FRONTIER_SAFETY_DISTANCE = 8.0  # 前沿探索器与障碍物的安全距离 (提高, 使路径/前沿选择更远离墙体)

# 迷宫和机器人参数
MAZE_FILE = "2.json"  # 默认迷宫文件
ROBOT_ODOM_NOISE = (0.01, math.radians(0.01))  # trans_noise, self.rot_noise = odom_noise (0.01, math.radians(1)))
VIRTUAL_WALL_RESOLUTION_FACTOR = 2  # 虚拟墙分辨率因子
VIRTUAL_WALL_Y_OFFSET = -1  # 虚拟墙Y方向偏移

# 激光雷达参数
LIDAR_MAX_RANGE = 12.0  # 激光雷达扫描半径
LIDAR_ANGLE_RESOLUTION = 3.0  # 激光雷达角度分辨率（度）：改为每3度一束，约120束
LIDAR_NOISE = 0.03  # 激光雷达噪声

# 出口检测参数
MIN_NO_OBSTACLE_COUNT = 34  # 无障碍点数阈值，超过此数值认为走出迷宫

# 探索阈值参数
MIN_EXPLORATION_DISTANCE = 30.0  # 最小探索距离阈值
MIN_FRONTIERS_TO_EXPLORE = 10  # 最小探索前沿数量

# 运动控制精度参数
ROTATION_THRESHOLD = 1e-3  # 旋转角度阈值
MOVEMENT_THRESHOLD = 1e-6  # 移动距离阈值

# 速度底线配置
EXPLORE_MIN_SPEED = 0.24  # 探索阶段的最小前进速度

# 卡住判定与挤出恢复参数
STUCK_WINDOW_STEPS = 12             # 判定窗口步数
STUCK_SPIN_W_THRESH = 0.9           # 认为“原地打转”的角速度阈值(rad/s)
STUCK_V_SMALL = 0.05                # 认为“几乎不前进”的线速度阈值(m/s)
STUCK_PROGRESS_EPS = 0.10           # 判定窗口内总位移阈值(m)
STUCK_COOLDOWN_STEPS = 35           # 一次挤出后冷却步数，避免频繁触发
RECOVERY_STEP_LIMIT = 22            # 挤出模式最多持续步数
RECOVERY_MIN_ADVANCE = 0.25         # 挤出最小前进距离(m)
RECOVERY_MAX_ADVANCE = 0.80         # 挤出最大前进距离(m)
RECOVERY_EXTRA_MARGIN = 0.05        # 在膨胀半径基础上额外预留的安全裕度(m)

# 可视化参数
VISUALIZATION_PAUSE_TIME = 0.005  # 暂停时的等待时间
VISUALIZATION_UPDATE_TIME = 0.0001  # 可视化更新时间

# 未探索区域搜索参数
OBSTACLE_SEARCH_EXPANSION = 0.5  # 障碍物区域搜索范围扩大距离（米）

# 前沿探索节流参数
FRONTIER_LONG_PATH_THRESHOLD_CELLS = 120  # A*规划路径超过该栅格数，则触发短期冷却
FRONTIER_COOLDOWN_STEPS = 40              # 冷却期间暂停A*与前沿刷新（冻结提示）

# ==================== 降噪滤波参数 ====================
# 滤波器总开关
NOISE_FILTER_ENABLED = False  # 是否启用降噪滤波器

# 激光雷达降噪参数
LIDAR_FILTER_ENABLED = False  # 是否启用激光雷达降噪
LIDAR_FILTER_TYPE = 'gaussian'  # 激光雷达滤波类型: 'none', 'median', 'moving_average', 'gaussian'
LIDAR_FILTER_WINDOW_SIZE = 5  # 激光雷达滤波窗口大小

# 里程计降噪参数  
ODOM_FILTER_ENABLED = False  # 是否启用里程计降噪
ODOM_FILTER_TYPE = 'none'  # 里程计滤波类型: 'none', 'kalman', 'moving_average'
ODOM_FILTER_WINDOW_SIZE = 3  # 里程计滤波窗口大小

def check_exit_condition(scan, max_range=12.0, min_no_obstacle_count=97):
    """
    检查机器人是否走出迷宫
    简化判断条件：无障碍点数为97及以上即判定为在终点

    Args:
        scan: 激光雷达扫描数据列表
        max_range: 激光雷达最大探测距离
        min_no_obstacle_count: 最小无障碍点数阈值，超过此数值认为走出迷宫
    
    Returns:
        bool: True表示已走出迷宫，False表示仍在迷宫内
    """
    if not scan or len(scan) == 0:
        return False
    
    # 将扫描数据转换为布尔数组，True表示该角度没有检测到障碍物
    no_obstacle = [dist >= max_range for dist in scan]
    
    # 统计无障碍点的数量
    no_obstacle_count = sum(no_obstacle)
    total_points = len(scan)
    no_obstacle_ratio = no_obstacle_count / total_points
    
    # 添加调试信息
    #print(f"[DEBUG] 扫描点总数: {total_points}, 无障碍点数: {no_obstacle_count}, 比例: {no_obstacle_ratio*100:.1f}%")
    
    # 简化判断条件：无障碍点数为110及以上
    result = no_obstacle_count >= min_no_obstacle_count
    
    if result:
        print(f"[EXIT DETECTED] 检测到出口！")
        print(f"[EXIT DETECTED] - 无障碍点数: {no_obstacle_count} >= {min_no_obstacle_count}")
        print(f"[EXIT DETECTED] - 无障碍比例: {no_obstacle_ratio*100:.1f}%")
    
    return result

def main():
    # 检查是否有环境变量控制GPU使用
    device_type = os.environ.get("DEVICE_TYPE", "auto").lower()
    if device_type == "cpu":
        os.environ["FORCE_CPU"] = "1"
        print("由环境变量设置使用CPU模式")
    elif device_type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("由环境变量设置优先使用MPS (Apple Metal)")
    
    # 1. 加载迷宫地图和参数
    loader = MazeLoader()
    maze = loader.load(MAZE_FILE)  # 加载默认迷宫

    # === 自动生成入口虚拟墙壁 ===
    start_x, start_y = maze.start
    min_x, min_y, max_x, max_y = maze.bounds
    eps = maze.resolution * VIRTUAL_WALL_RESOLUTION_FACTOR
    is_left = abs(start_x - min_x) < eps
    is_bottom = abs(start_y - min_y) < eps
    virtual_walls = []
    # 自动查找入口左右两侧最近的墙端点
    left_candidates = []
    right_candidates = []
    for wall in maze.walls:
        for pt in wall:
            if abs(pt[1] - start_y) < eps:
                if pt[0] < start_x:
                    left_candidates.append(pt)
                elif pt[0] > start_x:
                    right_candidates.append(pt)
    left_wall_x = max(left_candidates, default=(start_x - 1,))[0] if left_candidates else start_x - 1
    right_wall_x = min(right_candidates, default=(start_x + 1,))[0] if right_candidates else start_x + 1
    print(left_wall_x, right_wall_x)
    # 生成虚拟墙
    virtual_x1 = left_wall_x 
    virtual_x2 = right_wall_x 
    virtual_y = start_y + VIRTUAL_WALL_Y_OFFSET
    virtual_walls.append(((virtual_x1, virtual_y), (virtual_x2, virtual_y)))
    virtual_walls.append(((virtual_x1, virtual_y), (virtual_x1, start_y)))
    virtual_walls.append(((virtual_x2, virtual_y), (virtual_x2, start_y)))
    # 添加虚拟墙到maze.walls
    maze.walls.extend(virtual_walls)

    # 2. 初始化机器人、传感器、SLAM等模块
    # 初始朝向设为 pi/2 （朝向y正方向：向上）
    start_pose = (start_x, start_y, math.pi/2)
    robot = Robot(start_pose, odom_noise=ROBOT_ODOM_NOISE)  # 设置一定里程计噪声
    lidar = Lidar(maze.walls, max_range=LIDAR_MAX_RANGE, angle_resolution=LIDAR_ANGLE_RESOLUTION, noise=LIDAR_NOISE)
    slam = ICPSlam(maze, start_pose)
    explorer = FrontierExplorer(safety_distance=FRONTIER_SAFETY_DISTANCE)  # 设置与障碍物的安全距离
    viz = Visualizer(maze, robot=robot, slam=slam)
    explore_traj_style = {
        "color": "orange",
        "linewidth": 1.2,
        "alpha": 0.85,
        "label": "Explore Traj"
    }
    return_traj_style = {
        "color": "deepskyblue",
        "linewidth": 1.2,
        "alpha": 0.85,
        "label": "Return Traj"
    }
    def log_section_times(tag, timings):
        if not timings:
            return
        # 过滤掉字典类型的值，只显示数值型timing
        summary = " | ".join(f"{name}={elapsed:.2f}ms" for name, elapsed in timings if isinstance(elapsed, (int, float)))
        # 展开 DWA 内部详细统计（如果存在）
        if "dwa_plan" in dict(timings):
            dwa_detail = None
            for name, _ in timings:
                if name == "dwa_planner_detail":
                    dwa_detail = _
                    break
            if dwa_detail and isinstance(dwa_detail, dict):
                dwa_breakdown = " | ".join(f"dwa_{k}={v:.2f}ms" for k, v in dwa_detail.items() if isinstance(v, (int, float)))
                summary += f" | {dwa_breakdown}"
        print(f"[Timing] {tag} {summary}")
    return_traj_split_idx = None
    
    # 初始化降噪滤波器
    noise_filter = NoiseFilter(
        enabled=NOISE_FILTER_ENABLED,
        lidar_filter_enabled=LIDAR_FILTER_ENABLED,
        odom_filter_enabled=ODOM_FILTER_ENABLED,
        lidar_filter_type=LIDAR_FILTER_TYPE,
        odom_filter_type=ODOM_FILTER_TYPE,
        lidar_window_size=LIDAR_FILTER_WINDOW_SIZE,
        odom_window_size=ODOM_FILTER_WINDOW_SIZE
    )
    
    # 将滤波器设置到传感器和机器人
    robot.set_noise_filter(noise_filter)
    lidar.set_noise_filter(noise_filter)
    
    # 输出滤波器状态信息
    if NOISE_FILTER_ENABLED:
        filter_status = noise_filter.get_status()
        print(f"[降噪滤波器] 状态: {filter_status}")
    else:
        print("[降噪滤波器] 滤波器已禁用")
    # 3. 初始扫描并建立初始地图
    noisy, clean = lidar.scan(robot.get_pose())
    scan = noisy
    # 打印一次雷达束数用于验证分辨率变更
    try:
        print(f"[LIDAR] beams={len(scan)}  resolution={LIDAR_ANGLE_RESOLUTION}°  (expect≈{int(360/LIDAR_ANGLE_RESOLUTION)})")
    except Exception:
        pass
    est_pose = slam.update((0.0, 0.0), scan)  # 使用SLAM返回的估计位姿
    # 初始可视化
    robot_pose = robot.get_pose()
    # 初始不再寻找前沿
    frontiers = None
    target_cell = None  # 当前局部自由目标格（其邻居含未知）
    path = None         # 到该自由格的路径（A*或BFS重建）
    # 初始绘制：此时尚未创建DWA实例，先不显示机器人半径
    viz.update(est_pose, scan, frontiers=None, target=None, path=None, occupancy=slam.get_occupancy(),
               predicted_traj=None, robot_radius=None, actual_traj=robot.trajectory,
               actual_traj_style=explore_traj_style)
    
    # 初始化探索状态标志和探索进度跟踪
    exploration_complete = False
    total_distance_traveled = 0.0  # 总移动距离
    # 不再使用前沿计数与截断
    frontiers_explored = 0
    min_exploration_distance = MIN_EXPLORATION_DISTANCE
    arrival_threshold = 0.3  # 到局部目标判定（米）
    reselect_threshold = 1.2  # 路径长度倍数触发重新搜索
    same_target_max = 20      # 连续同目标次数阈值
    stagnation_steps = 40     # 速度或位移停滞步数
    min_progress_dist = 0.08  # 判定前进的最小距离
    last_progress_pos = (robot.x, robot.y)
    stagnation_counter = 0
    same_target_counter = 0
    # 目标/路径不再缓存，每步重新搜索
    prev_target_cell = None
    same_target_counter = 0  # 重用统计：连续获得相同目标的次数
    step_counter = 0
    current_path = None  # 本步BFS得到的路径（含起点与目标自由格）
    # 前沿节流控制
    frontier_cooldown_steps = 0
    current_unknown_neighbor = None  # 保存最近一次前沿搜索得到的未知邻居
    frontier_hint_cell = None        # 用于A*失败或冷却时作为DWA提示/跟随目标

    # 初始设置占位，待 simple_cfg 创建后再依据机器人尺寸重新计算
    dynamic_wall_margin = 1  # cells (placeholder)

    # 预分配 BFS 访问信息，避免每步创建大量 Python 列表/字典造成内存膨胀
    occ0 = slam.get_occupancy()
    H, W = occ0.shape
    visited_np = np.zeros((H, W), dtype=np.uint8)  # 0/1 标记
    parent_x = np.full((H, W), -1, dtype=np.int32)
    parent_y = np.full((H, W), -1, dtype=np.int32)

    # 最近未知搜索函数（BFS在空闲区域上扩展，一旦邻接未知返回）
    def find_nearest_unexplored(occupancy, start):
        h, w = occupancy.shape
        sx, sy = int(start[0]), int(start[1])
        if not (0 <= sx < w and 0 <= sy < h):
            return None, None, None
        # 复用并清零访问标记；父指针仅在访问到的格子上写入
        visited_np.fill(0)
        dq = deque()
        dq.append((sx, sy))
        visited_np[sy, sx] = 1
        parent_x[sy, sx] = -1
        parent_y[sy, sx] = -1
        directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
        # 计算一定格子范围内是否存在墙体（占据）
        def is_safe_cell(x, y):
            for dx in range(-dynamic_wall_margin, dynamic_wall_margin+1):
                for dy in range(-dynamic_wall_margin, dynamic_wall_margin+1):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < w and 0 <= ny < h and occupancy[ny, nx] == 1:
                        return False
            return True
        fallback_candidate = None
        while dq:
            x, y = dq.popleft()
            # 检查邻居是否含未知
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h and occupancy[ny, nx] == -1:  # 邻居未知 -> 候选
                    # 回溯路径（使用父指针）
                    path_cells = []
                    cx, cy = x, y
                    while cx != -1 and cy != -1:
                        path_cells.append((cx, cy))
                        pcx, pcy = parent_x[cy, cx], parent_y[cy, cx]
                        if pcx == -1 and pcy == -1:
                            break
                        cx, cy = pcx, pcy
                    path_cells.reverse()
                    if len(path_cells) == 0 or path_cells[0] != (sx, sy):
                        path_cells.insert(0, (sx, sy))
                    # 优先返回满足安全距离的自由格；否则记录一个回退候选
                    if is_safe_cell(x, y):
                        return (x, y), (nx, ny), path_cells
                    else:
                        if fallback_candidate is None:
                            fallback_candidate = ((x, y), (nx, ny), path_cells)
            # 扩展自由格
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h and visited_np[ny, nx] == 0 and occupancy[ny, nx] == 0:
                    visited_np[ny, nx] = 1
                    parent_x[ny, nx] = x
                    parent_y[ny, nx] = y
                    dq.append((nx, ny))
        # 若未找到安全候选但存在回退候选，则使用之（打印提示）
        if fallback_candidate is not None:
            if step_counter % 25 == 0:
                print(f"[警告] 未找到满足安全距离的目标，使用靠墙回退候选 {fallback_candidate[0]} margin={dynamic_wall_margin}cells")
            return fallback_candidate
        return None, None, None
    
    # 直接使用集中后的默认配置即可（原工厂函数已合并为默认值）
    dwa_cfg = DWAConfig()
    dwa_planner = DWAPlanner(dwa_cfg)
    base_margin_m = dwa_cfg.robot_radius + dwa_cfg.safety_clearance
    dynamic_wall_margin = max(1, int(base_margin_m / maze.resolution) + 1)
    if dwa_cfg.debug:
        print(f"[DWA模式=orig] 安全格距离: {dynamic_wall_margin} (格长={maze.resolution:.2f}m)")

    # --- 卡住检测/挤出恢复 状态 ---
    cmd_hist = deque(maxlen=STUCK_WINDOW_STEPS)   # (v_cmd, w_cmd, |d_trans|)
    disp_hist = deque(maxlen=STUCK_WINDOW_STEPS)  # |d_trans|
    stuck_cooldown_steps = 0
    recovery_active = False
    recovery_steps_left = 0
    recovery_target_world = None  # (tx, ty)
    recovery_target_cell = None   # 用于可视化

    # 4. 前沿探索主循环
    # 全局路径与前瞻步长（用于DWA参考）
    global_path = None
    base_lookahead_steps = 10  # 默认前瞻栅格数
    min_lookahead_steps = 4     # 弯曲段时的最小前瞻
    max_lookahead_steps = 25    # 直线段时的最大前瞻

    def compute_dynamic_lookahead(path_cells, current_idx,
                                  min_steps=min_lookahead_steps,
                                  max_steps=max_lookahead_steps):
        """根据局部路径曲率自适应选择前瞻步数。"""
        if not path_cells or len(path_cells) <= 1:
            return 1
        candidate = base_lookahead_steps
        start = max(0, current_idx - 1)
        end = min(len(path_cells) - 1, current_idx + 6)
        headings = []
        for i in range(start, end):
            x0, y0 = path_cells[i]
            x1, y1 = path_cells[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            if dx == 0 and dy == 0:
                continue
            headings.append(math.atan2(dy, dx))
        if len(headings) >= 2:
            diffs = []
            for i in range(len(headings) - 1):
                diff = math.atan2(
                    math.sin(headings[i + 1] - headings[i]),
                    math.cos(headings[i + 1] - headings[i])
                )
                diffs.append(abs(diff))
            if diffs:
                avg_turn = sum(diffs) / len(diffs)
                max_turn = max(diffs)
                curvature = 0.6 * avg_turn + 0.4 * max_turn
                straight_threshold = math.radians(8.0)
                curve_threshold = math.radians(35.0)
                if curvature <= straight_threshold:
                    factor = 0.0
                elif curvature >= curve_threshold:
                    factor = 1.0
                else:
                    factor = ((curvature - straight_threshold) /
                              (curve_threshold - straight_threshold))
                candidate = max_steps - factor * (max_steps - min_steps)
                candidate = int(round(candidate))
        candidate = max(min_steps, min(max_steps, candidate))
        remaining = len(path_cells) - 1 - current_idx
        if remaining <= 0:
            return 1
        candidate = min(candidate, remaining)
        return max(1, candidate)
    while True:
        # 1. 暂停检查
        if viz.paused:
            plt.pause(VISUALIZATION_PAUSE_TIME)
            continue
        loop_step_label = f"step={step_counter}"
        loop_start = time.perf_counter()
        section_times = []
        t_section = loop_start
        # 2. 采样扫描并更新探索距离条件
        noisy, clean = lidar.scan(robot.get_pose())
        scan = noisy
        has_sufficient_exploration = (total_distance_traveled >= min_exploration_distance)
        section_times.append(("scan", (time.perf_counter() - t_section) * 1000.0))
        t_section = time.perf_counter()
        # 3. 出口检测
        exit_triggered = False
        if has_sufficient_exploration and check_exit_condition(scan, max_range=lidar.max_range, min_no_obstacle_count=MIN_NO_OBSTACLE_COUNT):
            print(f"检测到超过180度的连续无障碍区域 - 已探索距离: {total_distance_traveled:.1f}m")
            est_pose = slam.update((0.0, 0.0), scan)
            viz.update(est_pose, scan, frontiers=None, target=None, path=None, occupancy=slam.get_occupancy(),
                       actual_traj=robot.trajectory, actual_traj_style=explore_traj_style)
            should_return_to_start = True
            exit_triggered = True
        section_times.append(("exit_check_pre", (time.perf_counter() - t_section) * 1000.0))
        t_section = time.perf_counter()
        if exit_triggered:
            section_times.append(("loop_total", (time.perf_counter() - loop_start) * 1000.0))
            log_section_times(loop_step_label, section_times)
            break
        # 4. 前沿与路径更新（增加节流逻辑）
        # 使用SLAM估计位姿计算所在栅格
        rx_idx = int((est_pose[0] - maze.bounds[0]) / maze.resolution)
        ry_idx = int((est_pose[1] - maze.bounds[1]) / maze.resolution)

        # 冷却策略：冷却期暂停A*与前沿刷新；冷却结束时刷新前沿并运行A*
        occupancy = slam.get_occupancy()
        if frontier_cooldown_steps == 0:
            # 刷新最近前沿与未知邻居（仅在非冷却期）
            target_cell_latest, unknown_neighbor_new, _ = find_nearest_unexplored(occupancy, (rx_idx, ry_idx))
            if target_cell_latest is None:
                print("没有可达的未知区域，探索结束。")
                section_times.append(("frontier_update", (time.perf_counter() - t_section) * 1000.0))
                section_times.append(("loop_total", (time.perf_counter() - loop_start) * 1000.0))
                log_section_times(loop_step_label, section_times)
                break
            frontier_hint_cell = target_cell_latest
            current_unknown_neighbor = unknown_neighbor_new
            if target_cell_latest == prev_target_cell:
                same_target_counter += 1
            else:
                same_target_counter = 0
            prev_target_cell = target_cell_latest

            # 运行A*（仅在非冷却期）
            safety_cells = max(1, int((dwa_cfg.robot_radius + dwa_cfg.safety_clearance) / maze.resolution))
            global_path = explorer.plan_path(occupancy, (rx_idx, ry_idx), target_cell_latest, safety_distance=safety_cells)
            current_path = global_path if global_path else None
            if global_path and (len(global_path) - 1) > FRONTIER_LONG_PATH_THRESHOLD_CELLS:
                frontier_cooldown_steps = FRONTIER_COOLDOWN_STEPS
                if step_counter % 20 == 0:
                    print(f"[前沿节流] A*路径过长({len(global_path)-1}格) -> 冷却 {FRONTIER_COOLDOWN_STEPS} 步（期间暂停A*与前沿刷新）")
        else:
            # 冷却中：不刷新前沿、不运行A*，仅递减计数器并沿旧提示/路径前进
            frontier_cooldown_steps = max(0, frontier_cooldown_steps - 1)
            if step_counter % 20 == 0:
                if current_path:
                    msg_len = f"旧A*路径长度={len(current_path)-1}格，使用旧路径提示"
                elif frontier_hint_cell is not None:
                    msg_len = "无A*路径，使用冻结的前沿提示"
                else:
                    msg_len = "无A*路径与前沿提示，使用当前位置作为临时目标"
                print(f"[前沿节流] 冷却中({frontier_cooldown_steps}步剩余)，暂停A*与前沿刷新，{msg_len}")
        section_times.append(("frontier_update", (time.perf_counter() - t_section) * 1000.0))
        t_section = time.perf_counter()

        # 选择DWA子目标：若存在A*路径（无论是否冷却中）均基于路径最近点+前瞻；否则用前沿提示
        if current_path and len(current_path) > 1:
            # 找到离当前机器人格子最近的路径索引
            nearest_idx = 0
            if len(current_path) > 8:
                # 简单全路径最近搜索；路径较长也可接受，必要时可窗口化优化
                min_d2 = 1e18
                for i, (px, py) in enumerate(current_path):
                    dx = px - rx_idx
                    dy = py - ry_idx
                    d2 = dx*dx + dy*dy
                    if d2 < min_d2:
                        min_d2 = d2
                        nearest_idx = i
            dynamic_steps = compute_dynamic_lookahead(current_path, nearest_idx)
            follow_idx = min(len(current_path) - 1, nearest_idx + dynamic_steps)
            follow_cell = current_path[follow_idx]
        else:
            follow_cell = frontier_hint_cell if frontier_hint_cell is not None else (rx_idx, ry_idx)
        # 若处于挤出模式，覆盖可视化目标为挤出目标
        viz_target_cell = None
        if recovery_active and recovery_target_world is not None:
            r_ix = int((recovery_target_world[0] - maze.bounds[0]) / maze.resolution)
            r_iy = int((recovery_target_world[1] - maze.bounds[1]) / maze.resolution)
            recovery_target_cell = (r_ix, r_iy)
            viz_target_cell = recovery_target_cell
        else:
            viz_target_cell = follow_cell
        if step_counter % 20 == 0 or same_target_counter == 0:
            print(f"[目标] 自由格: {prev_target_cell} 邻接未知: {current_unknown_neighbor} (忽略A*:{'是' if frontier_cooldown_steps>0 or current_path is None else '否'}) 连续相同={same_target_counter}")
        section_times.append(("target_selection", (time.perf_counter() - t_section) * 1000.0))
        t_section = time.perf_counter()

        # 将跟随的栅格点转换为世界坐标，作为DWA目标点（挤出模式下由临时目标覆盖）
        if recovery_active and recovery_target_world is not None:
            gx, gy = float(recovery_target_world[0]), float(recovery_target_world[1])
        else:
            gx = maze.bounds[0] + (follow_cell[0] + 0.5) * maze.resolution
            gy = maze.bounds[1] + (follow_cell[1] + 0.5) * maze.resolution

        obstacles = occupancy_to_obstacles(
            slam.get_occupancy(),
            maze.bounds,
            maze.resolution,
            stride=3
        )
        # DWA路径提示：若处于挤出模式则使用“当前位置→临时目标”直线；否则若存在A*路径（无论是否冷却中）则使用之；否则使用“当前位置→前沿提示”的直线
        path_hint_world = None
        if recovery_active and recovery_target_world is not None:
            path_hint_world = np.asarray([(est_pose[0], est_pose[1]), (gx, gy)], dtype=float)
        else:
            if current_path and len(current_path) >= 2:
                pts = []
                for cx, cy in current_path:
                    wx = maze.bounds[0] + (cx + 0.5) * maze.resolution
                    wy = maze.bounds[1] + (cy + 0.5) * maze.resolution
                    pts.append((wx, wy))
                path_hint_world = np.asarray(pts, dtype=float)
            elif frontier_hint_cell is not None:
                fx = maze.bounds[0] + (frontier_hint_cell[0] + 0.5) * maze.resolution
                fy = maze.bounds[1] + (frontier_hint_cell[1] + 0.5) * maze.resolution
                path_hint_world = np.asarray([(est_pose[0], est_pose[1]), (fx, fy)], dtype=float)
        section_times.append(("path_prepare", (time.perf_counter() - t_section) * 1000.0))
        t_section = time.perf_counter()
        # --- DWA 执行 (保持在 while True 循环内) ---
        # 使用SLAM估计位姿 + 机器人当前速度作为DWA状态
        state = np.array([
            est_pose[0],
            est_pose[1],
            est_pose[2],
            robot.linear_vel,
            robot.angular_vel
        ])
        (v_cmd, w_cmd), _traj = dwa_planner.plan(state, (gx, gy), obstacles, path_hint=path_hint_world)

        dwa_elapsed = (time.perf_counter() - t_section) * 1000.0
        section_times.append(("dwa_plan", dwa_elapsed))
        dwa_detail = getattr(dwa_planner, "last_timing", None)
        if isinstance(dwa_detail, dict) and dwa_detail:
            detail_order = [
                ("pre_calc", "dwa_pre"),
                ("dynamic_window", "dwa_dw"),
                ("sample_main", "dwa_sample"),
                ("sample_relax", "dwa_resample"),
                ("smoothing", "dwa_smooth"),
                ("post_update", "dwa_post"),
                ("total", "dwa_internal_total")
            ]
            for key, label in detail_order:
                if key in dwa_detail:
                    section_times.append((label, float(dwa_detail[key])))
        t_section = time.perf_counter()

        if v_cmd > 1e-6:
            dw_max = None
            if hasattr(dwa_planner, "last_dw") and isinstance(dwa_planner.last_dw, (list, tuple)) and len(dwa_planner.last_dw) >= 2:
                dw_max = dwa_planner.last_dw[1]
            max_allow = dw_max if (dw_max is not None and math.isfinite(dw_max)) else EXPLORE_MIN_SPEED
            boost_speed = min(EXPLORE_MIN_SPEED, max_allow)
            dist_to_goal = math.hypot(est_pose[0] - gx, est_pose[1] - gy)
            if boost_speed > v_cmd and dist_to_goal > 0.6:
                v_cmd = boost_speed
                dwa_planner._last_u = (v_cmd, w_cmd)
                if _traj is not None:
                    _traj = dwa_planner._predict_trajectory(state, v_cmd, w_cmd)
                if step_counter % 40 == 0:
                    dw_max_disp = dw_max if (dw_max is not None and math.isfinite(dw_max)) else float('nan')
                    print(f"[探索] 提升前进速度 -> {v_cmd:.2f} m/s (dw_max={dw_max_disp:.2f} dist_to_goal={dist_to_goal:.2f}m)")

        # 先用当前估计位姿绘制预测轨迹（起点一致，避免视觉错位）
        viz.update(
            est_pose,
            scan,
            frontiers=None,
            target=viz_target_cell,
            path=current_path,
            occupancy=slam.get_occupancy(),
            predicted_traj=_traj,
            robot_radius=dwa_planner.cfg.robot_radius,
            actual_traj=robot.trajectory,
            actual_traj_style=explore_traj_style
        )
        section_times.append(("visualize", (time.perf_counter() - t_section) * 1000.0))
        t_section = time.perf_counter()
        # 添加 DWA 详细统计
        if hasattr(dwa_planner, 'last_timing') and isinstance(dwa_planner.last_timing, dict):
            section_times.append(("dwa_planner_detail", dwa_planner.last_timing.copy()))
        
        # 再执行控制并更新SLAM
        d_trans, d_rot = robot.velocity_step(float(v_cmd), float(w_cmd), float(dwa_cfg.dt))
        if (step_counter % 20 == 0) or (v_cmd < -1e-3):
            print(f"[CTRL] step={step_counter} v={float(v_cmd):.2f} w={float(w_cmd):.2f} d_trans={d_trans:.3f} d_rot={d_rot:.3f}")
        total_distance_traveled += abs(d_trans)
        noisy, clean = lidar.scan(robot.get_pose())
        scan = noisy
        est_pose = slam.update((d_trans, d_rot), scan)
        section_times.append(("motion_update", (time.perf_counter() - t_section) * 1000.0))
        t_section = time.perf_counter()

        step_counter += 1
        if dwa_cfg.debug and step_counter % 15 == 0:
            print(f"[DWA] step={step_counter} v={v_cmd:.2f} w={w_cmd:.2f}")

        # --- 卡住检测数据更新 ---
        cmd_hist.append((float(v_cmd), float(w_cmd), abs(d_trans)))
        disp_hist.append(abs(d_trans))

        step_progress = math.hypot(robot.x - last_progress_pos[0], robot.y - last_progress_pos[1])
        if step_progress > min_progress_dist:
            last_progress_pos = (robot.x, robot.y)
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # --- 挤出恢复：检测原地打转/微小进退，朝最远净空方向短距离移动 ---
        if recovery_active and recovery_target_world is not None:
            # 到达或超时则结束挤出
            dist_to_recover = math.hypot(robot.x - recovery_target_world[0], robot.y - recovery_target_world[1])
            recovery_steps_left = max(0, recovery_steps_left - 1)
            if dist_to_recover < 0.20 or recovery_steps_left == 0:
                print(f"[恢复-完成] 挤出结束，剩余步={recovery_steps_left} dist={dist_to_recover:.2f}m")
                recovery_active = False
                recovery_target_world = None
                recovery_target_cell = None
                stuck_cooldown_steps = STUCK_COOLDOWN_STEPS
        else:
            if stuck_cooldown_steps > 0:
                stuck_cooldown_steps -= 1
            # 仅在未处于挤出模式时评估是否卡住
            spin_like = False
            tiny_progress = False
            if len(cmd_hist) == STUCK_WINDOW_STEPS:
                mean_w = sum(abs(w) for _, w, _ in cmd_hist) / STUCK_WINDOW_STEPS
                mean_v = sum(abs(v) for v, _, _ in cmd_hist) / STUCK_WINDOW_STEPS
                spin_like = (mean_w > STUCK_SPIN_W_THRESH and mean_v < STUCK_V_SMALL)
            if len(disp_hist) == STUCK_WINDOW_STEPS:
                total_disp = sum(disp_hist)
                tiny_progress = (total_disp < STUCK_PROGRESS_EPS)
            should_recover = ((stagnation_counter > stagnation_steps and same_target_counter > same_target_max) or
                              (stagnation_counter > 2 * stagnation_steps) or spin_like or tiny_progress)
            if should_recover and stuck_cooldown_steps == 0:
                # 选取当前扫描中最远净空方向
                angle_step = math.radians(LIDAR_ANGLE_RESOLUTION)
                try:
                    idx = int(np.argmax(clean)) if len(clean) > 0 else 0
                    max_clear = float(clean[idx]) if len(clean) > 0 else LIDAR_MAX_RANGE
                except Exception:
                    idx = 0
                    max_clear = LIDAR_MAX_RANGE
                rx, ry, rth = robot.get_pose()
                ang_world = rth + idx * angle_step
                safe_margin = base_margin_m + RECOVERY_EXTRA_MARGIN
                advance = max(0.0, max_clear - safe_margin)
                advance = max(0.0, min(RECOVERY_MAX_ADVANCE, advance))
                if advance >= RECOVERY_MIN_ADVANCE:
                    tx = rx + advance * math.cos(ang_world)
                    ty = ry + advance * math.sin(ang_world)
                    recovery_target_world = (tx, ty)
                    recovery_steps_left = RECOVERY_STEP_LIMIT
                    recovery_active = True
                    print(f"[恢复-挤出] 卡住判定(spin={spin_like}, disp<={STUCK_PROGRESS_EPS:.2f}m:{tiny_progress}) -> 朝 {idx*LIDAR_ANGLE_RESOLUTION:.0f}° 方向移动 {advance:.2f}m")
                    # 重置滞留相关计数，下一步开始执行挤出
                    stagnation_counter = 0
                    same_target_counter = 0
                else:
                    # 无足够净空，跳过本次挤出并进入短冷却
                    print(f"[恢复-挤出] 判定卡住但净空不足(max={max_clear:.2f}m, 需>{safe_margin+RECOVERY_MIN_ADVANCE:.2f}m)，跳过")
                    stuck_cooldown_steps = max(stuck_cooldown_steps, STUCK_WINDOW_STEPS)
        section_times.append(("stuck_recovery", (time.perf_counter() - t_section) * 1000.0))
        t_section = time.perf_counter()
        # 7. 再次出口检测（单步后）
        exit_triggered_post = False
        if total_distance_traveled >= min_exploration_distance and check_exit_condition(scan, max_range=lidar.max_range, min_no_obstacle_count=MIN_NO_OBSTACLE_COUNT):
            print("DWA控制过程中检测到超过180度连续无障碍区域！")
            should_return_to_start = True
            exit_triggered_post = True
        section_times.append(("exit_check_post", (time.perf_counter() - t_section) * 1000.0))
        t_section = time.perf_counter()
        if exit_triggered_post:
            section_times.append(("loop_total", (time.perf_counter() - loop_start) * 1000.0))
            log_section_times(loop_step_label, section_times)
            break
        section_times.append(("loop_total", (time.perf_counter() - loop_start) * 1000.0))
        log_section_times(loop_step_label, section_times)
                
    # 根据探索结束的原因决定后续行为
    if 'should_return_to_start' in locals() and should_return_to_start:
        print(f"探索过程中检测到超过180度连续无障碍区域，现在检查是否有未探索区域...")
        print(f"探索总结 - 总移动距离: {total_distance_traveled:.1f}m, 总共探索了 {frontiers_explored} 个前沿点")
        
        # 计算已知障碍物区域的边界
        occupancy = slam.get_occupancy()
        obstacle_coords = []
        
        # 找到所有已知障碍物（值为1）的坐标
        for y in range(occupancy.shape[0]):
            for x in range(occupancy.shape[1]):
                if occupancy[y, x] == 1:  # 障碍物
                    # 转换为世界坐标
                    world_x = maze.bounds[0] + x * maze.resolution
                    world_y = maze.bounds[1] + y * maze.resolution
                    obstacle_coords.append((world_x, world_y))
        
        if obstacle_coords:
            # 计算障碍物区域的最小和最大坐标，并扩大搜索范围
            min_obstacle_x = min(coord[0] for coord in obstacle_coords) - OBSTACLE_SEARCH_EXPANSION  # 扩大范围
            max_obstacle_x = max(coord[0] for coord in obstacle_coords) + OBSTACLE_SEARCH_EXPANSION
            min_obstacle_y = min(coord[1] for coord in obstacle_coords) - OBSTACLE_SEARCH_EXPANSION
            max_obstacle_y = max(coord[1] for coord in obstacle_coords) + OBSTACLE_SEARCH_EXPANSION
            print("检测到障碍物区域，边界如下：")
            print(f"障碍物区域边界: X[{min_obstacle_x:.1f}, {max_obstacle_x:.1f}], Y[{min_obstacle_y:.1f}, {max_obstacle_y:.1f}]")
            
            # 转换为栅格索引
            min_obs_x_idx = int((min_obstacle_x - maze.bounds[0]) / maze.resolution)
            max_obs_x_idx = int((max_obstacle_x - maze.bounds[0]) / maze.resolution)
            min_obs_y_idx = int((min_obstacle_y - maze.bounds[1]) / maze.resolution)
            max_obs_y_idx = int((max_obstacle_y - maze.bounds[1]) / maze.resolution)
            
            # 在障碍物边界范围内寻找未探索区域
            unexplored_in_range = []
            for y in range(max(0, min_obs_y_idx), min(occupancy.shape[0], max_obs_y_idx + 1)):
                for x in range(max(0, min_obs_x_idx), min(occupancy.shape[1], max_obs_x_idx + 1)):
                    if occupancy[y, x] == -1:  # 未探索区域
                        unexplored_in_range.append((x, y))
            
            print(f"在障碍物边界范围内发现 {len(unexplored_in_range)} 个未探索格子")
            
            if unexplored_in_range:
                # 尝试找到最近的未探索区域并规划路径
                current_idx_x = int((robot.x - maze.bounds[0]) / maze.resolution)
                current_idx_y = int((robot.y - maze.bounds[1]) / maze.resolution)
                
                min_distance = float('inf')
                closest_unexplored = None
                
                for ux, uy in unexplored_in_range:
                    distance = math.hypot(ux - current_idx_x, uy - current_idx_y)
                    if distance < min_distance:
                        min_distance = distance
                        closest_unexplored = (ux, uy)
                
                if closest_unexplored:
                    print(f"尝试规划到最近未探索区域 {closest_unexplored} 的路径...")
                    unexplored_path = explorer.plan_path(occupancy, (current_idx_x, current_idx_y), closest_unexplored)
                    
                    if unexplored_path and len(unexplored_path) > 1:
                        print(f"找到通往未探索区域的路径，长度: {len(unexplored_path)-1} 步")
                        print("前往未探索区域进行补充扫描...")
                        
                        # 移动到未探索区域并扫描
                        for step in unexplored_path[1:]:
                            ix, iy = step
                            target_x = maze.bounds[0] + (ix + 0.5) * maze.resolution
                            target_y = maze.bounds[1] + (iy + 0.5) * maze.resolution
                            dx = target_x - robot.x
                            dy = target_y - robot.y
                            desired_theta = math.atan2(dy, dx)
                            d_theta = desired_theta - robot.theta
                            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
                            
                            if abs(d_theta) > ROTATION_THRESHOLD:
                                robot.rotate(d_theta)
                                noisy, clean = lidar.scan(robot.get_pose())
                                scan = noisy
                                slam.update((0.0, d_theta), scan)
                                viz.update(robot.get_pose(), scan, frontiers=None, target=closest_unexplored, path=unexplored_path, occupancy=slam.get_occupancy(),
                                           actual_traj=robot.trajectory, actual_traj_style=explore_traj_style)
                            
                            distance = math.hypot(target_x - robot.x, target_y - robot.y)
                            if distance > MOVEMENT_THRESHOLD:
                                robot.move(distance)
                                noisy, clean = lidar.scan(robot.get_pose())
                                scan = noisy
                                slam.update((distance, 0.0), scan)
                                viz.update(robot.get_pose(), scan, frontiers=None, target=closest_unexplored, path=unexplored_path, occupancy=slam.get_occupancy(),
                                           actual_traj=robot.trajectory, actual_traj_style=explore_traj_style)
                        
                        print("补充扫描完成，现在返回起点...")
                    else:
                        print("无法找到通往未探索区域的路径，直接返回起点")
                else:
                    print("未找到可达的未探索区域，直接返回起点")
            else:
                print("障碍物边界范围内没有未探索区域，直接返回起点")
        else:
            print("未发现障碍物区域，直接返回起点")
    else:
        print("探索完成：迷宫内部区域已完全探索。")
        print(f"探索总结 - 总移动距离: {total_distance_traveled:.1f}m, 总共探索了 {frontiers_explored} 个前沿点")
    
    # 最后返回起点
    print("规划返回起点路径...")
    start_idx_x = int((maze.start[0] - maze.bounds[0]) / maze.resolution)
    start_idx_y = int((maze.start[1] - maze.bounds[1]) / maze.resolution)
    current_idx_x = int((robot.x - maze.bounds[0]) / maze.resolution)
    current_idx_y = int((robot.y - maze.bounds[1]) / maze.resolution)
    
    # 生成带安全距离的路径（逐步放宽安全距离，确保能够回家）
    occupancy_return = slam.get_occupancy().copy()
    # 确保当前格与起点格被视为空闲，避免因为噪点被判为不可通行
    if 0 <= current_idx_y < occupancy_return.shape[0] and 0 <= current_idx_x < occupancy_return.shape[1]:
        occupancy_return[current_idx_y, current_idx_x] = 0
    if 0 <= start_idx_y < occupancy_return.shape[0] and 0 <= start_idx_x < occupancy_return.shape[1]:
        occupancy_return[start_idx_y, start_idx_x] = 0

    safety_cells_nominal = max(1, int(round((dwa_cfg.robot_radius + dwa_cfg.safety_clearance) / maze.resolution)))
    safety_cells_nominal_f = float(safety_cells_nominal)
    safety_candidates = [float(safety_cells_nominal)]
    # 若默认安全距离无解，则逐步放宽，最终退化到0栅格
    for shrink in range(safety_cells_nominal - 1, -1, -1):
        safety_candidates.append(float(shrink))

    back_path = None
    used_safety = None
    for sd in safety_candidates:
        candidate = explorer.plan_path(
            occupancy_return,
            (current_idx_x, current_idx_y),
            (start_idx_x, start_idx_y),
            safety_distance=sd
        )
        if candidate:
            back_path = candidate
            used_safety = sd
            break

    if back_path:
        used_safety_val = float(used_safety) if used_safety is not None else 0.0
        if math.isclose(used_safety_val, safety_cells_nominal_f, rel_tol=1e-6):
            print(f"使用探索阶段的安全距离 {used_safety_val:.1f} 栅格规划返程路线")
        else:
            print(f"原安全距离 {safety_cells_nominal_f:.1f} 栅格无解，改用 {used_safety_val:.1f} 栅格规划返程路线")
    else:
        print(f"无法在安全距离 {safety_cells_nominal_f:.1f}~0.0 栅格范围内规划返程路径")

    back_path_no_safety = explorer.plan_path_no_safety(
        occupancy_return,
        (current_idx_x, current_idx_y),
        (start_idx_x, start_idx_y)
    )

    if back_path_no_safety:
        path_length_meters = explorer.calculate_path_length(back_path_no_safety, maze.resolution)
        print(f"无安全距离最短路径长度: {path_length_meters:.2f} 米 ({len(back_path_no_safety)-1} 栅格步数)")

    if back_path:
        viz.set_emergency_path(back_path)
    elif back_path_no_safety:
        viz.set_emergency_path(back_path_no_safety)

    def drive_path_with_dwa(path_cells, label="返回路径"):
        nonlocal est_pose, scan, total_distance_traveled, step_counter, return_traj_split_idx
        if not path_cells or len(path_cells) < 2:
            return False
        path_cells = list(path_cells)
        path_pts = []
        for cx, cy in path_cells:
            wx = maze.bounds[0] + (cx + 0.5) * maze.resolution
            wy = maze.bounds[1] + (cy + 0.5) * maze.resolution
            path_pts.append((wx, wy))
        path_arr = np.asarray(path_pts, dtype=float)
        return_threshold = 0.18
        max_iters = max(len(path_arr) * 80, 700)
        est_pose = robot.get_pose()
        print(f"[{label}] 使用DWA沿返程路径前进，共 {len(path_arr)-1} 段，最大步数 {max_iters}")

        dwa_planner._last_u = (0.0, 0.0)
        if hasattr(dwa_planner, '_dwell_count'):
            dwa_planner._dwell_count = 0
        if hasattr(dwa_planner, '_reverse_block_count'):
            dwa_planner._reverse_block_count = 0

        return_min_speed = 0.22
        return_speed_cap = 0.75
        stall_counter = 0
        path_update_counter = 0  # 计数器：每50次更新一次路径

        for idx in range(max_iters):
            while viz.paused:
                plt.pause(VISUALIZATION_PAUSE_TIME)

            loop_label = f"{label} idx={idx}"
            loop_start = time.perf_counter()
            section_times = []
            t_section = loop_start

            est_pose = robot.get_pose()
            section_times.append(("pose_fetch", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()

            # 每50次更新一次返回路径（类似探索阶段的逻辑）
            if path_update_counter % 50 == 0:
                # 重新规划从当前位置到起点的路径
                current_idx_x_update = int((est_pose[0] - maze.bounds[0]) / maze.resolution)
                current_idx_y_update = int((est_pose[1] - maze.bounds[1]) / maze.resolution)
                occupancy_update = slam.get_occupancy().copy()
                
                # 确保当前格和起点格被视为空闲
                if 0 <= current_idx_y_update < occupancy_update.shape[0] and 0 <= current_idx_x_update < occupancy_update.shape[1]:
                    occupancy_update[current_idx_y_update, current_idx_x_update] = 0
                if 0 <= start_idx_y < occupancy_update.shape[0] and 0 <= start_idx_x < occupancy_update.shape[1]:
                    occupancy_update[start_idx_y, start_idx_x] = 0
                
                # 使用当前安全距离重新规划
                safety_cells_return = max(1, int(round((dwa_cfg.robot_radius + dwa_cfg.safety_clearance) / maze.resolution)))
                updated_path = explorer.plan_path(occupancy_update, (current_idx_x_update, current_idx_y_update), (start_idx_x, start_idx_y), safety_distance=float(safety_cells_return))
                
                if updated_path and len(updated_path) >= 2:
                    # 更新路径
                    path_cells = list(updated_path)
                    path_pts = []
                    for cx, cy in path_cells:
                        wx = maze.bounds[0] + (cx + 0.5) * maze.resolution
                        wy = maze.bounds[1] + (cy + 0.5) * maze.resolution
                        path_pts.append((wx, wy))
                    path_arr = np.asarray(path_pts, dtype=float)
                    if idx % 50 == 0:
                        print(f"[{label}] 第{idx}步：更新返回路径，新路径长度 {len(path_arr)-1} 段")
            
            path_update_counter += 1

            dists = np.hypot(path_arr[:, 0] - est_pose[0], path_arr[:, 1] - est_pose[1])
            nearest_idx = int(np.argmin(dists))
            lookahead = compute_dynamic_lookahead(path_cells, nearest_idx)
            follow_idx = min(len(path_arr) - 1, nearest_idx + lookahead)
            follow_cell = path_cells[min(len(path_cells) - 1, nearest_idx)]
            goal_point = (float(path_arr[follow_idx, 0]), float(path_arr[follow_idx, 1]))

            gx, gy = goal_point
            obstacles = occupancy_to_obstacles(slam.get_occupancy(), maze.bounds, maze.resolution, stride=3)

            # 构造与探索阶段一致的路径提示（世界坐标路径）
            path_hint_world = path_arr

            section_times.append(("target_prepare", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()

            state = np.array([est_pose[0], est_pose[1], est_pose[2], robot.linear_vel, robot.angular_vel], dtype=float)
            (v_cmd, w_cmd), traj = dwa_planner.plan(state, (gx, gy), obstacles, path_hint=path_hint_world)
            dwa_elapsed = (time.perf_counter() - t_section) * 1000.0
            section_times.append(("dwa_plan", dwa_elapsed))
            dwa_detail = getattr(dwa_planner, "last_timing", None)
            if isinstance(dwa_detail, dict) and dwa_detail:
                detail_order = [
                    ("pre_calc", "dwa_pre"),
                    ("dynamic_window", "dwa_dw"),
                    ("sample_main", "dwa_sample"),
                    ("sample_relax", "dwa_resample"),
                    ("smoothing", "dwa_smooth"),
                    ("post_update", "dwa_post"),
                    ("total", "dwa_internal_total")
                ]
                for key, label in detail_order:
                    if key in dwa_detail:
                        section_times.append((label, float(dwa_detail[key])))
            t_section = time.perf_counter()

            if v_cmd > 1e-6:
                dw_min = dw_max = None
                if hasattr(dwa_planner, 'last_dw') and isinstance(dwa_planner.last_dw, (list, tuple)) and len(dwa_planner.last_dw) >= 2:
                    dw_min, dw_max = dwa_planner.last_dw[0], dwa_planner.last_dw[1]
                dist_to_goal = math.hypot(est_pose[0] - gx, est_pose[1] - gy)
                dist_to_start = math.hypot(est_pose[0] - maze.start[0], est_pose[1] - maze.start[1])
                adaptive_floor = return_min_speed + min(0.4, dist_to_start * 0.2)
                adaptive_floor = min(return_speed_cap, adaptive_floor)
                if dw_max is not None and math.isfinite(dw_max):
                    speed_ceiling = max(0.0, min(dw_max, return_speed_cap))
                else:
                    speed_ceiling = return_speed_cap
                target_speed = 0.0
                if speed_ceiling > 0.05:
                    target_speed = max(0.0, min(adaptive_floor, speed_ceiling))
                if target_speed > v_cmd and dist_to_goal > 0.6:
                    v_cmd = target_speed
                    dwa_planner._last_u = (v_cmd, w_cmd)
                    if hasattr(dwa_planner, '_dwell_count'):
                        dwa_planner._dwell_count = 0
                    if traj is not None:
                        traj = dwa_planner._predict_trajectory(state, v_cmd, w_cmd)
                    if idx % 40 == 0:
                        dw_min_disp = dw_min if (dw_min is not None and math.isfinite(dw_min)) else float('nan')
                        dw_max_disp = dw_max if (dw_max is not None and math.isfinite(dw_max)) else float('nan')
                        print(f"[{label}] 提升前进速度 -> {v_cmd:.2f} m/s (dw=[{dw_min_disp:.2f},{dw_max_disp:.2f}] dist_to_start={dist_to_start:.2f}m)")

            if abs(v_cmd) < 0.05:
                stall_counter += 1
            else:
                stall_counter = 0

            if stall_counter >= 6:
                dist_to_start = math.hypot(est_pose[0] - maze.start[0], est_pose[1] - maze.start[1])
                dw_min = dw_max = None
                if hasattr(dwa_planner, 'last_dw') and isinstance(dwa_planner.last_dw, (list, tuple)) and len(dwa_planner.last_dw) >= 2:
                    dw_min, dw_max = dwa_planner.last_dw[0], dwa_planner.last_dw[1]
                adaptive_floor = return_min_speed + min(0.4, dist_to_start * 0.2)
                adaptive_floor = min(return_speed_cap, adaptive_floor)
                forced_speed = adaptive_floor
                if forced_speed > 0.05:
                    v_cmd = forced_speed
                    w_cmd *= 0.5
                    dwa_planner._last_u = (v_cmd, w_cmd)
                    if traj is not None:
                        traj = dwa_planner._predict_trajectory(state, v_cmd, w_cmd)
                    dw_min_disp = dw_min if (dw_min is not None and math.isfinite(dw_min)) else float('nan')
                    dw_max_disp = dw_max if (dw_max is not None and math.isfinite(dw_max)) else float('nan')
                    print(f"[{label}] 强制解除滞留 -> {v_cmd:.2f} m/s (dw=[{dw_min_disp:.2f},{dw_max_disp:.2f}] dist_to_start={dist_to_start:.2f}m)")
                    stall_counter = 0

            section_times.append(("post_plan_adjust", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()

            viz_target_cell = path_cells[min(follow_idx, len(path_cells) - 1)]
            explore_segment = []
            return_segment = robot.trajectory
            extra_trajs = None
            if return_traj_split_idx is not None:
                explore_segment = robot.trajectory[:return_traj_split_idx]
                return_segment = robot.trajectory[return_traj_split_idx:]
                if len(explore_segment) >= 2:
                    extra_trajs = [{
                        'points': explore_segment,
                        'style': explore_traj_style
                    }]
            viz.update(
                est_pose,
                scan,
                frontiers=None,
                target=viz_target_cell,
                path=path_cells,
                occupancy=slam.get_occupancy(),
                predicted_traj=traj,
                robot_radius=dwa_planner.cfg.robot_radius,
                actual_traj=return_segment,
                actual_traj_style=return_traj_style,
                extra_trajs=extra_trajs
            )

            section_times.append(("visualization", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()

            d_trans, d_rot = robot.velocity_step(float(v_cmd), float(w_cmd), float(dwa_cfg.dt))
            total_distance_traveled += abs(d_trans)
            if idx % 20 == 0 or v_cmd < -1e-3:
                print(f"[{label}] idx={idx} v={float(v_cmd):.2f} w={float(w_cmd):.2f} follow={follow_idx}/{len(path_arr)-1}")

            section_times.append(("motion_update", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()

            noisy, clean = lidar.scan(robot.get_pose())
            scan = noisy
            est_pose = robot.get_pose()
            step_counter += 1

            section_times.append(("sensor_update", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()

            dist_to_start_now = math.hypot(est_pose[0] - maze.start[0], est_pose[1] - maze.start[1])
            reached_start = dist_to_start_now < return_threshold
            section_times.append(("exit_check", (time.perf_counter() - t_section) * 1000.0))
            section_times.append(("loop_total", (time.perf_counter() - loop_start) * 1000.0))
            log_section_times(loop_label, section_times)

            if reached_start:
                print(f"[{label}] 已到达起点附近（< {return_threshold:.2f} m）")
                return True

        print(f"[{label}] 未能在限制步数内完成返回，剩余距离 {math.hypot(est_pose[0]-maze.start[0], est_pose[1]-maze.start[1]):.2f} m")
        return False

    if back_path:
        used_safety_val = float(used_safety) if used_safety is not None else 0.0
        return_traj_split_idx = len(robot.trajectory)
        returned = drive_path_with_dwa(back_path, label="安全返回路径")
        if returned:
            print("Robot returned to start.")
        else:
            print("返程DWA未能在限定步数内抵达起点。")
    else:
        print("未能规划带安全距离的返程路径，无法执行返程。")
    
    # 根据结束条件输出相应信息
    if 'should_return_to_start' in locals() and should_return_to_start:
        print("仿真结束：检测到超过180度连续无障碍区域，机器人已返回起点！")
    else:
        print("仿真结束：迷宫探索完成，机器人已返回起点。")
        
    # 导出最终地图（按需求关闭 final_path.csv 导出）
    viz.save_map("final_map.png")
    # 已禁用：不再导出最终路径 CSV
    # viz.save_path("final_path.csv")
    
    # 释放GPU资源
    if hasattr(slam, 'release_resources'):
        slam.release_resources()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if "NotImplementedError" in str(e) and "MPS" in str(e):
            print("\n发生MPS设备错误，尝试使用CPU模式重新运行...")
            print("错误详情:", str(e))
            print("\n提示: 可以设置环境变量 DEVICE_TYPE=cpu 强制使用CPU模式")
            os.environ["FORCE_CPU"] = "1"
            print("正在使用CPU模式重新启动...\n")
            main()
        else:
            # 其他类型的错误，正常抛出
            raise
