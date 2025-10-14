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

# ==================== 机器人几何参数 ====================
WHEEL_TRACK = 0.16  # 两轮中心距(m)，差分底盘轮距
LIDAR_REAR_OFFSET = 0.12  # 轮中心连线到后方激光雷达中心的距离(m)
LIDAR_RADIUS = 0.055  # 激光雷达自身半径(m)
ROBOT_BODY_DIAMETER = 0.20  # 车体直径(m)
ROBOT_BODY_RADIUS = ROBOT_BODY_DIAMETER / 2.0
ROBOT_COLLISION_RADIUS = max(ROBOT_BODY_RADIUS, LIDAR_REAR_OFFSET + LIDAR_RADIUS)  # 需避障的最小半径
BASE_SAFETY_CLEARANCE = 0.03  # 车体外额外预留的安全裕度(m)
OCCUPANCY_GRID_RESOLUTION = 0.025  # 占据栅格分辨率(m)，更高的分辨率带来更细腻的虚拟栅格
ROBOT_VISUAL_RADIUS = ROBOT_BODY_RADIUS  # 可视化中展示的真实车体半径

# ==================== 系统参数配置 ====================
# 路径规划参数
SAFETY_DISTANCE_FACTOR = 0.7  # 路径截断百分比，表示只执行路径的前70%
FRONTIER_SAFETY_DISTANCE_METERS = ROBOT_COLLISION_RADIUS + BASE_SAFETY_CLEARANCE  # 前沿探索器需要保持的物理安全距离

# 迷宫和机器人参数
MAZE_FILE = "4.json"  # 默认迷宫文件
ROBOT_ODOM_NOISE = (0.01, math.radians(0.01))  # trans_noise, self.rot_noise = odom_noise (0.01, math.radians(1)))
VIRTUAL_WALL_RESOLUTION_FACTOR = 2  # 虚拟墙分辨率因子
VIRTUAL_WALL_Y_OFFSET = -1  # 虚拟墙Y方向偏移

# 激光雷达参数没有可达的未知区域，探索结束。
LIDAR_MAX_RANGE = 12.0  # 激光雷达扫描半径
LIDAR_ANGLE_RESOLUTION = 3.0  # 激光雷达角度分辨率（度）：改为每3度一束，约120束
LIDAR_NOISE = 0.01  # 激光雷达噪声

# 出口检测参数
MIN_NO_OBSTACLE_COUNT = 34  # 无障碍点数阈值，超过此数值认为走出迷宫

# 探索阈值参数
MIN_EXPLORATION_DISTANCE = 60.0  # 最小探索距离阈值
MIN_FRONTIERS_TO_EXPLORE = 20  # 最小探索前沿数量

# 返程路径容错参数
RETURN_MAX_UNKNOWN_CELLS = 10  # 允许A*返程路径穿越的未知栅格数量上限

# 障碍物区域补扫容错参数
OBSTACLE_SEARCH_MAX_UNKNOWN_CELLS = 3  # 补扫A*允许穿越的未知栅格数量

# 运动控制精度参数
ROTATION_THRESHOLD = 1e-3  # 旋转角度阈值
MOVEMENT_THRESHOLD = 1e-6  # 移动距离阈值

# 速度底线配置
EXPLORE_MIN_SPEED = 0.24  # 探索阶段的最小前进速度

# 卡住判定与挤出恢复参数
STUCK_WINDOW_STEPS = 16             # 判定窗口步数
STUCK_SPIN_W_THRESH = 1.1           # 认为“原地打转”的角速度阈值(rad/s)
STUCK_V_SMALL = 0.03                # 认为“几乎不前进”的线速度阈值(m/s)
STUCK_PROGRESS_EPS = 0.06           # 判定窗口内总位移阈值(m)
STUCK_COOLDOWN_STEPS = 35           # 一次挤出后冷却步数，避免频繁触发
RECOVERY_STEP_LIMIT = 22            # 挤出模式最多持续步数
RECOVERY_MIN_ADVANCE = 0.25         # s挤出最小前进距离(m)
RECOVERY_MAX_ADVANCE = 0.80         # 挤出最大前进距离(m)
RECOVERY_EXTRA_MARGIN = BASE_SAFETY_CLEARANCE        # 在膨胀半径基础上额外预留的安全裕度(m)

# 可视化参数
VISUALIZATION_PAUSE_TIME = 0.005  # 暂停时的等待时间
VISUALIZATION_UPDATE_TIME = 0.0001  # 可视化更新时间

# 未探索区域搜索参数
OBSTACLE_SEARCH_EXPANSION = BASE_SAFETY_CLEARANCE  # 障碍物区域搜索范围扩大距离（米）
# 前沿探索节流参数
FRONTIER_LONG_PATH_THRESHOLD_CELLS = 120  # A*规划路径超过该栅格数，则触发短期冷却
FRONTIER_COOLDOWN_STEPS = 20              # 冷却期间暂停A*与前沿刷新（冻结提示）
FRONTIER_UPDATE_INTERVAL = 2              # 前沿刷新间隔（每两轮刷新一次）
FRONTIER_COOLDOWN_REFERENCE_INDEX = 80    # 冷却期间参考的旧路径索引（截取原路径前缀）

# 冷却提前解除条件：基于路径进度提前允许刷新
FRONTIER_COOLDOWN_RELEASE_RATIO = 0.20    # 进度达到该比例即可提前结束冷却
FRONTIER_COOLDOWN_RELEASE_MIN_REMAIN = 20  # 剩余步数低于该值也提前结束
FRONTIER_COOLDOWN_EARLY_RELEASE_STEPS = 6 # 冷却维持超过该步数后强制解除
FRONTIER_COOLDOWN_REENTER_GRACE_STEPS = 12  # 冷却被强制解除后允许的重启宽限步数

# 路径重规划验证参数
REPLAN_PATH_VALIDATION_SPAN = 20          # 校验当前路径前方多少栅格是否仍安全

# 返程路径节流参数
RETURN_REPLAN_ANCHOR_LOOKAHEAD = 12  # 返程重规划时在初始路径上向前取的锚点偏移
RETURN_REPLAN_MIN_ADVANCE = 4        # 返程重规划至少前进的初始路径栅格数

# 补扫阶段可视化与覆盖率阈值
DFS_VISUALIZATION_ENABLED = False     # 是否展示补扫阶段DFS的逐步可视化
# 补扫阶段DFS覆盖率阈值（达到则判定无不可达区域）
DFS_REGION_COVERAGE_THRESHOLD = 0.90

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
    loader = MazeLoader(grid_resolution=OCCUPANCY_GRID_RESOLUTION)
    maze = loader.load(MAZE_FILE)  # 加载默认迷宫

    # === 入口边界检查参数（替代虚拟墙） ===
    start_x, start_y = maze.start
    min_x, min_y, max_x, max_y = maze.bounds
    eps = maze.resolution * VIRTUAL_WALL_RESOLUTION_FACTOR
    
    # 计算入口的安全边界，防止小车走出迷宫
    # 假设入口在底部，我们设置一个Y坐标的最小值
    entrance_safety_margin = ROBOT_COLLISION_RADIUS + BASE_SAFETY_CLEARANCE  # 入口安全边距（米）
    entrance_min_y = start_y - entrance_safety_margin  # 不允许探索低于此Y值的区域
    
    # 自动查找入口左右两侧最近的墙端点（用于确定入口范围）
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
    entrance_x_range = (left_wall_x, right_wall_x)  # 入口X范围
    
    print(f"入口位置: ({start_x:.2f}, {start_y:.2f}), X范围: [{left_wall_x:.2f}, {right_wall_x:.2f}], 安全边界Y >= {entrance_min_y:.2f}")

    # 2. 初始化机器人、传感器、SLAM等模块
    # 初始朝向设为 pi/2 （朝向y正方向：向上）
    start_pose = (start_x, start_y, math.pi/2)
    robot = Robot(start_pose, odom_noise=ROBOT_ODOM_NOISE)  # 设置一定里程计噪声
    robot.start_threaded()
    lidar = Lidar(maze.walls, max_range=LIDAR_MAX_RANGE, angle_resolution=LIDAR_ANGLE_RESOLUTION, noise=LIDAR_NOISE)
    slam = ICPSlam(maze, start_pose)
    frontier_safety_cells = max(
        1,
        int(math.ceil(FRONTIER_SAFETY_DISTANCE_METERS / maze.resolution))
    )
    explorer = FrontierExplorer(safety_distance=float(frontier_safety_cells))  # 设置与障碍物的安全距离
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
    frontier_update_tick = 0
    # 前沿节流控制
    frontier_cooldown_steps = 0
    frontier_cooldown_elapsed = 0
    frontier_cooldown_reenter_grace = 0
    current_unknown_neighbor = None  # 保存最近一次前沿搜索得到的未知邻居
    frontier_hint_cell = None        # 用于A*失败或冷却时作为DWA提示/跟随目标
    cooldown_reference_cell = None   # 冷却阶段使用的旧路径参考点

    # 初始设置占位，待 simple_cfg 创建后再依据机器人尺寸重新计算
    dynamic_wall_margin = 1  # cells (placeholder)

    # 预分配 BFS 访问信息，避免每步创建大量 Python 列表/字典造成内存膨胀
    occ0 = slam.get_occupancy()
    H, W = occ0.shape
    visited_np = np.zeros((H, W), dtype=np.uint8)  # 0/1 标记
    parent_x = np.full((H, W), -1, dtype=np.int32)
    parent_y = np.full((H, W), -1, dtype=np.int32)

    # 最近未知搜索函数（BFS在空闲区域上扩展，一旦邻接未知返回）
    def find_nearest_unexplored(occupancy, start, bounds_idx=None, unknown_limit=0,
                                visited_out=None, debug_update=None, debug_interval=80,
                                search_mode="bfs"):
        h, w = occupancy.shape
        sx, sy = int(start[0]), int(start[1])
        if not (0 <= sx < w and 0 <= sy < h):
            return None, None, None

        visited_np.fill(255)
        use_stack = (search_mode == "dfs")
        if use_stack:
            container = [(sx, sy, 0)]
            pop_item = container.pop
            push_item = container.append
        else:
            container = deque()
            container.append((sx, sy, 0))
            pop_item = container.popleft
            push_item = container.append
        visited_np[sy, sx] = 0
        parent_x[sy, sx] = -1
        parent_y[sy, sx] = -1
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]

        def has_unknown_neighbor(x, y):
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and occupancy[ny, nx] == -1:
                    return True
            return False

        if bounds_idx is not None:
            min_bx, max_bx, min_by, max_by = bounds_idx
            within_bounds = lambda cx, cy, _min_bx=min_bx, _max_bx=max_bx, _min_by=min_by, _max_by=max_by: (
                _min_bx <= cx <= _max_bx and _min_by <= cy <= _max_by
            )
            soft_margin = 3
            min_bx_soft = max(0, min_bx - soft_margin)
            max_bx_soft = min(w - 1, max_bx + soft_margin)
            min_by_soft = max(0, min_by - soft_margin)
            max_by_soft = min(h - 1, max_by + soft_margin)
            within_soft_bounds = lambda cx, cy, _min_bx=min_bx_soft, _max_bx=max_bx_soft, _min_by=min_by_soft, _max_by=max_by_soft: (
                _min_bx <= cx <= _max_bx and _min_by <= cy <= _max_by
            )
        else:
            within_bounds = lambda cx, cy: True
            within_soft_bounds = lambda cx, cy: True

        debug_points = None
        if visited_out is not None:
            visited_out.clear()
            debug_points = visited_out
        elif debug_update is not None:
            debug_points = []
        if debug_points is not None:
            debug_points.append((sx, sy))
            if debug_update is not None:
                debug_update(debug_points)

        debug_counter = 0
        max_unknown_allowed = unknown_limit if unknown_limit is not None else None

        def is_safe_cell(x, y):
            for dx in range(-dynamic_wall_margin, dynamic_wall_margin + 1):
                for dy in range(-dynamic_wall_margin, dynamic_wall_margin + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and occupancy[ny, nx] == 1:
                        return False
            return True

        def is_inside_maze(x, y):
            world_x = maze.bounds[0] + (x + 0.5) * maze.resolution
            world_y = maze.bounds[1] + (y + 0.5) * maze.resolution
            if world_y < entrance_min_y:
                return False
            if abs(world_y - start_y) < entrance_safety_margin * 2:
                if world_x < entrance_x_range[0] - 0.3 or world_x > entrance_x_range[1] + 0.3:
                    return False
            return True

        fallback_candidate = None
        while container:
            x, y, used_unknown = pop_item()
            debug_counter += 1

            if not is_inside_maze(x, y):
                continue

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if occupancy[ny, nx] != -1:
                    continue
                if not within_bounds(nx, ny):
                    continue
                if not is_inside_maze(nx, ny):
                    continue

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
                if is_safe_cell(x, y):
                    if debug_update is not None and debug_points is not None:
                        debug_update(debug_points)
                    return (x, y), (nx, ny), path_cells
                elif fallback_candidate is None:
                    fallback_candidate = ((x, y), (nx, ny), path_cells)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if not within_soft_bounds(nx, ny):
                    continue
                cell_val = occupancy[ny, nx]
                if cell_val == 1:
                    continue
                next_unknown = used_unknown + (1 if cell_val == -1 else 0)
                if max_unknown_allowed is not None and next_unknown > max_unknown_allowed:
                    continue
                if visited_np[ny, nx] <= next_unknown:
                    continue
                if not is_inside_maze(nx, ny):
                    continue
                visited_np[ny, nx] = next_unknown
                parent_x[ny, nx] = x
                parent_y[ny, nx] = y
                push_item((nx, ny, next_unknown))
                if debug_points is not None:
                    debug_points.append((nx, ny))
                    if debug_update is not None and (debug_counter % max(1, debug_interval) == 0):
                        debug_update(debug_points)

        if fallback_candidate is not None:
            safe_target = None
            safe_path = None
            (fx, fy), fallback_unknown, fallback_path = fallback_candidate
            if fallback_path:
                for idx in range(len(fallback_path) - 1, -1, -1):
                    cx, cy = fallback_path[idx]
                    if is_safe_cell(cx, cy) and has_unknown_neighbor(cx, cy):
                        safe_target = (cx, cy)
                        safe_path = fallback_path[:idx + 1]
                        break
            if safe_target is not None:
                sx_safe, sy_safe = safe_target
                safe_unknown = None
                for dx, dy in directions:
                    nx, ny = sx_safe + dx, sy_safe + dy
                    if 0 <= nx < w and 0 <= ny < h and occupancy[ny, nx] == -1:
                        safe_unknown = (nx, ny)
                        break
                if safe_unknown is None:
                    safe_target = None
                else:
                    fallback_unknown = safe_unknown
            if safe_target is not None:
                if step_counter % 25 == 0:
                    print(f"[警告] 未找到满足安全距离的前沿，改用最近安全点 {safe_target} (原候选 {(fx, fy)})")
                if debug_update is not None and debug_points is not None:
                    debug_update(debug_points)
                return safe_target, fallback_unknown, safe_path
            if step_counter % 25 == 0:
                print("[警告] 未找到满足安全距离的目标，且无可用安全替代点")
            if debug_update is not None and debug_points is not None:
                debug_update(debug_points)
            return None, None, None

        if debug_update is not None and debug_points is not None:
            debug_update(debug_points)
        return None, None, None
    
    # 直接使用集中后的默认配置即可（原工厂函数已合并为默认值）
    dwa_cfg = DWAConfig()
    dwa_cfg.robot_radius = ROBOT_COLLISION_RADIUS
    dwa_cfg.safety_clearance = BASE_SAFETY_CLEARANCE
    dwa_planner = DWAPlanner(dwa_cfg)
    base_margin_m = dwa_cfg.robot_radius + dwa_cfg.safety_clearance
    dynamic_wall_margin = max(1, int(base_margin_m / maze.resolution) + 1)
    explore_safety_cells = float(frontier_safety_cells)
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
    base_lookahead_steps = 25   # 默认前瞻栅格数
    min_lookahead_steps = 20    # 弯曲段时的最小前瞻
    max_lookahead_steps = 30   # 直线段时的最大前瞻

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

    def plan_path_with_safety(occupancy_grid, start_cell, goal_cell,
                              safety_cells, max_unknown_allowed=0):
        """使用与探索一致的安全距离规划路径。"""
        if start_cell == goal_cell:
            return [start_cell], float(safety_cells)
        planned = explorer.plan_path(
            occupancy_grid,
            start_cell,
            goal_cell,
            safety_distance=safety_cells,
            max_unknown_cells=max_unknown_allowed
        )
        if planned and len(planned) >= 2:
            return planned, float(safety_cells)
        return None, None

    def find_nearest_safe_cell(path_cells, occupancy_grid, safety_cells, start_cell):
        """在给定路径上查找距离起点最近且满足安全距离的栅格。"""
        if not path_cells:
            return None
        sx, sy = start_cell
        best_cell = None
        best_dist2 = float('inf')
        for cx, cy in path_cells:
            if (cx, cy) == (sx, sy):
                continue
            if not explorer._is_safe(occupancy_grid, cx, cy, safety_distance=safety_cells):
                continue
            dist2 = (cx - sx) * (cx - sx) + (cy - sy) * (cy - sy)
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_cell = (cx, cy)
        return best_cell

    def drive_path_with_dwa_segment(path_cells, label="路径跟随", arrival_tol=0.22,
                                    max_iter_factor=80,
                                    replan_callback=None,
                                    replan_interval=18,
                                    path_safety_cells=None,
                                    viz_context_provider=None):
        """使用DWA沿给定网格路径行驶，返回是否成功到达。

        viz_context_provider(Optional[Callable]): 每次循环提供可视化上下文的回调，
        返回 dict(actual_traj, actual_traj_style, extra_trajs)。"""
        nonlocal est_pose, scan, total_distance_traveled, step_counter
        if not path_cells or len(path_cells) < 2:
            return False

        def cells_to_world(cells):
            pts = []
            for cx, cy in cells:
                wx = maze.bounds[0] + (cx + 0.5) * maze.resolution
                wy = maze.bounds[1] + (cy + 0.5) * maze.resolution
                pts.append((wx, wy))
            return np.asarray(pts, dtype=float)

        path_cells = list(path_cells)
        path_arr = cells_to_world(path_cells)
        goal_cell = path_cells[-1]
        goal_world = path_arr[-1]
        last_replan_idx = -replan_interval
        max_iters = max(len(path_arr) * max_iter_factor, 600)
        print(f"[{label}] 使用DWA沿路径前进，共 {len(path_arr)-1} 段，最大步数 {max_iters}")

        def is_path_segment_safe(current_idx, occ_grid):
            if path_safety_cells is None:
                return False
            if not path_cells:
                return False
            h, w = occ_grid.shape
            start_idx = max(0, current_idx)
            end_idx = min(len(path_cells), current_idx + REPLAN_PATH_VALIDATION_SPAN)
            for idx_check in range(start_idx, end_idx):
                cx, cy = path_cells[idx_check]
                if not (0 <= cy < h and 0 <= cx < w):
                    return False
                if occ_grid[cy, cx] == 1:
                    return False
                if not explorer._is_safe(occ_grid, cx, cy, safety_distance=path_safety_cells):
                    return False
            return True

        dwa_planner._last_u = (0.0, 0.0)
        if hasattr(dwa_planner, '_dwell_count'):
            dwa_planner._dwell_count = 0
        if hasattr(dwa_planner, '_reverse_block_count'):
            dwa_planner._reverse_block_count = 0

        for idx in range(max_iters):
            while viz.paused:
                plt.pause(VISUALIZATION_PAUSE_TIME)

            loop_label = f"{label}-idx={idx}"
            loop_start = time.perf_counter()
            seg_times = []
            t_section = loop_start

            pause_wait = time.perf_counter() - t_section
            if pause_wait > 0:
                seg_times.append(("pause_wait", pause_wait * 1000.0))
            t_section = time.perf_counter()

            est_pose = robot.get_pose()
            dist_to_goal = math.hypot(est_pose[0] - goal_world[0], est_pose[1] - goal_world[1])
            seg_times.append(("pose_update", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()
            if dist_to_goal <= arrival_tol:
                seg_times.append(("loop_total", (time.perf_counter() - loop_start) * 1000.0))
                log_section_times(loop_label, seg_times)
                print(f"[{label}] 到达目标，终点剩余 {dist_to_goal:.2f}m")
                return True

            if replan_callback and (idx == 0 or (idx - last_replan_idx) >= replan_interval):
                new_path = replan_callback()
                if new_path and len(new_path) >= 2:
                    path_cells = list(new_path)
                    path_arr = cells_to_world(path_cells)
                    goal_cell = path_cells[-1]
                    goal_world = path_arr[-1]
                    last_replan_idx = idx
                    seg_times.append(("path_replan", (time.perf_counter() - t_section) * 1000.0))
                    t_section = time.perf_counter()

            dists = np.hypot(path_arr[:, 0] - est_pose[0], path_arr[:, 1] - est_pose[1])
            nearest_idx = int(np.argmin(dists))
            if replan_callback and (idx == 0 or (idx - last_replan_idx) >= replan_interval):
                check_start = time.perf_counter()
                path_needs_replan = True
                occ_preview = None
                if path_safety_cells is not None:
                    occ_preview = slam.get_occupancy()
                    path_needs_replan = not is_path_segment_safe(nearest_idx, occ_preview)
                seg_times.append(("path_check", (time.perf_counter() - check_start) * 1000.0))
                if path_needs_replan:
                    replan_start = time.perf_counter()
                    new_path = replan_callback()
                    seg_times.append(("path_replan", (time.perf_counter() - replan_start) * 1000.0))
                    if new_path and len(new_path) >= 2:
                        path_cells = list(new_path)
                        path_arr = cells_to_world(path_cells)
                        goal_cell = path_cells[-1]
                        goal_world = path_arr[-1]
                        dists = np.hypot(path_arr[:, 0] - est_pose[0], path_arr[:, 1] - est_pose[1])
                        nearest_idx = int(np.argmin(dists))
                last_replan_idx = idx
                t_section = time.perf_counter()
            else:
                t_section = time.perf_counter()
                if replan_callback and idx == 0:
                    last_replan_idx = idx

            lookahead = compute_dynamic_lookahead(path_cells, nearest_idx)
            follow_idx = min(len(path_arr) - 1, nearest_idx + lookahead)
            short_term_cell = path_cells[follow_idx]
            gx, gy = path_arr[follow_idx]

            seg_start = max(0, nearest_idx - 1)
            seg_end = min(len(path_arr), follow_idx + 2)
            if seg_end - seg_start >= 2:
                path_hint_segment = path_arr[seg_start:seg_end]
            else:
                path_hint_segment = path_arr

            state = np.array([
                est_pose[0],
                est_pose[1],
                est_pose[2],
                robot.linear_vel,
                robot.angular_vel
            ])
            obstacles = occupancy_to_obstacles(
                slam.get_occupancy(),
                maze.bounds,
                maze.resolution,
                stride=3
            )
            seg_times.append(("target_prepare", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()
            (v_cmd, w_cmd), predicted_traj = dwa_planner.plan(
                state,
                (gx, gy),
                obstacles,
                path_hint=path_hint_segment
            )
            seg_times.append(("dwa_plan", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()

            if v_cmd > 1e-6:
                dw_max = None
                if hasattr(dwa_planner, "last_dw") and isinstance(dwa_planner.last_dw, (list, tuple)) and len(dwa_planner.last_dw) >= 2:
                    dw_max = dwa_planner.last_dw[1]
                max_allow = dw_max if (dw_max is not None and math.isfinite(dw_max)) else EXPLORE_MIN_SPEED
                boost_speed = min(EXPLORE_MIN_SPEED, max_allow)
                if boost_speed > v_cmd and dist_to_goal > 0.5:
                    v_cmd = boost_speed
                    dwa_planner._last_u = (v_cmd, w_cmd)
                    if predicted_traj is not None:
                        predicted_traj = dwa_planner._predict_trajectory(state, v_cmd, w_cmd)

            viz_extra = [
                {
                    "points": [
                        (est_pose[0], est_pose[1]),
                        (gx, gy)
                    ],
                    "style": {
                        "color": "cyan",
                        "linewidth": 1.4,
                        "alpha": 0.8,
                        "label": "短期目标连线"
                    }
                }
            ]

            actual_traj_points = robot.trajectory
            actual_traj_style = explore_traj_style
            extra_traj_list = list(viz_extra)
            if viz_context_provider is not None:
                try:
                    custom_viz = viz_context_provider(goal_cell, path_cells, nearest_idx)
                except Exception:
                    custom_viz = None
                if isinstance(custom_viz, dict):
                    if 'actual_traj' in custom_viz and custom_viz['actual_traj'] is not None:
                        actual_traj_points = custom_viz['actual_traj']
                    if 'actual_traj_style' in custom_viz and custom_viz['actual_traj_style'] is not None:
                        actual_traj_style = custom_viz['actual_traj_style']
                    extra_from_provider = custom_viz.get('extra_trajs') if isinstance(custom_viz, dict) else None
                    if extra_from_provider:
                        extra_traj_list.extend(extra_from_provider)

            viz.update(
                est_pose,
                scan,
                frontiers=None,
                target=goal_cell,
                path=path_cells,
                occupancy=slam.get_occupancy(),
                predicted_traj=predicted_traj,
                robot_radius=ROBOT_VISUAL_RADIUS,
                actual_traj=actual_traj_points,
                actual_traj_style=actual_traj_style,
                extra_trajs=extra_traj_list
            )
            seg_times.append(("visualize", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()

            d_trans, d_rot = robot.velocity_step(float(v_cmd), float(w_cmd), float(dwa_cfg.dt))
            if idx % 20 == 0:
                print(f"[{label}] idx={idx} v={float(v_cmd):.2f} w={float(w_cmd):.2f} d_trans={d_trans:.3f} d_rot={d_rot:.3f}")
            total_distance_traveled += abs(d_trans)
            seg_times.append(("motion_update", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()

            noisy, clean = lidar.scan(robot.get_pose())
            scan = noisy
            est_pose = slam.update((d_trans, d_rot), scan)
            step_counter += 1
            seg_times.append(("slam_update", (time.perf_counter() - t_section) * 1000.0))
            t_section = time.perf_counter()

            if math.isclose(v_cmd, 0.0, abs_tol=1e-3) and math.isclose(w_cmd, 0.0, abs_tol=1e-3):
                if dist_to_goal <= arrival_tol + 0.1:
                    seg_times.append(("loop_total", (time.perf_counter() - loop_start) * 1000.0))
                    log_section_times(loop_label, seg_times)
                    print(f"[{label}] 靠近目标但速度趋近于零，判定到达")
                    return True

            seg_times.append(("loop_total", (time.perf_counter() - loop_start) * 1000.0))
            log_section_times(loop_label, seg_times)

        print(f"[{label}] 超过最大步数 {max_iters} 仍未到达目标")
        return False
    while True:
        # 1. 暂停检查
        if viz.paused:
            plt.pause(VISUALIZATION_PAUSE_TIME)
            continue
        loop_step_label = f"step={step_counter}"
        loop_start = time.perf_counter()
        section_times = []
        t_section = loop_start
        if frontier_cooldown_reenter_grace > 0:
            frontier_cooldown_reenter_grace -= 1
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

        if cooldown_reference_cell is not None:
            if abs(rx_idx - cooldown_reference_cell[0]) <= 1 and abs(ry_idx - cooldown_reference_cell[1]) <= 1:
                if frontier_cooldown_steps > 0 and step_counter % 20 == 0:
                    print(f"[前沿节流] 已到达冷却参考点 {cooldown_reference_cell}，解除冷却限制")
                cooldown_reference_cell = None
                frontier_cooldown_steps = 0
                frontier_hint_cell = None
                frontier_cooldown_elapsed = 0
                frontier_cooldown_reenter_grace = FRONTIER_COOLDOWN_REENTER_GRACE_STEPS

        # 冷却策略：冷却期暂停A*与前沿刷新；冷却结束时刷新前沿并运行A*
        occupancy = slam.get_occupancy()
        if frontier_cooldown_steps == 0:
            frontier_cooldown_elapsed = 0
            should_refresh_frontier = False
            if global_path is None or frontier_hint_cell is None:
                should_refresh_frontier = True
            else:
                if (frontier_update_tick % FRONTIER_UPDATE_INTERVAL) == 0:
                    should_refresh_frontier = True
            frontier_update_tick += 1
            # 刷新最近前沿与未知邻居（仅在非冷却期）
            if should_refresh_frontier:
                provided_path = None
                bfs_path = None
                if cooldown_reference_cell is not None:
                    target_cell_latest = cooldown_reference_cell
                    unknown_neighbor_new = None
                    current_unknown_neighbor = None
                else:
                    target_cell_latest, unknown_neighbor_new, bfs_path = find_nearest_unexplored(occupancy, (rx_idx, ry_idx))
                    if target_cell_latest is None:
                        # 使用全局前沿检测作为回退策略
                        fallback_frontier, fallback_path = explorer.find_nearest_frontier(occupancy, (rx_idx, ry_idx))
                        if fallback_frontier is None or not fallback_path:
                            print("没有可达的未知区域，探索结束。")
                            section_times.append(("frontier_update", (time.perf_counter() - t_section) * 1000.0))
                            section_times.append(("loop_total", (time.perf_counter() - loop_start) * 1000.0))
                            log_section_times(loop_step_label, section_times)
                            break
                        target_cell_latest = fallback_frontier
                        current_unknown_neighbor = None
                        provided_path = list(fallback_path)
                    else:
                        current_unknown_neighbor = unknown_neighbor_new
                        if bfs_path:
                            provided_path = list(bfs_path)
                frontier_hint_cell = target_cell_latest
                # 记录目标重复情况
                if target_cell_latest == prev_target_cell:
                    same_target_counter += 1
                else:
                    same_target_counter = 0
                prev_target_cell = target_cell_latest

                # 运行A*（仅在非冷却期），始终使用指定的前沿安全距离
                safety_cells = float(frontier_safety_cells)
                planned_path = None
                planned_sd = None
                if provided_path is not None and len(provided_path) >= 2:
                    path_safe = True
                    for cx, cy in provided_path:
                        if not explorer._is_safe(occupancy, cx, cy, safety_distance=safety_cells):
                            path_safe = False
                            break
                    if path_safe:
                        planned_path = list(provided_path)
                        planned_sd = safety_cells
                if planned_path is None:
                    planned_path, planned_sd = plan_path_with_safety(
                        occupancy,
                        (rx_idx, ry_idx),
                        target_cell_latest,
                        safety_cells,
                        max_unknown_allowed=0
                    )
                global_path = planned_path if planned_path is not None else None
                current_path = global_path if global_path else None
                if global_path and (len(global_path) - 1) > FRONTIER_LONG_PATH_THRESHOLD_CELLS:
                    original_path_steps = len(global_path) - 1
                    cooldown_reference_idx = min(len(global_path) - 1, max(1, FRONTIER_COOLDOWN_REFERENCE_INDEX))
                    planned_prefix = list(global_path[:cooldown_reference_idx + 1]) if cooldown_reference_idx < len(global_path) - 1 else list(global_path)
                    current_path = planned_prefix
                    global_path = planned_prefix
                    if frontier_cooldown_reenter_grace <= 0:
                        cooldown_reference_cell = planned_prefix[-1]
                        frontier_hint_cell = cooldown_reference_cell
                        frontier_cooldown_steps = FRONTIER_COOLDOWN_STEPS
                        frontier_cooldown_elapsed = 0
                        frontier_cooldown_reenter_grace = 0
                        if step_counter % 20 == 0:
                            print(f"[前沿节流] A*路径过长({original_path_steps}格) -> 冷却 {FRONTIER_COOLDOWN_STEPS} 步，改用旧路径点 {cooldown_reference_cell} (截断后 {len(current_path)-1} 格)")
                    else:
                        frontier_hint_cell = planned_prefix[-1]
                        cooldown_reference_cell = None
                        frontier_cooldown_steps = 0
                        frontier_cooldown_elapsed = 0
                        if step_counter % 20 == 0:
                            print(
                                f"[前沿节流] A*路径过长({original_path_steps}格) 但处于宽限期(剩余 {frontier_cooldown_reenter_grace} 步)，跳过冷却"
                            )
            else:
                if step_counter % 20 == 0:
                    print("[前沿刷新] 按照间隔策略跳过本轮前沿更新，沿用既有路径。")
        else:
            # 冷却中：不刷新前沿、不运行A*，仅递减计数器并沿旧提示/路径前进
            frontier_cooldown_elapsed += 1
            frontier_cooldown_steps = max(0, frontier_cooldown_steps - 1)
            frontier_update_tick = 0
            release_reason = None
            release_hint_cell = None
            if current_path and len(current_path) >= 2:
                # 根据当前靠近的路径索引判断进度
                nearest_idx = 0
                min_d2 = 1e18
                for i, (px, py) in enumerate(current_path):
                    dx = px - rx_idx
                    dy = py - ry_idx
                    d2 = dx * dx + dy * dy
                    if d2 < min_d2:
                        min_d2 = d2
                        nearest_idx = i
                path_steps = len(current_path) - 1
                remaining_steps = max(0, path_steps - nearest_idx)
                progress_ratio = nearest_idx / max(1, path_steps)
                if frontier_cooldown_steps > 0 and (
                    progress_ratio >= FRONTIER_COOLDOWN_RELEASE_RATIO or
                    remaining_steps <= FRONTIER_COOLDOWN_RELEASE_MIN_REMAIN
                ):
                    release_reason = f"已沿截断路径前进 {progress_ratio*100:.1f}% (剩余 {remaining_steps} 格)"
                    release_hint_cell = current_path[min(len(current_path) - 1, nearest_idx + 1)]
            else:
                nearest_idx = 0
                remaining_steps = 0
                progress_ratio = 0.0

            if release_reason is None and frontier_cooldown_steps > 0 and \
                    frontier_cooldown_elapsed >= FRONTIER_COOLDOWN_EARLY_RELEASE_STEPS:
                release_reason = (
                    f"冷却已持续 {frontier_cooldown_elapsed} 步 (阈值 {FRONTIER_COOLDOWN_EARLY_RELEASE_STEPS})"
                )
                if current_path and len(current_path) >= 2:
                    release_hint_cell = current_path[min(len(current_path) - 1, nearest_idx + 1)]
                elif frontier_hint_cell is not None:
                    release_hint_cell = frontier_hint_cell

            if release_reason is not None:
                cooldown_reference_cell = None
                frontier_cooldown_steps = 0
                frontier_cooldown_elapsed = 0
                frontier_cooldown_reenter_grace = FRONTIER_COOLDOWN_REENTER_GRACE_STEPS
                if release_hint_cell is not None:
                    frontier_hint_cell = release_hint_cell
                print(f"[前沿节流] {release_reason}，提前解除冷却")
            if frontier_cooldown_steps > 0 and step_counter % 20 == 0:
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
            robot_radius=ROBOT_VISUAL_RADIUS,
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

        def compute_obstacle_search_bounds(current_occupancy):
            obstacle_coords = []
            for yy in range(current_occupancy.shape[0]):
                for xx in range(current_occupancy.shape[1]):
                    if current_occupancy[yy, xx] == 1:
                        world_x = maze.bounds[0] + xx * maze.resolution
                        world_y = maze.bounds[1] + yy * maze.resolution
                        obstacle_coords.append((world_x, world_y))

            if not obstacle_coords:
                return None, None

            min_obstacle_x = min(coord[0] for coord in obstacle_coords) - OBSTACLE_SEARCH_EXPANSION
            max_obstacle_x = max(coord[0] for coord in obstacle_coords) + OBSTACLE_SEARCH_EXPANSION
            min_obstacle_y = min(coord[1] for coord in obstacle_coords) - OBSTACLE_SEARCH_EXPANSION
            max_obstacle_y = max(coord[1] for coord in obstacle_coords) + OBSTACLE_SEARCH_EXPANSION

            min_obs_x_idx = int((min_obstacle_x - maze.bounds[0]) / maze.resolution)
            max_obs_x_idx = int((max_obstacle_x - maze.bounds[0]) / maze.resolution)
            min_obs_y_idx = int((min_obstacle_y - maze.bounds[1]) / maze.resolution)
            max_obs_y_idx = int((max_obstacle_y - maze.bounds[1]) / maze.resolution)

            return (min_obs_x_idx, max_obs_x_idx, min_obs_y_idx, max_obs_y_idx), (
                min_obstacle_x,
                min_obstacle_y,
                max_obstacle_x,
                max_obstacle_y,
            )

        bounds_idx, bounds_world = compute_obstacle_search_bounds(occupancy)
        if bounds_idx and bounds_world:
            min_obs_x_idx, max_obs_x_idx, min_obs_y_idx, max_obs_y_idx = bounds_idx
            min_obstacle_x, min_obstacle_y, max_obstacle_x, max_obstacle_y = bounds_world
            print("检测到障碍物区域，边界如下：")
            print(f"障碍物区域边界: X[{min_obstacle_x:.1f}, {max_obstacle_x:.1f}], Y[{min_obstacle_y:.1f}, {max_obstacle_y:.1f}]")
            viz.set_obstacle_search_region((min_obstacle_x, min_obstacle_y, max_obstacle_x, max_obstacle_y))

            # 在障碍物边界范围内寻找未探索区域
            unexplored_in_range = []
            for y in range(max(0, min_obs_y_idx), min(occupancy.shape[0], max_obs_y_idx + 1)):
                for x in range(max(0, min_obs_x_idx), min(occupancy.shape[1], max_obs_x_idx + 1)):
                    if occupancy[y, x] == -1:  # 未探索区域
                        unexplored_in_range.append((x, y))
            
            print(f"在障碍物边界范围内发现 {len(unexplored_in_range)} 个未探索格子")
            
            if unexplored_in_range:
                def find_boundary_entry(start_x, start_y):
                    clamp_x = min(max(start_x, min_obs_x_idx), max_obs_x_idx)
                    clamp_y = min(max(start_y, min_obs_y_idx), max_obs_y_idx)
                    if (min_obs_x_idx <= start_x <= max_obs_x_idx and
                            min_obs_y_idx <= start_y <= max_obs_y_idx):
                        dist_pairs = [
                            (abs(start_x - min_obs_x_idx), min_obs_x_idx, clamp_y),
                            (abs(start_x - max_obs_x_idx), max_obs_x_idx, clamp_y),
                            (abs(start_y - min_obs_y_idx), clamp_x, min_obs_y_idx),
                            (abs(start_y - max_obs_y_idx), clamp_x, max_obs_y_idx)
                        ]
                        _, clamp_x, clamp_y = min(dist_pairs, key=lambda item: item[0])
                    clamp_x = min(max(clamp_x, min_obs_x_idx), max_obs_x_idx)
                    clamp_y = min(max(clamp_y, min_obs_y_idx), max_obs_y_idx)

                    if 0 <= clamp_y < occupancy.shape[0] and 0 <= clamp_x < occupancy.shape[1]:
                        if occupancy[clamp_y, clamp_x] != 1:
                            return clamp_x, clamp_y

                    search_offsets = [(0, 0)]
                    search_offsets += [(dx, 0) for dx in range(-4, 5)]
                    search_offsets += [(0, dy) for dy in range(-4, 5)]
                    search_offsets += [
                        (dx, dy)
                        for r in range(1, 4)
                        for dx in (-r, r)
                        for dy in (-r, r)
                    ]
                    for dx, dy in search_offsets:
                        nx = clamp_x + dx
                        ny = clamp_y + dy
                        if not (min_obs_x_idx <= nx <= max_obs_x_idx and min_obs_y_idx <= ny <= max_obs_y_idx):
                            continue
                        if 0 <= ny < occupancy.shape[0] and 0 <= nx < occupancy.shape[1]:
                            if occupancy[ny, nx] != 1:
                                return nx, ny
                    return clamp_x, clamp_y
                visited_subtargets = set()
                next_start_cell = None
                coverage_sufficient = False
                while True:
                    sweep_section_times = []
                    sweep_t0 = time.perf_counter()

                    def finish_logging(tag="补扫阶段-循环"):
                        sweep_section_times.append(("total_loop", (time.perf_counter() - sweep_t0) * 1000.0))
                        log_section_times(tag, sweep_section_times)
                    occupancy = slam.get_occupancy()
                    bounds_idx, bounds_world = compute_obstacle_search_bounds(occupancy)
                    if not (bounds_idx and bounds_world):
                        print("动态刷新障碍物边界时未检测到障碍区域，结束补扫阶段。")
                        break

                    min_obs_x_idx, max_obs_x_idx, min_obs_y_idx, max_obs_y_idx = bounds_idx
                    min_obstacle_x, min_obstacle_y, max_obstacle_x, max_obstacle_y = bounds_world
                    viz.set_obstacle_search_region((min_obstacle_x, min_obstacle_y, max_obstacle_x, max_obstacle_y))
                    sweep_section_times.append(("bounds_update", (time.perf_counter() - sweep_t0) * 1000.0))
                    if next_start_cell is None:
                        current_idx_x = int((robot.x - maze.bounds[0]) / maze.resolution)
                        current_idx_y = int((robot.y - maze.bounds[1]) / maze.resolution)
                        boundary_start = find_boundary_entry(current_idx_x, current_idx_y)
                        if boundary_start != (current_idx_x, current_idx_y):
                            print(f"[BFS Debug] 起点调整到障碍边界 {boundary_start}")
                        search_origin = boundary_start
                    else:
                        search_origin = next_start_cell
                        current_idx_x, current_idx_y = search_origin
                        boundary_start = search_origin
                    sweep_section_times.append(("origin_prepare", (time.perf_counter() - sweep_t0) * 1000.0))

                    bfs_debug_cells = []
                    bfs_debug_state = {"reported": 0}
                    if DFS_VISUALIZATION_ENABLED:
                        viz.set_bfs_debug_points(None)

                    def bfs_debug_callback(cells):
                        if not DFS_VISUALIZATION_ENABLED:
                            return
                        viz.set_bfs_debug_points(cells, color='cyan')
                        viz.update(
                            robot.get_pose(),
                            scan,
                            frontiers=None,
                            target=None,
                            path=None,
                            occupancy=occupancy,
                            actual_traj=robot.trajectory,
                            actual_traj_style=explore_traj_style
                        )
                        if len(cells) - bfs_debug_state["reported"] >= 150:
                            bfs_debug_state["reported"] = len(cells)
                            print(f"[BFS Debug] 已扩展 {len(cells)} 个栅格")
                        plt.pause(0.03)

                    occupancy_for_search = occupancy.copy()
                    if visited_subtargets:
                        for vx, vy in visited_subtargets:
                            if 0 <= vy < occupancy_for_search.shape[0] and 0 <= vx < occupancy_for_search.shape[1]:
                                occupancy_for_search[vy, vx] = 0
                    sweep_section_times.append(("prepare_mask", (time.perf_counter() - sweep_t0) * 1000.0))

                    bfs_target, bfs_unknown_neighbor, bfs_path = find_nearest_unexplored(
                        occupancy_for_search,
                        boundary_start,
                        bounds_idx=(min_obs_x_idx, max_obs_x_idx, min_obs_y_idx, max_obs_y_idx),
                        unknown_limit=OBSTACLE_SEARCH_MAX_UNKNOWN_CELLS,
                        visited_out=bfs_debug_cells,
                        debug_update=bfs_debug_callback,
                        debug_interval=60,
                        search_mode="dfs"
                    )
                    sweep_section_times.append(("dfs_search", (time.perf_counter() - sweep_t0) * 1000.0))

                    if DFS_VISUALIZATION_ENABLED:
                        if bfs_debug_cells:
                            print(f"[BFS Debug] 搜索结束，共记录 {len(bfs_debug_cells)} 个栅格")
                            viz.set_bfs_debug_points(bfs_debug_cells, color='cyan')
                        else:
                            viz.set_bfs_debug_points(None)

                        viz.update(
                            robot.get_pose(),
                            scan,
                            frontiers=None,
                            target=None,
                            path=None,
                            occupancy=slam.get_occupancy(),
                            actual_traj=robot.trajectory,
                            actual_traj_style=explore_traj_style
                        )
                        if bfs_debug_cells:
                            plt.pause(0.6)

                    coverage_eval_start = time.perf_counter()
                    region_slice = occupancy_for_search
                    traversable_cells = int(np.count_nonzero(region_slice != 1))
                    if bounds_idx:
                        min_bx, max_bx, min_by, max_by = bounds_idx
                        region_slice = occupancy_for_search[min_by:max_by + 1, min_bx:max_bx + 1]
                        traversable_cells = int(np.count_nonzero(region_slice != 1))
                    traversable_cells = max(1, traversable_cells)
                    visited_unique = {(vx, vy) for vx, vy in bfs_debug_cells
                                      if 0 <= vy < occupancy_for_search.shape[0] and 0 <= vx < occupancy_for_search.shape[1]}
                    visited_traversable = sum(1 for vx, vy in visited_unique if occupancy_for_search[vy, vx] != 1)
                    coverage_ratio = visited_traversable / traversable_cells
                    sweep_section_times.append(("coverage_eval", (time.perf_counter() - coverage_eval_start) * 1000.0))
                    if coverage_ratio >= DFS_REGION_COVERAGE_THRESHOLD:
                        print(f"DFS补扫覆盖率达到 {coverage_ratio*100:.1f}% (阈值 {DFS_REGION_COVERAGE_THRESHOLD*100:.0f}%)，判定障碍区域已充分探索，直接返回起点。")
                        coverage_sufficient = True
                        finish_logging("补扫阶段-覆盖完成")
                        break

                    if not (bfs_target and bfs_path and len(bfs_path) > 1):
                        print("障碍区域内没有更多符合条件的未知前沿，结束补扫阶段。")
                        finish_logging("补扫阶段-无前沿")
                        break

                    if bfs_target in visited_subtargets:
                        print(f"目标 {bfs_target} 已尝试过，跳过并继续搜索下一个未知点。")
                        finish_logging("补扫阶段-重复目标")
                        continue

                    print(f"BFS在障碍物区域内找到可达目标 {bfs_target}，路径长度 {len(bfs_path)-1} 步")
                    selected_target = bfs_target
                    selected_path = list(bfs_path)
                    visited_subtargets.add(selected_target)

                    occupancy_local = slam.get_occupancy().copy()
                    if 0 <= current_idx_y < occupancy_local.shape[0] and 0 <= current_idx_x < occupancy_local.shape[1]:
                        occupancy_local[current_idx_y, current_idx_x] = 0
                    tx, ty = selected_target
                    if 0 <= ty < occupancy_local.shape[0] and 0 <= tx < occupancy_local.shape[1]:
                        occupancy_local[ty, tx] = 0

                    safety_cells = float(frontier_safety_cells)
                    a_star_path = explorer.plan_path(
                        occupancy_local,
                        (current_idx_x, current_idx_y),
                        selected_target,
                        safety_distance=safety_cells,
                        max_unknown_cells=OBSTACLE_SEARCH_MAX_UNKNOWN_CELLS
                    )

                    effective_target = selected_target
                    path_to_follow = None
                    if a_star_path and len(a_star_path) >= 2:
                        path_to_follow = a_star_path
                        print(f"使用A*生成的路径前往补扫目标，长度 {len(a_star_path)-1} 格")
                    else:
                        strict_path, _ = plan_path_with_safety(
                            occupancy_local,
                            (current_idx_x, current_idx_y),
                            selected_target,
                            safety_cells,
                            max_unknown_allowed=OBSTACLE_SEARCH_MAX_UNKNOWN_CELLS
                        )
                        if strict_path:
                            path_to_follow = strict_path
                        else:
                            print("A* 无法直达补扫目标，继续搜索其他未知候选。")
                            path_to_follow = None

                    viz.set_bfs_debug_points(None)
                    if path_to_follow is None or len(path_to_follow) < 2:
                        print("补扫阶段未找到满足安全距离的有效路径，继续搜索其他候选。")
                        visited_subtargets.add(effective_target)
                        next_start_cell = None
                        sweep_section_times.append(("path_planning", (time.perf_counter() - sweep_t0) * 1000.0))
                        finish_logging("补扫阶段-无路径")
                        if coverage_sufficient:
                            break
                        continue

                    def replanner():
                        occ_latest = slam.get_occupancy().copy()
                        cx = int((robot.x - maze.bounds[0]) / maze.resolution)
                        cy = int((robot.y - maze.bounds[1]) / maze.resolution)
                        if 0 <= cy < occ_latest.shape[0] and 0 <= cx < occ_latest.shape[1]:
                            occ_latest[cy, cx] = 0
                        tx, ty = effective_target
                        if 0 <= ty < occ_latest.shape[0] and 0 <= tx < occ_latest.shape[1]:
                            occ_latest[ty, tx] = 0
                        new_path, _ = plan_path_with_safety(
                            occ_latest,
                            (cx, cy),
                            effective_target,
                            safety_cells,
                            max_unknown_allowed=OBSTACLE_SEARCH_MAX_UNKNOWN_CELLS
                        )
                        return new_path

                    sweep_section_times.append(("path_planning", (time.perf_counter() - sweep_t0) * 1000.0))
                    reached = drive_path_with_dwa_segment(
                        path_to_follow,
                        label="障碍补扫路径",
                        replan_callback=replanner,
                        path_safety_cells=safety_cells
                    )
                    sweep_section_times.append(("dwa_follow", (time.perf_counter() - sweep_t0) * 1000.0))

                    if reached:
                        print("补充扫描完成，继续检查是否存在剩余未知区域...")
                        visited_subtargets.add(effective_target)
                        next_start_cell = effective_target
                    else:
                        print("补扫路径未能成功完成，返回起点前请留意地图覆盖情况")
                        visited_subtargets.add(effective_target)
                        next_start_cell = None
                    finish_logging()
                    if coverage_sufficient:
                        break
                else:
                    print("在指定障碍物区域内未找到满足安全距离的可达前沿，直接返回起点")
                    if bfs_debug_cells:
                        viz.set_bfs_debug_points(bfs_debug_cells, color='magenta')
                        plt.pause(0.8)
                    viz.set_bfs_debug_points(None)
            else:
                print("障碍物边界范围内没有未探索区域，直接返回起点")
        else:
            print("未发现障碍物区域，直接返回起点")
            viz.set_obstacle_search_region(None)
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
    # 若默认安全距离无解，则逐步放宽，但不低于1栅格
    for shrink in range(safety_cells_nominal - 1, 0, -1):
        safety_candidates.append(float(shrink))

    back_path = None
    used_safety = None
    for sd in safety_candidates:
        candidate = explorer.plan_path(
            occupancy_return,
            (current_idx_x, current_idx_y),
            (start_idx_x, start_idx_y),
            safety_distance=sd,
            max_unknown_cells=RETURN_MAX_UNKNOWN_CELLS
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
        min_safety = 1.0 if safety_cells_nominal_f >= 1.0 else safety_cells_nominal_f
        print(f"无法在安全距离 {safety_cells_nominal_f:.1f}~{min_safety:.1f} 栅格范围内规划返程路径")

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

        initial_path_cells = list(path_cells)
        initial_path_arr = np.array([
            [
                maze.bounds[0] + (cx + 0.5) * maze.resolution,
                maze.bounds[1] + (cy + 0.5) * maze.resolution
            ]
            for cx, cy in initial_path_cells
        ], dtype=float)

        path_safety_val = float(used_safety) if used_safety is not None else None

        def replan_return_path():
            pose_now = robot.get_pose()
            current_idx_x_update = int((pose_now[0] - maze.bounds[0]) / maze.resolution)
            current_idx_y_update = int((pose_now[1] - maze.bounds[1]) / maze.resolution)

            occupancy_update = slam.get_occupancy().copy()
            if 0 <= current_idx_y_update < occupancy_update.shape[0] and 0 <= current_idx_x_update < occupancy_update.shape[1]:
                occupancy_update[current_idx_y_update, current_idx_x_update] = 0
            if 0 <= start_idx_y < occupancy_update.shape[0] and 0 <= start_idx_x < occupancy_update.shape[1]:
                occupancy_update[start_idx_y, start_idx_x] = 0

            safety_cells_return = max(1, int(round((dwa_cfg.robot_radius + dwa_cfg.safety_clearance) / maze.resolution)))

            anchor_path_cells = None
            anchor_idx = None
            if len(initial_path_cells) >= 2 and initial_path_arr.shape[0] == len(initial_path_cells):
                dists_initial = np.hypot(initial_path_arr[:, 0] - pose_now[0], initial_path_arr[:, 1] - pose_now[1])
                nearest_on_initial = int(np.argmin(dists_initial))
                lookahead_offset = max(RETURN_REPLAN_ANCHOR_LOOKAHEAD, RETURN_REPLAN_MIN_ADVANCE)
                anchor_idx = min(len(initial_path_cells) - 1, nearest_on_initial + lookahead_offset)
                if anchor_idx <= nearest_on_initial:
                    anchor_idx = min(len(initial_path_cells) - 1, nearest_on_initial + RETURN_REPLAN_MIN_ADVANCE)
                anchor_cell = tuple(initial_path_cells[anchor_idx])
                anchor_path_cells = explorer.plan_path(
                    occupancy_update,
                    (current_idx_x_update, current_idx_y_update),
                    anchor_cell,
                    safety_distance=float(safety_cells_return),
                    max_unknown_cells=RETURN_MAX_UNKNOWN_CELLS
                )

            updated_path = None
            if anchor_path_cells and len(anchor_path_cells) >= 2 and anchor_idx is not None:
                updated_path = list(anchor_path_cells)
                remainder = initial_path_cells[anchor_idx + 1:]
                if remainder:
                    if updated_path[-1] != initial_path_cells[anchor_idx]:
                        updated_path.append(initial_path_cells[anchor_idx])
                    updated_path.extend(remainder)
            else:
                updated_path = explorer.plan_path(
                    occupancy_update,
                    (current_idx_x_update, current_idx_y_update),
                    (start_idx_x, start_idx_y),
                    safety_distance=float(safety_cells_return),
                    max_unknown_cells=RETURN_MAX_UNKNOWN_CELLS
                )

            if (not updated_path or len(updated_path) < 2) and path_safety_val is not None:
                updated_path = explorer.plan_path(
                    occupancy_update,
                    (current_idx_x_update, current_idx_y_update),
                    (start_idx_x, start_idx_y),
                    safety_distance=path_safety_val,
                    max_unknown_cells=RETURN_MAX_UNKNOWN_CELLS
                )

            if not updated_path or len(updated_path) < 2:
                updated_path = explorer.plan_path_no_safety(
                    occupancy_update,
                    (current_idx_x_update, current_idx_y_update),
                    (start_idx_x, start_idx_y)
                )

            return updated_path

        if return_traj_split_idx is None:
            return_traj_split_idx = len(robot.trajectory)

        def return_viz_context(goal_cell, active_path_cells, nearest_idx):
            actual_points = robot.trajectory
            actual_style = return_traj_style
            extra_segments = []
            if return_traj_split_idx is not None:
                explore_segment = robot.trajectory[:return_traj_split_idx]
                return_segment = robot.trajectory[return_traj_split_idx:]
                if len(return_segment) >= 2:
                    actual_points = return_segment
                if len(explore_segment) >= 2:
                    extra_segments.append({
                        'points': explore_segment,
                        'style': explore_traj_style
                    })
            return {
                'actual_traj': actual_points,
                'actual_traj_style': actual_style,
                'extra_trajs': extra_segments or None
            }

        return drive_path_with_dwa_segment(
            path_cells,
            label=label,
            arrival_tol=0.18,
            max_iter_factor=80,
            replan_callback=replan_return_path,
            replan_interval=18,
            path_safety_cells=path_safety_val,
            viz_context_provider=return_viz_context
        )

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
    
    robot.stop_threaded()

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
