import time
import math
import os
import matplotlib.pyplot as plt
from maze_loader import MazeLoader
from robot import Robot
from lidar import Lidar
from icp_slam import ICPSlam    
from frontier_explorer import FrontierExplorer
from visualizer import Visualizer
from noise_filter import NoiseFilter
import numpy as np

# ==================== 系统参数配置 ====================
# 路径规划参数
SAFETY_DISTANCE_FACTOR = 0.7  # 路径截断百分比，表示只执行路径的前70%
FRONTIER_SAFETY_DISTANCE = 6.0  # 前沿探索器与障碍物的安全距离

# 迷宫和机器人参数
MAZE_FILE = "3.json"  # 默认迷宫文件
ROBOT_ODOM_NOISE = (0.01, math.radians(0.01))  # trans_noise, self.rot_noise = odom_noise (0.01, math.radians(1)))
VIRTUAL_WALL_RESOLUTION_FACTOR = 2  # 虚拟墙分辨率因子
VIRTUAL_WALL_Y_OFFSET = -1  # 虚拟墙Y方向偏移

# 激光雷达参数
LIDAR_MAX_RANGE = 12.0  # 激光雷达扫描半径
LIDAR_ANGLE_RESOLUTION = 1.0  # 激光雷达角度分辨率
LIDAR_NOISE = 0.035  # 激光雷达噪声

# 出口检测参数
MIN_NO_OBSTACLE_COUNT = 100  # 无障碍点数阈值，超过此数值认为走出迷宫

# 探索阈值参数
MIN_EXPLORATION_DISTANCE = 20.0  # 最小探索距离阈值
MIN_FRONTIERS_TO_EXPLORE = 10  # 最小探索前沿数量

# 运动控制精度参数
ROTATION_THRESHOLD = 1e-3  # 旋转角度阈值
MOVEMENT_THRESHOLD = 1e-6  # 移动距离阈值

# 可视化参数
VISUALIZATION_PAUSE_TIME = 0.005  # 暂停时的等待时间
VISUALIZATION_UPDATE_TIME = 0.0001  # 可视化更新时间

# 未探索区域搜索参数
OBSTACLE_SEARCH_EXPANSION = 0.5  # 障碍物区域搜索范围扩大距离（米）

# ==================== 降噪滤波参数 ====================
# 滤波器总开关
NOISE_FILTER_ENABLED = False  # 是否启用降噪滤波器

# 激光雷达降噪参数
LIDAR_FILTER_TYPE = 'median'  # 激光雷达滤波类型: 'none', 'median', 'moving_average', 'gaussian'
LIDAR_FILTER_WINDOW_SIZE = 5  # 激光雷达滤波窗口大小

# 里程计降噪参数  
ODOM_FILTER_TYPE = 'kalman'  # 里程计滤波类型: 'none', 'kalman', 'moving_average'
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
    # 初始朝向设为0（朝向x正方向）
    start_pose = (start_x, start_y, 0.0)
    robot = Robot(start_pose, odom_noise=ROBOT_ODOM_NOISE)  # 设置一定里程计噪声
    lidar = Lidar(maze.walls, max_range=LIDAR_MAX_RANGE, angle_resolution=LIDAR_ANGLE_RESOLUTION, noise=LIDAR_NOISE)
    slam = ICPSlam(maze, start_pose)
    explorer = FrontierExplorer(safety_distance=FRONTIER_SAFETY_DISTANCE)  # 设置与障碍物的安全距离
    viz = Visualizer(maze, robot=robot, slam=slam)
    
    # 初始化降噪滤波器
    noise_filter = NoiseFilter(
        enabled=NOISE_FILTER_ENABLED,
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
    scan = lidar.scan(robot.get_pose())
    slam.update((0.0, 0.0), scan)
    # 初始可视化
    robot_pose = robot.get_pose()
    frontiers = explorer.find_frontiers(slam.get_occupancy())
    target_cell = None
    path = None
    viz.update(robot_pose, scan, frontiers=frontiers, target=target_cell, path=path, occupancy=slam.get_occupancy())
    
    # 初始化探索状态标志和探索进度跟踪
    exploration_complete = False
    total_distance_traveled = 0.0  # 总移动距离
    frontiers_explored = 0  # 已探索的前沿数量
    min_exploration_distance = MIN_EXPLORATION_DISTANCE  # 最小探索距离阈值
    min_frontiers_to_explore = MIN_FRONTIERS_TO_EXPLORE  # 最小探索前沿数量
    
    # 安全移动参数
    safety_distance_factor = SAFETY_DISTANCE_FACTOR  # 路径截断百分比
    
    # 4. 前沿探索主循环
    while True:
        # 检查暂停状态
        if viz.paused:
            # 暂停循环，直到恢复
            plt.pause(VISUALIZATION_PAUSE_TIME)  # 减少暂停时的等待时间，提高响应速度
            continue
        
        # 注意：路径规划会保持与障碍物的安全距离，防止穿墙
        # 只有在进行了充分探索后才检查是否检测到超过180度的连续无障碍区域
        scan = lidar.scan(robot.get_pose())
        has_sufficient_exploration = (total_distance_traveled >= min_exploration_distance and 
                                     frontiers_explored >= min_frontiers_to_explore)
        
        if has_sufficient_exploration and check_exit_condition(scan, max_range=lidar.max_range, min_no_obstacle_count=MIN_NO_OBSTACLE_COUNT):
            print(f"检测到超过180度的连续无障碍区域 - 已探索距离: {total_distance_traveled:.1f}m, 已探索前沿: {frontiers_explored}个")
            print("探索充分，开始返回起点！")
            # 更新SLAM和可视化
            slam.update((0.0, 0.0), scan)
            robot_pose = robot.get_pose()
            frontiers = explorer.find_frontiers(slam.get_occupancy())
            viz.update(robot_pose, scan, frontiers=frontiers, target=None, path=None, occupancy=slam.get_occupancy())
            # 设置标志表示需要返回起点
            should_return_to_start = True
            break
        
        # 查找最近的前沿
        # 获取机器人当前所在栅格索引
        rx_idx = int((robot.x - maze.bounds[0]) / maze.resolution)
        ry_idx = int((robot.y - maze.bounds[1]) / maze.resolution)
        frontier_cell, path = explorer.find_nearest_frontier(slam.get_occupancy(), (rx_idx, ry_idx))
        if frontier_cell is None:
            # 没有前沿但还未走出迷宫，可能是探索完成但仍在迷宫内
            print("没有更多前沿但仍在迷宫内 - 尝试寻找出口...")
            # 可以在这里添加寻找出口的逻辑，暂时继续执行原逻辑
            print("迷宫内部探索完成。")
            break
        
        # 打印路径信息来验证对角线移动
        print(f"Planning path to frontier {frontier_cell}")
        if path and len(path) > 1:
            diagonal_moves = 0
            for i in range(len(path) - 1):
                dx = abs(path[i+1][0] - path[i][0])
                dy = abs(path[i+1][1] - path[i][1])
                if dx == 1 and dy == 1:
                    diagonal_moves += 1
            print(f"Path length: {len(path)}, Diagonal moves: {diagonal_moves}")
        
        # 在可视化中标记当前目标前沿
        target_cell = frontier_cell
        # 增加已探索前沿计数
        frontiers_explored += 1
        print(f"开始探索第 {frontiers_explored} 个前沿点: {frontier_cell}")
        
        # 5. 沿规划路径移动机器人
        if path:  # 确保路径存在
            # 对路径进行百分比截断，只执行前safety_distance_factor比例的路径
            if len(path) > 1:  # 确保路径至少有两个点
                path_length = len(path) - 1  # 减去当前位置
                truncated_length = max(1, int(path_length * safety_distance_factor))  # 至少保留一步
                truncated_path = path[:truncated_length+1]  # +1是因为path[0]是当前位置
                print(f"路径截断: 原路径长度={path_length}，截断后长度={truncated_length} (保留{safety_distance_factor*100:.0f}%)")
                path = truncated_path  # 使用截断后的路径
            
            for step in path[1:]:  # path[0] 是当前位置
                # 计算目标栅格中心的世界坐标
                ix, iy = step
                target_x = maze.bounds[0] + (ix + 0.5) * maze.resolution
                target_y = maze.bounds[1] + (iy + 0.5) * maze.resolution
                # 计算需要旋转的角度
                dx = target_x - robot.x
                dy = target_y - robot.y
                desired_theta = math.atan2(dy, dx)
                # 计算最小旋转角度差
                d_theta = desired_theta - robot.theta
                # 将角度差规范化到[-pi, pi]
                d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
                if abs(d_theta) > ROTATION_THRESHOLD:
                    # 执行旋转
                    old_odom_theta = robot.odom_theta
                    robot.rotate(d_theta)
                    # 计算里程计角增量
                    dtheta_odom = robot.odom_theta - old_odom_theta
                    # 获取旋转后的扫描数据并更新SLAM
                    scan = lidar.scan(robot.get_pose())
                    slam.update((0.0, dtheta_odom), scan)
                    # 更新可视化
                    viz.update(robot.get_pose(), scan, frontiers=frontiers, target=target_cell, path=path, occupancy=slam.get_occupancy())
                # 前进到目标格中心，不再缩短距离
                distance = math.hypot(target_x - robot.x, target_y - robot.y)
                #print(f"目标点: ({target_x:.2f}, {target_y:.2f}), 距离: {distance:.2f}")
                if distance > 0:
                    # 执行移动，使用完整距离
                    old_odom_x, old_odom_y = robot.odom_x, robot.odom_y
                    robot.move(distance)  # 移动到目标位置
                    # 计算里程计距离增量（直线移动，朝向不变）
                    d_trans = math.hypot(robot.odom_x - old_odom_x, robot.odom_y - old_odom_y)
                    total_distance_traveled += d_trans  # 累计总移动距离
                    scan = lidar.scan(robot.get_pose())
                    slam.update((d_trans, 0.0), scan)
                    
                    # 只有在进行了充分探索后才检查180度条件
                    has_sufficient_exploration = (total_distance_traveled >= min_exploration_distance and 
                                                 frontiers_explored >= min_frontiers_to_explore)
                    
                    if has_sufficient_exploration and check_exit_condition(scan, max_range=lidar.max_range, min_no_obstacle_count=MIN_NO_OBSTACLE_COUNT):
                        print(f"移动过程中检测到超过180度连续无障碍区域！")
                        print(f"探索统计 - 总距离: {total_distance_traveled:.1f}m, 已探索前沿: {frontiers_explored}个")
                        robot_pose = robot.get_pose()
                        frontiers = explorer.find_frontiers(slam.get_occupancy())
                        viz.update(robot_pose, scan, frontiers=frontiers, target=target_cell, path=path, occupancy=slam.get_occupancy())
                        # 设置标志表示需要返回起点
                        should_return_to_start = True
                        break  # 跳出移动循环
                    
                    # 更新可视化
                    robot_pose = robot.get_pose()
                    frontiers = explorer.find_frontiers(slam.get_occupancy())
                    viz.update(robot_pose, scan, frontiers=frontiers, target=target_cell, path=path, occupancy=slam.get_occupancy())
            
            # 如果在移动过程中检测到需要返回起点的条件，跳出外层循环
            if 'should_return_to_start' in locals() and should_return_to_start:
                break
                
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
                                scan = lidar.scan(robot.get_pose())
                                slam.update((0.0, d_theta), scan)
                                viz.update(robot.get_pose(), scan, frontiers=None, target=closest_unexplored, path=unexplored_path, occupancy=slam.get_occupancy())
                            
                            distance = math.hypot(target_x - robot.x, target_y - robot.y)
                            if distance > MOVEMENT_THRESHOLD:
                                robot.move(distance)
                                scan = lidar.scan(robot.get_pose())
                                slam.update((distance, 0.0), scan)
                                viz.update(robot.get_pose(), scan, frontiers=None, target=closest_unexplored, path=unexplored_path, occupancy=slam.get_occupancy())
                        
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
    
    # 生成带安全距离的路径
    back_path = explorer.plan_path(slam.get_occupancy(), (current_idx_x, current_idx_y), (start_idx_x, start_idx_y))
    
    # 生成不带安全距离的路径用于显示
    back_path_no_safety = explorer.plan_path_no_safety(slam.get_occupancy(), (current_idx_x, current_idx_y), (start_idx_x, start_idx_y))
    
    if back_path_no_safety:
        # 计算无安全距离路径的长度
        path_length_meters = explorer.calculate_path_length(back_path_no_safety, maze.resolution)
        print(f"无安全距离最短路径长度: {path_length_meters:.2f} 米 ({len(back_path_no_safety)-1} 栅格步数)")
        
        # 在可视化中显示红色路径
        viz.set_emergency_path(back_path_no_safety)
    
    if back_path:
        print("Returning to start...")
        print(f"安全返回路径长度: {len(back_path)-1} 步，直接走到起点")
        
        for step in back_path[1:]:
                ix, iy = step
                target_x = maze.bounds[0] + (ix + 0.5) * maze.resolution
                target_y = maze.bounds[1] + (iy + 0.5) * maze.resolution
                dx = target_x - robot.x
                dy = target_y - robot.y
                desired_theta = math.atan2(dy, dx)
                d_theta = desired_theta - robot.theta
                d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
                if abs(d_theta) > ROTATION_THRESHOLD:
                    old_odom_theta = robot.odom_theta
                    robot.rotate(d_theta)
                    dtheta_odom = robot.odom_theta - old_odom_theta
                    # 返回时获取扫描数据但不用于SLAM建图，仅用于可视化
                    scan = lidar.scan(robot.get_pose())
                    # slam.update((0.0, dtheta_odom), scan)  # 注释掉SLAM更新
                    viz.update(robot.get_pose(), scan, frontiers=None, target=None, path=back_path, occupancy=slam.get_occupancy())
                distance = math.hypot(target_x - robot.x, target_y - robot.y)
                if distance > MOVEMENT_THRESHOLD:
                    # 执行移动，使用完整距离
                    old_odom_x, old_odom_y = robot.odom_x, robot.odom_y
                    robot.move(distance)
                    d_trans = math.hypot(robot.odom_x - old_odom_x, robot.odom_y - old_odom_y)
                    # 返回时获取扫描数据但不用于SLAM建图，仅用于可视化
                    scan = lidar.scan(robot.get_pose())
                    # slam.update((d_trans, 0.0), scan)  # 注释掉SLAM更新
                    viz.update(robot.get_pose(), scan, frontiers=None, target=None, path=back_path, occupancy=slam.get_occupancy())
        print("Robot returned to start.")
    else:
        print("无法规划返回起点的路径。")
    
    # 根据结束条件输出相应信息
    if 'should_return_to_start' in locals() and should_return_to_start:
        print("仿真结束：检测到超过180度连续无障碍区域，机器人已返回起点！")
    else:
        print("仿真结束：迷宫探索完成，机器人已返回起点。")
        
    # 导出最终地图和路径
    viz.save_map("final_map.png")
    viz.save_path("final_path.csv")
    
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
