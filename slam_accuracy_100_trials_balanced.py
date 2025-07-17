#!/usr/bin/env python3
"""
SLAM建模准确度100次批量测试程序 - 平衡版
运行100次完整的迷宫SLAM建模，分析每次的准确度，并绘制准确度曲线图
使用MPS加速，调整探索参数以平衡覆盖率和效率
"""

import time
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from maze_loader import MazeLoader
from robot import Robot
from lidar import Lidar
from icp_slam import ICPSlam    
from frontier_explorer import FrontierExplorer
from visualizer import Visualizer
from noise_filter import NoiseFilter

# 强制使用MPS设备
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print("强制启用MPS (Apple Metal) 加速")

# ==================== 系统参数配置 ====================
# 迷宫和机器人参数
MAZE_FILE = "3.json"  # 迷宫文件
ROBOT_ODOM_NOISE = (0.01, math.radians(0.01))  # 里程计噪声
VIRTUAL_WALL_RESOLUTION_FACTOR = 2
VIRTUAL_WALL_Y_OFFSET = -1

# 激光雷达参数
LIDAR_MAX_RANGE = 12.0
LIDAR_ANGLE_RESOLUTION = 1.0
LIDAR_NOISE = 0.035

# 探索参数 - 平衡版本
SAFETY_DISTANCE_FACTOR = 0.7
FRONTIER_SAFETY_DISTANCE = 6.0
MIN_EXPLORATION_DISTANCE = 8.0   # 降低最小探索距离
MIN_FRONTIERS_TO_EXPLORE = 5     # 降低最小探索前沿数量
MIN_NO_OBSTACLE_COUNT = 80       # 出口检测阈值

# 运动控制精度参数
ROTATION_THRESHOLD = 1e-3
MOVEMENT_THRESHOLD = 1e-6

# 测试参数
NUM_TRIALS = 100  # 运行次数
ENABLE_PROCESS_VISUALIZATION = False  # 关闭过程可视化以加速

def compare_maps(true_maze, slam_occupancy, maze_bounds, resolution):
    """
    比较真实地图和SLAM建模地图，计算偏差率
    
    Args:
        true_maze: Maze对象，包含真实的墙壁信息
        slam_occupancy: SLAM建模的占用栅格图
        maze_bounds: 地图边界 (min_x, min_y, max_x, max_y)
        resolution: 栅格分辨率
    
    Returns:
        dict: 包含各种偏差指标的字典
    """
    # 创建真实地图的占用栅格
    height, width = slam_occupancy.shape
    true_occupancy = np.full((height, width), -1, dtype=np.int8)  # -1表示未知，0表示自由，1表示障碍物
    
    min_x, min_y, max_x, max_y = maze_bounds
    
    # 将真实墙壁转换为栅格表示
    for wall in true_maze.walls:
        (x1, y1), (x2, y2) = wall
        # 使用Bresenham算法在墙壁线段上生成栅格点
        steps = max(abs(x2 - x1), abs(y2 - y1)) / resolution
        steps = int(steps) + 1
        
        for i in range(steps):
            t = i / max(1, steps - 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # 转换为栅格索引
            grid_x = int((x - min_x) / resolution)
            grid_y = int((y - min_y) / resolution)
            
            # 检查边界
            if 0 <= grid_x < width and 0 <= grid_y < height:
                true_occupancy[grid_y, grid_x] = 1  # 标记为障碍物
    
    # 将真实地图中非障碍物区域标记为自由空间（简化处理）
    for y in range(height):
        for x in range(width):
            if true_occupancy[y, x] == -1:  # 未标记的区域
                # 检查是否在迷宫边界内的合理范围
                world_x = min_x + x * resolution
                world_y = min_y + y * resolution
                if min_x <= world_x <= max_x and min_y <= world_y <= max_y:
                    true_occupancy[y, x] = 0  # 标记为自由空间
    
    # 统计比较结果
    total_cells = 0
    correct_cells = 0
    obstacle_mismatch = 0
    free_mismatch = 0
    slam_explored_cells = 0
    
    true_obstacles = 0
    slam_obstacles = 0
    true_free = 0
    slam_free = 0
    
    for y in range(height):
        for x in range(width):
            true_val = true_occupancy[y, x]
            slam_val = slam_occupancy[y, x]
            
            # 只比较SLAM已探索的区域
            if slam_val != -1:  # SLAM已探索
                slam_explored_cells += 1
                total_cells += 1
                
                if true_val == 1:  # 真实障碍物
                    true_obstacles += 1
                elif true_val == 0:  # 真实自由空间
                    true_free += 1
                
                if slam_val == 1:  # SLAM检测到障碍物
                    slam_obstacles += 1
                elif slam_val == 0:  # SLAM检测到自由空间
                    slam_free += 1
                
                # 检查匹配情况
                if true_val == slam_val:
                    correct_cells += 1
                else:
                    if true_val == 1 and slam_val == 0:
                        obstacle_mismatch += 1  # 真实障碍物被误判为自由空间
                    elif true_val == 0 and slam_val == 1:
                        free_mismatch += 1  # 真实自由空间被误判为障碍物
    
    # 计算各种指标
    if total_cells > 0:
        accuracy = correct_cells / total_cells
        error_rate = 1 - accuracy
        
        # 障碍物检测精度
        if true_obstacles > 0:
            obstacle_detection_rate = (true_obstacles - obstacle_mismatch) / true_obstacles
        else:
            obstacle_detection_rate = 1.0
        
        # 自由空间检测精度
        if true_free > 0:
            free_detection_rate = (true_free - free_mismatch) / true_free
        else:
            free_detection_rate = 1.0
        
        # 覆盖率：SLAM探索的区域占总地图的比例
        total_map_cells = height * width
        coverage_rate = slam_explored_cells / total_map_cells
    else:
        accuracy = 0.0
        error_rate = 1.0
        obstacle_detection_rate = 0.0
        free_detection_rate = 0.0
        coverage_rate = 0.0
    
    return {
        'accuracy': accuracy,
        'error_rate': error_rate,
        'obstacle_detection_rate': obstacle_detection_rate,
        'free_detection_rate': free_detection_rate,
        'coverage_rate': coverage_rate,
        'total_cells': total_cells,
        'correct_cells': correct_cells,
        'obstacle_mismatch': obstacle_mismatch,
        'free_mismatch': free_mismatch,
        'true_obstacles': true_obstacles,
        'slam_obstacles': slam_obstacles,
        'true_free': true_free,
        'slam_free': slam_free,
        'slam_explored_cells': slam_explored_cells
    }

def check_exit_condition(scan, max_range=12.0, min_no_obstacle_count=80):
    """检查机器人是否走出迷宫"""
    if not scan or len(scan) == 0:
        return False
    
    no_obstacle = [dist >= max_range for dist in scan]
    no_obstacle_count = sum(no_obstacle)
    
    result = no_obstacle_count >= min_no_obstacle_count
    return result

def run_single_slam_trial(trial_num):
    """
    运行单次SLAM建模试验
    
    Args:
        trial_num: 试验编号
    
    Returns:
        dict: 包含准确度指标的字典
    """
    if trial_num % 10 == 0:
        print(f"\n========== 第 {trial_num+1}/{NUM_TRIALS} 次试验 ==========")
    else:
        print(f"第 {trial_num+1}/{NUM_TRIALS} 次试验", end=" ")
    
    try:
        # 1. 加载迷宫地图和参数
        loader = MazeLoader()
        maze = loader.load(MAZE_FILE)

        # === 自动生成入口虚拟墙壁 ===
        start_x, start_y = maze.start
        min_x, min_y, max_x, max_y = maze.bounds
        eps = maze.resolution * VIRTUAL_WALL_RESOLUTION_FACTOR
        
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
        
        # 生成虚拟墙
        virtual_walls = []
        virtual_x1 = left_wall_x 
        virtual_x2 = right_wall_x 
        virtual_y = start_y + VIRTUAL_WALL_Y_OFFSET
        virtual_walls.append(((virtual_x1, virtual_y), (virtual_x2, virtual_y)))
        virtual_walls.append(((virtual_x1, virtual_y), (virtual_x1, start_y)))
        virtual_walls.append(((virtual_x2, virtual_y), (virtual_x2, start_y)))
        # 添加虚拟墙到maze.walls
        maze.walls.extend(virtual_walls)

        # 2. 初始化机器人、传感器、SLAM等模块
        start_pose = (start_x, start_y, 0.0)
        robot = Robot(start_pose, odom_noise=ROBOT_ODOM_NOISE)
        lidar = Lidar(maze.walls, max_range=LIDAR_MAX_RANGE, angle_resolution=LIDAR_ANGLE_RESOLUTION, noise=LIDAR_NOISE)
        slam = ICPSlam(maze, start_pose)
        explorer = FrontierExplorer(safety_distance=FRONTIER_SAFETY_DISTANCE)
        
        # 创建可视化器但不显示
        viz = Visualizer(maze, robot=robot, slam=slam)
        plt.close(viz.fig)  # 立即关闭图形窗口
        
        # 初始化降噪滤波器
        noise_filter = NoiseFilter(
            enabled=True,
            lidar_filter_enabled=False,
            odom_filter_enabled=False
        )
        
        robot.set_noise_filter(noise_filter)
        lidar.set_noise_filter(noise_filter)
        
        # 3. 初始扫描并建立初始地图
        scan = lidar.scan(robot.get_pose())
        slam.update((0.0, 0.0), scan)
        
        # 初始化探索状态标志和探索进度跟踪
        total_distance_traveled = 0.0
        frontiers_explored = 0
        
        # 4. 前沿探索主循环
        max_iterations = 300  # 平衡迭代次数
        iteration = 0
        no_frontier_count = 0  # 连续无前沿计数器
        
        while iteration < max_iterations:
            iteration += 1
            
            # 检查探索完成条件
            scan = lidar.scan(robot.get_pose())
            has_sufficient_exploration = (total_distance_traveled >= MIN_EXPLORATION_DISTANCE and 
                                         frontiers_explored >= MIN_FRONTIERS_TO_EXPLORE)
            
            # 放宽出口检测条件，或者在一定距离后认为探索完成
            if (has_sufficient_exploration and check_exit_condition(scan, max_range=lidar.max_range, min_no_obstacle_count=MIN_NO_OBSTACLE_COUNT)) or total_distance_traveled >= 50.0:
                if trial_num % 10 == 0:
                    print(f"探索完成条件满足 - 已探索距离: {total_distance_traveled:.1f}m, 已探索前沿: {frontiers_explored}个")
                break
            
            # 查找最近的前沿
            rx_idx = int((robot.x - maze.bounds[0]) / maze.resolution)
            ry_idx = int((robot.y - maze.bounds[1]) / maze.resolution)
            frontier_cell, path = explorer.find_nearest_frontier(slam.get_occupancy(), (rx_idx, ry_idx))
            
            if frontier_cell is None:
                no_frontier_count += 1
                # 如果连续多次没有前沿且已有一定探索，则认为探索完成
                if no_frontier_count >= 3 and total_distance_traveled >= MIN_EXPLORATION_DISTANCE:
                    if trial_num % 10 == 0:
                        print("连续无前沿且达到最小探索距离 - 探索完成")
                    break
                # 否则继续尝试
                continue
            else:
                no_frontier_count = 0  # 重置计数器
            
            frontiers_explored += 1
            
            # 5. 沿规划路径移动机器人
            if path is None or len(path) <= 1:
                continue
                
            prev_pose = robot.get_pose()
            
            for next_cell in path[1:]:  # 跳过当前位置
                target_x = maze.bounds[0] + next_cell[0] * maze.resolution
                target_y = maze.bounds[1] + next_cell[1] * maze.resolution
                target_pos = (target_x, target_y)
                
                # 计算需要旋转的角度和移动距离
                current_x, current_y, current_theta = robot.get_pose()
                dx, dy = target_pos[0] - current_x, target_pos[1] - current_y
                target_theta = math.atan2(dy, dx)
                angular_diff = target_theta - current_theta
                
                # 标准化角度差
                while angular_diff > math.pi:
                    angular_diff -= 2 * math.pi
                while angular_diff < -math.pi:
                    angular_diff += 2 * math.pi
                
                # 旋转到目标方向
                if abs(angular_diff) > ROTATION_THRESHOLD:
                    robot.rotate(angular_diff)
                
                # 移动到目标位置
                distance = math.sqrt(dx*dx + dy*dy)
                if distance > MOVEMENT_THRESHOLD:
                    robot.move(distance)
                
                # 更新累计移动距离
                total_distance_traveled += distance
                
                # 扫描环境并更新SLAM
                current_pose = robot.get_pose()
                odometry = (current_pose[0] - prev_pose[0], current_pose[1] - prev_pose[1])
                scan = lidar.scan(current_pose)
                slam.update(odometry, scan)
                prev_pose = current_pose
        
        if trial_num % 10 == 0:
            print(f"探索完成 - 总迭代次数: {iteration}, 总移动距离: {total_distance_traveled:.1f}m, 探索前沿: {frontiers_explored}个")
        
        # 进行地图比较
        comparison_results = compare_maps(maze, slam.get_occupancy(), maze.bounds, maze.resolution)
        
        # 添加试验信息
        comparison_results['trial_num'] = trial_num + 1
        comparison_results['total_distance'] = total_distance_traveled
        comparison_results['frontiers_explored'] = frontiers_explored
        comparison_results['iterations'] = iteration
        
        if trial_num % 10 == 0:
            print(f"试验 {trial_num+1} 完成: 准确率={comparison_results['accuracy']*100:.2f}%, 覆盖率={comparison_results['coverage_rate']*100:.2f}%")
        else:
            print(f"- 准确率={comparison_results['accuracy']*100:.1f}%, 覆盖率={comparison_results['coverage_rate']*100:.1f}%")
        
        # 释放GPU资源
        if hasattr(slam, 'release_resources'):
            slam.release_resources()
        
        # 关闭matplotlib图形
        plt.close('all')
        
        return comparison_results
        
    except Exception as e:
        print(f"试验 {trial_num+1} 出错: {e}")
        
        # 返回默认的错误结果
        return {
            'trial_num': trial_num + 1,
            'accuracy': 0.0,
            'error_rate': 1.0,
            'obstacle_detection_rate': 0.0,
            'free_detection_rate': 0.0,
            'coverage_rate': 0.0,
            'total_distance': 0.0,
            'frontiers_explored': 0,
            'iterations': 0,
            'total_cells': 0,
            'correct_cells': 0,
            'slam_explored_cells': 0
        }

def plot_accuracy_curves(results):
    """绘制准确度曲线图"""
    trial_nums = [r['trial_num'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]  # 转换为百分比
    error_rates = [r['error_rate'] * 100 for r in results]
    coverage_rates = [r['coverage_rate'] * 100 for r in results]
    obstacle_detection_rates = [r['obstacle_detection_rate'] * 100 for r in results]
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 总体准确率曲线
    ax1.plot(trial_nums, accuracies, 'b-', linewidth=1.5, alpha=0.8, label='Accuracy Rate')
    ax1.axhline(y=np.mean(accuracies), color='r', linestyle='--', linewidth=2, label=f'Average: {np.mean(accuracies):.2f}%')
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Accuracy Rate (%)', fontsize=12)
    ax1.set_title('SLAM Mapping Accuracy Rate Curve (100 Trials)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 100)
    
    # 2. 偏差率曲线
    ax2.plot(trial_nums, error_rates, 'r-', linewidth=1.5, alpha=0.8, label='Error Rate')
    ax2.axhline(y=np.mean(error_rates), color='b', linestyle='--', linewidth=2, label=f'Average: {np.mean(error_rates):.2f}%')
    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('SLAM Mapping Error Rate Curve (100 Trials)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 100)
    
    # 3. 覆盖率曲线
    ax3.plot(trial_nums, coverage_rates, 'g-', linewidth=1.5, alpha=0.8, label='Coverage Rate')
    ax3.axhline(y=np.mean(coverage_rates), color='orange', linestyle='--', linewidth=2, label=f'Average: {np.mean(coverage_rates):.2f}%')
    ax3.set_xlabel('Trial Number', fontsize=12)
    ax3.set_ylabel('Coverage Rate (%)', fontsize=12)
    ax3.set_title('SLAM Exploration Coverage Rate Curve (100 Trials)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 100)
    
    # 4. 障碍物检测率曲线
    ax4.plot(trial_nums, obstacle_detection_rates, 'purple', linewidth=1.5, alpha=0.8, label='Obstacle Detection Rate')
    ax4.axhline(y=np.mean(obstacle_detection_rates), color='cyan', linestyle='--', linewidth=2, label=f'Average: {np.mean(obstacle_detection_rates):.2f}%')
    ax4.set_xlabel('Trial Number', fontsize=12)
    ax4.set_ylabel('Obstacle Detection Rate (%)', fontsize=12)
    ax4.set_title('SLAM Obstacle Detection Rate Curve (100 Trials)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('slam_accuracy_analysis_100_trials_balanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n准确度曲线图已保存为: slam_accuracy_analysis_100_trials_balanced.png")

def analyze_results(results):
    """分析100次试验的统计结果"""
    print(f"\n{'='*70}")
    print(f"100次SLAM建模试验统计分析 - 平衡版")
    print(f"{'='*70}")
    
    # 计算统计指标
    accuracies = [r['accuracy'] * 100 for r in results if r['accuracy'] > 0]
    error_rates = [r['error_rate'] * 100 for r in results if r['error_rate'] < 1]
    coverage_rates = [r['coverage_rate'] * 100 for r in results if r['coverage_rate'] > 0]
    obstacle_detection_rates = [r['obstacle_detection_rate'] * 100 for r in results if r['obstacle_detection_rate'] > 0]
    distances = [r['total_distance'] for r in results if r['total_distance'] > 0]
    frontiers = [r['frontiers_explored'] for r in results if r['frontiers_explored'] > 0]
    
    if accuracies:
        print(f"\n总体准确率统计:")
        print(f"  - 平均值: {np.mean(accuracies):.2f}%")
        print(f"  - 标准差: {np.std(accuracies):.2f}%")
        print(f"  - 最大值: {np.max(accuracies):.2f}%")
        print(f"  - 最小值: {np.min(accuracies):.2f}%")
        print(f"  - 中位数: {np.median(accuracies):.2f}%")
        print(f"  - 90分位数: {np.percentile(accuracies, 90):.2f}%")
        print(f"  - 10分位数: {np.percentile(accuracies, 10):.2f}%")
    
    if coverage_rates:
        print(f"\n探索覆盖率统计:")
        print(f"  - 平均值: {np.mean(coverage_rates):.2f}%")
        print(f"  - 标准差: {np.std(coverage_rates):.2f}%")
        print(f"  - 最大值: {np.max(coverage_rates):.2f}%")
        print(f"  - 最小值: {np.min(coverage_rates):.2f}%")
        print(f"  - 中位数: {np.median(coverage_rates):.2f}%")
    
    if obstacle_detection_rates:
        print(f"\n障碍物检测率统计:")
        print(f"  - 平均值: {np.mean(obstacle_detection_rates):.2f}%")
        print(f"  - 标准差: {np.std(obstacle_detection_rates):.2f}%")
        print(f"  - 最大值: {np.max(obstacle_detection_rates):.2f}%")
        print(f"  - 最小值: {np.min(obstacle_detection_rates):.2f}%")
    
    if distances:
        print(f"\n探索距离统计:")
        print(f"  - 平均值: {np.mean(distances):.1f}m")
        print(f"  - 标准差: {np.std(distances):.1f}m")
        print(f"  - 最大值: {np.max(distances):.1f}m")
        print(f"  - 最小值: {np.min(distances):.1f}m")
    
    if frontiers:
        print(f"\n探索前沿数统计:")
        print(f"  - 平均值: {np.mean(frontiers):.1f}个")
        print(f"  - 标准差: {np.std(frontiers):.1f}个")
        print(f"  - 最大值: {np.max(frontiers)}个")
        print(f"  - 最小值: {np.min(frontiers)}个")
    
    # 成功试验统计
    successful_trials = len([r for r in results if r['accuracy'] > 0])
    print(f"\n试验成功率: {successful_trials}/{NUM_TRIALS} ({successful_trials}%)")
    
    # 高质量试验统计（准确率>60%且覆盖率>15%）
    high_quality_trials = len([r for r in results if r['accuracy'] > 0.6 and r['coverage_rate'] > 0.15])
    print(f"高质量试验数 (准确率>60%且覆盖率>15%): {high_quality_trials}/{NUM_TRIALS} ({high_quality_trials}%)")
    
    # 找出最佳和最差的试验
    if accuracies:
        best_trial = max(results, key=lambda x: x['accuracy'])
        worst_trial = min([r for r in results if r['accuracy'] > 0], key=lambda x: x['accuracy'])
        
        print(f"\n最佳试验 (第{best_trial['trial_num']}次):")
        print(f"  - 准确率: {best_trial['accuracy']*100:.2f}%")
        print(f"  - 覆盖率: {best_trial['coverage_rate']*100:.2f}%")
        print(f"  - 障碍物检测率: {best_trial['obstacle_detection_rate']*100:.2f}%")
        print(f"  - 探索前沿数: {best_trial['frontiers_explored']}")
        print(f"  - 总移动距离: {best_trial['total_distance']:.1f}m")
        print(f"  - 迭代次数: {best_trial['iterations']}")
        
        print(f"\n最差试验 (第{worst_trial['trial_num']}次):")
        print(f"  - 准确率: {worst_trial['accuracy']*100:.2f}%")
        print(f"  - 覆盖率: {worst_trial['coverage_rate']*100:.2f}%")
        print(f"  - 障碍物检测率: {worst_trial['obstacle_detection_rate']*100:.2f}%")
        print(f"  - 探索前沿数: {worst_trial['frontiers_explored']}")
        print(f"  - 总移动距离: {worst_trial['total_distance']:.1f}m")
        print(f"  - 迭代次数: {worst_trial['iterations']}")

def main():
    """主函数：运行100次SLAM建模试验"""
    print("开始100次SLAM建模准确度批量测试 - 平衡版...")
    print("使用MPS (Apple Metal) 加速")
    print(f"迷宫文件: {MAZE_FILE}")
    print(f"过程可视化: {'启用' if ENABLE_PROCESS_VISUALIZATION else '关闭'}")
    print(f"探索参数: 最小距离={MIN_EXPLORATION_DISTANCE}m, 最小前沿={MIN_FRONTIERS_TO_EXPLORE}个")
    
    results = []
    start_time = time.time()
    
    for trial in range(NUM_TRIALS):
        trial_start = time.time()
        result = run_single_slam_trial(trial)
        trial_end = time.time()
        
        result['trial_duration'] = trial_end - trial_start
        results.append(result)
        
        # 显示进度（每10次输出详细信息）
        if (trial + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time_per_trial = elapsed / (trial + 1)
            estimated_total = avg_time_per_trial * NUM_TRIALS
            remaining = estimated_total - elapsed
            
            print(f"进度: {trial+1}/{NUM_TRIALS}, 平均用时: {avg_time_per_trial:.1f}s/次, 预计剩余: {remaining/60:.1f}分钟")
            
            # 阶段性统计
            completed_results = [r for r in results if r['accuracy'] > 0]
            if completed_results:
                avg_accuracy = np.mean([r['accuracy'] * 100 for r in completed_results])
                avg_coverage = np.mean([r['coverage_rate'] * 100 for r in completed_results])
                print(f"  >>> 阶段性统计: 平均准确率={avg_accuracy:.2f}%, 平均覆盖率={avg_coverage:.2f}%")
    
    total_time = time.time() - start_time
    print(f"\n所有试验完成！总用时: {total_time/60:.1f}分钟")
    
    # 分析结果
    analyze_results(results)
    
    # 绘制准确度曲线图
    plot_accuracy_curves(results)
    
    # 保存详细结果到文件
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f'slam_accuracy_results_100_trials_balanced_{timestamp}.json'
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"详细结果已保存到: {results_filename}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {e}")
        import traceback
        traceback.print_exc()
