#!/usr/bin/env python3
"""
SLAM参数调优程序 - 控制变量法
对关键参数进行逐一调优，每组参数运行100次实验，绘制准确率随参数变化的曲线图
使用MPS加速，关闭过程可视化
"""

import time
import math
import os
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import json
from maze_loader import MazeLoader
from robot import Robot
from lidar import Lidar
from icp_slam import ICPSlam    
from frontier_explorer import FrontierExplorer
from visualizer import Visualizer
from noise_filter import NoiseFilter
import psutil
import gc
import traceback
import uuid
import glob
import icp_slam

# 关闭matplotlib所有交互功能
plt.ioff()

# 导入torch但延迟设备初始化
import torch

def initialize_device():
    """初始化并检测最佳计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用CUDA加速: {torch.cuda.get_device_name(0)}")
        print(f"CUDA显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # 设置CUDA内存分配策略
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("使用MPS (Apple Metal) 加速")
    else:
        device = torch.device("cpu")
        print("使用CPU计算")

    # 全局设备配置
    os.environ["SLAM_DEVICE"] = str(device)
    return device

# 延迟初始化设备，避免模块导入时的潜在问题
device = None

# ==================== 内存管理工具函数 ====================
def get_memory_usage():
    """获取当前内存和GPU显存使用情况"""
    info = {"timestamp": time.strftime("%H:%M:%S")}
    try:
        process = psutil.Process(os.getpid())
        info["cpu_memory_mb"] = process.memory_info().rss / (1024 * 1024)
        info["cpu_memory_percent"] = process.memory_percent()
    except:
        info["cpu_memory_mb"] = 0
        info["cpu_memory_percent"] = 0
    
    # GPU显存使用情况
    if torch.cuda.is_available():
        info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        info["gpu_allocated_mb"] = 0
        info["gpu_reserved_mb"] = 0
        info["gpu_max_allocated_mb"] = 0
    
    return info

def force_cleanup():
    """强制执行内存清理"""
    try:
        # 关闭所有matplotlib图形
        plt.close('all')
        
        # GPU显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理临时文件
        temp_patterns = ["trial_temp_*.json", "incremental_*.json", "temp_*.json"]
        for pattern in temp_patterns:
            for temp_file in glob.glob(pattern):
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
    except Exception as e:
        print(f"强制清理时出错: {e}")

def monitor_memory_usage(operation_name="操作"):
    """内存使用监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 执行前记录
            before = get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 执行后记录和清理
                after = get_memory_usage()
                
                # 计算内存变化
                cpu_diff = after["cpu_memory_mb"] - before["cpu_memory_mb"]
                gpu_diff = after["gpu_allocated_mb"] - before["gpu_allocated_mb"]
                
                if abs(cpu_diff) > 10 or abs(gpu_diff) > 10:  # 只在显著变化时打印
                    print(f"  [{operation_name}] 内存变化: CPU {cpu_diff:+.1f}MB, GPU {gpu_diff:+.1f}MB")
                
                # 如果内存使用过高，执行清理
                if after["cpu_memory_mb"] > 2000 or after["gpu_allocated_mb"] > 1000:
                    print(f"  内存使用较高，执行清理...")
                    force_cleanup()
        
        return wrapper
    return decorator

# ==================== 基础参数配置 ====================
# 迷宫和机器人参数
MAZE_FILE = "3.json"
ROBOT_ODOM_NOISE = (0.01, math.radians(0.01))
VIRTUAL_WALL_RESOLUTION_FACTOR = 2
VIRTUAL_WALL_Y_OFFSET = -1

# 激光雷达参数
LIDAR_MAX_RANGE = 12.0
LIDAR_ANGLE_RESOLUTION = 1.0
LIDAR_NOISE = 0.035

# 出口检测参数
MIN_NO_OBSTACLE_COUNT = 95

# 运动控制精度参数
ROTATION_THRESHOLD = 1e-3
MOVEMENT_THRESHOLD = 1e-6

# 测试参数
NUM_TRIALS_PER_PARAM = 50  # 每组参数运行的试验次数
ENABLE_PROCESS_VISUALIZATION = False  # 关闭过程可视化以加速

# ==================== 待调优参数定义 ====================
PARAM_CONFIGS = {
    'MAX_RANGE_FACTOR': {
        'range': np.arange(0.10, 0.91, 0.02),  # 0.1到0.9，步长0.02，共40个点
        'default': 0.49,
        'description': '超过最大范围的比例阈值'
    },
    'ADJACENCY_DIFF_THRESHOLD': {
        'range': np.arange(0.005, 0.51, 0.002),  # 0.005到0.51，步长0.002，共253个点
        'default': 0.01,
        'description': '相邻测距点之间的差异阈值 (米)'
    },
    'ICP_MAX_ITER': {
        'range': np.arange(100, 2001, 10),  # 100到2000，步长10，共191个点
        'default': 500,
        'description': 'ICP最大迭代次数'
    },
    'SAFETY_DISTANCE_FACTOR': {
        'range': np.arange(0.20, 0.91, 0.02),  # 0.2到0.9，步长0.02，共35个点
        'default': 0.7,
        'description': '路径截断百分比'
    },
    'MIN_EXPLORATION_DISTANCE': {
        'range': np.arange(2, 35, 1),  # 2到34，步长1，共33个点
        'default': 10.0,
        'description': '最小探索距离阈值'
    },
    'MIN_FRONTIERS_TO_EXPLORE': {
        'range': np.arange(2, 30, 1),  # 2到29，步长1，共28个点
        'default': 5,
        'description': '最小探索前沿数量'
    }
}

def compare_maps(true_maze, slam_occupancy, maze_bounds, resolution):
    """比较真实地图和SLAM建模地图，计算偏差率"""
    # 创建真实地图的占用栅格
    height, width = slam_occupancy.shape
    true_occupancy = np.full((height, width), -1, dtype=np.int8)
    
    min_x, min_y, max_x, max_y = maze_bounds
    
    # 将真实墙壁转换为栅格表示
    for wall in true_maze.walls:
        (x1, y1), (x2, y2) = wall
        steps = max(abs(x2 - x1), abs(y2 - y1)) / resolution
        steps = int(steps) + 1
        
        for i in range(steps):
            t = i / max(1, steps - 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            grid_x = int((x - min_x) / resolution)
            grid_y = int((y - min_y) / resolution)
            
            if 0 <= grid_x < width and 0 <= grid_y < height:
                true_occupancy[grid_y, grid_x] = 1
    
    # 标记自由空间
    for y in range(height):
        for x in range(width):
            if true_occupancy[y, x] == -1:
                world_x = min_x + x * resolution
                world_y = min_y + y * resolution
                if min_x <= world_x <= max_x and min_y <= world_y <= max_y:
                    true_occupancy[y, x] = 0
    
    # 统计比较结果
    total_cells = 0
    correct_cells = 0
    slam_explored_cells = 0
    
    for y in range(height):
        for x in range(width):
            true_val = true_occupancy[y, x]
            slam_val = slam_occupancy[y, x]
            
            if slam_val != -1:  # SLAM已探索
                slam_explored_cells += 1
                total_cells += 1
                
                if true_val == slam_val:
                    correct_cells += 1
    
    # 计算各种指标
    if total_cells > 0:
        accuracy = correct_cells / total_cells
        coverage_rate = slam_explored_cells / (height * width)
    else:
        accuracy = 0.0
        coverage_rate = 0.0
    
    return {
        'accuracy': accuracy,
        'coverage_rate': coverage_rate,
        'total_cells': total_cells,
        'correct_cells': correct_cells,
        'slam_explored_cells': slam_explored_cells
    }

def check_exit_condition(scan, max_range=12.0, min_no_obstacle_count=95):
    """检查机器人是否走出迷宫"""
    if not scan or len(scan) == 0:
        return False
    
    no_obstacle = [dist >= max_range for dist in scan]
    no_obstacle_count = sum(no_obstacle)
    
    return no_obstacle_count >= min_no_obstacle_count

@monitor_memory_usage("SLAM试验")
def run_single_slam_trial(params):
    """
    运行单次SLAM建模试验，支持CUDA加速并自动释放显存
    
    Args:
        params: 参数字典，包含所有要测试的参数值
    
    Returns:
        dict: 包含准确度指标的字典
    """
    # 记录GPU显存使用情况（如果使用CUDA）
    initial_memory = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
    
    # 创建变量追踪，用于异常时的清理
    objects_to_cleanup = {}
    
    try:
        # 1. 加载迷宫地图和参数
        loader = MazeLoader()
        maze = loader.load(MAZE_FILE)
        objects_to_cleanup['loader'] = loader
        objects_to_cleanup['maze'] = maze

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
        maze.walls.extend(virtual_walls)

        # 2. 初始化机器人、传感器、SLAM等模块
        start_pose = (start_x, start_y, 0.0)
        robot = Robot(start_pose, odom_noise=ROBOT_ODOM_NOISE)
        objects_to_cleanup['robot'] = robot
        
        lidar = Lidar(maze.walls, max_range=LIDAR_MAX_RANGE, angle_resolution=LIDAR_ANGLE_RESOLUTION, noise=LIDAR_NOISE)
        objects_to_cleanup['lidar'] = lidar
        
        slam = ICPSlam(maze, start_pose)
        objects_to_cleanup['slam'] = slam
        
        # 应用参数到SLAM算法（直接修改全局变量）
        icp_slam.MAX_RANGE_FACTOR = params.get('MAX_RANGE_FACTOR', 0.49)
        icp_slam.ADJACENCY_DIFF_THRESHOLD = params.get('ADJACENCY_DIFF_THRESHOLD', 0.01)
        icp_slam.ICP_MAX_ITER = params.get('ICP_MAX_ITER', 500)
        
        # 也直接设置slam对象的属性
        slam.icp_max_iter = params.get('ICP_MAX_ITER', 500)
        
        # 设置前沿探索器安全距离
        frontier_safety_distance = 6.0
        explorer = FrontierExplorer(safety_distance=frontier_safety_distance)
        objects_to_cleanup['explorer'] = explorer
        
        # 创建可视化器但不显示
        viz = Visualizer(maze, robot=robot, slam=slam)
        objects_to_cleanup['viz'] = viz
        plt.close('all')  # 立即关闭所有图形窗口
        
        # 初始化降噪滤波器
        noise_filter = NoiseFilter(
            enabled=True,
            lidar_filter_enabled=False,
            odom_filter_enabled=False
        )
        objects_to_cleanup['noise_filter'] = noise_filter
        
        robot.set_noise_filter(noise_filter)
        lidar.set_noise_filter(noise_filter)
        
        # 3. 初始扫描并建立初始地图
        scan = lidar.scan(robot.get_pose())
        slam.update((0.0, 0.0), scan)
        
        # 初始化探索状态标志和探索进度跟踪
        total_distance_traveled = 0.0
        frontiers_explored = 0
        
        # 从参数中获取探索阈值
        min_exploration_distance = params.get('MIN_EXPLORATION_DISTANCE', 10.0)
        min_frontiers_to_explore = params.get('MIN_FRONTIERS_TO_EXPLORE', 5)
        safety_distance_factor = params.get('SAFETY_DISTANCE_FACTOR', 0.7)
        
        # 4. 前沿探索主循环
        max_iterations = 300  # 限制最大迭代次数
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 检查探索完成条件
            scan = lidar.scan(robot.get_pose())
            has_sufficient_exploration = (total_distance_traveled >= min_exploration_distance and 
                                         frontiers_explored >= min_frontiers_to_explore)
            
            if has_sufficient_exploration and check_exit_condition(scan, max_range=lidar.max_range, min_no_obstacle_count=MIN_NO_OBSTACLE_COUNT):
                break
            
            # 查找最近的前沿
            rx_idx = int((robot.x - maze.bounds[0]) / maze.resolution)
            ry_idx = int((robot.y - maze.bounds[1]) / maze.resolution)
            frontier_cell, path = explorer.find_nearest_frontier(slam.get_occupancy(), (rx_idx, ry_idx))
            
            if frontier_cell is None:
                if total_distance_traveled >= min_exploration_distance:
                    break
                continue
            
            frontiers_explored += 1
            
            # 5. 沿规划路径移动机器人
            if path is None or len(path) <= 1:
                continue
                
            # 应用安全距离因子截断路径
            if len(path) > 1:
                path_length = len(path) - 1
                truncated_length = max(1, int(path_length * safety_distance_factor))
                path = path[:truncated_length+1]
                
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
                
                # 每10步清理一次显存
                if iteration % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 检查是否满足退出条件
                has_sufficient_exploration = (total_distance_traveled >= min_exploration_distance and 
                                             frontiers_explored >= min_frontiers_to_explore)
                
                if has_sufficient_exploration and check_exit_condition(scan, max_range=lidar.max_range, min_no_obstacle_count=MIN_NO_OBSTACLE_COUNT):
                    break
            else:
                continue  # 只有在内循环正常结束时才继续外循环
            break  # 如果内循环因为break退出，也退出外循环
        
        # 进行地图比较
        comparison_results = compare_maps(maze, slam.get_occupancy(), maze.bounds, maze.resolution)
        
        # 添加试验信息
        comparison_results['total_distance'] = total_distance_traveled
        comparison_results['frontiers_explored'] = frontiers_explored
        comparison_results['iterations'] = iteration
        
        # 记录显存使用情况
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            comparison_results['peak_gpu_memory_mb'] = peak_memory / (1024 * 1024)
            comparison_results['initial_gpu_memory_mb'] = initial_memory / (1024 * 1024) if initial_memory else 0
        
        # 立即保存单次试验结果到临时文件
        trial_data = {
            'timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'params': params.copy(),
            'results': comparison_results
        }
          # 生成唯一的试验文件名（使用time.time()获取微秒精度）
        trial_filename = f"trial_temp_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}.json"
        try:
            with open(trial_filename, 'w') as f:
                json.dump(trial_data, f, indent=2)
        except Exception as e:
            print(f"保存试验临时文件失败: {e}")
        
        return comparison_results
        
    except Exception as e:
        print(f"试验出错: {e}")
        traceback.print_exc()
        
        # 返回默认的错误结果
        return {
            'accuracy': 0.0,
            'coverage_rate': 0.0,
            'total_distance': 0.0,
            'frontiers_explored': 0,
            'iterations': 0,
            'total_cells': 0,            'correct_cells': 0,
            'slam_explored_cells': 0,
            'error': str(e)
        }
    
    finally:
        # 强制释放所有GPU资源和内存 - 在finally块中确保总是执行
        try:
            # 释放SLAM GPU资源
            if 'slam' in objects_to_cleanup and hasattr(objects_to_cleanup['slam'], 'release_resources'):
                objects_to_cleanup['slam'].release_resources()
            
            # 删除所有大对象
            for obj_name, obj in objects_to_cleanup.items():
                try:
                    del obj
                except:
                    pass
            objects_to_cleanup.clear()
            
            # 删除局部变量中的大对象
            for var_name in ['scan', 'path', 'frontier_cell', 'trial_data', 'comparison_results']:
                if var_name in locals():
                    try:
                        del locals()[var_name]
                    except:
                        pass
            
            # 清理matplotlib
            plt.close('all')
            
            # 多次强制清理GPU缓存
            if torch.cuda.is_available():
                for _ in range(3):  # 多次清理以确保彻底
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 等待所有GPU操作完成
                torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
            
            # 强制垃圾回收


            gc.collect()
            
            # 检查GPU内存释放情况
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                if current_memory > 0:
                    print(f"警告: 试验结束后仍有 {current_memory / (1024*1024):.1f} MB GPU内存未释放")
                    # 尝试更激进的清理

                    if hasattr(icp_slam, 'clear_global_cache'):
                        icp_slam.clear_global_cache()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()  # 清理IPC缓存
                    
                    # 再次检查
                    final_memory = torch.cuda.memory_allocated()
                    if final_memory < current_memory:
                        print(f"  已释放 {(current_memory - final_memory) / (1024*1024):.1f} MB GPU内存")
                
            # 清理临时试验文件
            try:

                temp_files = glob.glob("trial_temp_*.json")
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            except:
                pass
                
        except Exception as cleanup_error:
            print(f"资源清理时出错: {cleanup_error}")

def load_checkpoint(checkpoint_filename):
    """加载检查点文件，恢复之前的测试进度"""
    try:
        with open(checkpoint_filename, 'r') as f:
            checkpoint_data = json.load(f)
        return checkpoint_data
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_checkpoint(checkpoint_filename, current_param, param_index, value_index, current_trial, completed_results):
    """保存当前测试进度到检查点文件"""
    checkpoint_data = {
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        'current_param': current_param,
        'param_index': param_index,
        'value_index': value_index,
        'current_trial': current_trial,  # 当前参数值下已完成的试验次数
        'completed_results': completed_results,
        'total_params': len(PARAM_CONFIGS),
        'param_configs': {param: config for param, config in PARAM_CONFIGS.items()}
    }
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    def recursive_convert(data):
        if isinstance(data, dict):
            return {key: recursive_convert(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [recursive_convert(item) for item in data]
        else:
            return convert_numpy(data)

    checkpoint_data = recursive_convert(checkpoint_data)
    try:
        with open(checkpoint_filename, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"  检查点已保存: {checkpoint_filename}")
    except Exception as e:
        print(f"  保存检查点失败: {e}")

def test_parameter(param_name, param_values, default_params, checkpoint_filename=None, start_from_index=0, start_from_trial=0):
    """
    测试单个参数的不同值，结果立即保存到文件以节省内存
    支持断点续传功能
    
    Args:
        param_name: 参数名称
        param_values: 参数值列表
        default_params: 默认参数字典
        checkpoint_filename: 检查点文件名
        start_from_index: 从哪个参数值索引开始测试（用于断点续传）
        start_from_trial: 从哪个试验次数开始测试（用于断点续传）
    
    Returns:
        tuple: (param_values, accuracies, coverage_rates)
    """
    print(f"\n{'='*60}")
    print(f"开始测试参数: {param_name}")
    print(f"参数范围: {param_values[0]:.3f} - {param_values[-1]:.3f}")
    print(f"参数点数: {len(param_values)}")
    if start_from_index > 0:
        print(f"断点续传: 从第 {start_from_index + 1} 个参数值开始")
    if start_from_trial > 0:
        print(f"断点续传: 从第 {start_from_trial + 1} 次试验开始")
    print(f"{'='*60}")
    
    # 创建参数专用的结果文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f'param_tuning_{param_name.lower()}_{timestamp}.json'
    
    # 如果是断点续传，尝试从现有文件加载数据
    existing_data = {'parameter_results': []}
    if start_from_index > 0:
        existing_files = glob.glob(f'param_tuning_{param_name.lower()}_*.json')
        if existing_files:
            # 使用最新的文件
            existing_files.sort()
            results_filename = existing_files[-1]
            try:
                with open(results_filename, 'r') as f:
                    existing_data = json.load(f)
                print(f"  续传模式: 加载现有结果文件 {results_filename}")
            except Exception as e:
                print(f"  加载现有文件失败: {e}")
    
    accuracies = []
    coverage_rates = []
    
    # 从已有结果中恢复数据
    if existing_data['parameter_results']:
        for result in existing_data['parameter_results']:
            accuracies.append(result['avg_accuracy'])
            coverage_rates.append(result['avg_coverage_rate'])
        print(f"  已恢复 {len(accuracies)} 个参数值的结果")
    
    total_start_time = time.time()
    
    for i, param_value in enumerate(param_values):
        # 跳过已完成的参数值
        if i < start_from_index:
            continue
            
        print(f"\n--- 测试 {param_name} = {param_value:.3f} ({i+1}/{len(param_values)}) ---")
        
        # 创建当前测试的参数组合
        test_params = default_params.copy()
        test_params[param_name] = param_value
        
        # 运行多次试验 - 不保存在内存中
        trial_accuracies = []
        trial_coverage_rates = []
        trial_results = []  # 临时存储本组试验结果
          # 确定从哪个试验开始
        trial_start_num = start_from_trial if i == start_from_index else 0
        
        trial_start_time = time.time()
        
        for trial in range(trial_start_num, NUM_TRIALS_PER_PARAM):
            if trial % 5 == 0:  # 更频繁显示进度
                # 显示GPU显存使用情况（如果可用）
                gpu_info = ""
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                    max_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                    gpu_info = f" [GPU: {current_memory:.0f}/{max_memory:.0f}MB]"
                
                print(f"  试验进度: {trial+1}/{NUM_TRIALS_PER_PARAM}{gpu_info}")
            
            # 在试验前清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            result = run_single_slam_trial(test_params)
            trial_accuracies.append(result['accuracy'])
            trial_coverage_rates.append(result['coverage_rate'])
            
            # 保存详细的单次试验结果到临时列表
            trial_result_data = {
                'trial': trial + 1,
                'accuracy': result['accuracy'],
                'coverage_rate': result['coverage_rate'],
                'total_distance': result['total_distance'],
                'frontiers_explored': result['frontiers_explored'],
                'iterations': result['iterations']
            }
            
            # 如果有GPU内存信息，也记录下来
            if 'peak_gpu_memory_mb' in result:
                trial_result_data['peak_gpu_memory_mb'] = result['peak_gpu_memory_mb']
                trial_result_data['initial_gpu_memory_mb'] = result['initial_gpu_memory_mb']
            
            trial_results.append(trial_result_data)
            
            # 每次试验后立即保存到磁盘（增量保存）
            incremental_filename = f'incremental_{param_name.lower()}_{i}_{trial}.json'
            try:
                with open(incremental_filename, 'w') as f:
                    json.dump(trial_result_data, f, indent=2)
            except Exception as e:
                print(f"  保存增量结果失败: {e}")
            
            # 每次试验后保存检查点（仅当是断点续传的参数值时）
            if checkpoint_filename and i == start_from_index:
                save_checkpoint(checkpoint_filename, param_name, 
                              list(PARAM_CONFIGS.keys()).index(param_name), i, trial + 1, {})
              # 每次试验后进行内存清理和监控
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                # 检查GPU内存泄漏
                current_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
                if current_gpu_memory > 500:  # 如果GPU内存超过500MB，发出警告
                    print(f"  警告: GPU内存使用较高 ({current_gpu_memory:.1f}MB)，执行深度清理...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
            
            # 每10次试验执行一次深度清理
            if (trial + 1) % 10 == 0:
                force_cleanup()
                print(f"  第{trial+1}次试验后执行深度清理")
            
            # 删除临时试验文件
            try:
                temp_files = glob.glob(f"trial_temp_*.json") + glob.glob(f"incremental_{param_name.lower()}_{i}_{trial}.json")
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            except:
                pass
        
        trial_end_time = time.time()
        
        # 计算平均值
        avg_accuracy = np.mean(trial_accuracies)
        avg_coverage = np.mean(trial_coverage_rates)
        std_accuracy = np.std(trial_accuracies)
        
        accuracies.append(avg_accuracy)
        coverage_rates.append(avg_coverage)
        
        # 立即保存当前参数值的所有试验结果到文件
        param_group_data = {
            'param_name': param_name,
            'param_value': float(param_value),
            'param_index': i + 1,
            'num_trials': NUM_TRIALS_PER_PARAM,
            'avg_accuracy': float(avg_accuracy),
            'avg_coverage_rate': float(avg_coverage),
            'std_accuracy': float(std_accuracy),
            'trial_time': trial_end_time - trial_start_time,
            'detailed_trials': trial_results
        }
        
        # 追加写入到参数结果文件
        try:
            # 添加新的参数组数据
            existing_data['parameter_results'].append(param_group_data)
            
            # 写回文件
            with open(results_filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            print(f"  已保存结果到: {results_filename}")
            
        except Exception as e:
            print(f"  保存结果时出错: {e}")
        
        # 保存检查点
        if checkpoint_filename:
            save_checkpoint(checkpoint_filename, param_name, 
                          list(PARAM_CONFIGS.keys()).index(param_name), i + 1, 0, {})
        
        # 清理内存：删除试验结果数据
        del trial_results, trial_accuracies, trial_coverage_rates
        gc.collect()
        
        # 估算剩余时间
        elapsed = time.time() - total_start_time
        completed_values = i - start_from_index + 1
        if completed_values > 0:
            avg_time_per_param = elapsed / completed_values
            remaining_params = len(param_values) - (i + 1)
            estimated_remaining = avg_time_per_param * remaining_params
            
            print(f"  平均准确率: {avg_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
            print(f"  平均覆盖率: {avg_coverage*100:.2f}%")
            print(f"  本组用时: {trial_end_time - trial_start_time:.1f}s")
            if remaining_params > 0:
                print(f"  预计剩余: {estimated_remaining/60:.1f}分钟")
    
    total_end_time = time.time()
    print(f"\n{param_name} 参数测试完成！总用时: {(total_end_time - total_start_time)/60:.1f}分钟")
    print(f"详细结果已保存到: {results_filename}")
    
    return param_values, accuracies, coverage_rates

def plot_parameter_curves(param_name, param_values, accuracies, coverage_rates, param_description):
    """绘制参数调优曲线图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 准确率曲线
    ax1.plot(param_values, np.array(accuracies) * 100, 'b-o', linewidth=2, markersize=4, alpha=0.8)
    ax1.set_xlabel(f'{param_name}', fontsize=12)
    ax1.set_ylabel('Accuracy Rate (%)', fontsize=12)
    ax1.set_title(f'SLAM Accuracy vs {param_description}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 找到最佳准确率
    best_idx = np.argmax(accuracies)
    best_param = param_values[best_idx]
    best_accuracy = accuracies[best_idx] * 100
    
    ax1.axvline(x=best_param, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_param:.3f} ({best_accuracy:.2f}%)')
    ax1.legend()
    
    # 覆盖率曲线
    ax2.plot(param_values, np.array(coverage_rates) * 100, 'g-o', linewidth=2, markersize=4, alpha=0.8)
    ax2.set_xlabel(f'{param_name}', fontsize=12)
    ax2.set_ylabel('Coverage Rate (%)', fontsize=12)
    ax2.set_title(f'SLAM Coverage vs {param_description}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 找到最佳覆盖率
    best_coverage_idx = np.argmax(coverage_rates)
    best_coverage_param = param_values[best_coverage_idx]
    best_coverage_rate = coverage_rates[best_coverage_idx] * 100
    
    ax2.axvline(x=best_coverage_param, color='r', linestyle='--', alpha=0.7, 
                label=f'Best: {best_coverage_param:.3f} ({best_coverage_rate:.2f}%)')
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图片
    filename = f'param_tuning_{param_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形而不显示
    
    print(f"参数调优曲线图已保存为: {filename}")
    
    return best_param, best_accuracy, best_coverage_param, best_coverage_rate

def main():
    """主函数：运行参数调优，支持断点续传"""
    global device
    
    print("开始SLAM参数调优 - 控制变量法")
    
    # 初始化设备
    device = initialize_device()
    
    print(f"每组参数运行 {NUM_TRIALS_PER_PARAM} 次试验")
    print("结果将立即保存到文件以节省内存")
    print("支持断点续传 - 可随时中断并恢复测试")
    
    # 显示初始内存使用情况
    initial_memory = get_memory_usage()
    print(f"初始内存: CPU {initial_memory['cpu_memory_mb']:.1f}MB, GPU {initial_memory['gpu_allocated_mb']:.1f}MB")
    
    # 执行初始清理
    force_cleanup()
    print("已执行初始内存清理")
    
    # 设置默认参数
    default_params = {param: config['default'] for param, config in PARAM_CONFIGS.items()}
    
    # 创建检查点文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f'slam_param_checkpoint_{timestamp}.json'
    summary_filename = f'param_tuning_summary_{timestamp}.json'
    
    # 检查是否存在之前的检查点
    existing_checkpoints = glob.glob('slam_param_checkpoint_*.json')
    if existing_checkpoints:
        existing_checkpoints.sort()
        latest_checkpoint = existing_checkpoints[-1]
        
        print(f"\n发现现有检查点文件: {latest_checkpoint}")
        choice = input("是否从上次中断的地方继续测试？(y/n): ").lower().strip()
        
        if choice in ['y', 'yes', '是']:
            checkpoint_data = load_checkpoint(latest_checkpoint)
            if checkpoint_data:
                print(f"正在恢复测试进度...")
                print(f"上次中断在参数: {checkpoint_data['current_param']}")
                print(f"参数索引: {checkpoint_data['param_index'] + 1}/{checkpoint_data['total_params']}")
                print(f"参数值索引: {checkpoint_data['value_index']}")
                print(f"试验进度: {checkpoint_data.get('current_trial', 0)}/{NUM_TRIALS_PER_PARAM}")
                
                # 使用原来的时间戳和文件名
                original_timestamp = latest_checkpoint.split('_')[-1].replace('.json', '')
                checkpoint_filename = latest_checkpoint
                summary_filename = f'param_tuning_summary_{original_timestamp}.json'
                
                start_param_index = checkpoint_data['param_index']
                start_value_index = checkpoint_data['value_index']
                start_trial_index = checkpoint_data.get('current_trial', 0)
            else:
                print("检查点文件损坏，将重新开始测试")
                start_param_index = 0
                start_value_index = 0
                start_trial_index = 0
        else:
            print("开始新的测试")
            start_param_index = 0
            start_value_index = 0
    else:
        print("开始新的测试")
        start_param_index = 0
        start_value_index = 0
    
    # 存储参数调优总结（只保存最佳结果，不保存详细数据）
    summary_results = {}
    
    # 如果是续传模式，尝试加载已有的总结文件
    if start_param_index > 0:
        try:
            with open(summary_filename, 'r') as f:
                existing_summary = json.load(f)
                summary_results = existing_summary.get('summary_results', {})
            print(f"已恢复 {len(summary_results)} 个参数的总结结果")
        except Exception as e:
            print(f"无法加载总结文件: {e}")
    
    total_start_time = time.time()
    
    # 逐个测试每个参数
    param_names = list(PARAM_CONFIGS.keys())
    for param_idx, (param_name, param_config) in enumerate(PARAM_CONFIGS.items()):
        # 跳过已完成的参数
        if param_idx < start_param_index:
            continue
            
        param_values = param_config['range']
        param_description = param_config['description']
        
        print(f"\n正在处理参数: {param_name} ({param_idx + 1}/{len(PARAM_CONFIGS)})")
        
        # 确定从哪个参数值开始测试
        value_start_index = start_value_index if param_idx == start_param_index else 0
        
        try:
            # 测试当前参数（结果会自动保存到专用文件）
            values, accuracies, coverage_rates = test_parameter(
                param_name, param_values, default_params, 
                checkpoint_filename, value_start_index
            )
            
            # 绘制曲线图
            best_param, best_accuracy, best_coverage_param, best_coverage_rate = plot_parameter_curves(
                param_name, values, accuracies, coverage_rates, param_description
            )
            
            # 只保存总结信息到内存
            summary_results[param_name] = {
                'best_accuracy_param': float(best_param),
                'best_accuracy_value': float(best_accuracy),
                'best_coverage_param': float(best_coverage_param),
                'best_coverage_value': float(best_coverage_rate),
                'description': param_description,                'num_tested_values': len(values),
                'value_range': [float(values[0]), float(values[-1])],
                'completed': True
            }
            
            print(f"\n{param_name} 最佳结果:")
            print(f"  最佳准确率: {best_param:.3f} -> {best_accuracy:.2f}%")
            print(f"  最佳覆盖率: {best_coverage_param:.3f} -> {best_coverage_rate:.2f}%")
            
            # 显示内存使用情况
            current_memory = get_memory_usage()
            print(f"  当前内存: CPU {current_memory['cpu_memory_mb']:.1f}MB, GPU {current_memory['gpu_allocated_mb']:.1f}MB")
            
        except KeyboardInterrupt:
            print(f"\n\n程序被用户中断")
            print(f"当前进度已保存到检查点文件: {checkpoint_filename}")
            print(f"可以随时重新运行程序并选择继续测试")
            
            # 执行最终清理
            force_cleanup()
            return
        except Exception as e:
            print(f"\n参数 {param_name} 测试出错: {e}")
            traceback.print_exc()
            print("继续下一个参数...")
            continue
        
        # 立即保存总结到文件
        summary_data = {
            'config': {
                'num_trials_per_param': NUM_TRIALS_PER_PARAM,
                'maze_file': MAZE_FILE,
                'timestamp': timestamp,
                'total_parameters': len(PARAM_CONFIGS),
                'completed_parameters': len([r for r in summary_results.values() if r.get('completed', False)]),
                'last_updated': time.strftime("%Y%m%d_%H%M%S")
            },
            'parameter_configs': PARAM_CONFIGS,
            'summary_results': summary_results
        }
        
        # 转换numpy对象
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            return obj
        
        def recursive_convert(data):
            if isinstance(data, dict):
                return {key: recursive_convert(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        summary_data = recursive_convert(summary_data)
        
        with open(summary_filename, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # 更新检查点 - 标记当前参数已完成
        save_checkpoint(checkpoint_filename, param_name, param_idx + 1, 0, 0, summary_results)
        
        # 清理内存
        del values, accuracies, coverage_rates
        gc.collect()
        
        print(f"参数 {param_name} 处理完成，总结已保存到: {summary_filename}")
        
        # 重置value_start_index，确保下一个参数从头开始
        start_value_index = 0
    
    total_end_time = time.time()
    
    # 输出最终总结
    print(f"\n{'='*70}")
    print("参数调优完成总结")
    print(f"{'='*70}")
    print(f"总用时: {(total_end_time - total_start_time)/60:.1f}分钟")
    
    completed_params = [name for name, result in summary_results.items() if result.get('completed', False)]
    print(f"已完成参数: {len(completed_params)}/{len(PARAM_CONFIGS)}")
    
    if completed_params:
        print(f"\n各参数最佳值总结:")
        optimal_params = {}
        for param_name in completed_params:
            results = summary_results[param_name]
            optimal_params[param_name] = results['best_accuracy_param']
            print(f"  {param_name}: {results['best_accuracy_param']:.3f} (准确率: {results['best_accuracy_value']:.2f}%)")
        
        # 保存最终的最优参数配置
        optimal_config = {
            'timestamp': timestamp,
            'total_time_minutes': (total_end_time - total_start_time) / 60,
            'completed_parameters': len(completed_params),
            'total_parameters': len(PARAM_CONFIGS),
            'optimal_parameters': optimal_params,
            'parameter_details': summary_results
        }
        
        optimal_filename = f'optimal_parameters_{timestamp}.json'
        with open(optimal_filename, 'w') as f:
            json.dump(optimal_config, f, indent=2)
        
        print(f"\n结果文件说明:")
        print(f"  总结文件: {summary_filename}")
        print(f"  最优参数: {optimal_filename}")
        print(f"  检查点文件: {checkpoint_filename}")
        print(f"  详细数据: param_tuning_*_{timestamp}.json (每个参数一个文件)")
        print(f"  曲线图片: param_tuning_*.png (每个参数一个图片)")
        
        print(f"\n功能特性:")
        print(f"  ✓ 内存优化: 详细试验数据保存到文件，内存占用最小化")
        print(f"  ✓ 断点续传: 可随时中断并从上次位置继续测试")
        print(f"  ✓ 实时保存: 每个参数值测试完成后立即保存结果")
          # 删除检查点文件（测试完成）
        try:
            os.remove(checkpoint_filename)
            print(f"  ✓ 测试完成，已清理检查点文件")
        except:
            pass
    else:
        print("没有完成任何参数的测试")
    
    print(f"\n内存优化: 所有详细试验数据已保存到文件，内存占用已最小化")
    
    # 显示最终内存使用情况
    final_memory = get_memory_usage()
    print(f"最终内存: CPU {final_memory['cpu_memory_mb']:.1f}MB, GPU {final_memory['gpu_allocated_mb']:.1f}MB")
    
    # 执行最终清理
    force_cleanup()
    print("已执行最终内存清理")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        force_cleanup()  # 中断时也执行清理
    except Exception as e:
        print(f"\n程序出错: {e}")
        traceback.print_exc()
        force_cleanup()  # 出错时也执行清理
