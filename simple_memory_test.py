#!/usr/bin/env python3
"""
简化版迷宫探索程序 - 专注于内存管理测试
"""

import time
import math
import os
import gc
import psutil
import matplotlib.pyplot as plt
from maze_loader import MazeLoader
from robot import Robot
from lidar import Lidar
from icp_slam import ICPSlam
from frontier_explorer import FrontierExplorer
from visualizer import Visualizer

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # 转换为MB

def cleanup_memory():
    """清理内存"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def main():
    print("=== 简化版迷宫探索 - 内存管理测试 ===")
    
    # 设置CPU模式以避免GPU相关的复杂性
    os.environ["FORCE_CPU"] = "1"
    
    print(f"初始内存: {get_memory_usage():.1f}MB")
    
    # 初始化
    loader = MazeLoader()
    maze = loader.load()
    start_x, start_y = maze.start
    start_pose = (start_x, start_y, 0.0)
    
    robot = Robot(start_pose, odom_noise=(0.01, math.radians(1)))
    lidar = Lidar(maze.walls, max_range=12.0, angle_resolution=1.0, noise=0.01)
    slam = ICPSlam(maze, start_pose)
    explorer = FrontierExplorer(safety_distance=6.0)
    
    print(f"组件初始化后内存: {get_memory_usage():.1f}MB")
    
    # 设置更激进的内存管理参数
    slam.max_map_points = 5000  # 降低最大点数
    slam.cleanup_interval = 20   # 更频繁的清理
    
    # 简化的探索循环 - 只运行几次迭代
    max_iterations = 100
    
    for iteration in range(max_iterations):
        # 扫描和SLAM更新
        scan = lidar.scan(robot.get_pose())
        slam.update((0.1, 0.01), scan)  # 模拟小幅移动
        
        # 每10次迭代检查内存
        if iteration % 10 == 0:
            memory_usage = get_memory_usage()
            slam_stats = slam.get_memory_stats()
            print(f"第{iteration}次迭代 - 内存: {memory_usage:.1f}MB, "
                  f"地图点数: {slam_stats['map_points_count']}")
            
            # 如果内存超过阈值，主动清理
            if memory_usage > 500:
                print("主动执行内存清理...")
                slam.cleanup_memory()
                cleanup_memory()
                print(f"清理后内存: {get_memory_usage():.1f}MB")
        
        # 模拟机器人移动
        robot.move(0.1)
        robot.rotate(0.1)
    
    print(f"探索完成后内存: {get_memory_usage():.1f}MB")
    
    # 全面清理
    print("开始全面清理...")
    
    # 清理SLAM
    slam.cleanup_memory()
    slam.map_points.clear()
    if hasattr(slam, 'map_points_tensor') and slam.map_points_tensor is not None:
        del slam.map_points_tensor
        slam.map_points_tensor = None
    
    # 删除对象
    del slam, robot, lidar, explorer, loader, maze
    
    # 关闭图形
    plt.close('all')
    
    # 最终清理
    cleanup_memory()
    
    print(f"最终内存: {get_memory_usage():.1f}MB")

if __name__ == "__main__":
    main()
