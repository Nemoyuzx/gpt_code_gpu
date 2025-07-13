#!/usr/bin/env python3
"""
噪声滤波器测试脚本
用于验证激光雷达和里程计数据的降噪效果
"""

import numpy as np
import matplotlib.pyplot as plt
from noise_filter import NoiseFilter
import math

def test_lidar_filter():
    """测试激光雷达数据滤波"""
    print("测试激光雷达数据滤波...")
    
    # 创建不同类型的滤波器
    filters = {
        'none': NoiseFilter(enabled=True, lidar_filter_type='none'),
        'median': NoiseFilter(enabled=True, lidar_filter_type='median', lidar_window_size=5),
        'moving_average': NoiseFilter(enabled=True, lidar_filter_type='moving_average', lidar_window_size=5),
        'gaussian': NoiseFilter(enabled=True, lidar_filter_type='gaussian', lidar_window_size=5)
    }
    
    # 生成测试数据：基础信号 + 噪声
    base_signal = [5.0 + 2.0 * math.sin(i * 0.1) for i in range(100)]
    noisy_signal = [val + np.random.normal(0, 0.5) for val in base_signal]
    
    # 添加一些脉冲噪声
    for i in [20, 40, 60, 80]:
        noisy_signal[i] += np.random.choice([-3, 3])
    
    results = {}
    
    # 对每种滤波器进行测试
    for name, filter_obj in filters.items():
        filtered_data = []
        for data_point in noisy_signal:
            # 模拟激光雷达扫描数据（只有一个点）
            scan_data = [data_point]
            filtered_scan = filter_obj.filter_lidar_data(scan_data)
            filtered_data.append(filtered_scan[0])
        results[name] = filtered_data
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(base_signal, 'g-', label='Original Signal', linewidth=2)
    plt.plot(noisy_signal, 'r--', label='Noisy Signal', alpha=0.7)
    plt.plot(results['none'], 'b:', label='No Filter')
    plt.legend()
    plt.title('No Filter')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(base_signal, 'g-', label='Original Signal', linewidth=2)
    plt.plot(noisy_signal, 'r--', label='Noisy Signal', alpha=0.7)
    plt.plot(results['median'], 'b-', label='Median Filter')
    plt.legend()
    plt.title('Median Filter')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(base_signal, 'g-', label='Original Signal', linewidth=2)
    plt.plot(noisy_signal, 'r--', label='Noisy Signal', alpha=0.7)
    plt.plot(results['moving_average'], 'b-', label='Moving Average')
    plt.legend()
    plt.title('Moving Average Filter')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(base_signal, 'g-', label='Original Signal', linewidth=2)
    plt.plot(noisy_signal, 'r--', label='Noisy Signal', alpha=0.7)
    plt.plot(results['gaussian'], 'b-', label='Gaussian Filter')
    plt.legend()
    plt.title('Gaussian Filter')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lidar_filter_test.png', dpi=150)
    plt.show()
    
    # 计算MSE
    print("\n激光雷达滤波器性能（MSE相对于原始信号）:")
    for name, filtered in results.items():
        mse = np.mean([(f - o)**2 for f, o in zip(filtered, base_signal)])
        print(f"{name:15}: {mse:.4f}")

def test_odometry_filter():
    """测试里程计数据滤波"""
    print("\n测试里程计数据滤波...")
    
    # 创建滤波器
    kalman_filter = NoiseFilter(enabled=True, odom_filter_type='kalman', odom_window_size=3)
    avg_filter = NoiseFilter(enabled=True, odom_filter_type='moving_average', odom_window_size=3)
    no_filter = NoiseFilter(enabled=True, odom_filter_type='none')
    
    # 生成测试轨迹：圆形路径
    true_trajectory = []
    noisy_trajectory = []
    
    for i in range(100):
        t = i * 0.1
        # 真实轨迹（圆形）
        true_x = 5 * math.cos(t)
        true_y = 5 * math.sin(t)
        true_theta = t + math.pi/2
        
        # 添加噪声
        noisy_x = true_x + np.random.normal(0, 0.2)
        noisy_y = true_y + np.random.normal(0, 0.2)
        noisy_theta = true_theta + np.random.normal(0, 0.1)
        
        true_trajectory.append((true_x, true_y, true_theta))
        noisy_trajectory.append((noisy_x, noisy_y, noisy_theta))
    
    # 滤波处理
    kalman_results = []
    avg_results = []
    no_filter_results = []
    
    for noisy_x, noisy_y, noisy_theta in noisy_trajectory:
        kalman_x, kalman_y, kalman_theta = kalman_filter.filter_odometry_data(noisy_x, noisy_y, noisy_theta)
        avg_x, avg_y, avg_theta = avg_filter.filter_odometry_data(noisy_x, noisy_y, noisy_theta)
        no_x, no_y, no_theta = no_filter.filter_odometry_data(noisy_x, noisy_y, noisy_theta)
        
        kalman_results.append((kalman_x, kalman_y, kalman_theta))
        avg_results.append((avg_x, avg_y, avg_theta))
        no_filter_results.append((no_x, no_y, no_theta))
    
    # 绘制结果
    plt.figure(figsize=(10, 10))
    
    # 提取x, y坐标
    true_x = [t[0] for t in true_trajectory]
    true_y = [t[1] for t in true_trajectory]
    noisy_x = [t[0] for t in noisy_trajectory]
    noisy_y = [t[1] for t in noisy_trajectory]
    kalman_x = [t[0] for t in kalman_results]
    kalman_y = [t[1] for t in kalman_results]
    avg_x = [t[0] for t in avg_results]
    avg_y = [t[1] for t in avg_results]
    
    plt.plot(true_x, true_y, 'g-', label='True Trajectory', linewidth=3)
    plt.plot(noisy_x, noisy_y, 'r:', label='Noisy Trajectory', alpha=0.7)
    plt.plot(kalman_x, kalman_y, 'b-', label='Kalman Filter', linewidth=2)
    plt.plot(avg_x, avg_y, 'm--', label='Moving Average', linewidth=2)
    
    plt.legend()
    plt.title('Odometry Filter Comparison')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('odometry_filter_test.png', dpi=150)
    plt.show()
    
    # 计算位置误差
    print("\n里程计滤波器性能（平均位置误差）:")
    
    def calc_position_error(filtered, true):
        errors = [math.sqrt((f[0]-t[0])**2 + (f[1]-t[1])**2) for f, t in zip(filtered, true)]
        return np.mean(errors)
    
    print(f"无滤波:       {calc_position_error(no_filter_results, true_trajectory):.4f} m")
    print(f"卡尔曼滤波:   {calc_position_error(kalman_results, true_trajectory):.4f} m")
    print(f"移动平均:     {calc_position_error(avg_results, true_trajectory):.4f} m")

def main():
    """主测试函数"""
    print("开始噪声滤波器测试...")
    print("=" * 50)
    
    # 测试激光雷达滤波
    test_lidar_filter()
    
    # 测试里程计滤波
    test_odometry_filter()
    
    print("\n测试完成！结果图片已保存。")

if __name__ == "__main__":
    main()
