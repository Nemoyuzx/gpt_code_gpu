import numpy as np
from collections import deque
import math

class NoiseFilter:
    """
    噪声滤波器 - 处理激光雷达数据和里程计数据的噪声去除
    支持多种滤波算法：移动平均、中值滤波、卡尔曼滤波等
    """
    
    def __init__(self, enabled=True, lidar_filter_enabled=True, odom_filter_enabled=True,
                 lidar_filter_type='median', odom_filter_type='kalman',
                 lidar_window_size=5, odom_window_size=3):
        """
        初始化噪声滤波器
        
        Args:
            enabled: 是否启用滤波器总开关
            lidar_filter_enabled: 是否启用激光雷达滤波
            odom_filter_enabled: 是否启用里程计滤波
            lidar_filter_type: 激光雷达滤波类型 ('none', 'median', 'moving_average', 'gaussian')
            odom_filter_type: 里程计滤波类型 ('none', 'kalman', 'moving_average')
            lidar_window_size: 激光雷达滤波窗口大小
            odom_window_size: 里程计滤波窗口大小
        """
        self.enabled = enabled
        self.lidar_filter_enabled = lidar_filter_enabled
        self.odom_filter_enabled = odom_filter_enabled
        self.lidar_filter_type = lidar_filter_type
        self.odom_filter_type = odom_filter_type
        self.lidar_window_size = lidar_window_size
        self.odom_window_size = odom_window_size
        
        # 激光雷达数据历史缓存
        self.lidar_history = deque(maxlen=lidar_window_size)
        
        # 里程计数据历史缓存
        self.odom_x_history = deque(maxlen=odom_window_size)
        self.odom_y_history = deque(maxlen=odom_window_size)
        self.odom_theta_history = deque(maxlen=odom_window_size)
        
        # 卡尔曼滤波器状态（用于里程计）
        self._init_kalman_filter()
        
        print(f"[NoiseFilter] 滤波器初始化 - 状态: {'启用' if enabled else '禁用'}")
        if enabled:
            print(f"[NoiseFilter] 激光雷达滤波: {'启用' if lidar_filter_enabled else '禁用'} - 类型: {lidar_filter_type}, 窗口大小: {lidar_window_size}")
            print(f"[NoiseFilter] 里程计滤波: {'启用' if odom_filter_enabled else '禁用'} - 类型: {odom_filter_type}, 窗口大小: {odom_window_size}")
    
    def _init_kalman_filter(self):
        """初始化卡尔曼滤波器参数"""
        # 状态向量 [x, y, theta, vx, vy, vtheta]
        self.kalman_state = np.zeros(6)
        
        # 状态协方差矩阵
        self.kalman_P = np.eye(6) * 0.1
        
        # 过程噪声协方差
        self.kalman_Q = np.eye(6) * 0.01
        
        # 观测噪声协方差
        self.kalman_R = np.eye(3) * 0.1
        
        # 状态转移矩阵 (假设恒定速度模型)
        dt = 0.1  # 时间步长
        self.kalman_F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 观测矩阵
        self.kalman_H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        self.kalman_initialized = False
    
    def filter_lidar_data(self, scan_data):
        """
        对激光雷达数据进行滤波
        
        Args:
            scan_data: 原始激光雷达扫描数据列表
            
        Returns:
            filtered_data: 滤波后的扫描数据
        """
        if not self.enabled or not self.lidar_filter_enabled or self.lidar_filter_type == 'none':
            return scan_data
        
        if not scan_data:
            return scan_data
        
        # 将当前数据添加到历史缓存
        self.lidar_history.append(np.array(scan_data))
        
        if len(self.lidar_history) < 2:
            return scan_data
        
        filtered_data = scan_data.copy()
        
        if self.lidar_filter_type == 'median':
            filtered_data = self._median_filter_lidar()
        elif self.lidar_filter_type == 'moving_average':
            filtered_data = self._moving_average_filter_lidar()
        elif self.lidar_filter_type == 'gaussian':
            filtered_data = self._gaussian_filter_lidar()
        
        return filtered_data
    
    def _median_filter_lidar(self):
        """中值滤波 - 有效去除脉冲噪声"""
        if len(self.lidar_history) < 3:
            return list(self.lidar_history[-1])
        
        # 对每个角度的历史数据进行中值滤波
        history_array = np.array(list(self.lidar_history))
        filtered = np.median(history_array, axis=0)
        return list(filtered)
    
    def _moving_average_filter_lidar(self):
        """移动平均滤波 - 平滑数据"""
        history_array = np.array(list(self.lidar_history))
        filtered = np.mean(history_array, axis=0)
        return list(filtered)
    
    def _gaussian_filter_lidar(self):
        """高斯加权滤波 - 对近期数据给予更高权重"""
        if len(self.lidar_history) < 2:
            return list(self.lidar_history[-1])
        
        history_array = np.array(list(self.lidar_history))
        
        # 生成高斯权重（越新的数据权重越大）
        weights = np.exp(-0.5 * np.arange(len(self.lidar_history))**2)
        weights = weights[::-1]  # 反转，使最新数据权重最大
        weights = weights / np.sum(weights)  # 归一化
        
        # 加权平均
        filtered = np.average(history_array, axis=0, weights=weights)
        return list(filtered)
    
    def filter_odometry_data(self, x, y, theta):
        """
        对里程计数据进行滤波
        
        Args:
            x, y, theta: 当前里程计位置和角度
            
        Returns:
            filtered_x, filtered_y, filtered_theta: 滤波后的位置和角度
        """
        if not self.enabled or not self.odom_filter_enabled or self.odom_filter_type == 'none':
            return x, y, theta
        
        if self.odom_filter_type == 'kalman':
            return self._kalman_filter_odom(x, y, theta)
        elif self.odom_filter_type == 'moving_average':
            return self._moving_average_filter_odom(x, y, theta)
        
        return x, y, theta
    
    def _kalman_filter_odom(self, x, y, theta):
        """卡尔曼滤波处理里程计数据"""
        # 观测向量
        z = np.array([x, y, theta])
        
        if not self.kalman_initialized:
            # 初始化状态
            self.kalman_state[:3] = z
            self.kalman_initialized = True
            return x, y, theta
        
        # 预测步骤
        self.kalman_state = self.kalman_F @ self.kalman_state
        self.kalman_P = self.kalman_F @ self.kalman_P @ self.kalman_F.T + self.kalman_Q
        
        # 更新步骤
        y_residual = z - self.kalman_H @ self.kalman_state
        
        # 处理角度的周期性
        y_residual[2] = math.atan2(math.sin(y_residual[2]), math.cos(y_residual[2]))
        
        S = self.kalman_H @ self.kalman_P @ self.kalman_H.T + self.kalman_R
        K = self.kalman_P @ self.kalman_H.T @ np.linalg.inv(S)
        
        self.kalman_state = self.kalman_state + K @ y_residual
        self.kalman_P = (np.eye(6) - K @ self.kalman_H) @ self.kalman_P
        
        return self.kalman_state[0], self.kalman_state[1], self.kalman_state[2]
    
    def _moving_average_filter_odom(self, x, y, theta):
        """移动平均滤波处理里程计数据"""
        # 添加到历史缓存
        self.odom_x_history.append(x)
        self.odom_y_history.append(y)
        self.odom_theta_history.append(theta)
        
        if len(self.odom_x_history) < 2:
            return x, y, theta
        
        # 计算移动平均
        filtered_x = np.mean(list(self.odom_x_history))
        filtered_y = np.mean(list(self.odom_y_history))
        
        # 角度需要特殊处理（圆形平均）
        theta_sin = np.mean([math.sin(t) for t in self.odom_theta_history])
        theta_cos = np.mean([math.cos(t) for t in self.odom_theta_history])
        filtered_theta = math.atan2(theta_sin, theta_cos)
        
        return filtered_x, filtered_y, filtered_theta
    
    def reset(self):
        """重置滤波器状态"""
        self.lidar_history.clear()
        self.odom_x_history.clear()
        self.odom_y_history.clear()
        self.odom_theta_history.clear()
        self.kalman_initialized = False
        self._init_kalman_filter()
        print("[NoiseFilter] 滤波器状态已重置")
    
    def get_status(self):
        """获取滤波器状态信息"""
        return {
            'enabled': self.enabled,
            'lidar_filter_enabled': self.lidar_filter_enabled,
            'odom_filter_enabled': self.odom_filter_enabled,
            'lidar_filter': self.lidar_filter_type,
            'odom_filter': self.odom_filter_type,
            'lidar_history_size': len(self.lidar_history),
            'odom_history_size': len(self.odom_x_history),
            'kalman_initialized': self.kalman_initialized
        }
