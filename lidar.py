import math
import numpy as np

class Lidar:
    """激光雷达模拟器。生成360度距离扫描数据。"""
    def __init__(self, walls, max_range=12.0, angle_resolution=1.0, noise=0.0):
        """
        walls: 墙壁线段列表，每个为((x1,y1),(x2,y2))，在同一全局坐标系下。
        max_range: 最大探测距离 (米)。
        angle_resolution: 角分辨率 (度)。
        noise: 距离测量噪声标准差 (米)。
        """
        self.walls = walls
        self.max_range = max_range
        self.angle_resolution = angle_resolution
        self.noise = noise
        self.last_start_offset_deg = 0.0
        
        # 降噪滤波器引用（由外部设置）
        self.noise_filter = None

    def scan(self, pose):
        """
        返回 (noisy_distances, clean_distances)
        clean_distances: 每束光在加噪前的最近交点距离
        noisy_distances: 在 clean 的基础上加入噪声和截断后的测量值（与你原逻辑一致）
        """
        x, y, theta = pose
        num_beams = int(360 / self.angle_resolution)
        angle_step = math.radians(self.angle_resolution)
        start_offset = float(np.random.uniform(0.0, math.radians(1.0)))
        self.last_start_offset_deg = math.degrees(start_offset)

        clean = []
        noisy = []

        for i in range(num_beams):
            angle = theta + start_offset + i * angle_step
            angle = math.atan2(math.sin(angle), math.cos(angle))
            dx, dy = math.cos(angle), math.sin(angle)

            closest_dist = self.max_range
            for (p1, p2) in self.walls:
                x1, y1 = p1
                x2, y2 = p2
                v_x, v_y = dx, dy
                w_x, w_y = (x2 - x1), (y2 - y1)
                denom = v_x * w_y - v_y * w_x
                if abs(denom) < 1e-6:
                    continue
                t = ((x1 - x) * w_y - (y1 - y) * w_x) / denom
                u = ((x1 - x) * v_y - (y1 - y) * v_x) / denom
                if t >= 0 and 0 <= u <= 1:
                    if t < closest_dist:
                        closest_dist = t

            # 保存干净距离（未加噪）
            clean.append(closest_dist)

            # 构造含噪、截断测量
            measured_dist = closest_dist
            if self.noise > 0:
                measured_dist += np.random.normal(0, self.noise)
                if measured_dist < 0:
                    measured_dist = 0.0
            if measured_dist > self.max_range:
                measured_dist = self.max_range
            noisy.append(measured_dist)

        return noisy, clean

    
    def set_noise_filter(self, noise_filter):
        """设置降噪滤波器"""
        self.noise_filter = noise_filter
    
        
    def is_far_range(self, dist, threshold_ratio=0.8):
        """
        判断测量距离是否超出最大探测范围的指定比例
        
        参数:
        - dist: 测量距离
        - threshold_ratio: 阈值比例，默认为0.8（80%）
        
        返回:
        - True: 如果距离超出最大范围的threshold_ratio
        - False: 如果距离在范围内
        """
        return dist >= self.max_range * threshold_ratio
