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

        # 预计算墙段的向量化表示 (P1, W)，用于向量化求交。
        # 形状均为 (M, 2)。
        if walls:
            walls_arr = np.asarray(walls, dtype=np.float64)  # (M, 2, 2)
            self._wall_p1 = walls_arr[:, 0, :]               # (M, 2)
            self._wall_w = walls_arr[:, 1, :] - walls_arr[:, 0, :]  # (M, 2)
        else:
            self._wall_p1 = np.zeros((0, 2), dtype=np.float64)
            self._wall_w = np.zeros((0, 2), dtype=np.float64)

        # 预计算 beam 方向 (旋转前) 以加速 scan。
        num_beams = int(360 / angle_resolution)
        beam_idx = np.arange(num_beams, dtype=np.float64)
        self._beam_base_angles = beam_idx * math.radians(angle_resolution)  # (N,)
        self._num_beams = num_beams

        # 降噪滤波器引用（由外部设置）
        self.noise_filter = None

    def scan(self, pose):
        """
        返回 (noisy_distances, clean_distances)
        clean_distances: 每束光在加噪前的最近交点距离
        noisy_distances: 在 clean 的基础上加入噪声和截断后的测量值（与原逻辑一致）

        向量化实现：对所有 beam × 墙段同时求解参数化交点。
        """
        x, y, theta = pose
        num_beams = self._num_beams
        max_range = self.max_range

        if self._wall_w.shape[0] == 0:
            clean = np.full(num_beams, max_range, dtype=np.float64)
        else:
            # beam 方向 (N, 2)
            angles = theta + self._beam_base_angles
            dx = np.cos(angles)
            dy = np.sin(angles)

            w = self._wall_w        # (M, 2)
            p1 = self._wall_p1      # (M, 2)
            wx = w[:, 0]            # (M,)
            wy = w[:, 1]

            # denom[i, j] = dx_i * wy_j - dy_i * wx_j  -> (N, M)
            denom = np.outer(dx, wy) - np.outer(dy, wx)

            # (x1 - x), (y1 - y)  广播至 (N, M)
            qx = p1[:, 0] - x       # (M,)
            qy = p1[:, 1] - y

            # 为避免除零告警，在平行情况下临时将 denom 替换为 1，
            # 最后通过 valid 掩码滤掉这些解。
            parallel = np.abs(denom) <= 1e-9
            denom_safe = np.where(parallel, 1.0, denom)

            # t = (qx * wy - qy * wx) / denom 与 beam 无关的分子 (M,)
            numer_t = qx * wy - qy * wx                  # (M,)
            t_mat = numer_t[None, :] / denom_safe        # (N, M)

            # u = (qx * dy - qy * dx) / denom
            numer_u = np.outer(dx, qy) - np.outer(dy, qx)  # (N, M) = dx*qy - dy*qx
            # 上式实际 = dx_i * qy_j - dy_i * qx_j，但我们要 qx * dy - qy * dx
            # 即 -numer_u，因此：
            u_mat = -numer_u / denom_safe

            # 有效交点掩码
            valid = (~parallel) & (t_mat >= 0.0) & (u_mat >= 0.0) & (u_mat <= 1.0)

            # 无效位置填为 +inf，便于取 min
            t_mat = np.where(valid, t_mat, np.inf)
            closest = np.min(t_mat, axis=1)             # (N,)
            closest = np.where(closest < max_range, closest, max_range)
            clean = closest

        # 构造含噪、截断测量
        if self.noise > 0:
            noisy = clean + np.random.normal(0.0, self.noise, size=num_beams)
            np.clip(noisy, 0.0, max_range, out=noisy)
        else:
            noisy = np.minimum(clean, max_range)

        # 保持与旧接口一致：返回 Python list
        return noisy.tolist(), clean.tolist()

    
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
