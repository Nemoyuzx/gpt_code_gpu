import math
import numpy as np
import torch
from collections import deque
import threading

# 常量参数
MAX_RANGE_FACTOR = 1            # 超过最大测距范围的倍数阈值
ADJACENCY_DIFF_THRESHOLD = 1    # 相邻激光点距离差阈值

class ICPSlam:
    """混合式 ICP-EKF SLAM 模块，结合ICP前端配准和EKF后端定位，支持回环检测与全局优化。"""
    def __init__(self, maze, init_pose):
        """
        maze: Maze 对象，提供占据栅格地图 (maze.grid)、地图分辨率 (maze.resolution)、地图边界 (maze.bounds) 等属性。
        init_pose: 初始位姿 (x, y, theta) 元组。
        """
        # 地图初始化
        self.occupancy = maze.grid                  # 占据栅格地图(-1未知,0空闲,1占据)
        self.resolution = maze.resolution
        self.min_x = maze.bounds[0]
        self.min_y = maze.bounds[1]
        
        # SLAM当前估计位姿 (由ICP前端提供)
        self.x, self.y, self.theta = init_pose
        
        # EKF滤波器状态 (后端融合结果)
        self.ekf_state = np.array([init_pose[0], init_pose[1], init_pose[2]])
        self.ekf_covariance = np.eye(3) * 0.01
        
        # 轨迹历史用于回环检测
        self.trajectory_history = deque(maxlen=1000)   # 保存最近1000帧位姿
        self.trajectory_history.append(self.ekf_state.copy())
        
        # 地图点云用于ICP匹配
        self.map_points = []            # 障碍物点集合（全局坐标列表）
        self.map_points_tensor = None   # 点云PyTorch张量 (在GPU上加速计算)
        
        # 回环检测参数
        self.loop_closure_threshold = 2.0  # 空间距离阈值，小于此视为回环
        self.min_loop_interval = 50       # 最小回环帧间隔，避免重复检测
        self.last_loop_frame = -100       # 上次发生回环的帧计数
        self.frame_count = 0              # 当前帧计数
        
        # ICP算法参数
        self.icp_max_iter = 10                # ICP最大迭代次数
        self.icp_tolerance = 1e-4             # ICP收敛阈值
        self.icp_correspondence_thresh = 1.0  # 最近邻对应距离阈值 (米)
        
        # EKF噪声参数
        self.motion_noise = np.diag([0.01, 0.01, 0.005])   # 运动模型噪声协方差
        self.observation_noise = np.diag([0.05, 0.05, 0.01])  # 观测模型噪声协方差
        
        # 设备选择 (优先GPU/MPS)
        self.device = torch.device("cpu")
        self.use_mps_fallback = False
        try:
            import os
            force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
            if not force_cpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"[ICPSlam] 使用CUDA设备: {torch.cuda.get_device_name(0)}")
            elif not force_cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.use_mps_fallback = True
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                print("[ICPSlam] 使用MPS设备进行加速 (Apple Metal)，对不支持的操作将降级到CPU")
            else:
                print("[ICPSlam] 使用CPU设备" + (" (用户强制)" if force_cpu else " (未检测到GPU)"))
        except Exception as e:
            self.device = torch.device("cpu")
            print(f"[ICPSlam] GPU初始化失败，回退到CPU: {str(e)}")
        
        # 传感器最大测距范围参数：优先使用Maze提供值，否则默认12.0米
        self.max_range = 12.0
        if hasattr(maze, 'max_range'):
            self.max_range = maze.max_range
        
        # 点云地图管理参数
        self.max_map_points = 100000     # 地图点云上限数量，可根据内存/需求调整
        self.downsample_voxel_size = 0.2 # 体素降采样分辨率(米)，可调
        self.downsample_interval = 50    # 点云降采样执行间隔（帧），可调
        
        # 线程锁初始化（用于地图和点云更新的互斥）
        self.lock = threading.Lock()
    
    def to_tensor(self, data):
        """将numpy数组或Tensor转为当前设备上的Tensor。"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def to_numpy(self, tensor):
        """将Tensor转回numpy数组（在CPU上）。"""
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu().numpy() if tensor is not None else None
    
    def to_cpu(self, tensor):
        """将Tensor安全地移到CPU，用于MPS不支持的操作。"""
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu()
    
    def compute_determinant(self, matrix):
        """计算矩阵行列式，兼容MPS设备。"""
        if self.use_mps_fallback:
            matrix_cpu = self.to_cpu(matrix)
            return torch.linalg.det(matrix_cpu)
        else:
            return torch.linalg.det(matrix)
    
    def ekf_predict(self, odom_delta):
        """EKF预测：根据里程计增量预测状态均值和协方差。"""
        d_trans, d_rot = odom_delta
        # 基于当前朝向预测位移增量
        dt_x = d_trans * math.cos(self.ekf_state[2])
        dt_y = d_trans * math.sin(self.ekf_state[2])
        dt_theta = d_rot
        # 状态预测
        predicted_state = self.ekf_state.copy()
        predicted_state[0] += dt_x
        predicted_state[1] += dt_y
        predicted_state[2] += dt_theta
        predicted_state[2] = math.atan2(math.sin(predicted_state[2]), math.cos(predicted_state[2]))  # 规范化角度
        # 雅可比矩阵F
        F = np.array([
            [1, 0, -d_trans * math.sin(self.ekf_state[2])],
            [0, 1,  d_trans * math.cos(self.ekf_state[2])],
            [0, 0, 1]
        ])
        # 协方差预测
        predicted_covariance = F @ self.ekf_covariance @ F.T + self.motion_noise
        return predicted_state, predicted_covariance
    
    def ekf_update(self, icp_pose):
        """EKF更新：融合ICP计算得到的观测位姿。"""
        z = np.array(icp_pose)  # ICP观测值 [x, y, theta]
        h = self.ekf_state.copy()  # 预测观测（此处直接用状态）
        innovation = z - h                   # 创新（观测残差）
        innovation[2] = math.atan2(math.sin(innovation[2]), math.cos(innovation[2]))  # 归一化角度差
        H = np.eye(3)                        # 观测矩阵H（直接观测状态）
        S = H @ self.ekf_covariance @ H.T + self.observation_noise
        K = self.ekf_covariance @ H.T @ np.linalg.inv(S)   # 卡尔曼增益
        # 状态更新
        self.ekf_state = self.ekf_state + K @ innovation
        self.ekf_state[2] = math.atan2(math.sin(self.ekf_state[2]), math.cos(self.ekf_state[2]))
        # 协方差更新
        self.ekf_covariance = (np.eye(3) - K @ H) @ self.ekf_covariance
    
    def detect_loop_closure(self):
        """简单回环检测：当前位置是否接近曾经到过的位置。"""
        current_pos = self.ekf_state[:2]
        # 间隔不足min_loop_interval帧则不进行检测
        if self.frame_count - self.last_loop_frame < self.min_loop_interval:
            return None
        # 在历史轨迹中查找空间距离接近的位置
        for i, hist_pose in enumerate(self.trajectory_history):
            if i >= len(self.trajectory_history) - self.min_loop_interval:
                break  # 排除最近的若干帧
            hist_pos = hist_pose[:2]
            distance = np.linalg.norm(current_pos - hist_pos)
            if distance < self.loop_closure_threshold:
                # 发现回环
                self.last_loop_frame = self.frame_count
                return hist_pose
        return None
    
    def global_pose_correction(self, loop_pose):
        """改进的回环全局位姿矫正：对回环路径段进行整体调整。"""
        try:
            loop_index = list(self.trajectory_history).index(loop_pose)
        except ValueError:
            loop_index = None
        if loop_index is None:
            return  # 未找到匹配的回环位姿
        # 提取从回环位置到当前的轨迹段位姿列表
        poses_segment = list(self.trajectory_history)[loop_index:]
        current_pose = self.ekf_state.copy()
        poses_segment.append(current_pose)  # 将当前位姿暂时加入计算
        # 计算当前位姿与回环位姿之间的总误差（在世界坐标系下）
        loop_pose_arr = loop_pose
        error_x = current_pose[0] - loop_pose_arr[0]
        error_y = current_pose[1] - loop_pose_arr[1]
        error_theta = math.atan2(math.sin(current_pose[2] - loop_pose_arr[2]), math.cos(current_pose[2] - loop_pose_arr[2]))
        # 计算轨迹段累计距离，用于按比例分配误差
        distances = [0.0]
        for j in range(len(poses_segment) - 1):
            dx = poses_segment[j+1][0] - poses_segment[j][0]
            dy = poses_segment[j+1][1] - poses_segment[j][1]
            dist = math.hypot(dx, dy)
            distances.append(distances[-1] + dist)
        total_dist = distances[-1] if distances else 0.0
        if total_dist <= 0:
            # 若总距离为0（重合），则改用序号比例
            total_dist = len(poses_segment) - 1
            distances = [i for i in range(len(poses_segment))]
        # 按距离比例平滑矫正每个位姿
        new_poses_segment = []
        for j, pose in enumerate(poses_segment):
            if j == 0:
                new_poses_segment.append(pose.copy())  # 回环段起点保持不变
            else:
                frac = distances[j] / total_dist
                new_x = pose[0] - frac * error_x
                new_y = pose[1] - frac * error_y
                new_theta = pose[2] - frac * error_theta
                new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))
                new_poses_segment.append(np.array([new_x, new_y, new_theta]))
        # 将矫正后的位姿写回轨迹历史对应位置
        for offset, new_pose in enumerate(new_poses_segment[:-1]):
            idx = loop_index + offset
            self.trajectory_history[idx] = new_pose
        # 更新当前位姿为矫正结果
        corrected_pose = new_poses_segment[-1]
        self.ekf_state = corrected_pose.copy()
        # 降低协方差增益，增加对当前位姿的信任
        self.ekf_covariance *= 0.5
        print(f"[SLAM] 回环检测成功！全局位姿已矫正")
        # 将SLAM位姿同步为EKF矫正后的结果
        self.x, self.y, self.theta = self.ekf_state
    
    def update(self, odom_delta, scan):
        """
        SLAM主更新函数：输入里程计增量 (d_trans, d_rot) 和激光扫描数据 scan 列表，
        融合ICP与EKF更新位姿和地图，返回当前位姿估计 (x, y, theta)。
        """
        self.frame_count += 1
        d_trans, d_rot = odom_delta
        
        # 1️⃣ 利用里程计预测本帧初始位姿（供ICP初始值）
        self.theta += d_rot
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        self.x += d_trans * math.cos(self.theta)
        self.y += d_trans * math.sin(self.theta)
        
        # 2️⃣ EKF状态预测
        predicted_state, predicted_covariance = self.ekf_predict(odom_delta)
        self.ekf_state = predicted_state
        self.ekf_covariance = predicted_covariance
        
        # 3️⃣ 激光数据预处理：滑动窗口中值滤波去除孤立噪点
        scan_len = len(scan)
        filtered_scan = [0] * scan_len
        for i in range(scan_len):
            neighbors = []
            for offset in [-1, 0, 1]:
                idx = (i + offset) % scan_len
                dist = scan[idx]
                if dist < self.get_max_range():
                    neighbors.append(dist)
            med = np.median(neighbors) if neighbors else self.get_max_range()
            filtered_scan[i] = med if med < self.get_max_range() else scan[i]
        scan = filtered_scan
        
        # 4️⃣ 计算激光扫描点的全局坐标
        angles_np = np.radians(np.linspace(0, 360, len(scan), endpoint=False))
        pts_local = []
        far_threshold = self.get_max_range() * MAX_RANGE_FACTOR
        adjacent_diff_threshold = ADJACENCY_DIFF_THRESHOLD
        for i, dist in enumerate(scan):
            if dist >= self.get_max_range() or dist > far_threshold:
                continue
            # 相邻点跳变检查
            should_skip = False
            prev_idx = (i - 1) % scan_len
            next_idx = (i + 1) % scan_len
            if scan[prev_idx] < self.get_max_range() and scan[prev_idx] <= far_threshold:
                if abs(dist - scan[prev_idx]) > adjacent_diff_threshold:
                    should_skip = True
            if scan[next_idx] < self.get_max_range() and scan[next_idx] <= far_threshold:
                if abs(dist - scan[next_idx]) > adjacent_diff_threshold:
                    should_skip = True
            if should_skip:
                continue
            # 计算全局坐标
            angle = self.theta + angles_np[i]
            px = self.x + dist * math.cos(angle)
            py = self.y + dist * math.sin(angle)
            pts_local.append([px, py])
        
        # 5️⃣ ICP配准计算位姿修正
        icp_corrected_pose = (self.x, self.y, self.theta)  # 默认不校正
        if pts_local and self.map_points:
            pts_local = np.array(pts_local)
            src = self.to_tensor(pts_local)  # 当前激光点集（源）
            # 准备目标点集张量
            if self.map_points_tensor is None or len(self.map_points) != len(self.map_points_tensor):
                self.map_points_tensor = self.to_tensor(self.map_points)
            tgt = self.map_points_tensor      # 地图点云（目标）
            # ICP迭代
            for it in range(self.icp_max_iter):
                dist_matrix = torch.cdist(src, tgt)
                min_dists, min_indices = torch.min(dist_matrix, dim=1)
                valid_mask = min_dists < self.icp_correspondence_thresh
                if torch.sum(valid_mask) < 3:
                    print("[ICP] 匹配点不足3个，ICP提前终止")
                    break
                # 提取有效配对点
                paired_src = src[valid_mask]
                paired_tgt = tgt[min_indices[valid_mask]]
                # 计算质心
                src_center = torch.mean(paired_src, dim=0)
                tgt_center = torch.mean(paired_tgt, dim=0)
                src_centered = paired_src - src_center
                tgt_centered = paired_tgt - tgt_center
                # SVD求解最佳刚体变换
                H = src_centered.T @ tgt_centered
                if self.use_mps_fallback:
                    H_cpu = H.to('cpu')
                    U_cpu, S_cpu, Vt_cpu = torch.linalg.svd(H_cpu)
                    R_cpu = Vt_cpu.T @ U_cpu.T
                    if torch.linalg.det(R_cpu) < 0:
                        Vt_cpu[-1, :] *= -1
                        R_cpu = Vt_cpu.T @ U_cpu.T
                    R = R_cpu.to(self.device)
                else:
                    U, S, Vt = torch.linalg.svd(H)
                    R = Vt.T @ U.T
                    if self.compute_determinant(R) < 0:
                        Vt_fixed = Vt.clone()
                        Vt_fixed[-1, :] *= -1
                        R = Vt_fixed.T @ U.T
                t = tgt_center - R @ src_center
                # 应用变换到源点集合
                src_new = (R @ src.T).T + t
                # 判断收敛
                if torch.max(torch.norm(src_new - src, dim=1)) < self.icp_tolerance:
                    src = src_new
                    break
                src = src_new
            # 将ICP计算的位姿更新到机器人位姿
            R_cpu = R.cpu().numpy()
            t_cpu = t.cpu().numpy()
            angle_correction = math.atan2(R_cpu[1, 0], R_cpu[0, 0])
            self.theta = math.atan2(math.sin(self.theta + angle_correction), math.cos(self.theta + angle_correction))
            self.x += t_cpu[0]
            self.y += t_cpu[1]
            # 同步转换当前帧所有点（用于地图更新）
            if len(pts_local) > 0:
                pts_local_tensor = self.to_tensor(pts_local)
                pts_local_transformed = (R @ pts_local_tensor.T).T + t
                pts_local = self.to_numpy(pts_local_transformed)
        icp_corrected_pose = (self.x, self.y, self.theta)
        
        # 6️⃣ EKF更新融合 ICP 位姿观测
        self.ekf_update(icp_corrected_pose)
        
        # 7️⃣ 回环检测与全局矫正
        loop_pose = self.detect_loop_closure()
        if loop_pose is not None:
            self.global_pose_correction(loop_pose)
        # 将当前EKF位姿加入轨迹记录
        self.trajectory_history.append(self.ekf_state.copy())
        
        # 8️⃣ 地图更新（占据栅格 + 点云）
        with self.lock:
            # 使用EKF校正后的状态更新 SLAM 位姿
            self.x, self.y, self.theta = self.ekf_state
            # 机器人所在栅格索引
            rx = int((self.x - self.min_x) / self.resolution)
            ry = int((self.y - self.min_y) / self.resolution)
            freed_cells = []  # 记录本帧变为空闲的栅格列表
            for i, dist in enumerate(scan):
                should_skip_map_update = False
                # 沿用与前面相同的相邻跳变过滤逻辑
                if dist < self.get_max_range() and dist <= far_threshold:
                    if i > 0 and scan[i-1] < self.get_max_range() and scan[i-1] <= far_threshold:
                        if abs(dist - scan[i-1]) > adjacent_diff_threshold:
                            should_skip_map_update = True
                    if i < len(scan) - 1 and scan[i+1] < self.get_max_range() and scan[i+1] <= far_threshold:
                        if abs(dist - scan[i+1]) > adjacent_diff_threshold:
                            should_skip_map_update = True
                    if i == 0 and len(scan) > 1:
                        last_dist = scan[-1]
                        if last_dist < self.get_max_range() and last_dist <= far_threshold:
                            if abs(dist - last_dist) > adjacent_diff_threshold:
                                should_skip_map_update = True
                    elif i == len(scan) - 1 and len(scan) > 1:
                        first_dist = scan[0]
                        if first_dist < self.get_max_range() and first_dist <= far_threshold:
                            if abs(dist - first_dist) > adjacent_diff_threshold:
                                should_skip_map_update = True
                # 计算射线末端全局坐标
                beam_angle = self.theta + (angles_np[i] if len(scan) > 0 else 0.0)
                beam_angle = math.atan2(math.sin(beam_angle), math.cos(beam_angle))
                if dist >= self.get_max_range():
                    # 激光未命中障碍（最大范围），以最大距离计算末端坐标
                    end_x = self.x + self.get_max_range() * math.cos(beam_angle)
                    end_y = self.y + self.get_max_range() * math.sin(beam_angle)
                else:
                    end_x = self.x + dist * math.cos(beam_angle)
                    end_y = self.y + dist * math.sin(beam_angle)
                # 栅格坐标
                tx = int((end_x - self.min_x) / self.resolution)
                ty = int((end_y - self.min_y) / self.resolution)
                # 边界裁剪
                max_x_idx = self.occupancy.shape[1] - 1
                max_y_idx = self.occupancy.shape[0] - 1
                tx = max(0, min(tx, max_x_idx))
                ty = max(0, min(ty, max_y_idx))
                rx = max(0, min(rx, max_x_idx))
                ry = max(0, min(ry, max_y_idx))
                # 获取射线经过的栅格坐标序列
                line = self._bresenham(rx, ry, tx, ty)
                if dist < self.get_max_range() and dist <= far_threshold and not should_skip_map_update:
                    # 有障碍命中且未被过滤
                    for cx, cy in line[:-1]:
                        if self.occupancy[cy, cx] == -1:
                            self.occupancy[cy, cx] = 0  # 射线中途经过的未知栅格标记为空闲
                    ox, oy = line[-1]
                    # 更新障碍栅格（若之前不是障碍）
                    if self.occupancy[oy, ox] != 1:
                        self.occupancy[oy, ox] = 1
                        # 添加新的障碍点到地图点云
                        new_point = [end_x, end_y]
                        self.map_points.append(new_point)
                        if self.map_points_tensor is not None:
                            new_point_tensor = self.to_tensor([new_point])
                            self.map_points_tensor = torch.cat([self.map_points_tensor, new_point_tensor], dim=0)
                else:
                    # 无障碍（激光未命中或被过滤）：整条射线均为空闲
                    for cx, cy in line:
                        # 若该栅格之前标记为障碍，现在应移除
                        if self.occupancy[cy, cx] == 1:
                            freed_cells.append((cx, cy))
                        if self.occupancy[cy, cx] == -1:
                            self.occupancy[cy, cx] = 0
            # 根据freed_cells移除对应的历史障碍点
            if freed_cells and len(self.map_points) <= self.max_map_points:
                freed_set = set(freed_cells)
                new_map_points = []
                for p in self.map_points:
                    cx = int((p[0] - self.min_x) / self.resolution)
                    cy = int((p[1] - self.min_y) / self.resolution)
                    if (cx, cy) not in freed_set:
                        new_map_points.append(p)
                self.map_points = new_map_points
                self.map_points_tensor = self.to_tensor(self.map_points) if self.map_points else None
            # 若点云过大或到达降采样间隔，执行降采样压缩
            if len(self.map_points) > self.max_map_points or \
               (self.downsample_interval and self.frame_count % self.downsample_interval == 0):
                if self.map_points:
                    pts = np.array(self.map_points)
                    # 仅保留对应栅格仍为占据的点
                    idxs = ((pts[:, 0] - self.min_x) / self.resolution).astype(int)
                    idys = ((pts[:, 1] - self.min_y) / self.resolution).astype(int)
                    occ_vals = []
                    for cx, cy in zip(idxs, idys):
                        if 0 <= cy < self.occupancy.shape[0] and 0 <= cx < self.occupancy.shape[1]:
                            occ_vals.append(self.occupancy[cy, cx])
                        else:
                            occ_vals.append(-1)
                    occ_vals = np.array(occ_vals)
                    pts = pts[occ_vals == 1]
                    # 体素栅格滤波降采样
                    voxel_index = np.floor(pts / self.downsample_voxel_size).astype(int)
                    voxel_dict = {}
                    for idx, point in zip(voxel_index, pts):
                        key = (idx[0], idx[1])
                        if key not in voxel_dict:
                            voxel_dict[key] = [point, 1]
                        else:
                            voxel_dict[key][0] += point
                            voxel_dict[key][1] += 1
                    downsampled_points = []
                    for key, (sum_point, count) in voxel_dict.items():
                        avg_point = sum_point / count
                        downsampled_points.append(avg_point.tolist())
                    # 更新地图点云为降采样结果
                    self.map_points = downsampled_points
                    self.map_points_tensor = self.to_tensor(self.map_points) if self.map_points else None
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()  # 释放多余GPU显存缓存
        
        # 返回当前SLAM位姿估计
        return (self.x, self.y, self.theta)
    
    def get_max_range(self):
        """获取激光传感器最大测距范围。"""
        return self.max_range
    
    def get_occupancy(self):
        """安全地获取当前占据栅格地图（拷贝）。"""
        with self.lock:
            return self.occupancy.copy()
    
    def release_resources(self):
        """释放资源（显存），在程序结束时调用。"""
        if hasattr(self, 'map_points_tensor') and self.map_points_tensor is not None:
            del self.map_points_tensor
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def _bresenham(self, x0, y0, x1, y1):
        """Bresenham直线算法：获取从(x0,y0)到(x1,y1)的离散栅格坐标列表。"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points
    
    def set_pose(self, new_pose, reset_covariance=True):
        """手动设置当前位姿（用于真值注入或人工重定位调试）。"""
        self.x, self.y, self.theta = new_pose
        self.ekf_state = np.array([new_pose[0], new_pose[1], new_pose[2]])
        if reset_covariance:
            self.ekf_covariance = np.eye(3) * 0.01
        print(f"[ICPSlam] 手动将位姿设置为: {new_pose}")
