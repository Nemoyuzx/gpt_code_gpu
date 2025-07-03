import math
import numpy as np
import torch  # 引入 PyTorch 库以使用张量和GPU加速

#超过最大范围比例
MAX_RANGE_FACTOR = 0.9  # 超过最大范围的比例阈值，用于忽略远距离点
#相邻测距点差异阈值
ADJACENCY_DIFF_THRESHOLD = 0.2  # 相邻测距点之间的差异阈值 (米)

class ICPSlam:
    """ICP SLAM建图与定位模块。利用激光数据和运动模型进行SLAM。支持GPU加速。"""
    def __init__(self, maze, init_pose):
        """
        maze: Maze对象，提供占据栅格地图 (未知初始化) 和环境参数。
        init_pose: 初始位姿 (x, y, theta)。
        """
        self.occupancy = maze.grid  # 占据栅格地图 (引用自Maze，-1未知,0空闲,1占据)
        self.resolution = maze.resolution
        self.min_x = maze.bounds[0]
        self.min_y = maze.bounds[1]
        # 当前SLAM估计的机器人位姿
        self.x, self.y, self.theta = init_pose
        # 保存地图中的点云（全局坐标）用于ICP匹配
        self.map_points = []  # list of [x,y] obstacle points
        self.map_points_tensor = None  # 地图点云的PyTorch张量版本
        # ICP参数
        self.icp_max_iter = 30
        self.icp_tolerance = 1e-4
        self.icp_correspondence_thresh = 0.2  # 匹配对应点的距离阈值 (米)
        
        # 设备检测与选择
        self.device = torch.device("cpu")  # 默认使用CPU
        self.use_mps_fallback = False      # 标记是否MPS需要特殊处理
        
        try:
            # 检查是否有环境变量设置强制使用CPU
            import os
            force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
            
            if not force_cpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"[ICPSlam] 使用CUDA设备: {torch.cuda.get_device_name(0)}")
            elif not force_cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # 在MPS设备上，某些操作不支持，需要特殊处理
                self.device = torch.device("mps")
                self.use_mps_fallback = True
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 启用MPS降级
                print("[ICPSlam] 使用MPS设备进行加速 (Apple Metal)，对不支持的操作将降级到CPU")
            else:
                print("[ICPSlam] 使用CPU设备" + (" (用户强制)" if force_cpu else " (未检测到GPU)"))
        except Exception as e:
            # 如果设备初始化失败，回退到CPU
            self.device = torch.device("cpu")
            print(f"[ICPSlam] GPU初始化失败，回退到CPU: {str(e)}")
            
    def to_tensor(self, data):
        """将NumPy数组转换为PyTorch张量并移到当前设备"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)
        
    def to_numpy(self, tensor):
        """将PyTorch张量转换为NumPy数组"""
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu().numpy() if tensor is not None else None
        
    def to_cpu(self, tensor):
        """将张量安全地移至CPU"""
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu()
        
    def compute_determinant(self, matrix):
        """计算行列式，处理MPS设备兼容性问题"""
        if self.use_mps_fallback:
            # MPS设备上不支持linalg.det，移至CPU计算
            matrix_cpu = self.to_cpu(matrix)
            return torch.linalg.det(matrix_cpu)
        else:
            # 在CUDA或CPU上直接计算
            return torch.linalg.det(matrix)

    def update(self, odom_delta, scan):
        """
        使用新的里程计增量odom_delta (tuple: (delta_distance, delta_theta))
        和激光扫描数据scan (距离列表) 更新SLAM估计和地图。
        返回更新后的位姿估计 (x, y, theta)。
        """
        d_trans, d_rot = odom_delta
        # 步骤1: 预测位姿 (根据里程计增量更新估计位姿)
        self.theta += d_rot
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        # 假设d_trans沿当前朝向方向
        self.x += d_trans * math.cos(self.theta)
        self.y += d_trans * math.sin(self.theta)
        
        # 步骤2: ICP匹配校正 (使用scan与已有地图点云匹配修正位姿)
        # 将激光扫描点转换为全局坐标（基于预测位姿）
        # 准备角度数据
        angles_np = np.radians(np.arange(0, 360, 1.0))
        if len(scan) != len(angles_np):
            angles_np = np.radians(np.linspace(0, 360, len(scan), endpoint=False))
        
        # 可以考虑在GPU上进行这部分计算，但由于涉及多重条件判断，目前在CPU上处理
        pts_local = []
        # 计算80%的最大范围阈值
        far_threshold = self.get_max_range() * MAX_RANGE_FACTOR
        # 相邻测距点差异阈值 - 如果相邻两个点距离差超过此值，则忽略该点
        adjacent_diff_threshold = ADJACENCY_DIFF_THRESHOLD  # 米，可根据实际情况调整
        
        for i, dist in enumerate(scan):
            if dist >= self.get_max_range() or dist > far_threshold:
                # 距离为最大范围或超过80%最大范围，未击中障碍或距离太远，跳过作为特征点（不加入ICP匹配）
                continue
                
            # 检查与相邻点的距离差异，过滤掉跳跃点
            should_skip = False
            # 检查与前一个点的差异
            if i > 0 and scan[i-1] < self.get_max_range() and scan[i-1] <= far_threshold:
                diff_prev = abs(dist - scan[i-1])
                if diff_prev > adjacent_diff_threshold:
                    should_skip = True
            
            # 检查与后一个点的差异
            if i < len(scan) - 1 and scan[i+1] < self.get_max_range() and scan[i+1] <= far_threshold:
                diff_next = abs(dist - scan[i+1])
                if diff_next > adjacent_diff_threshold:
                    should_skip = True
            
            # 对于环形激光雷达，还要检查第一个和最后一个点的连接
            if i == 0 and len(scan) > 1:  # 第一个点，检查与最后一个点的差异
                last_dist = scan[-1]
                if last_dist < self.get_max_range() and last_dist <= far_threshold:
                    diff_wrap = abs(dist - last_dist)
                    if diff_wrap > adjacent_diff_threshold:
                        should_skip = True
            elif i == len(scan) - 1 and len(scan) > 1:  # 最后一个点，检查与第一个点的差异
                first_dist = scan[0]
                if first_dist < self.get_max_range() and first_dist <= far_threshold:
                    diff_wrap = abs(dist - first_dist)
                    if diff_wrap > adjacent_diff_threshold:
                        should_skip = True
            
            if should_skip:
                continue
                
            angle = self.theta + angles_np[i]
            px = self.x + dist * math.cos(angle)
            py = self.y + dist * math.sin(angle)
            pts_local.append([px, py])
        
        # 只有当有点被添加时才创建数组
        if not pts_local:
            # 没有有效的扫描点，不进行ICP
            return (self.x, self.y, self.theta)
            
        pts_local = np.array(pts_local)
        # 如果已有地图点云不为空，则执行ICP配准
        if len(self.map_points) > 0 and pts_local.size > 0:
            # 将点云数据转换为PyTorch张量并移到选定设备上
            src = self.to_tensor(pts_local)  # 新扫描点（源）
            
            # 如果地图点云张量未初始化或需要更新，则更新张量版本
            if self.map_points_tensor is None or len(self.map_points) != len(self.map_points_tensor):
                self.map_points_tensor = self.to_tensor(self.map_points)
            
            tgt = self.map_points_tensor  # 使用缓存的张量版本地图点云
            
            # ICP迭代
            for it in range(self.icp_max_iter):
                # 为每个源点找到最近的目标点索引（GPU加速版本）
                src_pts = src
                
                # 使用torch.cdist计算所有点对之间的距离矩阵
                dist_matrix = torch.cdist(src_pts, tgt)  # [src_size, tgt_size]
                min_dists, min_indices = torch.min(dist_matrix, dim=1)  # 每个源点的最小距离和索引
                
                # 根据阈值过滤对应点
                valid_mask = min_dists < self.icp_correspondence_thresh
                if torch.sum(valid_mask) < 3:
                    # 有效对应点太少，不足以计算精确变换，跳出
                    break
                
                # 筛选有效的配对点
                paired_src = src_pts[valid_mask]
                paired_tgt = tgt[min_indices[valid_mask]]
                
                # 计算质心
                src_center = torch.mean(paired_src, dim=0)
                tgt_center = torch.mean(paired_tgt, dim=0)
                
                # 去中心化
                src_centered = paired_src - src_center
                tgt_centered = paired_tgt - tgt_center
                
                # 求解最优旋转（使用SVD）- GPU加速版本
                H = src_centered.T @ tgt_centered  # 矩阵乘法
                
                # 对不支持的操作进行特殊处理
                if self.use_mps_fallback:
                    # 在MPS上某些操作不支持，将张量移到CPU进行计算
                    H_cpu = H.to('cpu')
                    U_cpu, S_cpu, Vt_cpu = torch.linalg.svd(H_cpu)
                    # 计算当前迭代的旋转矩阵 R
                    R_cpu = Vt_cpu.T @ U_cpu.T
                    # 若出现反射（det(R)为负），调整Vt最后一行符号以确保得到合法旋转
                    det_R = torch.linalg.det(R_cpu)
                    if det_R < 0:
                        Vt_fixed = Vt_cpu.clone()
                        Vt_fixed[-1, :] *= -1
                        R_cpu = Vt_fixed.T @ U_cpu.T
                    # 将结果移回设备
                    R = R_cpu.to(self.device)
                else:
                    # CUDA或CPU设备上直接计算
                    U, S, Vt = torch.linalg.svd(H)
                    R = Vt.T @ U.T
                    
                    # 若产生反射，调整R
                    det_R = self.compute_determinant(R)
                    if det_R < 0:
                        Vt_fixed = Vt.clone()
                        Vt_fixed[-1, :] *= -1
                        R = Vt_fixed.T @ U.T
                
                # 计算平移
                t = tgt_center - R @ src_center
                
                # 对src点集应用变换
                src_new = (R @ src_pts.T).T + t
                
                # 计算变化量用于判断收敛
                diff = torch.max(torch.norm(src_new - src_pts, dim=1))
                
                # 更新源点（为下次迭代）
                src = src_new
                
                if diff < self.icp_tolerance:
                    break
            
            # 将最终的R和t转回CPU进行位姿更新
            R_cpu = R.cpu().numpy()
            t_cpu = t.cpu().numpy()
            
            # 更新当前位姿估计
            # 旋转角校正量
            angle_correction = math.atan2(R_cpu[1, 0], R_cpu[0, 0])
            self.theta = math.atan2(math.sin(self.theta + angle_correction), math.cos(self.theta + angle_correction))
            
            # 平移校正量
            self.x += t_cpu[0]
            self.y += t_cpu[1]
            
            # 使用修正后的pose更新当前扫描点坐标（全局）
            if pts_local.size > 0:
                pts_local_tensor = self.to_tensor(pts_local)
                pts_local_transformed = (R @ pts_local_tensor.T).T + t
                pts_local = self.to_numpy(pts_local_transformed)
        # 步骤3: 更新地图（占据栅格和点云）
        # 获取机器人在栅格中的索引
        rx = int((self.x - self.min_x) / self.resolution)
        ry = int((self.y - self.min_y) / self.resolution)
        # 80%的最大范围阈值
        far_threshold = self.get_max_range() * MAX_RANGE_FACTOR
        # 相邻测距点差异阈值 - 与ICP部分保持一致
        adjacent_diff_threshold = 2.0  # 米
        
        # 更新占据栅格地图，根据扫描结果
        for i, dist in enumerate(scan):
            # 应用相同的相邻点差异过滤逻辑
            should_skip_map_update = False
            if dist < self.get_max_range() and dist <= far_threshold:
                # 检查与相邻点的距离差异
                if i > 0 and scan[i-1] < self.get_max_range() and scan[i-1] <= far_threshold:
                    diff_prev = abs(dist - scan[i-1])
                    if diff_prev > adjacent_diff_threshold:
                        should_skip_map_update = True
                
                if i < len(scan) - 1 and scan[i+1] < self.get_max_range() and scan[i+1] <= far_threshold:
                    diff_next = abs(dist - scan[i+1])
                    if diff_next > adjacent_diff_threshold:
                        should_skip_map_update = True
                
                # 环形连接检查
                if i == 0 and len(scan) > 1:
                    last_dist = scan[-1]
                    if last_dist < self.get_max_range() and last_dist <= far_threshold:
                        diff_wrap = abs(dist - last_dist)
                        if diff_wrap > adjacent_diff_threshold:
                            should_skip_map_update = True
                elif i == len(scan) - 1 and len(scan) > 1:
                    first_dist = scan[0]
                    if first_dist < self.get_max_range() and first_dist <= far_threshold:
                        diff_wrap = abs(dist - first_dist)
                        if diff_wrap > adjacent_diff_threshold:
                            should_skip_map_update = True
            
            beam_angle = self.theta + (angles_np[i] if 'angles_np' in locals() else math.radians(i))
            # 归一化角度
            beam_angle = math.atan2(math.sin(beam_angle), math.cos(beam_angle))
            # 计算射线末端点（若达到距离上限则取max_range点）
            if dist >= self.get_max_range():
                # 未检测到障碍
                end_x = self.x + self.get_max_range() * math.cos(beam_angle)
                end_y = self.y + self.get_max_range() * math.sin(beam_angle)
            else:
                # 检测到障碍，计算击中点
                end_x = self.x + dist * math.cos(beam_angle)
                end_y = self.y + dist * math.sin(beam_angle)
            # 转换为栅格索引
            tx = int((end_x - self.min_x) / self.resolution)
            ty = int((end_y - self.min_y) / self.resolution)
            # 限制索引在地图范围内
            max_x_idx = self.occupancy.shape[1] - 1
            max_y_idx = self.occupancy.shape[0] - 1
            if tx < 0: tx = 0
            if tx > max_x_idx: tx = max_x_idx
            if ty < 0: ty = 0
            if ty > max_y_idx: ty = max_y_idx
            
            # 确保机器人位置也在栅格范围内
            if rx < 0: rx = 0
            if rx > max_x_idx: rx = max_x_idx
            if ry < 0: ry = 0
            if ry > max_y_idx: ry = max_y_idx
            # 获取栅格直线路径
            line = self._bresenham(rx, ry, tx, ty)
            if dist < self.get_max_range() and dist <= far_threshold and not should_skip_map_update:
                # 有障碍命中且在80%范围内且未被过滤：最后一点为障碍
                for cell in line[:-1]:
                    cx, cy = cell
                    # 如果当前未知，则标记为空闲
                    if self.occupancy[cy, cx] == -1:
                        self.occupancy[cy, cx] = 0
                # 最后一个点标记为占据
                ox, oy = line[-1]
                self.occupancy[oy, ox] = 1
                # 记录障碍物点坐标（全局坐标）
                new_point = [end_x, end_y]
                self.map_points.append(new_point)
                
                # 更新张量版本的地图点云（增量更新）
                if hasattr(self, 'map_points_tensor') and self.map_points_tensor is not None:
                    # 如果已存在张量版本，追加新点
                    new_point_tensor = self.to_tensor([new_point])
                    self.map_points_tensor = torch.cat([self.map_points_tensor, new_point_tensor], dim=0)
                # 如果张量版本不存在，下次ICP时会完整更新
            else:
                # 未命中障碍：整条射线区域均为空闲
                for cell in line:
                    cx, cy = cell
                    if self.occupancy[cy, cx] == -1:
                        self.occupancy[cy, cx] = 0
        return (self.x, self.y, self.theta)

    def get_max_range(self):
        """获取传感器最大范围。假设扫描数据长度和角度分辨率固定360度。"""
        # 由于scan本身不包含max_range信息，这里假定scan中最大值代表max_range。
        # 实际实现中可存储max_range参数。此处简化处理。
        return 12.0

    def get_occupancy(self):
        """返回当前占据栅格地图 (numpy数组)。"""
        return self.occupancy
        
    def release_resources(self):
        """释放GPU资源，在程序结束前调用"""
        # 清除张量缓存
        if hasattr(self, 'map_points_tensor') and self.map_points_tensor is not None:
            del self.map_points_tensor
        # 清空CUDA缓存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _bresenham(self, x0, y0, x1, y1):
        """Bresenham算法获取两个格点之间的离散栅格线。"""
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
