import math
import numpy as np
import torch
import os   # 新增: 用于文件读写操作
import gc   # 新增: 用于显式进行垃圾回收



#超过最大范围比例
MAX_RANGE_FACTOR = 0.49  # 超过最大范围的比例阈值，用于忽略远距离点
#相邻测距点差异阈值
ADJACENCY_DIFF_THRESHOLD = 0.01  # 相邻测距点之间的差异阈值 (米)

ICP_MAX_ITER = 500  # ICP最大迭代次数
ICP_TOLERANCE = 1e-7  # ICP收敛容忍
ICP_CORRESPONDENCE_THRESH = 0.000001  # ICP对应点匹配距离

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
        self.icp_max_iter = ICP_MAX_ITER
        self.icp_tolerance = ICP_TOLERANCE  # 收敛容忍度
        self.icp_correspondence_thresh = ICP_CORRESPONDENCE_THRESH  # 对应点匹配距离阈值
          # 设备检测与选择
        self.device = torch.device("cpu")  # 默认使用CPU
        self.use_mps_fallback = False      # 标记是否MPS需要特殊处理
        
        try:
            # 检查是否有环境变量设置强制使用CPU
            force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
            slam_device = os.environ.get("SLAM_DEVICE", "")
            
            if not force_cpu:
                if slam_device:
                    # 使用环境变量指定的设备
                    self.device = torch.device(slam_device.split(':')[0])  # 去掉设备索引
                    if slam_device.startswith("cuda") and torch.cuda.is_available():
                        print(f"[ICPSlam] 使用CUDA设备: {torch.cuda.get_device_name(0)}")
                    elif slam_device == "mps":
                        self.use_mps_fallback = True
                        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                        print("[ICPSlam] 使用MPS设备进行加速 (Apple Metal)")
                    else:
                        print(f"[ICPSlam] 使用指定设备: {slam_device}")
                elif torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print(f"[ICPSlam] 使用CUDA设备: {torch.cuda.get_device_name(0)}")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # 在MPS设备上，某些操作不支持，需要特殊处理
                    self.device = torch.device("mps")
                    self.use_mps_fallback = True
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 启用MPS降级
                    print("[ICPSlam] 使用MPS设备进行加速 (Apple Metal)，对不支持的操作将降级到CPU")
                else:
                    print("[ICPSlam] 使用CPU设备 (未检测到GPU)")
            else:
                print("[ICPSlam] 使用CPU设备 (用户强制)")
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
        使用新的里程计增量 odom_delta (tuple: (delta_distance, delta_theta))
        和激光扫描数据 scan (距离列表) 更新 SLAM 估计和地图。
        返回更新后的位姿估计 (x, y, theta)。
        """
        d_trans, d_rot = odom_delta
        # 步骤1: 预测位姿 (根据里程计增量更新估计位姿)
        self.theta += d_rot
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        # 假设 d_trans 沿当前朝向方向
        self.x += d_trans * math.cos(self.theta)
        self.y += d_trans * math.sin(self.theta)

        # 步骤2: ICP 匹配校正 (使用 scan 与已有地图点云匹配修正位姿)
        # 将激光扫描点转换为全局坐标（基于预测位姿）
        angles_np = np.radians(np.arange(0, 360, 1.0))
        if len(scan) != len(angles_np):
            angles_np = np.radians(np.linspace(0, 360, len(scan), endpoint=False))

        pts_local = []
        far_threshold = self.get_max_range() * MAX_RANGE_FACTOR  # 80% 最大范围阈值
        adjacent_diff_threshold = ADJACENCY_DIFF_THRESHOLD       # 相邻点距离差阈值

        for i, dist in enumerate(scan):
            if dist >= self.get_max_range() or dist > far_threshold:
                # 超过最大测距或阈值，未击中障碍或太远，跳过
                continue
            # 检查与相邻点的距离差异，滤除陡升/陡降点
            should_skip = False
            if i > 0 and scan[i-1] < self.get_max_range() and scan[i-1] <= far_threshold:
                if abs(dist - scan[i-1]) > adjacent_diff_threshold:
                    should_skip = True
            if i < len(scan) - 1 and scan[i+1] < self.get_max_range() and scan[i+1] <= far_threshold:
                if abs(dist - scan[i+1]) > adjacent_diff_threshold:
                    should_skip = True
            # 环形激光雷达首尾相邻点差异检查
            if i == 0 and len(scan) > 1:
                last_dist = scan[-1]
                if last_dist < self.get_max_range() and last_dist <= far_threshold:
                    if abs(dist - last_dist) > adjacent_diff_threshold:
                        should_skip = True
            elif i == len(scan) - 1 and len(scan) > 1:
                first_dist = scan[0]
                if first_dist < self.get_max_range() and first_dist <= far_threshold:
                    if abs(dist - first_dist) > adjacent_diff_threshold:
                        should_skip = True
            if should_skip:
                continue
            # 计算该激光点的全局坐标
            angle = self.theta + angles_np[i]
            px = self.x + dist * math.cos(angle)
            py = self.y + dist * math.sin(angle)
            pts_local.append([px, py])

        if not pts_local:
            # 没有有效扫描点，直接返回预测位姿（无ICP校正）
            return (self.x, self.y, self.theta)
        pts_local = np.array(pts_local, dtype=np.float32)

        # 如果存在已有地图点云，则进行 ICP 匹配校正
        if (len(self.map_points) > 0 or os.path.exists("map_points.npy")) and pts_local.size > 0:
            # 开始 ICP 前，将地图点云加载至 GPU/CPU 张量
            # 将新扫描点转换为 PyTorch 张量
            src = self.to_tensor(pts_local)  # 源点云 (shape: [N_src, 2])
            # 准备目标点云张量 (地图点)
            if self.map_points_tensor is not None and len(self.map_points) > 0:
                tgt = self.map_points_tensor  # 若内存中已有张量版本则直接使用
            else:
                # 从磁盘加载已有地图点（如果存在）
                map_points_np = None
                try:
                    map_points_np = np.load("map_points.npy").astype(np.float32)
                except FileNotFoundError:
                    map_points_np = np.array(self.map_points, dtype=np.float32)
                tgt = self.to_tensor(map_points_np)  # 转为张量送入设备
                del map_points_np  # 释放CPU内存
            # 在 no_grad 环境下进行 ICP 迭代，以减少显存开销
            with torch.no_grad():
                # 提前初始化 R 和 t，若 ICP 对应点不足可保持单位变换
                R = torch.eye(2, device=self.device)
                t = torch.zeros(2, device=self.device)
                # 用于避免未定义变量的占位初始化
                H = None; U = Vt = None
                U_cpu = Vt_cpu = None
                # ICP 迭代过程
                for it in range(self.icp_max_iter):
                    # 计算源点集到目标点集的距离矩阵并寻找最近邻
                    dist_matrix = torch.cdist(src, tgt)  # [N_src, N_tgt]
                    min_dists, min_indices = torch.min(dist_matrix, dim=1)
                    # 筛选出距离在阈值内的有效对应点对
                    valid_mask = min_dists < self.icp_correspondence_thresh
                    if torch.sum(valid_mask) < 3:
                        # 对应点太少，无法计算精确变换，退出 ICP
                        break
                    paired_src = src[valid_mask]
                    paired_tgt = tgt[min_indices[valid_mask]]
                    # 计算配对点集的质心
                    src_center = torch.mean(paired_src, dim=0)
                    tgt_center = torch.mean(paired_tgt, dim=0)
                    # 去中心化点集坐标
                    src_centered = paired_src - src_center
                    tgt_centered = paired_tgt - tgt_center
                    # 计算协方差矩阵 H
                    H = src_centered.T @ tgt_centered
                    # 求解最优旋转矩阵 R（根据设备类型选择计算方式）
                    if self.use_mps_fallback:
                        # 在 MPS 设备上，切换到 CPU 计算 SVD
                        H_cpu = H.cpu()
                        U_cpu, _, Vt_cpu = torch.linalg.svd(H_cpu)
                        R_cpu_tensor = Vt_cpu.T @ U_cpu.T
                        # 如果产生反射（行列式为负），调整旋转矩阵
                        if torch.linalg.det(R_cpu_tensor) < 0:
                            Vt_cpu[-1, :] *= -1
                            R_cpu_tensor = Vt_cpu.T @ U_cpu.T
                        R = R_cpu_tensor.to(self.device)
                    else:
                        U, _, Vt = torch.linalg.svd(H)
                        R = Vt.T @ U.T
                        # 若产生反射，调整最后一行符号确保合法旋转
                        if self.compute_determinant(R) < 0:
                            Vt_fixed = Vt.clone()
                            Vt_fixed[-1, :] *= -1
                            R = Vt_fixed.T @ U.T
                    # 计算平移向量 t
                    t = tgt_center - R @ src_center
                    # 将变换应用于源点集合并计算最大位移差
                    src_new = (R @ src.T).T + t
                    diff = torch.max(torch.norm(src_new - src, dim=1))
                    # 更新 src，为下次迭代使用更新后的点集
                    src = src_new
                    if diff < self.icp_tolerance:
                        # 收敛判定：变化量小于阈值，结束迭代
                        break
                # 提取最终结果到 CPU（numpy）用于更新机器人位姿
                R_cpu = R.cpu().numpy()
                t_cpu = t.cpu().numpy()
                # 显式删除大张量释放显存
                for name in [
                    'dist_matrix', 'min_dists', 'min_indices', 'valid_mask',
                    'paired_src', 'paired_tgt', 'src_center', 'tgt_center',
                    'src_centered', 'tgt_centered', 'src_new', 'src', 'tgt',
                    'U', 'Vt', 'U_cpu', 'Vt_cpu'
                ]:
                    if name in locals():
                        del locals()[name]

                if 'U' in locals(): del U, Vt  # 删除 SVD 中间结果
                if 'U_cpu' in locals(): del U_cpu, Vt_cpu
                # 清空未使用的 GPU 显存缓存
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            # 利用 ICP 计算得到的 R_cpu 和 t_cpu 修正机器人位姿
            angle_correction = math.atan2(R_cpu[1, 0], R_cpu[0, 0])
            self.theta = math.atan2(math.sin(self.theta + angle_correction),
                                    math.cos(self.theta + angle_correction))
            self.x += t_cpu[0]
            self.y += t_cpu[1]
            # 使用修正后的位姿更新当前激光点的全局坐标（numpy 计算）
            pts_local = pts_local.dot(R_cpu.T) + t_cpu
        # （若未进入 ICP，例如地图为空，仅根据里程计预测，则直接使用预测位姿进行建图）

        # 步骤3: 更新占据栅格地图和地图点云列表（将新扫描结果整合进地图）
        rx = int((self.x - self.min_x) / self.resolution)
        ry = int((self.y - self.min_y) / self.resolution)
        far_threshold = self.get_max_range() * MAX_RANGE_FACTOR
        adjacent_diff_threshold = 2.0  # 与ICP部分一致或更宽松的阈值
        for i, dist in enumerate(scan):
            should_skip_map_update = False
            if dist < self.get_max_range() and dist <= far_threshold:
                # 与相邻点差异的过滤（与上面类似逻辑）
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
            # 计算该激光束末端的全局坐标 (end_x, end_y)
            beam_angle = self.theta + (angles_np[i] if 'angles_np' in locals() else math.radians(i))
            beam_angle = math.atan2(math.sin(beam_angle), math.cos(beam_angle))  # 归一化角度
            if dist >= self.get_max_range():
                # 未检测到障碍，用最大范围点作为末端
                end_x = self.x + self.get_max_range() * math.cos(beam_angle)
                end_y = self.y + self.get_max_range() * math.sin(beam_angle)
            else:
                # 检测到障碍
                end_x = self.x + dist * math.cos(beam_angle)
                end_y = self.y + dist * math.sin(beam_angle)
            # 将末端点转换为栅格地图索引
            tx = int((end_x - self.min_x) / self.resolution);  ty = int((end_y - self.min_y) / self.resolution)
            max_x_idx = self.occupancy.shape[1] - 1;          max_y_idx = self.occupancy.shape[0] - 1
            if tx < 0: tx = 0
            if tx > max_x_idx: tx = max_x_idx
            if ty < 0: ty = 0
            if ty > max_y_idx: ty = max_y_idx
            # 确保机器人自身所在格也在范围内
            if rx < 0: rx = 0
            if rx > max_x_idx: rx = max_x_idx
            if ry < 0: ry = 0
            if ry > max_y_idx: ry = max_y_idx
            # 获取射线经过的栅格路径
            line = self._bresenham(rx, ry, tx, ty)
            if dist < self.get_max_range() and dist <= far_threshold and not should_skip_map_update:
                # 射线击中了障碍物（在范围内且未被过滤）
                # 将路径上除最后一点外的格子标记为空闲
                for cx, cy in line[:-1]:
                    if self.occupancy[cy, cx] == -1:
                        self.occupancy[cy, cx] = 0
                # 最后一个格子是障碍物
                ox, oy = line[-1]
                # 若该障碍格此前未标记过，则标记占据并记录点
                if self.occupancy[oy, ox] != 1:
                    self.occupancy[oy, ox] = 1
                    new_point = [end_x, end_y]
                    self.map_points.append(new_point)
                    # （不再这里增量更新 map_points_tensor，改为统一在磁盘保存）
            else:
                # 未命中障碍（或被过滤）：该射线经过区域均标记为空闲
                for cx, cy in line:
                    if self.occupancy[cy, cx] == -1:
                        self.occupancy[cy, cx] = 0

        # 将地图点云数据保存到磁盘，释放内存（历史点云不常驻内存）
        if len(self.map_points) > 0:
            # 将当前新增的点与已有点云合并保存
            if os.path.exists("map_points.npy"):
                try:
                    old_points = np.load("map_points.npy").astype(np.float32)
                except Exception:
                    old_points = np.empty((0, 2), dtype=np.float32)
                if len(old_points) > 0:
                    new_points = np.array(self.map_points, dtype=np.float32)
                    all_points = np.concatenate((old_points, new_points), axis=0)
                else:
                    # 若旧文件存在但无数据（或读取失败），直接使用新点
                    all_points = np.array(self.map_points, dtype=np.float32)
                np.save("map_points.npy", all_points)
            else:
                # 若不存在旧文件，直接保存当前点云
                np.save("map_points.npy", np.array(self.map_points, dtype=np.float32))
        # 清理内存中的点云列表和张量缓存
        self.map_points.clear()
        if hasattr(self, 'map_points_tensor') and self.map_points_tensor is not None:
            del self.map_points_tensor
            self.map_points_tensor = None
        # 最后再显式调用垃圾回收，释放Python对象占用的内存
        gc.collect()
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
        try:
            # 清除张量缓存
            if hasattr(self, 'map_points_tensor') and self.map_points_tensor is not None:
                del self.map_points_tensor
                self.map_points_tensor = None
            
            # 清除地图点列表
            if hasattr(self, 'map_points'):
                self.map_points.clear()
            
            # 清除其他可能的张量
            for attr_name in ['occupancy', 'grid']:
                if hasattr(self, attr_name):
                    attr_value = getattr(self, attr_name)
                    if torch.is_tensor(attr_value):
                        del attr_value
            
            # 多次清空GPU缓存以确保彻底清理
            if self.device.type == 'cuda':
                for _ in range(3):
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()  # 清理进程间通信缓存
            elif self.device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            # 多次执行垃圾回收
            for _ in range(3):
                gc.collect()
            
        except Exception as e:
            print(f"[ICPSlam] 释放资源时出错: {e}")
    
    @staticmethod
    def clear_global_cache():
        """清理全局缓存的静态方法"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[ICPSlam] 清理全局缓存时出错: {e}")
    
    def get_device_info(self):
        """获取当前设备信息"""
        info = {
            'device': str(self.device),
            'device_type': self.device.type
        }
        
        if self.device.type == 'cuda':
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['device_name'] = torch.cuda.get_device_name(0)
                info['memory_allocated'] = torch.cuda.memory_allocated() / (1024**2)  # MB
                info['memory_reserved'] = torch.cuda.memory_reserved() / (1024**2)  # MB
                info['max_memory_allocated'] = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        
        return info

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
