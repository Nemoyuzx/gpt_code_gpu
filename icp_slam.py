import math
import numpy as np
import torch
import os   # 新增: 用于环境变量和设备判断
import gc   # 新增: 用于显式进行垃圾回收
import resource  # 新增: 获取内存占用（Unix/macOS）
import datetime  # 新增: 时间戳
import csv       # 新增: 写入CSV
import psutil  # 可选依赖
from typing import Optional, Sequence



#0.49
#超过最大范围比例
MAX_RANGE_FACTOR = 0.9  # 超过最大范围的比例阈值，用于忽略远距离点
#相邻测距点差异阈值
ADJACENCY_DIFF_THRESHOLD = 1  # 相邻测距点之间的差异阈值 (米)

ICP_MAX_ITER = int(os.environ.get("ICP_MAX_ITER", "200"))  # ICP最大迭代次数，可通过环境变量调整
ICP_TOLERANCE = float(os.environ.get("ICP_TOLERANCE", "1e-3"))  # ICP收敛容忍，默认放宽以加速收敛
ICP_CORRESPONDENCE_THRESH = float(os.environ.get("ICP_CORR_THRESH", "12"))  # ICP对应点匹配距离上限 (米)
ICP_DEBUG = os.environ.get("ICP_DEBUG", "0") == "1"  # 是否输出ICP调试信息

ICP_ACCUM_TRANS_THRESHOLD = float(os.environ.get("ICP_ACCUM_TRANS", "0.02"))
ICP_ACCUM_ROT_THRESHOLD = float(os.environ.get("ICP_ACCUM_ROT", "0.05"))
ICP_MAX_ANGLE_CORRECTION = float(os.environ.get("ICP_MAX_ANGLE_CORR", str(math.radians(95.0))))
ICP_MAX_POS_CORRECTION = float(os.environ.get("ICP_MAX_POS_CORR", "0.6"))
ICP_FAIL_SKIP_FRAMES = int(os.environ.get("ICP_FAIL_SKIP", "3"))

# 为了限制内存：ICP匹配时目标点云的最大样本数W，以及全局地图点云的上限
MAX_TGT_POINTS_FOR_ICP = int(os.environ.get("ICP_TGT_MAX", "30000"))
MAX_MAP_POINTS_GLOBAL = int(os.environ.get("MAP_POINTS_MAX", "50000"))

class ICPSlam:
    """ICP SLAM建图与定位模块。利用激光数据和运动模型进行SLAM。支持GPU加速。"""

    def __init__(self, maze, init_pose, *, laser_angle_offset_deg: float = 0.0):
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
        # 激光雷达安装角度偏移（车体坐标系下的零度方向修正）
        self.laser_angle_offset = math.radians(float(laser_angle_offset_deg))

        # 保存地图中的点云（全局坐标）用于ICP匹配
        self.map_points = []  # list of [x,y] obstacle points
        self.map_points_tensor = None  # 地图点云的PyTorch张量版本
        # ICP参数
        self.icp_max_iter = ICP_MAX_ITER
        self.icp_tolerance = ICP_TOLERANCE  # 收敛容忍度
        self.icp_correspondence_thresh = ICP_CORRESPONDENCE_THRESH  # 对应点匹配距离阈值

        self.icp_accum_trans_threshold = max(0.0, ICP_ACCUM_TRANS_THRESHOLD)
        self.icp_accum_rot_threshold = max(0.0, ICP_ACCUM_ROT_THRESHOLD)
        self._motion_accum_trans = 0.0
        self._motion_accum_rot = 0.0
        self.icp_max_angle_correction = max(0.0, ICP_MAX_ANGLE_CORRECTION)
        self.icp_max_pos_correction = max(0.0, ICP_MAX_POS_CORRECTION)
        self.icp_fail_skip_frames = max(0, ICP_FAIL_SKIP_FRAMES)
        
        # 设备检测与选择
        self.device = torch.device("cpu")  # 默认使用CPU
        self.use_mps_fallback = False      # 标记是否MPS需要特殊处理
        
        try:
            # 检查是否有环境变量设置强制使用CPU
            
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

        self.icp_consecutive_failures = 0
        self.icp_skip_frames = 0
        self.icp_last_status = "init"
            
    def _print_memory_usage(self, icp_iterations=None):
        """打印当前进程与设备的内存占用信息，可选附带本轮ICP迭代次数。"""
        # 进程常驻内存（RSS）—注意：ru_maxrss 是“峰值RSS”(high-water mark)
        try:
            # macOS 上 resource.ru_maxrss 单位为字节，Linux 为 KB；这里做两种情况的兼容
            usage = resource.getrusage(resource.RUSAGE_SELF)
            ru_maxrss = usage.ru_maxrss
            # 粗略判断：若值很大且不太可能是 KB，则按字节处理，否则按 KB 处理
            if ru_maxrss > 1e9:  # 明显是字节
                rss_mb = ru_maxrss / (1024.0 * 1024.0)
            else:  # 可能是 KB
                rss_mb = ru_maxrss / 1024.0
        except Exception:
            rss_mb = float('nan')

        # 可选：当前RSS（需要 psutil，若不可用则忽略）
        rss_cur_mb = None
        try:
            
            proc = psutil.Process(os.getpid())
            rss_cur_mb = proc.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            pass

        # 设备显存（CUDA/MPS）
        cuda_alloc_mb = cuda_reserved_mb = None
        mps_current_mb = mps_driver_mb = None
        try:
            if self.device.type == 'cuda' and torch.cuda.is_available():
                cuda_alloc_mb = torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0)
                cuda_reserved_mb = torch.cuda.memory_reserved(self.device) / (1024.0 * 1024.0)
        except Exception:
            pass
        try:
            # 仅在支持的 PyTorch 版本有效
            if self.device.type == 'mps' and hasattr(torch, 'mps'):
                if hasattr(torch.mps, 'current_allocated_memory'):
                    mps_current_mb = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
                if hasattr(torch.mps, 'driver_allocated_memory'):
                    mps_driver_mb = torch.mps.driver_allocated_memory() / (1024.0 * 1024.0)
        except Exception:
            pass

        # 地图相关内存估计
        map_pts = 0
        map_pts_mb = 0.0
        if self.map_points_tensor is not None:
            try:
                map_pts = int(self.map_points_tensor.shape[0])
                map_pts_mb = float(self.map_points_tensor.element_size() * self.map_points_tensor.nelement()) / (1024.0 * 1024.0)
            except Exception:
                pass
        occ_mb = 0.0
        try:
            if isinstance(self.occupancy, np.ndarray):
                occ_mb = self.occupancy.nbytes / (1024.0 * 1024.0)
        except Exception:
            pass

        if rss_cur_mb is not None:
            parts = [f"RSS(cur/peak): {rss_cur_mb:.1f}/{rss_mb:.1f} MB"]
        else:
            parts = [f"RSS_peak: {rss_mb:.1f} MB"]
        if cuda_alloc_mb is not None:
            parts.append(f"CUDA alloc/resv: {cuda_alloc_mb:.1f}/{cuda_reserved_mb:.1f} MB")
        if mps_current_mb is not None:
            if mps_driver_mb is not None:
                parts.append(f"MPS curr/driver: {mps_current_mb:.1f}/{mps_driver_mb:.1f} MB")
            else:
                parts.append(f"MPS curr: {mps_current_mb:.1f} MB")
        parts.append(f"map_points: {map_pts} (~{map_pts_mb:.1f} MB)")
        parts.append(f"occupancy: ~{occ_mb:.1f} MB")
        if icp_iterations is not None:
            parts.append(f"ICP iters: {int(icp_iterations)}")
        line = "[ICPSlam][Mem] " + " | ".join(parts)
        print(line)

        # 追加写入CSV（可通过环境变量 MEMLOG_CSV 指定路径）
        try:
            csv_path = os.environ.get("MEMLOG_CSV", "mem_usage_log.csv")
            # 准备行数据
            ts = datetime.datetime.now().isoformat(timespec='seconds')
            row = {
                'timestamp': ts,
                'rss_cur_mb': round(rss_cur_mb, 3) if rss_cur_mb is not None else None,
                'rss_peak_mb': round(rss_mb, 3) if rss_mb is not None else None,
                'cuda_alloc_mb': round(cuda_alloc_mb, 3) if cuda_alloc_mb is not None else None,
                'cuda_reserved_mb': round(cuda_reserved_mb, 3) if cuda_reserved_mb is not None else None,
                'mps_current_mb': round(mps_current_mb, 3) if mps_current_mb is not None else None,
                'mps_driver_mb': round(mps_driver_mb, 3) if mps_driver_mb is not None else None,
                'map_points': map_pts,
                'map_points_mb': round(map_pts_mb, 6),
                'occupancy_mb': round(occ_mb, 6),
                'icp_iterations': int(icp_iterations) if icp_iterations is not None else None,
            }
            # 若文件不存在或为空，写入表头
            need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if need_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            # 日志写入失败不影响主流程
            pass

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

    def _shrink_map_points(self, keep: int) -> None:
        if self.map_points_tensor is None:
            return
        total = int(self.map_points_tensor.shape[0])
        if total <= keep:
            return
        try:
            idx = torch.randperm(total, device=self.device)[:keep]
        except Exception:
            idx_cpu = torch.randperm(total)[:keep]
            try:
                idx = idx_cpu.to(self.device)
            except Exception:
                idx = idx_cpu
                self.map_points_tensor = self.map_points_tensor.cpu()
        self.map_points_tensor = self.map_points_tensor.index_select(0, idx)
        if self.map_points_tensor.device != self.device:
            self.map_points_tensor = self.map_points_tensor.to(self.device)

    def _handle_icp_failure(
        self,
        *,
        reason: str,
        valid_pairs: int,
        last_error: Optional[float],
        last_diff: Optional[float],
    ) -> None:
        self.icp_consecutive_failures += 1
        detail_parts = []
        if last_error is not None and math.isfinite(last_error):
            detail_parts.append(f"error={last_error:.4f}")
        if last_diff is not None and math.isfinite(last_diff):
            detail_parts.append(f"delta={last_diff:.4f}")
        detail_parts.append(f"pairs={valid_pairs}")
        print(
            f"[ICP][WARN] {reason} ({', '.join(detail_parts)}) -> consecutive_failures={self.icp_consecutive_failures}"
        )
        self.icp_last_status = "fail"
        target_keep = max(int(MAX_TGT_POINTS_FOR_ICP * 0.7), 15000)
        self._shrink_map_points(target_keep)
        if self.icp_consecutive_failures >= 3:
            self.icp_skip_frames = max(self.icp_skip_frames, 2)
            self.icp_last_status = "skipping"

    def _handle_icp_success(self) -> None:
        if self.icp_consecutive_failures > 0:
            print(f"[ICP] recovered after {self.icp_consecutive_failures} failure(s)")
        self.icp_consecutive_failures = 0
        self.icp_skip_frames = 0
        self.icp_last_status = "ok"

    def update(self, odom_delta, scan, angles_deg: Optional[Sequence[float]] = None):
        """
        使用新的里程计增量 odom_delta (tuple: (delta_distance, delta_theta))
        和激光扫描数据 scan (距离列表) 更新 SLAM 估计和地图。
        返回更新后的位姿估计 (x, y, theta)。
        """
        d_trans, d_rot = odom_delta
        
        # 运动阈值：当运动量小于此值时，认为机器人静止，跳过ICP以避免噪声导致的漂移
        MOTION_THRESHOLD_TRANS = float(os.environ.get("ICP_MIN_TRANS", "0.005"))  # 5mm
        MOTION_THRESHOLD_ROT = float(os.environ.get("ICP_MIN_ROT", "0.01"))      # ~0.57度
        
        # 检查是否实际有运动
        is_moving = (abs(d_trans) >= MOTION_THRESHOLD_TRANS or abs(d_rot) >= MOTION_THRESHOLD_ROT)

        # 步骤1: 预测位姿 (根据里程计增量更新估计位姿)
        self.theta += d_rot
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        # 假设 d_trans 沿当前朝向方向
        self.x += d_trans * math.cos(self.theta)
        self.y += d_trans * math.sin(self.theta)

        pred_x, pred_y, pred_theta = self.x, self.y, self.theta

        if is_moving:
            self._motion_accum_trans += abs(d_trans)
            self._motion_accum_rot += abs(d_rot)
        else:
            # 静止时缓慢衰减累计量，避免长时间停留后立刻触发ICP
            self._motion_accum_trans *= 0.5
            self._motion_accum_rot *= 0.5

        accum_ready = True
        if self.icp_accum_trans_threshold > 0.0 or self.icp_accum_rot_threshold > 0.0:
            accum_ready = False
            if self.icp_accum_trans_threshold > 0.0 and self._motion_accum_trans >= self.icp_accum_trans_threshold:
                accum_ready = True
            if self.icp_accum_rot_threshold > 0.0 and self._motion_accum_rot >= self.icp_accum_rot_threshold:
                accum_ready = True

        should_attempt_icp = is_moving and accum_ready

        # 步骤2: ICP 匹配校正 (仅在有实际运动时使用 scan 与已有地图点云匹配修正位姿)
        # 将激光扫描点转换为全局坐标（基于预测位姿）
        if angles_deg is not None and len(angles_deg) == len(scan):
            angles_array = np.array(angles_deg, dtype=float)
            finite_mask = np.isfinite(angles_array)
            if np.any(finite_mask):
                angles_array = np.mod(angles_array, 360.0)
                if not np.all(finite_mask):
                    fallback = np.linspace(0, 360, len(scan), endpoint=False)
                    angles_array = np.where(finite_mask, angles_array, fallback)
                angles_np = np.radians(angles_array)
            else:
                angles_np = np.radians(np.linspace(0, 360, len(scan), endpoint=False))
        else:
            angles_np = np.radians(np.linspace(0, 360, len(scan), endpoint=False))

        angle_offset = self.laser_angle_offset

        body_points = []
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
            # 计算该激光点在机器人坐标系下的坐标
            local_angle = angle_offset + angles_np[i]
            local_x = dist * math.cos(local_angle)
            local_y = dist * math.sin(local_angle)
            body_points.append([local_x, local_y])

        if not body_points:
            # 没有有效扫描点，直接返回预测位姿（无ICP校正）
            return (self.x, self.y, self.theta)
        body_points_np = np.array(body_points, dtype=np.float32)

        # 如果存在已有地图点云且机器人在运动，则进行 ICP 匹配校正
        # 静止时跳过ICP以避免传感器噪声导致的位姿漂移
        icp_iterations = 0
        icp_valid_pairs = 0
        icp_last_diff: Optional[float] = None
        icp_last_error: Optional[float] = None
        icp_converged = False
        icp_performed = False
        icp_pose_valid = False
        icp_failure_reason: Optional[str] = None
        skip_icp = False
        if should_attempt_icp and self.icp_skip_frames > 0:
            remaining = self.icp_skip_frames
            self.icp_skip_frames = max(0, self.icp_skip_frames - 1)
            skip_icp = True
            if self.icp_last_status != "skipping":
                print(f"[ICP][INFO] skipping ICP for recovery ({remaining} frame(s) left)")
            self.icp_last_status = "skipping"

        if (
            should_attempt_icp
            and not skip_icp
            and self.map_points_tensor is not None
            and self.map_points_tensor.numel() > 0
            and body_points_np.size > 0
        ):
            icp_performed = True
            # 开始 ICP 前，将地图点云加载至 GPU/CPU 张量
            src_body = self.to_tensor(body_points_np)  # 源点云（机器人坐标系，形状 [N_src, 2]）
            # 准备目标点云张量 (地图点)
            tgt = self.map_points_tensor  # 仅使用内存中的最新地图点云
            # 若目标点云过大，则随机子采样到上限，限制 cdist 峰值内存
            try:
                if tgt.shape[0] > MAX_TGT_POINTS_FOR_ICP:
                    idx = torch.randperm(tgt.shape[0], device=self.device)[:MAX_TGT_POINTS_FOR_ICP]
                    tgt = tgt.index_select(0, idx)
            except Exception:
                pass

            # 预测位姿变换（作为初始猜测）
            cos_theta = math.cos(self.theta)
            sin_theta = math.sin(self.theta)
            R_total = torch.tensor(
                [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
                dtype=torch.float32,
                device=self.device,
            )
            t_total = torch.tensor([self.x, self.y], dtype=torch.float32, device=self.device)

            # 在 no_grad 环境下进行 ICP 迭代，以减少显存开销
            with torch.no_grad():
                R = R_total
                t = t_total
                src_world_prev = (R @ src_body.T).T + t
                prev_error = float('inf')  # 追踪误差变化
                stagnation_count = 0  # 连续停滞计数
                # ICP 迭代过程
                for it in range(self.icp_max_iter):
                    icp_iterations = it + 1
                    # 计算源点集到目标点集的距离矩阵并寻找最近邻
                    dist_matrix = torch.cdist(src_world_prev, tgt)  # [N_src, N_tgt]
                    min_dists, min_indices = torch.min(dist_matrix, dim=1)
                    # 筛选出距离在阈值内的有效对应点对
                    valid_mask = min_dists < self.icp_correspondence_thresh
                    valid_count = int(torch.sum(valid_mask).item())
                    current_error = float(torch.mean(min_dists[valid_mask]).item()) if valid_count > 0 else float('inf')
                    icp_valid_pairs = valid_count
                    if math.isfinite(current_error):
                        icp_last_error = current_error
                    
                    paired_src_body = src_body[valid_mask]
                    paired_src_world = src_world_prev[valid_mask]
                    paired_tgt = tgt[min_indices[valid_mask]]
                    # 计算配对点集的质心
                    src_center_body = torch.mean(paired_src_body, dim=0)
                    tgt_center = torch.mean(paired_tgt, dim=0)
                    # 去中心化点集坐标
                    src_centered = paired_src_body - src_center_body
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
                        R_delta = R_cpu_tensor.to(self.device)
                    else:
                        U, _, Vt = torch.linalg.svd(H)
                        R_delta = Vt.T @ U.T
                        # 若产生反射，调整最后一行符号确保合法旋转
                        if self.compute_determinant(R_delta) < 0:
                            Vt_fixed = Vt.clone()
                            Vt_fixed[-1, :] *= -1
                            R_delta = Vt_fixed.T @ U.T
                    # 计算平移向量 t
                    t_delta = tgt_center - R_delta @ src_center_body
                    # 将新变换应用于源点集合并计算最大位移差
                    src_world_new = (R_delta @ src_body.T).T + t_delta
                    diff = torch.max(torch.norm(src_world_new - src_world_prev, dim=1))
                    icp_last_diff = float(diff)
                    # 更新当前估计，为下次迭代使用
                    R = R_delta
                    t = t_delta
                    src_world_prev = src_world_new
                    
                    # 检测停滞（误差不再显著下降）
                    error_reduction = prev_error - current_error
                    if abs(error_reduction) < 1e-6:  # 误差几乎不变
                        stagnation_count += 1
                        if stagnation_count >= 10:  # 连续10次停滞
                            if ICP_DEBUG:
                                print(f"[ICP][debug] iter={it+1} stagnation detected, early exit (error={current_error:.6f})")
                            break
                    else:
                        stagnation_count = 0
                    prev_error = current_error
                    
                    if diff < self.icp_tolerance:
                        # 收敛判定：变化量小于阈值，结束迭代
                        if ICP_DEBUG:
                            print(f"[ICP][debug] iter={it+1} converged (diff={float(diff):.6f} < tol={self.icp_tolerance})")
                        icp_converged = True
                        break
                    
                    # 达到最大迭代次数的警告
                    if it + 1 == self.icp_max_iter:
                        print(f"[ICP][WARN] reached MAX_ITER={self.icp_max_iter}, diff={float(diff):.6f}, error={current_error:.6f}, valid_corr={valid_count}/{len(src_body)} (tgt_pts={len(tgt)})")
                        break
                        
                # 提取最终结果到 CPU（numpy）用于更新机器人位姿
                R_cpu = R.cpu().numpy()
                t_cpu = t.cpu().numpy()
                # 显式删除大张量释放显存
                for name in [
                    'dist_matrix', 'min_dists', 'min_indices', 'valid_mask',
                    'paired_src_body', 'paired_src_world', 'paired_tgt',
                    'src_center_body', 'tgt_center', 'src_centered', 'tgt_centered',
                    'src_world_new', 'src_world_prev', 'src_body', 'tgt',
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
            # 利用 ICP 结果修正机器人状态（若校验通过）
            theta_candidate = math.atan2(R_cpu[1, 0], R_cpu[0, 0])
            delta_theta = abs(math.atan2(math.sin(theta_candidate - pred_theta), math.cos(theta_candidate - pred_theta)))
            delta_pos = math.hypot(float(t_cpu[0]) - pred_x, float(t_cpu[1]) - pred_y)
            pose_jump = False
            if self.icp_max_angle_correction > 0.0 and delta_theta > self.icp_max_angle_correction:
                pose_jump = True
            if self.icp_max_pos_correction > 0.0 and delta_pos > self.icp_max_pos_correction:
                pose_jump = True

            if pose_jump:
                icp_pose_valid = False
                icp_failure_reason = (
                    f"ICP姿态跳变 Δθ={math.degrees(delta_theta):.1f}° Δs={delta_pos:.3f}m"
                )
                if ICP_DEBUG:
                    print(f"[ICP][debug] rejected pose update: {icp_failure_reason}")
                if self.icp_fail_skip_frames > 0:
                    self.icp_skip_frames = max(self.icp_skip_frames, self.icp_fail_skip_frames)
            else:
                icp_pose_valid = True
                self.theta = theta_candidate
                self.x = float(t_cpu[0])
                self.y = float(t_cpu[1])
            # 使用修正后的位姿更新当前激光点的全局坐标（numpy 计算）
        else:
            # 若未进入 ICP（地图为空、静止或无运动），仅根据里程计预测位姿进行建图
            # 这种情况下位姿已经在步骤1更新，无需额外处理
            if ICP_DEBUG and not is_moving:
                print(f"[ICP][debug] 机器人静止 (d_trans={d_trans:.6f}, d_rot={d_rot:.6f})，跳过ICP以避免噪声漂移")

        if icp_performed:
            if icp_converged and icp_pose_valid:
                self._handle_icp_success()
            else:
                reason = icp_failure_reason or "ICP未收敛"
                if icp_failure_reason is None:
                    if icp_iterations >= self.icp_max_iter:
                        reason = "ICP达到最大迭代次数"
                    elif icp_valid_pairs < 3:
                        reason = "ICP有效对应点不足"
                    elif icp_last_diff is not None:
                        reason = "ICP迭代停滞"
                self._handle_icp_failure(
                    reason=reason,
                    valid_pairs=icp_valid_pairs,
                    last_error=icp_last_error,
                    last_diff=icp_last_diff,
                )
            self._motion_accum_trans = 0.0
            self._motion_accum_rot = 0.0
        elif is_moving and not skip_icp and self.icp_consecutive_failures > 0:
            self._handle_icp_success()

        # 步骤3: 更新占据栅格地图和地图点云列表（将新扫描结果整合进地图，内存中仅保留一份最新地图）
        rx = int((self.x - self.min_x) / self.resolution)
        ry = int((self.y - self.min_y) / self.resolution)
        far_threshold = self.get_max_range() * MAX_RANGE_FACTOR
        adjacent_diff_threshold = 2.0  # 与ICP部分一致或更宽松的阈值
        new_points = []  # 本次扫描新增的障碍点（全局坐标）
        max_range = self.get_max_range()
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

            if should_skip_map_update:
                # 跳过不可靠的测距，避免误清理遮挡后区域
                continue


            # 计算该激光束末端的全局坐标 (end_x, end_y)
            if 'angles_np' in locals() and i < len(angles_np):
                beam_angle = self.theta + angle_offset + angles_np[i]
            else:
                beam_angle = self.theta + angle_offset + math.radians(i)
            beam_angle = math.atan2(math.sin(beam_angle), math.cos(beam_angle))  # 归一化角度
            hit_obstacle = dist < max_range and dist <= far_threshold and not math.isinf(dist)
            effective_dist = dist if hit_obstacle else min(far_threshold, dist if dist < float('inf') else far_threshold)
            end_x = self.x + effective_dist * math.cos(beam_angle)
            end_y = self.y + effective_dist * math.sin(beam_angle)
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
            if hit_obstacle:
                # 射线击中了障碍物（在范围内且未被过滤）
                # 将路径上除最后一点外的格子标记为空闲
                for cx, cy in line[:-1]:
                    if self.occupancy[cy, cx] == 1:
                        # 遇到已知障碍，停止向前清空，避免噪声导致墙体被抹除
                        break
                    if self.occupancy[cy, cx] == -1:
                        self.occupancy[cy, cx] = 0
                # 最后一个格子是障碍物
                ox, oy = line[-1]
                # 若该障碍格此前未标记过，则标记占据并记录点
                if self.occupancy[oy, ox] != 1:
                    self.occupancy[oy, ox] = 1
                    new_point = [end_x, end_y]
                    new_points.append(new_point)
            else:
                # 未命中障碍：仅在可靠距离内将未知标记为空闲，遇到已知障碍立即停止
                for cx, cy in line:
                    if self.occupancy[cy, cx] == 1:
                        break
                    if self.occupancy[cy, cx] == -1:
                        self.occupancy[cy, cx] = 0
        # 将本次新增点与内存中的最新地图合并，仅保留一份张量
        if len(new_points) > 0:
            new_pts_np = np.array(new_points, dtype=np.float32)
            new_pts_tensor = self.to_tensor(new_pts_np)
            if self.map_points_tensor is not None and self.map_points_tensor.numel() > 0:
                self.map_points_tensor = torch.cat([self.map_points_tensor, new_pts_tensor], dim=0)
            else:
                self.map_points_tensor = new_pts_tensor
            # 释放中间变量
            del new_pts_np, new_pts_tensor
            # 控制全局地图点云上限，避免无限增长导致内存持续上升
            try:
                if self.map_points_tensor.shape[0] > MAX_MAP_POINTS_GLOBAL:
                    perm = torch.randperm(self.map_points_tensor.shape[0], device=self.device)[:MAX_MAP_POINTS_GLOBAL]
                    self.map_points_tensor = self.map_points_tensor.index_select(0, perm)
            except Exception:
                pass
        # 清理临时列表，尽快释放内存
        new_points.clear()

        # 显式调用垃圾回收，释放Python对象占用的内存
        gc.collect()
        # 每次更新后打印内存使用量
        self._print_memory_usage(icp_iterations)
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
