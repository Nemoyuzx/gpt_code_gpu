import math
import numpy as np
import torch
import os   # 新增: 用于环境变量和设备判断
import gc   # 新增: 用于显式进行垃圾回收
import resource  # 新增: 获取内存占用（Unix/macOS）
import psutil  # 可选依赖
from typing import Optional, Sequence
from datetime import datetime as dt



MAX_RANGE_FACTOR = 0.95  # 超过最大范围的比例阈值，用于忽略远距离点
#相邻测距点差异阈值
# 原为 2m，对尼龙环境中的连续墙面过于宽松；收紧到 0.5m 可更好地滤掉跨边缘
# 的离群点（这类点会给 ICP 带来偏差）。
ADJACENCY_DIFF_THRESHOLD = 0.5

ICP_MAX_ITER = int(os.environ.get("ICP_MAX_ITER", "30"))  # ICP最大迭代次数（默认30，实测收敛通常<20）
ICP_TOLERANCE = float(os.environ.get("ICP_TOLERANCE", "1e-4"))  # ICP收敛容忍，默认放宽以加速收敛
# 原为 30m（几乎不限制）。对应点距离上限过宽会让错匹配拉偏ICP，
# 收紧到 1.0m：在帧间位姿增量更安静的分辨率下这个值足够宽遗，又能有效剥离外点。
ICP_CORRESPONDENCE_THRESH = float(os.environ.get("ICP_CORR_THRESH", "1.0"))
# 逐代收紧对应阈值：初值相对较宽，随迭代指数衰减到 final，使 ICP 先粗对齐再精修。
ICP_CORR_THRESH_FINAL = float(os.environ.get("ICP_CORR_THRESH_FINAL", "0.2"))
ICP_CORR_THRESH_DECAY = float(os.environ.get("ICP_CORR_THRESH_DECAY", "0.85"))
# Trimmed ICP：每次迭代在硬阈值筛选之后，再按残差排序仅保留最好的这比例对应点。
# 1.0 表示关闭 trimming。0.8 表示丢掉残差最大的 20% 对应点。
ICP_TRIM_RATIO = float(os.environ.get("ICP_TRIM_RATIO", "0.8"))
ICP_DEBUG = os.environ.get("ICP_DEBUG", "0") == "1"  # 是否输出ICP调试信息（默认关闭以减少I/O开销）
ICP_MEM_LOG = os.environ.get("ICP_MEM_LOG", "0") == "1"  # 是否每帧打印内存使用（默认关闭）
ICP_GC_EVERY = int(os.environ.get("ICP_GC_EVERY", "0"))  # 每N帧显式gc；0表示关闭


ICP_ACCUM_TRANS_THRESHOLD = float(os.environ.get("ICP_ACCUM_TRANS", "0.001"))
ICP_ACCUM_ROT_THRESHOLD = float(os.environ.get("ICP_ACCUM_ROT", "0.00005"))
# 原为 270° / 1m，这么宽的跳变阈值几乎不会拒绝任何 ICP 解，恶 ICP
# 会静默地拉偏位姿。帧间实际修正应 << 0.3m / 30°，超出就视为失败。
ICP_MAX_ANGLE_CORRECTION = float(os.environ.get("ICP_MAX_ANGLE_CORR", str(math.radians(30.0))))
ICP_MAX_POS_CORRECTION = float(os.environ.get("ICP_MAX_POS_CORR", "0.3"))
ICP_FAIL_SKIP_FRAMES = int(os.environ.get("ICP_FAIL_SKIP", "1"))

# 激光雷达安装点相对车体中心的偏移（单位: 米），默认向后5cm，可通过环境变量覆盖
LIDAR_MOUNT_OFFSET_X = float(os.environ.get("LIDAR_MOUNT_OFFSET_X", "-0.019"))
LIDAR_MOUNT_OFFSET_Y = float(os.environ.get("LIDAR_MOUNT_OFFSET_Y", "0.0"))

# 为了限制内存：ICP匹配时目标点云的最大样本数，以及全局地图点云的上限
# 提高上限以保留更多地图信息，避免中途突然失败
MAX_TGT_POINTS_FOR_ICP = int(os.environ.get("ICP_TGT_MAX", "50000"))  # 从30000提高到50000
MAX_MAP_POINTS_GLOBAL = int(os.environ.get("MAP_POINTS_MAX", "10000000"))  # 从50000提高到100000

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
        # 激光雷达安装位置相对车体参考点的偏移（车体坐标系，x向前，y向左）
        self.lidar_mount_offset = (float(LIDAR_MOUNT_OFFSET_X), float(LIDAR_MOUNT_OFFSET_Y))

        # 保存地图中的点云（全局坐标）用于ICP匹配
        self.map_points = []  # list of [x,y] obstacle points
        self.map_points_tensor = None  # 地图点云的PyTorch张量版本
        # ICP参数
        self.icp_max_iter = ICP_MAX_ITER
        self.icp_tolerance = ICP_TOLERANCE  # 收敛容忍度
        self.icp_correspondence_thresh = ICP_CORRESPONDENCE_THRESH  # 对应点匹配距离阈值（初始值）
        self.icp_corr_thresh_final = max(0.0, ICP_CORR_THRESH_FINAL)
        self.icp_corr_thresh_decay = max(0.0, min(1.0, ICP_CORR_THRESH_DECAY))
        self.icp_trim_ratio = max(0.1, min(1.0, ICP_TRIM_RATIO))

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
            force_gpu = os.environ.get("FORCE_GPU", "0") == "1"
            # 经测量：在 2D 仿真（<~20k 地图点）下，MPS 的内核启动开销远大于计算本身，
            # CPU 通常快 5-10 倍。因此默认在未显式要求 GPU 时使用 CPU；
            # 显式 FORCE_GPU=1 或检测到 CUDA 时才使用加速器。
            if not force_cpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"[ICPSlam] 使用CUDA设备: {torch.cuda.get_device_name(0)}")
            elif force_gpu and not force_cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # 在MPS设备上，某些操作不支持，需要特殊处理
                self.device = torch.device("mps")
                self.use_mps_fallback = True
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 启用MPS降级
                print("[ICPSlam] 使用MPS设备进行加速 (Apple Metal, FORCE_GPU=1)")
            else:
                reason = (
                    " (用户强制)" if force_cpu
                    else " (小规模点云下 CPU 更快；设置 FORCE_GPU=1 可启用 MPS)"
                )
                print("[ICPSlam] 使用CPU设备" + reason)
        except Exception as e:
            # 如果设备初始化失败，回退到CPU
            self.device = torch.device("cpu")
            print(f"[ICPSlam] GPU初始化失败，回退到CPU: {str(e)}")

        self.icp_consecutive_failures = 0
        self.icp_skip_frames = 0
        self.icp_last_status = "init"
        
        # 定位专用模式：探索完成后，只进行ICP定位，不更新地图
        self.localize_only = False
            
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
        timestamp = dt.now().strftime("[%H:%M:%S.%f]")
        line = "[ICPSlam][Mem] " + " | ".join(parts)
        print(f"{timestamp} {line}")

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
        """智能削减地图点云，优先保留空间分布均匀的点"""
        if self.map_points_tensor is None:
            return
        total = int(self.map_points_tensor.shape[0])
        if total <= keep:
            return
        
        # 使用体素网格采样而非随机采样，保持地图特征
        try:
            # 将点云移到CPU进行体素化处理（避免GPU内存峰值）
            points_cpu = self.map_points_tensor.cpu().numpy()
            
            # 计算合适的体素大小
            min_coords = points_cpu.min(axis=0)
            max_coords = points_cpu.max(axis=0)
            map_span = max_coords - min_coords
            
            # 根据目标点数估算体素大小
            voxel_size = max(0.05, np.power(np.prod(map_span) / keep, 1.0/2.0))
            
            # 体素化：将点分配到网格中
            voxel_indices = ((points_cpu - min_coords) / voxel_size).astype(np.int32)
            
            # 使用字典记录每个体素中的点（保留最后一个点）
            voxel_dict = {}
            for i, voxel_idx in enumerate(voxel_indices):
                key = tuple(voxel_idx)
                voxel_dict[key] = i  # 保留每个体素的一个代表点
            
            # 提取保留的点索引
            keep_indices = list(voxel_dict.values())
            
            # 如果体素化后点数仍然过多，再进行随机采样
            if len(keep_indices) > keep:
                keep_indices = np.random.choice(keep_indices, keep, replace=False)
            
            # 更新地图点云
            self.map_points_tensor = torch.from_numpy(points_cpu[keep_indices]).to(self.device)
            
            print(f"[ICP][INFO] 地图点云从 {total} 削减到 {len(keep_indices)} (体素大小={voxel_size:.3f}m)")
            
        except Exception as e:
            # 如果体素化失败，回退到随机采样
            print(f"[ICP][WARN] 体素化失败，使用随机采样: {e}")
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
        # ICP失败时不要过度削减地图点云，保留更多信息用于恢复
        target_keep = max(int(MAX_TGT_POINTS_FOR_ICP * 0.9), 40000)  # 从0.7提高到0.9，从15000提高到40000
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

        # 步骤1: 预测位姿（对称运动模型，消除旋转+平移联合时的系统性偏差）：
        # 先转 d_rot/2，再沿中间方向平移 d_trans，最后转剩余 d_rot/2。
        self.theta += 0.5 * d_rot
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        self.x += d_trans * math.cos(self.theta)
        self.y += d_trans * math.sin(self.theta)
        self.theta += 0.5 * d_rot
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

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

        offset_body_x, offset_body_y = self.lidar_mount_offset
        max_range_val = self.get_max_range()
        far_threshold = max_range_val * MAX_RANGE_FACTOR  # 95% 最大范围阈值（建图用）
        # ICP 源点云的最大距离上限（米）。远距离激光返回横向噪声 ~ R·σθ，
        # 在 SVD 里会主导旋转估计；在转弯时尤其放大角度误差。把 ICP 的输入
        # 截到一个更近的范围（默认 4m），角度约束更干净，同时保留近处墙面
        # 为主要残差。建图阶段不受此限制，仍使用 far_threshold。
        icp_range_limit = float(os.environ.get("ICP_RANGE_LIMIT", "4.0"))
        icp_range_limit = min(icp_range_limit, far_threshold) if icp_range_limit > 0 else far_threshold
        adjacent_diff_threshold = ADJACENCY_DIFF_THRESHOLD       # 相邻点距离差阈值

        # 向量化构造 body 坐标系下的有效激光点：
        #   1. 距离在可靠范围内；
        #   2. 与前后（含环形首尾）相邻点的跳变小于阈值。
        scan_np = np.asarray(scan, dtype=np.float64)
        valid_dist = (scan_np < max_range_val) & (scan_np <= icp_range_limit)
        if valid_dist.any():
            # 相邻点（含环形）均为可靠点才参与比较
            prev_scan = np.roll(scan_np, 1)
            next_scan = np.roll(scan_np, -1)
            prev_valid = np.roll(valid_dist, 1)
            next_valid = np.roll(valid_dist, -1)
            jump_prev = prev_valid & (np.abs(scan_np - prev_scan) > adjacent_diff_threshold)
            jump_next = next_valid & (np.abs(scan_np - next_scan) > adjacent_diff_threshold)
            # 单侧或双侧跳变都剔除：这类点落在真实几何不连续处（墙角/门沿），
            # 单点位置由噪声决定，给 ICP 只会引入偏差。墙角几何将由两侧连续
            # 墙面上的稳定点隐式确定。
            keep_mask = valid_dist & ~jump_prev & ~jump_next
        else:
            keep_mask = valid_dist

        if keep_mask.any():
            kept_dist = scan_np[keep_mask]
            kept_angles = angle_offset + angles_np[keep_mask]
            bx = offset_body_x + kept_dist * np.cos(kept_angles)
            by = offset_body_y + kept_dist * np.sin(kept_angles)
            body_points_np = np.stack([bx, by], axis=1).astype(np.float32, copy=False)
        else:
            body_points_np = np.zeros((0, 2), dtype=np.float32)

        if body_points_np.shape[0] == 0:
            # 没有有效扫描点，直接返回预测位姿（无ICP校正）
            return (self.x, self.y, self.theta)

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
            # ---- 空间裁剪：ICP 只需要预测位姿附近的地图点 ----
            # 源点云最远距离 = ICP_RANGE_LIMIT，再加上对应搜索半径冗余，就是 tgt
            # 实际可能参与配对的最大距离。把 tgt 按 bbox 裁到这个范围可以把
            # cdist 的规模从"整张地图"缩到"局部窗口"，这是每帧最大的开销。
            try:
                pred_x_for_crop = float(self.x)
                pred_y_for_crop = float(self.y)
                icp_range_limit_m = float(os.environ.get("ICP_RANGE_LIMIT", "4.0"))
                if icp_range_limit_m <= 0:
                    icp_range_limit_m = self.get_max_range() * MAX_RANGE_FACTOR
                crop_radius = icp_range_limit_m + max(self.icp_correspondence_thresh, 1.0)
                tgt_x = tgt[:, 0]
                tgt_y = tgt[:, 1]
                bbox_mask = (
                    (tgt_x >= pred_x_for_crop - crop_radius)
                    & (tgt_x <= pred_x_for_crop + crop_radius)
                    & (tgt_y >= pred_y_for_crop - crop_radius)
                    & (tgt_y <= pred_y_for_crop + crop_radius)
                )
                cropped = tgt[bbox_mask]
                # 需要足够的支撑点才使用裁剪结果，否则退回全图
                if int(cropped.shape[0]) >= 50:
                    tgt = cropped
            except Exception:
                pass
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
                # 对应阈值逐代衰减：init -> final（指数衰减）
                corr_thresh_init = self.icp_correspondence_thresh
                corr_thresh_final = min(self.icp_corr_thresh_final, corr_thresh_init)
                corr_decay = self.icp_corr_thresh_decay
                trim_ratio = self.icp_trim_ratio
                for it in range(self.icp_max_iter):
                    icp_iterations = it + 1
                    # 本次迭代使用的对应阈值
                    current_corr_thresh = max(
                        corr_thresh_final,
                        corr_thresh_init * (corr_decay ** it),
                    )
                    # 计算源点集到目标点集的距离矩阵并寻找最近邻
                    dist_matrix = torch.cdist(src_world_prev, tgt)  # [N_src, N_tgt]
                    min_dists, min_indices = torch.min(dist_matrix, dim=1)
                    # 筛选出距离在阈值内的有效对应点对
                    valid_mask = min_dists < current_corr_thresh
                    valid_count = int(torch.sum(valid_mask).item())
                    # Trimmed ICP：在硬阈值之上，按残差再保留最好的 trim_ratio 部分，
                    # 消除离群对应点对 SVD 的拖拽。只在有足够对应点时启用。
                    if trim_ratio < 1.0 and valid_count >= 10:
                        kept_dists = min_dists[valid_mask]
                        k_keep = max(3, int(math.ceil(valid_count * trim_ratio)))
                        if k_keep < valid_count:
                            # 取前 k_keep 小的残差阈值
                            kth = torch.kthvalue(kept_dists, k_keep).values
                            trim_mask = min_dists <= kth
                            valid_mask = valid_mask & trim_mask
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
        # 如果设置了localize_only模式，则跳过地图更新，仅进行定位
        if not self.localize_only:
            offset_body_x, offset_body_y = self.lidar_mount_offset
            cos_theta = math.cos(self.theta)
            sin_theta = math.sin(self.theta)
            sensor_x = self.x + offset_body_x * cos_theta - offset_body_y * sin_theta
            sensor_y = self.y + offset_body_x * sin_theta + offset_body_y * cos_theta
            inv_res = 1.0 / self.resolution
            max_y_idx = self.occupancy.shape[0] - 1
            max_x_idx = self.occupancy.shape[1] - 1
            sx_idx = int((sensor_x - self.min_x) * inv_res)
            sy_idx = int((sensor_y - self.min_y) * inv_res)
            sx_idx = min(max(sx_idx, 0), max_x_idx)
            sy_idx = min(max(sy_idx, 0), max_y_idx)

            max_range_val = self.get_max_range()
            far_threshold = max_range_val * MAX_RANGE_FACTOR
            # 建图阶段使用与 ICP 一致的跳变阈值，避免在过滤宽松时把边缘离群点也画进地图
            adjacent_diff_threshold_map = ADJACENCY_DIFF_THRESHOLD

            # 向量化：相邻差分过滤 + 端点栅格坐标，一次性在 NumPy 中完成
            # 注意：这里的 scan_np/angles_np 已在步骤2中准备好
            reliable = (scan_np < max_range_val) & (scan_np <= far_threshold)
            prev_scan2 = np.roll(scan_np, 1)
            next_scan2 = np.roll(scan_np, -1)
            prev_ok = np.roll(reliable, 1)
            next_ok = np.roll(reliable, -1)
            jump_prev2 = prev_ok & (np.abs(scan_np - prev_scan2) > adjacent_diff_threshold_map)
            jump_next2 = next_ok & (np.abs(scan_np - next_scan2) > adjacent_diff_threshold_map)
            # 仅“可靠距离”要求做跳变过滤；远距离未命中射线也参与建图（标记free）
            map_mask = ~(reliable & (jump_prev2 | jump_next2))

            hit_obstacle_arr = reliable  # 击中障碍 = 在可靠距离内的测量
            # 有效投射距离：命中则为测量距离，否则截到 far_threshold
            effective_dist = np.where(hit_obstacle_arr, scan_np, far_threshold)
            beam_angles = self.theta + angle_offset + angles_np
            end_x_arr = sensor_x + effective_dist * np.cos(beam_angles)
            end_y_arr = sensor_y + effective_dist * np.sin(beam_angles)
            tx_arr = np.clip(((end_x_arr - self.min_x) * inv_res).astype(np.int32), 0, max_x_idx)
            ty_arr = np.clip(((end_y_arr - self.min_y) * inv_res).astype(np.int32), 0, max_y_idx)

            new_points = []  # 本次扫描新增的障碍点（全局坐标）
            occupancy = self.occupancy  # 本地引用加速
            raster = self._raster_line_np

            # 仅遍历通过过滤的 beam；内层用 NumPy 处理栅格序列
            idx_iter = np.nonzero(map_mask)[0]
            for i in idx_iter:
                xs, ys = raster(sx_idx, sy_idx, int(tx_arr[i]), int(ty_arr[i]))
                occ_vals = occupancy[ys, xs]
                # 找到本射线第一个"已知占据"的位置，遇到就停止清理
                hit_mask = occ_vals == 1
                if hit_mask.any():
                    first_hit = int(np.argmax(hit_mask))
                else:
                    first_hit = xs.size

                if hit_obstacle_arr[i]:
                    # 沿线将未知格清空为空闲（不含末端障碍自身）
                    end_idx = min(first_hit, xs.size - 1)
                    if end_idx > 0:
                        seg_x = xs[:end_idx]
                        seg_y = ys[:end_idx]
                        seg_vals = occ_vals[:end_idx]
                        unknown = seg_vals == -1
                        if unknown.any():
                            occupancy[seg_y[unknown], seg_x[unknown]] = 0
                    # 若未在途中撞上已知墙，才写入末端障碍
                    if first_hit >= xs.size - 1:
                        ox = int(xs[-1])
                        oy = int(ys[-1])
                        if occupancy[oy, ox] != 1:
                            occupancy[oy, ox] = 1
                            new_points.append([float(end_x_arr[i]), float(end_y_arr[i])])
                else:
                    # 未命中障碍：沿线将未知格标记空闲，遇已知占据即停
                    end_idx = first_hit
                    if end_idx > 0:
                        seg_x = xs[:end_idx]
                        seg_y = ys[:end_idx]
                        seg_vals = occ_vals[:end_idx]
                        unknown = seg_vals == -1
                        if unknown.any():
                            occupancy[seg_y[unknown], seg_x[unknown]] = 0
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

        # 显式调用垃圾回收（仅按配置间隔执行，避免每帧 gc 阻塞）
        if ICP_GC_EVERY > 0:
            self._gc_counter = getattr(self, "_gc_counter", 0) + 1
            if self._gc_counter >= ICP_GC_EVERY:
                self._gc_counter = 0
                gc.collect()
        # 内存使用量打印仅按需开启，避免频繁 I/O 拖慢主循环
        if ICP_MEM_LOG:
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

    @staticmethod
    def _raster_line_np(x0: int, y0: int, x1: int, y1: int):
        """向量化栅格线光栅化（使用线性插值 + 四舍五入）。

        与 Bresenham 在连通线段上的像素集合基本一致；对于射线投影建图
        ("遇到已知占据格即停止 / 将沿线未知格标记为空闲") 的语义而言
        等价，且通过 NumPy 一次性产出整数坐标，避免 Python 循环逐格访问。
        返回：xs, ys（np.int32 一维数组），长度 = max(|dx|,|dy|) + 1。
        """
        dx = int(x1 - x0)
        dy = int(y1 - y0)
        n = max(abs(dx), abs(dy)) + 1
        if n <= 1:
            return (
                np.array([x0], dtype=np.int32),
                np.array([y0], dtype=np.int32),
            )
        t = np.linspace(0.0, 1.0, n)
        xs = np.rint(x0 + t * dx).astype(np.int32, copy=False)
        ys = np.rint(y0 + t * dy).astype(np.int32, copy=False)
        return xs, ys
