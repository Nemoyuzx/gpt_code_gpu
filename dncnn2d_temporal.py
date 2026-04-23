# dncnn2d_temporal.py
import math
import random
from collections import deque
from typing import Callable, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

# ---------------- 设备选择工具 (支持 macOS MPS) ----------------
def select_device(explicit: Optional[str] = None) -> torch.device:
    """优先级: 显式 > CUDA > MPS > CPU"""
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _DummyScaler:
    """在非 CUDA 场景下替代 GradScaler，接口对齐。"""
    def scale(self, loss):
        return loss
    def unscale_(self, opt):
        return None
    def step(self, opt):
        opt.step()
    def update(self):
        return None


# ---------------- 2D-CBAM ----------------
class ChannelAttention2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SELU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = torch.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM2D(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention2D(channels, reduction)
        self.sa = SpatialAttention2D(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


# -------------- 残差块(Conv2d+BN+SELU) --------------
class ResBlock2D(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        p = (k - 1) // 2
        self.conv1 = nn.Conv2d(channels, channels, k, padding=p, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SELU(inplace=True)

    def forward(self, x):
        idt = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + idt)
        return out




# -------------- DnCNN+ (2D, 时间×角度) --------------
class DnCNN2DPlus(nn.Module):
    """DnCNN+ 2D (时间×角度) 去噪网络

    输入: noisy_norm [B,1,T,N]  (已 /max_range 归一化)
    输出: (clean_last, r_last)
        clean_last: [B,1,N]  最后一帧的去噪距离 (归一化)
        r_last:     [B,1,N]  对最后一帧预测的残差 (noisy_last - clean_last)

    可配置:
      base_channels: 主通道数
      num_blocks:    残差块数量
      use_cbam_at:   在第 use_cbam_at 个残差块后插入 CBAM (1-based), 0/None 关闭
    """
    def __init__(self, base_channels=64, num_blocks=12, use_cbam_at: int | None = 6, in_channels: int = 1):
        super().__init__()
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.use_cbam_at = use_cbam_at
        self.in_channels = in_channels
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.SELU(inplace=True),
        )
        body = []
        for i in range(num_blocks):
            body.append(ResBlock2D(base_channels))
            if use_cbam_at and (i + 1) == use_cbam_at:
                body.append(CBAM2D(base_channels, reduction=16, spatial_kernel=7))
        self.body = nn.Sequential(*body)
        self.tail = nn.Conv2d(base_channels, 1, 3, padding=1, bias=False)

    def forward(self, noisy_norm):  # noisy_norm: [B,C,T,N] (C=1 or 2)
        # 如果是条件噪声模型 (C=2)，第一通道是距离，第二通道是噪声条件，残差基于第一通道计算
        if noisy_norm.dim() != 4:
            raise ValueError(f"expected 4D input [B,C,T,N], got {noisy_norm.shape}")
        if noisy_norm.size(1) > 1:
            dist_ch = noisy_norm[:, :1, :, :]
        else:
            dist_ch = noisy_norm
        feat = self.head(noisy_norm)
        feat = self.body(feat)
        r_hat = self.tail(feat)               # [B,1,T,N]
        r_last = r_hat[:, :, -1, :]           # [B,1,N]
        noisy_last = dist_ch[:, :, -1, :]     # [B,1,N]
        clean_last = noisy_last - r_last
        return clean_last, r_last


    # FiLM 版本及相关实验代码已移除，保留单一的通道拼接条件模型实现。


# -------------- 时空版滤波器（在线推理） --------------
class TemporalDnCNN2DFilter:
    """
    在线滤波：
    - 维护一个长度为 T 的队列（最近 T 帧）
    - 每次传入 1×N 的距离向量，凑满 T 帧后输出当前帧的去噪结果
    """
    def __init__(self, T, max_range, model_path=None, scan_length=None, device=None,
                 base_channels: int = 64, num_blocks: int = 12, use_cbam_at: int | None = 6,
                 condition_noise: bool = False, noise_norm_divisor: float | None = None):
        self.T = int(T)
        self.max_range = float(max_range)
        self.scan_length = scan_length
        self.buf = deque(maxlen=self.T)
        self.device = select_device(device)
        self.condition_noise = condition_noise
        self.noise_norm_divisor = noise_norm_divisor if noise_norm_divisor else max_range  # 默认用量程归一
        in_ch = 2 if condition_noise else 1
        self.net = DnCNN2DPlus(base_channels=base_channels, num_blocks=num_blocks, use_cbam_at=use_cbam_at, in_channels=in_ch).to(self.device).eval()
        if model_path is not None:
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))

    @torch.no_grad()
    def filter_lidar_data(self, distances_list: List[float], noise_sigma: float | None = None) -> List[float]:
        arr = np.asarray(distances_list, dtype=np.float32)
        if self.scan_length is not None and len(arr) != self.scan_length:
            raise ValueError(f"scan length {len(arr)} != expected {self.scan_length}")
        arr = np.clip(arr, 0.0, self.max_range)
        self.buf.append(arr)

        # 缓冲未满：直接返回原始帧（或可返回 None 由上层处理）
        if len(self.buf) < self.T:
            lst = arr.tolist()  # type: ignore[assignment]
            # 明确断言为 List[float]
            return [float(v) for v in lst]  # type: ignore[list-item]

        # 组装 [1,1,T,N]
        stack = np.stack(list(self.buf), axis=0)  # [T,N]
        stack_norm = (stack / self.max_range)[None, None, :, :]  # [1,1,T,N]
        if self.condition_noise:
            ns = float(noise_sigma if noise_sigma is not None else 0.0)
            ns_norm = ns / self.noise_norm_divisor
            noise_ch = np.full_like(stack_norm, ns_norm, dtype=np.float32)
            inp = np.concatenate([stack_norm, noise_ch], axis=1)
        else:
            inp = stack_norm
        x = torch.from_numpy(inp).to(self.device)
        clean_last, _ = self.net(x)               # [1,1,N]
        clean = (clean_last.squeeze(0).squeeze(0).cpu().numpy()) * self.max_range
        clean = np.clip(clean, 0.0, self.max_range)
        return clean.tolist()


# ----------------- 数据集/批次工厂（合成序列） -----------------
def make_seq_batch_func_factory(sim, T: int, step_xy_std=0.05, step_theta_std=0.03):
    """
    基于你的模拟器 sim 生成序列数据:
    - T 帧连续位姿: 起点随机，然后做小步随机游走（更贴近真实）
    - 返回: noisy_seq_list, clean_seq_list
      其中每个元素是 shape [T,N] 的 np.ndarray
    """
    xs = [p[0] for seg in sim.walls for p in seg]
    ys = [p[1] for seg in sim.walls for p in seg]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    def sample_start_pose():
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        theta = random.uniform(-math.pi, math.pi)
        return [x, y, theta]

    num_beams = int(360 / sim.angle_resolution)

    def make_batch(batch_size: int):
        noisy_list, clean_list = [], []
        for _ in range(batch_size):
            pose = sample_start_pose()
            noisy_seq = np.zeros((T, num_beams), dtype=np.float32)
            clean_seq = np.zeros((T, num_beams), dtype=np.float32)
            for t in range(T):
                # 连续小运动（可按需要改成你的运动模型）
                if t > 0:
                    pose[0] += np.random.normal(0, step_xy_std)
                    pose[1] += np.random.normal(0, step_xy_std)
                    pose[2] += np.random.normal(0, step_theta_std)
                    # 角度规范化
                    pose[2] = math.atan2(math.sin(pose[2]), math.cos(pose[2]))
                noisy, clean = sim.scan_with_clean(tuple(pose))
                noisy_seq[t, :] = np.asarray(noisy, dtype=np.float32)
                clean_seq[t, :] = np.asarray(clean, dtype=np.float32)
            noisy_list.append(noisy_seq)
            clean_list.append(clean_seq)
        return noisy_list, clean_list

    return make_batch


class LidarSeqDataset(torch.utils.data.Dataset):
    """时序 LiDAR 序列数据集

    每个样本:
        noisy_seq [T,N], clean_seq [T,N] (均为浮点米值)
    生成张量 (归一化):
        noisy_norm:   [1,T,N]
        residual_last:[1,N]   (noisy_last - clean_last)
        valid_mask:   [1,N]   (clean_last < max_range 判定可学习区域)
        clean_last:   [1,N]   (可选, 直接监督去噪输出)
    """
    def __init__(self, noisy_seqs: List[np.ndarray], clean_seqs: List[np.ndarray], max_range: float):
        assert len(noisy_seqs) == len(clean_seqs)
        self.noisy = noisy_seqs
        self.clean = clean_seqs
        self.max_range = float(max_range)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = np.clip(self.noisy[idx].astype(np.float32), 0.0, self.max_range)   # [T,N]
        clean = np.clip(self.clean[idx].astype(np.float32), 0.0, self.max_range)   # [T,N]

        noisy /= self.max_range
        clean /= self.max_range

        residual_last = noisy[-1, :] - clean[-1, :]           # [N]
        valid_mask = (clean[-1, :] < 0.999).astype(np.float32)

        sample = {
            'noisy': noisy[None, :, :],            # [1,T,N]
            'residual_last': residual_last[None, :],# [1,N]
            'valid': valid_mask[None, :],           # [1,N]
            'clean_last': clean[-1:, :]             # [1,N]
        }
        return {k: torch.from_numpy(v) for k, v in sample.items()}


# ----------------- 训练循环 -----------------
def train_temporal_dncnn2d(model_save_path: str,
                           max_range: float,
                           make_seq_batch_func: Callable[[int], Tuple[List[np.ndarray], List[np.ndarray]] | Tuple[List[np.ndarray], List[np.ndarray], float]],
                           steps=20000,
                           batch_size=32,
                           lr=1e-3,
                           device=None,
                           base_channels: int = 64,
                           num_blocks: int = 12,
                           use_cbam_at: int | None = 6,
                           cond_noise: bool = False,
                           noise_norm_divisor: float | None = None,
                           loss_type: str = 'mse',
                           hybrid_alpha: float = 0.5,
                           distance_loss_weight: float = 0.0,
                           warmup_steps: int = 0,
                           scheduler: str | None = 'cosine',
                           grad_clip: float = 1.0,
                           ema_decay: float | None = None,
                           save_ema: bool = True,
                           val_noise: float | None = None,
                           val_size: int = 0,
                           eval_interval: int = 500,
                           save_best: bool = True,
                           best_suffix: str = '.best.pt',
                           sigma_weight_power: float = 0.0,
                           resume_path: str | None = None):
    """可配置训练循环 (向后兼容默认参数)。

    distance_loss_weight>0 时，会在 residual 监督外，对 clean_last 做额外 MSE 约束。
    """
    

    dev = select_device(device)
    in_ch = 2 if cond_noise else 1
    net = DnCNN2DPlus(base_channels=base_channels, num_blocks=num_blocks, use_cbam_at=use_cbam_at, in_channels=in_ch).to(dev)
    if resume_path:
        try:
            state = torch.load(resume_path, map_location=dev)
            net.load_state_dict(state, strict=True)
            print(f"[resume] loaded weights from {resume_path}")
        except Exception as e:
            print(f"[resume] failed to load {resume_path}: {e}")
    # EMA 模型（用于稳定低噪声小残差场景）
    use_ema = (ema_decay is not None) and (ema_decay > 0) and (ema_decay < 1.0)
    ema_net = None
    if use_ema:
        ema_net = DnCNN2DPlus(base_channels=base_channels, num_blocks=num_blocks, use_cbam_at=use_cbam_at, in_channels=in_ch).to(dev)
        ema_net.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    if dev.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        use_amp = True
    else:
        scaler = _DummyScaler()
        use_amp = False
    sched = None
    if scheduler == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps - warmup_steps), eta_min=lr * 0.1)

    # 固定验证集（合成一次）
    val_batch = None
    if val_size > 0:
        ret_val = make_seq_batch_func(val_size)
        if isinstance(ret_val, tuple) and len(ret_val) == 3:
            noisy_v, clean_v, _vn = ret_val  # type: ignore[misc]
        else:
            noisy_v, clean_v = ret_val  # type: ignore[assignment]
        val_batch = LidarSeqDataset(noisy_v, clean_v, max_range)

    net.train()
    ema = None
    best_val = None

    def compute_loss(r_last, residual_last, valid, clean_pred=None, clean_gt=None, batch_sigma: float | None = None):
        # r_last/residual_last: [B,1,N] valid:[B,1,N]
        diff = r_last - residual_last
        if loss_type == 'mse':
            base = (diff ** 2)
        elif loss_type == 'l1':
            base = diff.abs()
        elif loss_type == 'smoothl1':
            base = F.smooth_l1_loss(r_last, residual_last, reduction='none')
        elif loss_type == 'hybrid':
            mse_part = (diff ** 2)
            l1_part = diff.abs()
            base = hybrid_alpha * mse_part + (1.0 - hybrid_alpha) * l1_part
        else:
            raise ValueError(f"unknown loss_type {loss_type}")
        loss_residual = (base * valid).mean()
        if distance_loss_weight > 0 and clean_pred is not None and clean_gt is not None:
            dist_loss = ((clean_pred - clean_gt) ** 2 * valid).mean()
            total = loss_residual + distance_loss_weight * dist_loss
            return total, loss_residual, dist_loss
        total = loss_residual
        return total, loss_residual, torch.tensor(0.0, device=dev)


    def evaluate():
        if val_batch is None:
            return None
        target_model = ema_net if (use_ema and ema_net is not None) else net
        target_model.eval()
        with torch.no_grad():
            loader_v = torch.utils.data.DataLoader(val_batch, batch_size=len(val_batch), shuffle=False)
            for sample in loader_v:
                noisy = sample['noisy'].to(dev)
                residual_last = sample['residual_last'].to(dev)
                valid = sample['valid'].to(dev)
                clean_last_gt = sample['clean_last'].to(dev)
                if cond_noise:
                    vn = float(val_noise if (val_noise is not None) else 0.0)
                    div = (noise_norm_divisor if (noise_norm_divisor is not None) else max_range)
                    vn_norm = vn / div
                    noise_level = torch.full((noisy.shape[0],1,noisy.shape[2],noisy.shape[3]), vn_norm, device=dev)
                    noisy_in = torch.cat([noisy, noise_level], dim=1)
                else:
                    noisy_in = noisy
                clean_last_pred, r_last = target_model(noisy_in)
                val_loss, _, _ = compute_loss(r_last, residual_last, valid, clean_last_pred, clean_last_gt, batch_sigma=val_noise)
                return float(val_loss.item())
    # 评估返回 None 表示没有验证集
    for step in range(1, steps + 1):
        if step == 1:
            print("[debug] entered training loop")
        batch_noise_value = None
        ret = make_seq_batch_func(batch_size)
        if isinstance(ret, tuple) and len(ret) == 3:
            noisy_seqs, clean_seqs, batch_noise_value = ret  # type: ignore[misc]
        else:
            noisy_seqs, clean_seqs = ret  # type: ignore[assignment]
        ds = LidarSeqDataset(noisy_seqs, clean_seqs, max_range)
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)

        
        for sample in loader:
            noisy = sample['noisy'].to(dev, non_blocking=True)
            residual_last = sample['residual_last'].to(dev, non_blocking=True)
            valid = sample['valid'].to(dev, non_blocking=True)
            clean_last_gt = sample['clean_last'].to(dev, non_blocking=True)

            # warmup: 线性增长 lr
            if warmup_steps > 0 and step <= warmup_steps:
                warm_scale = step / warmup_steps
                for pg in opt.param_groups:
                    pg['lr'] = lr * warm_scale

            opt.zero_grad(set_to_none=True)
            if dev.type == 'cuda':
                amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)
            else:
                
                amp_ctx = nullcontext()
            with amp_ctx:
                if cond_noise:
                    if batch_noise_value is not None:
                        div = (noise_norm_divisor if (noise_norm_divisor is not None) else max_range)
                        bn_norm = float(batch_noise_value) / div
                        noise_level = torch.full((noisy.shape[0],1,noisy.shape[2],noisy.shape[3]), bn_norm, device=dev)
                    else:
                        noise_level = torch.zeros(noisy.shape[0],1,noisy.shape[2],noisy.shape[3], device=dev)
                    noisy_in = torch.cat([noisy, noise_level], dim=1)
                else:
                    noisy_in = noisy
                clean_last_pred, r_last = net(noisy_in)
                loss, loss_residual, loss_distance = compute_loss(
                    r_last, residual_last, valid,
                    clean_pred=clean_last_pred if distance_loss_weight > 0 else None,
                    clean_gt=clean_last_gt if distance_loss_weight > 0 else None,
                    batch_sigma=batch_noise_value,
                )
                if sigma_weight_power > 0 and batch_noise_value is not None and batch_noise_value > 0:
                    loss = loss * (batch_noise_value ** (-sigma_weight_power))

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            # EMA 更新
            if use_ema and ema_net is not None:
                with torch.no_grad():
                    msd = net.state_dict()
                    ed = float(ema_decay)  # type: ignore[arg-type]
                    for k, v in ema_net.state_dict().items():
                        v.copy_(v * ed + msd[k] * (1.0 - ed))

        # scheduler (放在一个 outer step 结束后)
        if sched and (warmup_steps == 0 or step > warmup_steps):
            sched.step()

        l = float(loss.detach().item())
        ema = l if ema is None else (0.98 * ema + 0.02 * l)

        if step % 100 == 0 or step == 1:
            current_lr = opt.param_groups[0]['lr']
            msg = f"[step {step}] lr={current_lr:.3e} loss={l:.6f} ema={ema:.6f}"
            if distance_loss_weight > 0:
                msg += f" (res={loss_residual.item():.6f}, dist={loss_distance.item():.6f})"
            if sigma_weight_power > 0 and batch_noise_value is not None and batch_noise_value > 0:
                msg += f" w_sigma={(batch_noise_value ** (-sigma_weight_power)):.3f}"
            print(msg)

        # 验证 & 保存
        if val_batch is not None and (step % eval_interval == 0 or step == steps):
            val_loss = evaluate()
            if val_loss is not None:
                print(f"  -> val_loss={val_loss:.6f}")
                improved = (best_val is None) or (val_loss < best_val - 1e-6)
                if improved and save_best:
                    best_val = val_loss
                    best_path = model_save_path + best_suffix
                    torch.save(net.state_dict(), best_path)
                    print(f"  >> saved best (raw): {best_path} (val={best_val:.6f})")
                    if use_ema and save_ema and ema_net is not None:
                        best_ema_path = model_save_path + '.ema' + best_suffix
                        torch.save(ema_net.state_dict(), best_ema_path)
                        print(f"  >> saved best (EMA): {best_ema_path}")

        if step % 2000 == 0:
            torch.save(net.state_dict(), model_save_path)

    # 最终保存 (最后一版)
    torch.save(net.state_dict(), model_save_path)
    print("saved final (raw):", model_save_path)
    if use_ema and save_ema and ema_net is not None:
        ema_final_path = model_save_path + '.ema.pt'
        torch.save(ema_net.state_dict(), ema_final_path)
        print("saved final (EMA):", ema_final_path)
    if save_best and best_val is not None:
        print(f"best val={best_val:.6f} -> {model_save_path + best_suffix}")
        if use_ema and save_ema:
            print(f"(EMA best maybe at {model_save_path + '.ema' + best_suffix})")
