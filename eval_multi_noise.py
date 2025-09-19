"""eval_multi_noise.py
多噪声条件 Temporal DnCNN2DPlus 评估脚本。

功能:
  - 给定权重文件(EMA或raw)与噪声sigma列表, 在模拟环境下合成若干序列评估每个sigma的去噪效果。
  - 指标: noisy最后一帧 vs clean 的 MSE, 模型输出clean vs clean 的 MSE, 以及改善百分比。

用法示例:
  python eval_multi_noise.py \
      --maze-config 2.json \
      --weights weights/dncnn_multi_noise_finetune_rollback.pt.ema.best.pt \
      --sigmas 0.05 0.1 0.2 0.3 0.5 \
      --samples 200 --T 5 --base-channels 56 --num-blocks 12 --use-cbam-at 6 --cond-noise

对比两个权重:
  python eval_multi_noise.py --compare \
      --weights-a weights/dncnn_multi_noise_finetune_rollback.pt.ema.best.pt \
      --weights-b weights/dncnn_multi_noise_finetune_rollback_continue.pt.ema.best.pt \
      --sigmas 0.05 0.1 0.2 0.3 0.5 --samples 200 --T 5 --base-channels 56 --num-blocks 12 --use-cbam-at 6 --cond-noise

注意:
  - 该脚本重新合成随机序列 (与训练一致的随机游走策略)。
  - MSE 仅在 valid_mask (clean < max_range*0.999) 范围内统计。
"""

from __future__ import annotations

import argparse
import math
import random
from typing import List, Tuple

import numpy as np
import torch

from maze_loader import MazeLoader
from lidar import Lidar
from dncnn2d_temporal import DnCNN2DPlus, select_device


class ProjectSim:
    def __init__(self, walls, max_range: float, angle_resolution: float, noise: float):
        self.walls = walls
        self.max_range = float(max_range)
        self.angle_resolution = float(angle_resolution)
        self.noise = float(noise)
        self._lidar = Lidar(walls, max_range=self.max_range, angle_resolution=self.angle_resolution, noise=self.noise)

    def scan_with_clean(self, pose):
        noisy, clean = self._lidar.scan(pose)
        return noisy, clean


def build_sequences(sim: ProjectSim, T: int, samples: int, step_xy_std: float, step_theta_std: float, seed: int | None = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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
    noisy_list, clean_list = [], []
    for _ in range(samples):
        pose = sample_start_pose()
        noisy_seq = np.zeros((T, num_beams), dtype=np.float32)
        clean_seq = np.zeros((T, num_beams), dtype=np.float32)
        for t in range(T):
            if t > 0:
                pose[0] += np.random.normal(0, step_xy_std)
                pose[1] += np.random.normal(0, step_xy_std)
                pose[2] += np.random.normal(0, step_theta_std)
                pose[2] = math.atan2(math.sin(pose[2]), math.cos(pose[2]))
            noisy, clean = sim.scan_with_clean(tuple(pose))
            noisy_seq[t, :] = noisy
            clean_seq[t, :] = clean
        noisy_list.append(noisy_seq)
        clean_list.append(clean_seq)
    return noisy_list, clean_list


@torch.no_grad()
def evaluate_weight(weight_path: str, sigmas: List[float], args) -> dict:
    dev = select_device(None)
    in_ch = 2 if args.cond_noise else 1
    net = DnCNN2DPlus(base_channels=args.base_channels, num_blocks=args.num_blocks, use_cbam_at=(args.use_cbam_at if args.use_cbam_at > 0 else None), in_channels=in_ch).to(dev)
    net.load_state_dict(torch.load(weight_path, map_location=dev))
    net.eval()

    loader = MazeLoader()
    maze = loader.load(args.maze_config)

    results = {}
    for sigma in sigmas:
        sim = ProjectSim(maze.walls, max_range=args.max_range, angle_resolution=args.angle_resolution, noise=sigma)
        noisy_list, clean_list = build_sequences(sim, args.T, args.samples, args.step_xy_std, args.step_theta_std, seed=args.seed)
        noisy_arr = np.stack(noisy_list, axis=0)  # [B,T,N]
        clean_arr = np.stack(clean_list, axis=0)
        B, T, N = noisy_arr.shape
        noisy_norm = np.clip(noisy_arr / args.max_range, 0.0, 1.0)
        clean_norm = np.clip(clean_arr / args.max_range, 0.0, 1.0)
        noisy_tensor = torch.from_numpy(noisy_norm).float().to(dev)  # [B,T,N]
        clean_tensor = torch.from_numpy(clean_norm).float().to(dev)

        residual_last = noisy_tensor[:, -1, :] - clean_tensor[:, -1, :]  # [B,N]
        valid_mask = (clean_tensor[:, -1, :] < 0.999).float()

        # 组装网络输入 [B,C,T,N]
        inp = noisy_tensor.unsqueeze(1)  # [B,1,T,N]
        if args.cond_noise:
            div = (args.noise_divisor if args.noise_divisor else args.max_range)
            ns_norm = float(sigma) / div
            noise_ch = torch.full_like(inp, ns_norm)
            inp = torch.cat([inp, noise_ch], dim=1)

        clean_pred, r_last = net(inp)
        noisy_last = noisy_tensor[:, -1, :].unsqueeze(1)  # [B,1,N]
        clean_gt_last = clean_tensor[:, -1, :].unsqueeze(1)

        # MSE on residual baseline: (noisy_last - clean_gt)^2
        mse_noisy = ((noisy_last - clean_gt_last) ** 2 * valid_mask.unsqueeze(1)).mean().item()
        mse_model = ((clean_pred - clean_gt_last) ** 2 * valid_mask.unsqueeze(1)).mean().item()
        improvement = (mse_noisy - mse_model) / mse_noisy * 100.0 if mse_noisy > 0 else 0.0
        results[sigma] = {
            'mse_noisy': mse_noisy,
            'mse_model': mse_model,
            'improve_pct': improvement
        }
    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maze-config', type=str, default='2.json')
    ap.add_argument('--angle-resolution', type=float, default=1.0)
    ap.add_argument('--max-range', type=float, default=12.0)
    ap.add_argument('--T', type=int, default=5)
    ap.add_argument('--samples', type=int, default=200, help='每个sigma评估序列数')
    ap.add_argument('--sigmas', type=float, nargs='+', required=True)
    ap.add_argument('--weights', type=str, required=True, help='单权重评估或对比A权重')
    ap.add_argument('--weights-b', type=str, default=None, help='对比评估时的B权重')
    ap.add_argument('--compare', action='store_true', help='启用A vs B对比')
    ap.add_argument('--base-channels', type=int, default=64)
    ap.add_argument('--num-blocks', type=int, default=12)
    ap.add_argument('--use-cbam-at', type=int, default=6)
    ap.add_argument('--cond-noise', action='store_true')
    ap.add_argument('--noise-divisor', type=float, default=None)
    ap.add_argument('--step-xy-std', type=float, default=0.05)
    ap.add_argument('--step-theta-std', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=777)
    return ap.parse_args()


def main():
    args = parse_args()
    sigmas = args.sigmas
    print(f"Evaluating {args.weights} on sigmas: {sigmas}")
    res_a = evaluate_weight(args.weights, sigmas, args)
    if args.compare:
        if not args.weights_b:
            raise SystemExit('--compare 需要提供 --weights-b')
        print(f"Evaluating {args.weights_b} ...")
        res_b = evaluate_weight(args.weights_b, sigmas, args)
    else:
        res_b = None

    def fmt(r):
        return f"noisy={r['mse_noisy']:.6f} model={r['mse_model']:.6f} imp={r['improve_pct']:.2f}%"

    print('\n=== Results ===')
    for s in sigmas:
        ra = res_a[s]
        line = f"sigma={s:.3f}  A:{fmt(ra)}"
        if res_b is not None:
            rb = res_b[s]
            line += f"  |  B:{fmt(rb)}  Δimp={rb['improve_pct']-ra['improve_pct']:.2f}pt"
        print(line)

    # 简洁JSON可复用
    import json
    out = {'A': res_a, 'B': res_b, 'sigmas': sigmas}
    print('\nJSON_SUMMARY_START')
    print(json.dumps(out, indent=2))
    print('JSON_SUMMARY_END')


if __name__ == '__main__':
    main()
