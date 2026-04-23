"""evaluate_multi_noise.py
批量评估多个模型在多种噪声水平下的鲁棒性。

用法示例：
  python evaluate_multi_noise.py \
      --maze-config 2.json --T 5 --num-seqs 60 \
      --noise-levels 0.05 0.1 0.2 0.3 0.5 \
      --model weights/dncnn_low_noise_ema.pt.ema.best.pt,64,14,6,low_ema \
      --model weights/dncnn_mid_noise.pt.best.pt,48,10,5,mid \
      --model weights/dncnn_high_noise.pt.best.pt,48,10,5,high \
      --out-csv robustness.csv --out-markdown robustness.md --plot robustness.png

--model 说明： path,base_channels,num_blocks,use_cbam_at,label
  其中 use_cbam_at 可为负/0 表示无 CBAM。

输出：
  - CSV: 每行一个 (model_label, eval_noise) 指标
  - Markdown: 表格汇总
  - Plot: (可选) 不同模型随噪声的 MSE 改善百分比折线图
"""
from __future__ import annotations
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from maze_loader import MazeLoader
from lidar import Lidar
from dncnn2d_temporal import make_seq_batch_func_factory, TemporalDnCNN2DFilter

@dataclass
class ModelSpec:
    path: str
    base_channels: int
    num_blocks: int
    use_cbam_at: int | None
    label: str


def parse_model_spec(s: str) -> ModelSpec:
    parts = s.split(',')
    if len(parts) != 5:
        raise ValueError(f"模型参数格式错误: {s}")
    path = parts[0].strip()
    bc = int(parts[1])
    nb = int(parts[2])
    cb_raw = int(parts[3])
    use_cb = cb_raw if cb_raw > 0 else None
    label = parts[4].strip()
    return ModelSpec(path, bc, nb, use_cb, label)


def eval_one(model: ModelSpec, noise: float, args) -> Tuple[float,float,float,float,float]:
    # 构建带此噪声的模拟器
    maze = args._maze  # 预加载
    lidar_sim = Lidar(maze.walls, max_range=args.max_range, angle_resolution=args.angle_resolution, noise=noise)
    # 批次生成器
    class _SimWrap:
        def __init__(self, walls, max_range, angle_resolution, noise):
            self.walls = walls
            self.max_range = max_range
            self.angle_resolution = angle_resolution
            self.noise = noise
            self._lidar = lidar_sim
        def scan_with_clean(self, pose):
            return self._lidar.scan(pose)
    sim = _SimWrap(maze.walls, args.max_range, args.angle_resolution, noise)
    make_batch = make_seq_batch_func_factory(sim, T=args.T, step_xy_std=args.step_xy_std, step_theta_std=args.step_theta_std)
    noisy_list, clean_list = make_batch(args.num_seqs)

    filt = TemporalDnCNN2DFilter(T=args.T, max_range=args.max_range, model_path=model.path,
                                 base_channels=model.base_channels, num_blocks=model.num_blocks,
                                 use_cbam_at=model.use_cbam_at,
                                 condition_noise=args.cond_noise,
                                 noise_norm_divisor=(args.noise_divisor if args.noise_divisor else None))

    noisy_mses = []
    denoise_mses = []
    for seq_idx in range(len(noisy_list)):
        noisy_seq = noisy_list[seq_idx]
        clean_seq = clean_list[seq_idx]
        for t in range(args.T):
            if args.cond_noise:
                out = filt.filter_lidar_data(noisy_seq[t].tolist(), noise_sigma=noise)
            else:
                out = filt.filter_lidar_data(noisy_seq[t].tolist())
        den = np.array(out)
        noisy_last = noisy_seq[-1]
        clean_last = clean_seq[-1]
        valid = clean_last < (args.max_range * 0.999)
        if valid.sum() == 0:
            filt.buf.clear()
            continue
        nmse = float(np.mean((noisy_last[valid] - clean_last[valid]) ** 2))
        dmse = float(np.mean((den[valid] - clean_last[valid]) ** 2))
        noisy_mses.append(nmse)
        denoise_mses.append(dmse)
        filt.buf.clear()
    if not noisy_mses:
        return math.nan, math.nan, math.nan, math.nan, 0.0
    noisy_mse = float(np.mean(noisy_mses))
    denoise_mse = float(np.mean(denoise_mses))
    noisy_rmse = math.sqrt(noisy_mse)
    denoise_rmse = math.sqrt(denoise_mse)
    improve = (noisy_mse - denoise_mse) / noisy_mse * 100.0 if noisy_mse > 0 else 0.0
    return noisy_mse, denoise_mse, noisy_rmse, denoise_rmse, improve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--maze-config', type=str, default='2.json')
    p.add_argument('--angle-resolution', type=float, default=1.0)
    p.add_argument('--max-range', type=float, default=12.0)
    p.add_argument('--T', type=int, default=5)
    p.add_argument('--num-seqs', type=int, default=60)
    p.add_argument('--noise-levels', type=float, nargs='+', required=True, help='需要评估的噪声sigma列表')
    p.add_argument('--model', type=str, action='append', required=True, help='格式: path,base_channels,num_blocks,use_cbam_at,label 可多次')
    p.add_argument('--step-xy-std', type=float, default=0.05)
    p.add_argument('--step-theta-std', type=float, default=0.03)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--out-csv', type=str, default='robustness.csv')
    p.add_argument('--out-markdown', type=str, default='robustness.md')
    p.add_argument('--plot', type=str, default=None, help='输出折线图文件名 (可选)')
    # 条件噪声模型支持
    p.add_argument('--cond-noise', action='store_true', help='模型包含噪声条件通道 (in_channels=2)')
    p.add_argument('--noise-divisor', type=float, default=None, help='噪声归一化除数(默认用 max_range)')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    # 预加载迷宫
    maze = MazeLoader().load(args.maze_config)
    args._maze = maze  # 注入

    models: List[ModelSpec] = [parse_model_spec(s) for s in args.model]

    rows = []
    print('=== Multi-Noise Robustness Evaluation ===')
    header = ['model','noise','noisy_mse','denoise_mse','noisy_rmse','denoise_rmse','improve_pct']
    for m in models:
        print(f"-- Model: {m.label} ({m.path})")
        for nz in args.noise_levels:
            noisy_mse, denoise_mse, noisy_rmse, denoise_rmse, improve = eval_one(m, nz, args)
            print(f"  noise={nz:.3f}  noisy_mse={noisy_mse:.6f}  denoise_mse={denoise_mse:.6f}  improve={improve:.2f}%")
            rows.append({
                'model': m.label,
                'noise': nz,
                'noisy_mse': noisy_mse,
                'denoise_mse': denoise_mse,
                'denoise_rmse': denoise_rmse,
                'improve_pct': improve,
            })

    # 写 CSV
    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print('CSV saved:', args.out_csv)

    # 写 Markdown
    if args.out_markdown:
        # 按噪声排序 -> 构建 markdown
        lines = ['| Model | Noise | Noisy MSE | Denoise MSE | Improve % |', '|-------|-------|-----------|-------------|-----------|']
        for r in rows:
            lines.append(f"| {r['model']} | {r['noise']:.3f} | {r['noisy_mse']:.6f} | {r['denoise_mse']:.6f} | {r['improve_pct']:.2f}% |")
        with open(args.out_markdown, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('Markdown saved:', args.out_markdown)

    # 绘图
    if args.plot:
        # 噪声 -> 改善的折线
        for m in models:
            xs = [r['noise'] for r in rows if r['model'] == m.label]
            ys = [r['improve_pct'] for r in rows if r['model'] == m.label]
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            plt.plot(xs, ys, marker='o', label=m.label)
        plt.xlabel('Noise Sigma')
        plt.ylabel('MSE Improvement %')
        plt.title('Denoising Robustness vs Noise')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        plt.close()
        print('Plot saved:', args.plot)

if __name__ == '__main__':
    main()
