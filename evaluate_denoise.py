"""evaluate_denoise.py
评估 Temporal DnCNN2D 降噪效果，并生成示例图。

用法示例：
  python evaluate_denoise.py --model weights/demo_dncnn2d.pt --T 5 --num-seqs 60 \
      --maze-config 2.json --noise 0.05 --out-img denoise_example.png

输出：
  - 终端打印 原始(noisy) 与 去噪(denoised) 的 MSE / RMSE 对比及改善百分比。
  - 保存示例图：原始 vs 去噪 vs 真值。
"""
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from maze_loader import MazeLoader
from lidar import Lidar
from dncnn2d_temporal import make_seq_batch_func_factory, TemporalDnCNN2DFilter

class ProjectSim:
    def __init__(self, walls, max_range, angle_resolution, noise):
        self.walls = walls
        self.max_range = float(max_range)
        self.angle_resolution = float(angle_resolution)
        self.noise = float(noise)
        self._lidar = Lidar(walls, max_range=self.max_range, angle_resolution=self.angle_resolution, noise=self.noise)
    def scan_with_clean(self, pose):
        return self._lidar.scan(pose)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, help='训练好的模型权重路径')
    p.add_argument('--maze-config', type=str, default='2.json')
    p.add_argument('--angle-resolution', type=float, default=1.0)
    p.add_argument('--max-range', type=float, default=12.0)
    p.add_argument('--noise', type=float, default=0.05)
    p.add_argument('--T', type=int, default=5)
    p.add_argument('--num-seqs', type=int, default=50, help='评估序列数量')
    p.add_argument('--step-xy-std', type=float, default=0.05)
    p.add_argument('--step-theta-std', type=float, default=0.03)
    p.add_argument('--out-img', type=str, default='denoise_example.png')
    p.add_argument('--seed', type=int, default=2024)
    # 网络结构（需与训练时一致）
    p.add_argument('--base-channels', type=int, default=64)
    p.add_argument('--num-blocks', type=int, default=12)
    p.add_argument('--use-cbam-at', type=int, default=6)
    p.add_argument('--cond-noise', action='store_true', help='模型包含噪声条件通道 (in_channels=2)')
    p.add_argument('--noise-divisor', type=float, default=None, help='噪声归一化除数(默认用 max_range)')
    return p.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    # 构建模拟器
    maze = MazeLoader().load(args.maze_config)
    sim = ProjectSim(maze.walls, args.max_range, args.angle_resolution, args.noise)

    # 批函数 (用于构造评估序列)
    make_batch = make_seq_batch_func_factory(
        sim,
        T=args.T,
        step_xy_std=args.step_xy_std,
        step_theta_std=args.step_theta_std,
    )

    # 构造评估集（列表）
    batch_noisy, batch_clean = make_batch(args.num_seqs)
    # 初始化滤波器
    filt = TemporalDnCNN2DFilter(T=args.T, max_range=sim.max_range, model_path=args.model,
                                 base_channels=args.base_channels, num_blocks=args.num_blocks,
                                 use_cbam_at=(args.use_cbam_at if args.use_cbam_at > 0 else None),
                                 condition_noise=args.cond_noise,
                                 noise_norm_divisor=(args.noise_divisor if args.noise_divisor else None))

    noisy_errors = []
    denoise_errors = []
    example_plotted = False

    for seq_idx in range(len(batch_noisy)):
        noisy_seq = batch_noisy[seq_idx]  # shape [T,N]
        clean_seq = batch_clean[seq_idx]
        # 逐帧送入滤波器
        for t in range(args.T):
            if args.cond_noise:
                output = filt.filter_lidar_data(noisy_seq[t].tolist(), noise_sigma=args.noise)
            else:
                output = filt.filter_lidar_data(noisy_seq[t].tolist())
        # 输出对应最后一帧的去噪结果
        denoised_last = np.array(output)
        noisy_last = noisy_seq[-1]
        clean_last = clean_seq[-1]

        # 计算误差 (只在 clean_last < max_range 处统计，也可以全量)
        valid_mask = clean_last < (sim.max_range * 0.999)
        if valid_mask.sum() == 0:
            continue
        noisy_err = np.mean((noisy_last[valid_mask] - clean_last[valid_mask]) ** 2)
        denoise_err = np.mean((denoised_last[valid_mask] - clean_last[valid_mask]) ** 2)
        noisy_errors.append(noisy_err)
        denoise_errors.append(denoise_err)

        # 保存示例图（只做一次）
        if not example_plotted:
            beams = np.arange(len(noisy_last))
            plt.figure(figsize=(10,4))
            plt.plot(beams, clean_last, label='Clean (GT)', linewidth=2)
            plt.plot(beams, noisy_last, label='Noisy', alpha=0.7)
            plt.plot(beams, denoised_last, label='Denoised', linewidth=1.5)
            plt.xlabel('Beam Index')
            plt.ylabel('Distance (m)')
            plt.title('LiDAR Denoising Example (Last Frame)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.out_img, dpi=150)
            plt.close()
            example_plotted = True
            print(f"示例图已保存: {args.out_img}")

        # 重置滤波器缓冲（保证序列独立统计）
        filt.buf.clear()

    if not noisy_errors:
        print('无有效样本 (可能全部为 max_range 命中)，请调整评估设定。')
        return

    noisy_mse = float(np.mean(noisy_errors))
    denoise_mse = float(np.mean(denoise_errors))
    noisy_rmse = math.sqrt(noisy_mse)
    denoise_rmse = math.sqrt(denoise_mse)
    improvement = (noisy_mse - denoise_mse) / noisy_mse * 100.0 if noisy_mse > 0 else 0.0

    print('\n==== Denoising Evaluation ====')
    print(f'Model: {args.model}')
    print(f'Sequences evaluated: {len(noisy_errors)} (T={args.T})')
    print(f'Noisy   MSE={noisy_mse:.6f}  RMSE={noisy_rmse:.6f}')
    print(f'Denoise MSE={denoise_mse:.6f}  RMSE={denoise_rmse:.6f}')
    print(f'Improvement: {improvement:.2f}% MSE ↓')

if __name__ == '__main__':
    main()
