"""train_my_sim.py
使用项目中已有的 MazeLoader (墙壁参数) 与 Lidar 模拟器，训练时序 DnCNN2D 去噪模型。

示例：
  1) 快速测试运行（小步数验证脚本无误）：
     python train_my_sim.py --steps 20 --T 3 --batch-size 4 --model-save weights/test_dncnn2d.pt

  2) 正式训练：
     python train_my_sim.py --maze-config 2.json --T 5 --steps 20000 \
         --batch-size 32 --lr 1e-3 --noise 0.05 --model-save weights/dncnn2d.pt

功能特点：
  - 复用现有 `MazeLoader` 加载墙壁线段。
  - 使用项目中的 `Lidar` 类产生 (noisy, clean) 激光帧。
  - 通过 `dncnn2d_temporal.make_seq_batch_func_factory` 构造时间窗口批次。
  - 支持噪声抖动 (noise-jitter) 提升泛化。
  - 训练只回归窗口内最后一帧残差 (在线因果)。
"""

import os
import json
import argparse
import math
import random
from typing import List, Tuple

import numpy as np

from maze_loader import MazeLoader
from lidar import Lidar
from dncnn2d_temporal import (
    train_temporal_dncnn2d,
    make_seq_batch_func_factory,
)


class ProjectSim:
    """将现有 Lidar 封装为训练所需接口的模拟器。

    需要属性：
        walls: List[((x1,y1),(x2,y2))]
        max_range: float
        angle_resolution: float (度)
        noise: float
    需要方法：
        scan_with_clean(pose) -> (noisy_list, clean_list)
    """

    def __init__(self, walls, max_range: float, angle_resolution: float, noise: float):
        self.walls = walls
        self.max_range = float(max_range)
        self.angle_resolution = float(angle_resolution)
        self.noise = float(noise)
        self._lidar = Lidar(walls, max_range=self.max_range, angle_resolution=self.angle_resolution, noise=self.noise)

    def scan_with_clean(self, pose: Tuple[float, float, float]):
        noisy, clean = self._lidar.scan(pose)
        return noisy, clean


def parse_args():
    p = argparse.ArgumentParser(description="Train temporal DnCNN2D on project maze lidar simulation")
    # 数据 / 模拟器
    p.add_argument("--maze-config", type=str, default="2.json", help="迷宫/墙壁配置文件 (支持 .json 或自定义文本格式)")
    p.add_argument("--angle-resolution", type=float, default=1.0, help="雷达角分辨率(度)")
    p.add_argument("--max-range", type=float, default=12.0, help="雷达最大量程(米)")
    p.add_argument("--noise", type=float, default=0.05, help="测距高斯噪声 sigma (米)")
    p.add_argument("--noise-jitter", type=float, default=0.0, help="每batch乘法噪声抖动幅度，例如0.3表示在[0.7,1.3]*noise")

    # 训练超参（基础）
    p.add_argument("--T", type=int, default=5, help="时间窗口长度(帧数)")
    p.add_argument("--steps", type=int, default=20000, help="训练步数 (outer iterations，每步重新合成 batch)")
    p.add_argument("--batch-size", type=int, default=32, help="批大小")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--model-save", type=str, default="weights/dncnn2d.pt", help="模型权重保存路径")

    # 网络结构参数
    p.add_argument("--base-channels", type=int, default=64, help="网络基础通道数")
    p.add_argument("--num-blocks", type=int, default=12, help="残差块数量")
    p.add_argument("--use-cbam-at", type=int, default=6, help="在第k个残差块后插入CBAM (0或负数关闭)")
    p.add_argument("--cond-noise", action="store_true", help="启用噪声条件通道 (多噪声统一模型)")
    p.add_argument("--noise-divisor", type=float, default=None, help="噪声归一化除数(默认用max_range)")
    # FiLM 已移除
    p.add_argument("--multi-noise-list", type=float, nargs='+', default=None, help="多噪声列表: 例如 --multi-noise-list 0.05 0.1 0.2 (需配合 --cond-noise)")
    p.add_argument("--multi-noise-range", type=str, default=None, help="多噪声范围规格: start:end:step 例如 0.01:0.50:0.005 (需配合 --cond-noise; 与 --multi-noise-list 互斥，优先生效)")
    p.add_argument("--val-noise", type=float, default=None, help="验证时使用的噪声sigma (cond-noise下指定一个代表性噪声做val；None则使用 --noise)")
    p.add_argument("--sample-sigma-power", type=float, default=0.0, help="多噪声采样概率 ~ sigma^(-p)。p>0 偏向低噪声；p<0 偏向高噪声；0=均匀")

    # 损失 & 正则
    p.add_argument("--loss-type", type=str, default="mse", choices=["mse","l1","smoothl1","hybrid"], help="主损失类型")
    p.add_argument("--hybrid-alpha", type=float, default=0.5, help="hybrid损失中MSE权重(alpha)")
    p.add_argument("--distance-loss-weight", type=float, default=0.0, help="额外 clean_last 监督权重")
    p.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值(<=0关闭)")
    p.add_argument("--sigma-weight-power", type=float, default=0.0, help="按( sigma ^ -p )加权loss (多噪声+cond下生效, 0关闭)")
    p.add_argument("--resume-from", type=str, default=None, help="从已有权重继续训练 (raw 权重路径)")
    # EMA
    p.add_argument("--ema-decay", type=float, default=None, help="启用EMA: 设为(0,1)之间的小数，如0.999；None关闭")
    p.add_argument("--no-save-ema", action="store_true", help="不保存EMA模型")

    # 调度 & warmup
    p.add_argument("--warmup-steps", type=int, default=0, help="学习率warmup步数")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine","none"], help="学习率调度器")

    # 验证与保存
    p.add_argument("--val-size", type=int, default=0, help="验证集大小(0关闭验证)")
    p.add_argument("--eval-interval", type=int, default=500, help="验证间隔步数")
    p.add_argument("--no-save-best", action="store_true", help="不单独保存best模型")
    p.add_argument("--best-suffix", type=str, default=".best.pt", help="best模型后缀")

    # 随机游走运动噪声(合成序列位姿扰动)
    p.add_argument("--step-xy-std", type=float, default=0.05, help="每帧平移步长标准差 (米)")
    p.add_argument("--step-theta-std", type=float, default=0.03, help="每帧旋转标准差 (弧度)")

    # 其它
    p.add_argument("--seed", type=int, default=123, help="随机种子 (影响起始位姿与噪声)")
    p.add_argument("--print-interval", type=int, default=100, help="日志打印间隔")
    return p.parse_args()


def build_sim(args):
    loader = MazeLoader()
    maze = loader.load(args.maze_config)
    # MazeLoader 已可选平移，直接取 walls
    sim = ProjectSim(maze.walls, max_range=args.max_range, angle_resolution=args.angle_resolution, noise=args.noise)
    return sim


def main():
    args = parse_args()

    # 随机性控制 (主要是 numpy & python 的随机)
    np.random.seed(args.seed)

    sim = build_sim(args)

    # 基础批函数 (不含多噪声)
    base_make_batch = make_seq_batch_func_factory(
        sim,
        T=args.T,
        step_xy_std=args.step_xy_std,
        step_theta_std=args.step_theta_std,
    )

    # 多噪声条件统一模型: 随机采噪声 -> 三元返回 (noisy_list, clean_list, noise_sigma)
    make_batch = base_make_batch
    # 解析 range 形式 (优先) -> multi_noise_list
    if args.multi_noise_range and not args.multi_noise_list:
        try:
            s, e, st = args.multi_noise_range.split(':')
            s = float(s); e = float(e); st = float(st)
            if st <= 0 or e <= s:
                raise ValueError("range 参数非法")
            vals = [round(x, 6) for x in np.arange(s, e + st*0.5, st)]
            args.multi_noise_list = vals
            print(f"[multi-noise-range] 生成 {len(vals)} 个 sigma 值: [{vals[0]}, ..., {vals[-1]}]")
        except Exception as ex:
            raise SystemExit(f"解析 --multi-noise-range 失败: {ex}")

    if args.cond_noise and args.multi_noise_list:
        noise_pool = list(args.multi_noise_list)
        weights = None
        if args.sample_sigma_power != 0:
            # 概率 ∝ sigma^(-p)。
            # p>0 -> 倾向小sigma (原逻辑)；p<0 -> (-p) 为负，使得指数为正，倾向大sigma。
            raw_w = [ (s ** (-args.sample_sigma_power)) if s>0 else 0.0 for s in noise_pool ]
            ssum = sum(raw_w)
            if ssum > 0:
                weights = [w/ssum for w in raw_w]
        from random import choices as _choices
        def multi_noise_make_batch(batch_size: int):
            if weights is not None:
                sigma = _choices(noise_pool, weights, k=1)[0]
            else:
                sigma = random.choice(noise_pool)
            sim.noise = float(sigma)
            noisy_list, clean_list = base_make_batch(batch_size)
            # 始终返回 3 元组 (noisy_list, clean_list, float)
            return noisy_list, clean_list, float(sigma)
        make_batch = multi_noise_make_batch

    # 噪声抖动（对当前 sim.noise 再乘随机因子；若多噪声启用，作用于采样后的 sigma）
    if args.noise_jitter > 0:
        inner_make = make_batch
        base_sigma_backup = float(sim.noise)

        def jittered_make_batch(batch_size: int):
            lo = max(0.0, 1.0 - args.noise_jitter)
            hi = 1.0 + args.noise_jitter
            scale = float(np.random.uniform(lo, hi))
            if args.cond_noise and args.multi_noise_list:
                ret = inner_make(batch_size)
                # 静态类型提示：multi_noise_make_batch 逻辑下始终返回三元组
                if len(ret) != 3:  # 运行期安全检查
                    raise RuntimeError("multi-noise 模式下应返回3元组 (noisy, clean, sigma)")
                noisy_list, clean_list, sigma = ret  # type: ignore[misc]
                jitter_sigma = float(sigma * scale)
                sim.noise = jitter_sigma
                # 重新基于抖动后 sigma 采样一次，保持数据与返回的 jitter_sigma 对齐
                noisy_list, clean_list = base_make_batch(batch_size)
                return noisy_list, clean_list, jitter_sigma
            else:
                sim.noise = base_sigma_backup * scale
                return inner_make(batch_size)

        make_batch = jittered_make_batch

    # 创建保存目录
    save_dir = os.path.dirname(args.model_save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("==== Training Temporal DnCNN2D (ProjectSim) ====")
    print(f"maze-config: {args.maze_config}")
    print(f"T={args.T} steps={args.steps} batch={args.batch_size} lr={args.lr}")
    print(f"angle_resolution={args.angle_resolution} max_range={args.max_range} noise={args.noise}")
    if args.noise_jitter > 0:
        print(f"noise-jitter: +/- {args.noise_jitter*100:.1f}%")

    # 训练
    train_temporal_dncnn2d(
        model_save_path=args.model_save,
        max_range=sim.max_range,
        make_seq_batch_func=make_batch,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=None,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        use_cbam_at=(args.use_cbam_at if args.use_cbam_at > 0 else None),
        cond_noise=args.cond_noise,
        noise_norm_divisor=args.noise_divisor,
        val_noise=args.val_noise,
        loss_type=args.loss_type,
        hybrid_alpha=args.hybrid_alpha,
        distance_loss_weight=args.distance_loss_weight,
        warmup_steps=args.warmup_steps,
        scheduler=(None if args.scheduler == 'none' else args.scheduler),
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        save_ema=(not args.no_save_ema),
        val_size=args.val_size,
        eval_interval=args.eval_interval,
        save_best=(not args.no_save_best),
        best_suffix=args.best_suffix,
    sigma_weight_power=args.sigma_weight_power,
    resume_path=args.resume_from,
    )


if __name__ == "__main__":
    main()
