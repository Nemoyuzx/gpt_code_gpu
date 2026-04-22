#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read mem_usage_log.csv and plot memory usage.
- Subplot 1: RSS current/peak (MB)
- Subplot 2: map_points count (left axis) and map_points_mb (right axis)
- Subplot 3: CUDA/MPS memory (if available)

Usage:
    python plot_mem_usage.py --csv outputs/data/mem_usage_log.csv --out outputs/visualization/mem_usage_plot.png
    python plot_mem_usage.py --csv outputs/data/mem_usage_log.csv --no-show
"""
from __future__ import annotations
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
from output_paths import MEM_USAGE_CSV, MEM_USAGE_PLOT, ensure_parent


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def main():
    ap = argparse.ArgumentParser(description="Plot memory usage from mem_usage_log.csv")
    ap.add_argument("--csv", default=str(MEM_USAGE_CSV), help=f"Path to CSV file (default: {MEM_USAGE_CSV})")
    ap.add_argument("--out", default=str(MEM_USAGE_PLOT), help=f"Output image path (default: {MEM_USAGE_PLOT}); set empty to skip saving")
    ap.add_argument("--no-show", action="store_true", help="Do not show the window; only save to --out")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    ap.add_argument("--annotate", action="store_true", help="Annotate maxima on plots for quick reading")
    args = ap.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)

    try:
        # 尝试解析时间戳
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        sys.exit(2)

    # 解析时间列
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except Exception:
            pass
        x = df['timestamp']
    else:
        # 回退：使用行索引作为x轴
        x = pd.RangeIndex(start=0, stop=len(df))

    # 数值列转为数值
    num_cols = [
        'rss_cur_mb','rss_peak_mb',
        'cuda_alloc_mb','cuda_reserved_mb',
        'mps_current_mb','mps_driver_mb',
        'map_points','map_points_mb','occupancy_mb'
    ]
    df = to_numeric(df, num_cols)

    # 绘图
    plt.style.use('seaborn-v0_8') if 'seaborn-v0_8' in plt.style.available else None
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    # 辅助函数：获取x轴在给定位置的值（兼容 Series/RangeIndex）
    def x_at(pos: int):
        try:
            return x.iloc[pos]  # type: ignore[attr-defined]
        except Exception:
            try:
                return x[pos]
            except Exception:
                return pos

    # 小工具：拿到Series按位置的最大值索引（跳过NaN）
    def pos_of_max(series: pd.Series | pd.Index) -> int | None:
        try:
            arr = pd.to_numeric(series, errors='coerce') if isinstance(series, pd.Series) else pd.to_numeric(pd.Series(series), errors='coerce')
            vals = arr.to_numpy(dtype=float)
            if len(vals) == 0 or pd.isna(vals).all():
                return None
            return int(np.nanargmax(vals))
        except Exception:
            return None

    # Subplot 1: RSS
    ax1 = axes[0]
    has_cur = 'rss_cur_mb' in df.columns and df['rss_cur_mb'].notna().any()
    has_peak = 'rss_peak_mb' in df.columns and df['rss_peak_mb'].notna().any()
    if has_cur:
        ax1.plot(x, df['rss_cur_mb'], label='RSS current (MB)', color='tab:blue')
    if has_peak:
        ax1.plot(x, df['rss_peak_mb'], label='RSS peak (MB)', color='tab:orange', alpha=0.8)
    if args.annotate and (has_cur or has_peak):
        try:
            # 优先标注当前RSS最大值
            if has_cur and df['rss_cur_mb'].notna().any():
                idx = pos_of_max(df['rss_cur_mb'])
                if idx is not None:
                    ax1.annotate(f"max cur {df['rss_cur_mb'].iloc[idx]:.1f} MB", xy=(x_at(idx), df['rss_cur_mb'].iloc[idx]),
                             xytext=(10, 10), textcoords='offset points', color='tab:blue')
            if has_peak and df['rss_peak_mb'].notna().any():
                idxp = pos_of_max(df['rss_peak_mb'])
                if idxp is not None:
                    ax1.annotate(f"peak {df['rss_peak_mb'].iloc[idxp]:.1f} MB", xy=(x_at(idxp), df['rss_peak_mb'].iloc[idxp]),
                             xytext=(10, -15), textcoords='offset points', color='tab:orange')
        except Exception:
            pass
    if not (has_cur or has_peak):
        ax1.text(0.5, 0.5, 'RSS columns missing in CSV', transform=ax1.transAxes, ha='center', va='center')
    ax1.set_title('Process memory RSS')
    ax1.set_ylabel('MB')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Subplot 2: map_points scale
    ax2 = axes[1]
    has_pts = 'map_points' in df.columns and df['map_points'].notna().any()
    has_pts_mb = 'map_points_mb' in df.columns and df['map_points_mb'].notna().any()
    if has_pts:
        ax2.plot(x, df['map_points'], label='map_points (count)', color='tab:green')
        ax2.set_ylabel('Count')
    if has_pts_mb:
        ax2b = ax2.twinx()
        ax2b.plot(x, df['map_points_mb'], label='map_points (MB)', color='tab:red')
        ax2b.set_ylabel('MB')
        if args.annotate and df['map_points_mb'].notna().any():
            try:
                idm = pos_of_max(df['map_points_mb'])
                if idm is not None:
                    ax2b.annotate(f"max {df['map_points_mb'].iloc[idm]:.4f} MB", xy=(x_at(idm), df['map_points_mb'].iloc[idm]),
                              xytext=(10, 10), textcoords='offset points', color='tab:red')
            except Exception:
                pass
        # 合并图例
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')
    else:
        ax2.legend(loc='best')
    ax2.set_title('Map point cloud size')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Device memory (CUDA/MPS)
    ax3 = axes[2]
    drew_any = False
    if 'cuda_alloc_mb' in df.columns and df['cuda_alloc_mb'].notna().any():
        ax3.plot(x, df['cuda_alloc_mb'], label='CUDA alloc(MB)', color='tab:purple')
        drew_any = True
    if 'cuda_reserved_mb' in df.columns and df['cuda_reserved_mb'].notna().any():
        ax3.plot(x, df['cuda_reserved_mb'], label='CUDA reserved(MB)', color='tab:pink')
        drew_any = True
    if 'mps_current_mb' in df.columns and df['mps_current_mb'].notna().any():
        ax3.plot(x, df['mps_current_mb'], label='MPS current(MB)', color='tab:brown')
        drew_any = True
    if 'mps_driver_mb' in df.columns and df['mps_driver_mb'].notna().any():
        ax3.plot(x, df['mps_driver_mb'], label='MPS driver(MB)', color='tab:olive')
        drew_any = True
    if args.annotate and drew_any:
        try:
            for col, c in [('cuda_alloc_mb','tab:purple'),('cuda_reserved_mb','tab:pink'),('mps_current_mb','tab:brown'),('mps_driver_mb','tab:olive')]:
                if col in df.columns and df[col].notna().any():
                    idxm = pos_of_max(df[col])
                    if idxm is None:
                        continue
                    ax3.annotate(f"max {col} {df[col].iloc[idxm]:.2f}", xy=(x_at(idxm), df[col].iloc[idxm]),
                                 xytext=(10, 10), textcoords='offset points', color=c)
        except Exception:
            pass
    if not drew_any:
        ax3.text(0.5, 0.5, 'No CUDA/MPS memory data', transform=ax3.transAxes, ha='center', va='center')
    ax3.set_title('Device memory (if available)')
    ax3.set_ylabel('MB')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')

    # x轴格式
    if isinstance(x, pd.Series) and pd.api.types.is_datetime64_any_dtype(x):
        for ax in axes:
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate(rotation=20)
    else:
        for ax in axes:
            ax.set_xlabel('Sample index')

    # 保存与显示
    # 打印一段简明的数值总结
    try:
        def safe(v):
            return None if v is None or (hasattr(v,'__float__') and (v != v)) else v
        rss_min = df['rss_cur_mb'].min() if 'rss_cur_mb' in df.columns else None
        rss_max = df['rss_cur_mb'].max() if 'rss_cur_mb' in df.columns else None
        rss_last = df['rss_cur_mb'].iloc[-1] if 'rss_cur_mb' in df.columns else None
        pts_last = df['map_points'].iloc[-1] if 'map_points' in df.columns else None
        print("Summary:")
        print(f"  rows={len(df)}")
        if 'timestamp' in df.columns:
            print(f"  time: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
        print(f"  RSS current min/max/last (MB): {safe(rss_min)} / {safe(rss_max)} / {safe(rss_last)}")
        if 'rss_peak_mb' in df.columns:
            print(f"  RSS peak max (MB): {safe(df['rss_peak_mb'].max())}")
        print(f"  map_points last/max (count): {safe(pts_last)} / {safe(df['map_points'].max() if 'map_points' in df.columns else None)}")
    except Exception as e:
        print(f"Summary failed: {e}")

    if args.out:
        try:
            out_path = ensure_parent(args.out)
            fig.savefig(out_path, dpi=args.dpi)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Failed to save figure: {e}")
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
