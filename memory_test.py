#!/usr/bin/env python3
"""
内存管理测试脚本
用于验证内存释放是否正常工作
"""

import time
import gc
import psutil
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # 转换为MB

def cleanup_memory():
    """清理内存"""
    gc.collect()  # Python垃圾回收
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空CUDA缓存
    except:
        pass

def test_memory_management():
    print(f"开始内存使用: {get_memory_usage():.1f}MB")
    
    # 模拟大量数据创建和释放
    for i in range(5):
        print(f"\n=== 测试轮次 {i+1} ===")
        
        # 创建大量numpy数组（模拟地图点云）
        data_arrays = []
        for j in range(100):
            arr = np.random.rand(1000, 2)  # 1000个2D点
            data_arrays.append(arr)
        
        print(f"创建大量数组后内存: {get_memory_usage():.1f}MB")
        
        # 转换为PyTorch张量（模拟GPU操作）
        if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            device = torch.device("cuda" if torch.cuda.is_available() else "mps")
            tensors = []
            for arr in data_arrays:
                tensor = torch.tensor(arr, dtype=torch.float32, device=device)
                tensors.append(tensor)
            print(f"转换为GPU张量后内存: {get_memory_usage():.1f}MB")
            
            # 显式删除张量
            for tensor in tensors:
                del tensor
            del tensors
        
        # 删除数组
        for arr in data_arrays:
            del arr
        del data_arrays
        
        # 强制垃圾回收
        cleanup_memory()
        
        print(f"清理后内存: {get_memory_usage():.1f}MB")
        
        time.sleep(1)  # 等待系统完成清理
    
    # 创建和清理matplotlib图形
    print(f"\n=== 测试matplotlib内存 ===")
    figures = []
    for i in range(10):
        fig, ax = plt.subplots(figsize=(10, 10))
        # 绘制大量数据
        x = np.random.rand(10000)
        y = np.random.rand(10000)
        ax.scatter(x, y, s=1)
        figures.append(fig)
    
    print(f"创建10个大图后内存: {get_memory_usage():.1f}MB")
    
    # 清理图形
    for fig in figures:
        plt.close(fig)
        del fig
    del figures
    plt.close('all')
    
    cleanup_memory()
    print(f"清理matplotlib后内存: {get_memory_usage():.1f}MB")
    
    print(f"\n最终内存使用: {get_memory_usage():.1f}MB")

if __name__ == "__main__":
    test_memory_management()
