# 降噪滤波器独立控制使用说明

## 概述

降噪滤波器现在支持激光雷达和里程计的**独立控制**，您可以：
- 只处理激光雷达噪声，不处理里程计噪声
- 只处理里程计噪声，不处理激光雷达噪声  
- 同时处理两种噪声
- 完全禁用噪声处理

## 参数配置

### 主控开关
```python
NOISE_FILTER_ENABLED = True  # 降噪滤波器总开关
```

### 激光雷达降噪控制
```python
LIDAR_FILTER_ENABLED = True  # 是否启用激光雷达降噪
LIDAR_FILTER_TYPE = 'median'  # 滤波类型
LIDAR_FILTER_WINDOW_SIZE = 5  # 滤波窗口大小
```

**激光雷达滤波类型说明：**
- `'none'`: 不进行滤波
- `'median'`: 中值滤波（推荐，有效去除脉冲噪声）
- `'moving_average'`: 移动平均滤波（平滑数据）
- `'gaussian'`: 高斯加权滤波（给新数据更高权重）

### 里程计降噪控制
```python
ODOM_FILTER_ENABLED = False  # 是否启用里程计降噪
ODOM_FILTER_TYPE = 'none'   # 滤波类型
ODOM_FILTER_WINDOW_SIZE = 3  # 滤波窗口大小
```

**里程计滤波类型说明：**
- `'none'`: 不进行滤波
- `'kalman'`: 卡尔曼滤波（推荐，基于运动模型）
- `'moving_average'`: 移动平均滤波

## 常用配置示例

### 1. 只处理激光雷达噪声（当前配置）
```python
NOISE_FILTER_ENABLED = True
LIDAR_FILTER_ENABLED = True
LIDAR_FILTER_TYPE = 'median'
LIDAR_FILTER_WINDOW_SIZE = 5

ODOM_FILTER_ENABLED = False
ODOM_FILTER_TYPE = 'none'
```

### 2. 只处理里程计噪声
```python
NOISE_FILTER_ENABLED = True
LIDAR_FILTER_ENABLED = False
LIDAR_FILTER_TYPE = 'none'

ODOM_FILTER_ENABLED = True
ODOM_FILTER_TYPE = 'kalman'
ODOM_FILTER_WINDOW_SIZE = 3
```

### 3. 处理所有噪声
```python
NOISE_FILTER_ENABLED = True
LIDAR_FILTER_ENABLED = True
LIDAR_FILTER_TYPE = 'median'
LIDAR_FILTER_WINDOW_SIZE = 5

ODOM_FILTER_ENABLED = True
ODOM_FILTER_TYPE = 'kalman'
ODOM_FILTER_WINDOW_SIZE = 3
```

### 4. 禁用所有降噪
```python
NOISE_FILTER_ENABLED = False
```

## 性能影响

### 激光雷达滤波
- **优点**: 减少传感器噪声对SLAM建图的影响，提高地图质量
- **计算开销**: 很小，主要是简单的数值运算
- **实时性**: 基本无影响

### 里程计滤波
- **优点**: 平滑机器人运动轨迹，减少定位误差
- **计算开销**: 卡尔曼滤波有一定计算量
- **实时性**: 轻微影响

## 调试和验证

### 查看滤波器状态
运行程序时会自动输出滤波器状态：
```
[NoiseFilter] 滤波器初始化 - 状态: 启用
[NoiseFilter] 激光雷达滤波: 启用 - 类型: median, 窗口大小: 5
[NoiseFilter] 里程计滤波: 禁用 - 类型: none, 窗口大小: 3
```

### 测试脚本
- `test_lidar_only_filter.py`: 测试仅激光雷达滤波功能
- `test_noise_filter.py`: 测试完整滤波功能

### 验证方法
1. 观察SLAM建图质量是否改善
2. 检查机器人轨迹是否更平滑
3. 比较滤波前后的传感器数据噪声水平

## 建议设置

根据您的需求选择合适的配置：

**一般使用（推荐当前设置）：**
```python
LIDAR_FILTER_ENABLED = True   # 启用激光雷达滤波
ODOM_FILTER_ENABLED = False   # 禁用里程计滤波
```

**高噪声环境：**
```python
LIDAR_FILTER_ENABLED = True
LIDAR_FILTER_WINDOW_SIZE = 7  # 增大窗口

ODOM_FILTER_ENABLED = True
ODOM_FILTER_TYPE = 'kalman'
```

**实时性要求极高：**
```python
NOISE_FILTER_ENABLED = False  # 完全禁用
```

**调试模式：**
```python
LIDAR_FILTER_ENABLED = False
ODOM_FILTER_ENABLED = False
```
然后逐个启用观察效果。
