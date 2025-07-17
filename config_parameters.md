# SLAM探索机器人仿真系统 - 参数配置说明

## 系统参数配置概览

本文档列出了SLAM探索机器人仿真系统中所有可配置的参数，这些参数已被提取到文件开头作为常量定义，便于调整和维护。

### 路径规划参数
- `SAFETY_DISTANCE_FACTOR = 0.75` - 路径截断百分比，表示只执行路径的前75%
- `FRONTIER_SAFETY_DISTANCE = 6.0` - 前沿探索器与障碍物的安全距离

### 迷宫和机器人参数
- `MAZE_FILE = "1.json"` - 默认迷宫文件
- `ROBOT_ODOM_NOISE = (0, 0)` - 机器人里程计噪声 (x_noise, y_noise)
- `VIRTUAL_WALL_RESOLUTION_FACTOR = 2` - 虚拟墙分辨率因子
- `VIRTUAL_WALL_Y_OFFSET = -1` - 虚拟墙Y方向偏移

### 激光雷达参数
- `LIDAR_MAX_RANGE = 12.0` - 激光雷达最大探测距离
- `LIDAR_ANGLE_RESOLUTION = 1.0` - 激光雷达角度分辨率
- `LIDAR_NOISE = 0` - 激光雷达噪声

### 出口检测参数
- `EXIT_DETECTION_MIN_ANGLE = 180.0` - 检测出口的最小连续角度范围（度）

### 探索阈值参数
- `MIN_EXPLORATION_DISTANCE = 10.0` - 最小探索距离阈值
- `MIN_FRONTIERS_TO_EXPLORE = 2` - 最小探索前沿数量

### 运动控制精度参数
- `ROTATION_THRESHOLD = 1e-3` - 旋转角度阈值
- `MOVEMENT_THRESHOLD = 1e-6` - 移动距离阈值

### 可视化参数
- `VISUALIZATION_PAUSE_TIME = 0.005` - 暂停时的等待时间
- `VISUALIZATION_UPDATE_TIME = 0.0001` - 可视化更新时间

### 未探索区域搜索参数
- `OBSTACLE_SEARCH_EXPANSION = 5.0` - 障碍物区域搜索范围扩大距离（米）

### 降噪滤波参数
- `NOISE_FILTER_ENABLED = True` - 降噪滤波器总开关
- `LIDAR_FILTER_ENABLED = True` - 是否启用激光雷达降噪（独立控制）
- `LIDAR_FILTER_TYPE = 'median'` - 激光雷达滤波类型: 'none', 'median', 'moving_average', 'gaussian'
- `LIDAR_FILTER_WINDOW_SIZE = 5` - 激光雷达滤波窗口大小
- `ODOM_FILTER_ENABLED = False` - 是否启用里程计降噪（独立控制）
- `ODOM_FILTER_TYPE = 'none'` - 里程计滤波类型: 'none', 'kalman', 'moving_average'
- `ODOM_FILTER_WINDOW_SIZE = 3` - 里程计滤波窗口大小

## 降噪滤波算法说明

### 激光雷达降噪
1. **无滤波 (`none`)**: 不进行任何滤波处理
2. **中值滤波 (`median`)**: 使用历史窗口内数据的中值，有效去除脉冲噪声
3. **移动平均 (`moving_average`)**: 使用历史窗口内数据的均值，平滑数据
4. **高斯滤波 (`gaussian`)**: 对历史数据进行高斯加权，越新的数据权重越大

### 里程计降噪
1. **无滤波 (`none`)**: 不进行任何滤波处理
2. **卡尔曼滤波 (`kalman`)**: 使用卡尔曼滤波器，基于运动模型预测和观测更新
3. **移动平均 (`moving_average`)**: 使用历史数据的移动平均，角度使用圆形平均

### 使用建议
- **只处理激光雷达噪声**: 设置 `LIDAR_FILTER_ENABLED = True`, `ODOM_FILTER_ENABLED = False`
- **只处理里程计噪声**: 设置 `LIDAR_FILTER_ENABLED = False`, `ODOM_FILTER_ENABLED = True`
- **处理所有噪声**: 设置 `LIDAR_FILTER_ENABLED = True`, `ODOM_FILTER_ENABLED = True`
- **禁用所有滤波**: 设置 `NOISE_FILTER_ENABLED = False`
- **高噪声环境**: 启用对应滤波器，使用较大的窗口大小
- **实时性要求高**: 使用较小的窗口大小或禁用对应滤波器
- **精度要求高**: 推荐使用卡尔曼滤波处理里程计数据，中值滤波处理激光雷达数据

## 参数调整建议

### 性能调优
1. **加快仿真速度**：
   - 减少 `VISUALIZATION_UPDATE_TIME` 和 `VISUALIZATION_PAUSE_TIME`
   - 增大 `SAFETY_DISTANCE_FACTOR` 减少路径规划频次

2. **提高探索精度**：
   - 减小 `LIDAR_ANGLE_RESOLUTION`
   - 减小 `ROTATION_THRESHOLD` 和 `MOVEMENT_THRESHOLD`
   - 减小 `FRONTIER_SAFETY_DISTANCE`

3. **调整探索策略**：
   - 修改 `MIN_EXPLORATION_DISTANCE` 和 `MIN_FRONTIERS_TO_EXPLORE` 控制何时开始检测出口
   - 调整 `EXIT_DETECTION_MIN_ANGLE` 改变出口检测敏感度

### 鲁棒性调整
1. **增加噪声模拟**：
   - 设置 `ROBOT_ODOM_NOISE` 和 `LIDAR_NOISE` 模拟真实环境
   
2. **安全性调整**：
   - 增大 `FRONTIER_SAFETY_DISTANCE` 提高安全性
   - 减小 `SAFETY_DISTANCE_FACTOR` 使路径更保守

## 配置文件位置

- 主要参数配置：`main.py` 文件开头
- 可视化参数：`visualizer.py` 文件开头

通过修改这些参数，您可以根据不同的迷宫环境和性能要求调整系统行为。
