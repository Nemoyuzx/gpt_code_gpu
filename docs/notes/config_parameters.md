# SLAM探索机器人仿真系统 - 参数配置说明

## 系统参数配置概览

### 路径规划参数
- `FRONTIER_SAFETY_DISTANCE_METERS = ROBOT_COLLISION_RADIUS + BASE_SAFETY_CLEARANCE` - 前沿候选点必须距离障碍至少该安全距离才会被选取
- 前沿搜索在路径与目标选择阶段都会滤除不满足安全距离的栅格
- 返程阶段复用探索阶段的 DWA 路径跟随逻辑，以保持一致的轨迹控制特性

### 迷宫和机器人参数
- `VIRTUAL_WALL_RESOLUTION_FACTOR = 2` - 虚拟墙分辨率因子
- `VIRTUAL_WALL_Y_OFFSET = -1` - 虚拟墙Y方向偏移
- `OCCUPANCY_GRID_RESOLUTION = 0.02` - 占据栅格地图的分辨率（米/格）

### 激光雷达参数
- `LIDAR_ANGLE_RESOLUTION = 1.0` - 激光雷达角度分辨率
- `LIDAR_NOISE = 0.01` - 激光雷达噪声
`STUCK_WINDOW_STEPS = 16` - 判定窗口步数
`STUCK_SPIN_W_THRESH = 1.1` - 认为“原地打转”的角速度阈值(rad/s)
`STUCK_V_SMALL = 0.03` - 认为“几乎不前进”的线速度阈值(m/s)
`STUCK_PROGRESS_EPS = 0.06` - 判定窗口内总位移阈值(m)
### 探索阈值参数
- `MIN_EXPLORATION_DISTANCE = 10.0` - 最小探索距离阈值
- `MIN_FRONTIERS_TO_EXPLORE = 2` - 最小探索前沿数量

### 运动控制精度参数
- `ROTATION_THRESHOLD = 1e-3` - 旋转角度阈值
- `MOVEMENT_THRESHOLD = 1e-6` - 移动距离阈值

### DWA倒车相关参数
- `reverse_heading_threshold ≈ 90°` - 朝向偏差接近90°才允许常规倒车
- `reverse_deadband = 0.22` - 低于该速度幅度的倒车会被抑制
- `reverse_min_speed_scale = 0.7` - 倒车的最小速度比例（相对于前进最小速度）
- `reverse_rot_cost_scale = 0.85` / `reverse_spin_penalty_scale = 0.85` - 倒车时的转向代价/打转惩罚缩放，调高以减少倒车吸引力
- `reverse_turn_bonus_gain = 0.12` - 倒车转弯奖励，保持较低
- `reverse_continue_penalty_gain = 2.2` - 已校正后继续倒车的惩罚
- `direct_reverse_reward_gain = 0.3` - 直接倒车奖励权重，再次调低
- `direct_reverse_gap_threshold = 0.15m` - 仅当前向净空小于该值时触发直接倒车奖励
- `forward_pref_cost_gain = 0.45` - 前向净空充足且角度小仍倒车时的惩罚增益
- `forward_pref_gap_thresh = 0.50` - 仅当前向净空超过该值时才应用前进偏好惩罚

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
   - 减小 `FRONTIER_SAFETY_DISTANCE_METERS`

3. **调整探索策略**：
   - 修改 `MIN_EXPLORATION_DISTANCE` 和 `MIN_FRONTIERS_TO_EXPLORE` 控制何时开始检测出口
   - 调整 `EXIT_DETECTION_MIN_ANGLE` 改变出口检测敏感度

### 鲁棒性调整
1. **增加噪声模拟**：
   - 设置 `ROBOT_ODOM_NOISE` 和 `LIDAR_NOISE` 模拟真实环境
   
2. **安全性调整**：
   - 增大 `FRONTIER_SAFETY_DISTANCE_METERS` 提高安全性
   - 减小 `SAFETY_DISTANCE_FACTOR` 使路径更保守

## 配置文件位置

- 主要参数配置：`main.py` 文件开头
- 可视化参数：`visualizer.py` 文件开头

通过修改这些参数，您可以根据不同的迷宫环境和性能要求调整系统行为。
