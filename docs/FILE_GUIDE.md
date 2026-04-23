# 文件索引

下面按职责分组说明仓库中的主要文件用途。为避免把运行日志和缓存也混进来，这里只描述应该保留和阅读的源码、配置与模型文件。

## 入口与核心模块

| 文件 | 作用 |
| --- | --- |
| `main.py` | 项目主入口，负责模式切换、主循环、探索状态机、建图和控制协同。 |
| `maze_loader.py` | 加载 JSON / 文本迷宫配置，生成墙体、起点和占据栅格画布。 |
| `robot.py` | 差分底盘仿真模型，维护真实位姿、里程计位姿和速度积分。 |
| `lidar.py` | 2D 激光雷达模拟器，输出带噪和无噪扫描。 |
| `icp_slam.py` | 基于运动先验 + ICP 的定位与建图实现，支持 CPU / GPU。 |
| `frontier_explorer.py` | 前沿检测、聚类、缓存、可达性判断和目标选择。 |
| `fast_path_planner.py` | 栅格路径规划器，按距离自动选择 BFS / 双向 BFS / A*。 |
| `dwa.py` | 实际使用的 DWA 局部规划器，包含大量工程化控制约束。 |
| `dynamic_window_approach.py` | 教材式 DWA 参考实现，用于对照和实验。 |
| `grid_system.py` | 将环境划分为更高层的逻辑网格，辅助区域搜索与展示。 |
| `noise_filter.py` | 激光与里程计噪声处理。 |
| `output_paths.py` | 统一管理日志、可视化图像和数据导出的默认目录。 |

## 可视化与性能分析

| 文件 | 作用 |
| --- | --- |
| `visualizer.py` | 基础 Matplotlib 可视化，支持地图、轨迹、前沿、快捷键。 |
| `async_visualizer.py` | 线程版异步可视化包装。 |
| `multiprocess_visualizer.py` | 多进程可视化版本。 |
| `shm_visualizer.py` | 基于共享内存的高性能可视化版本。 |
| `analyze_viz_latency.py` | 可视化延迟分析脚本。 |
| `benchmark_viz_modes.py` | 对比不同可视化模式的吞吐和延迟。 |
| `plot_ble_scan.py` | 读取 BLE 激光日志并快速画图。 |
| `plot_mem_usage.py` | 读取内存监控 CSV 并输出图表。 |

## 实机 BLE 接入

| 文件 | 作用 |
| --- | --- |
| `bluetooth_connection.py` | BLE 设备扫描、连接、通知解析、日志写出和后台监听。 |
| `real_robot_bridge.py` | 把 BLE 激光/编码器数据转换成主循环可消费的数据结构。 |
| `connection_test.py` | BLE 联调和连通性测试工具。 |
| `record_cmd_run.py` | 短时发送运动命令并录制 BLE 数据。 |
| `extract_mpu_raw.py` | 从原始 BLE 日志里提取 MPU 相关帧。 |
| `extract_mpu_parsed.py` | 从解析后的日志里导出结构化 MPU 数据。 |

## 去噪模型与训练评估

| 文件 | 作用 |
| --- | --- |
| `dncnn2d_temporal.py` | 时序 DnCNN 模型、注意力块、数据集与推理接口。 |
| `train_my_sim.py` | 用项目内地图和雷达模拟器训练去噪模型。 |
| `evaluate_denoise.py` | 评估单模型去噪效果并输出示例图。 |
| `evaluate_multi_noise.py` | 比较多组模型在不同噪声水平下的性能。 |
| `eval_multi_noise.py` | 面向多噪声统一模型的评估脚本。 |
| `data_collect.py` | 生成或采集训练/测试过程用的数据。 |

## 配置、地图与权重

| 文件 / 目录 | 作用 |
| --- | --- |
| `1.json` `2.json` `3.json` `3_scaled.json` `4.json` | 不同实验用迷宫/墙体配置。 |
| `maze.json` | 额外示例地图配置。 |
| `maze_config.txt` | 文本格式的旧版地图配置示例。 |
| `weights/` | 训练得到的公开示例权重。 |
| `conda-environment.yml` | 推荐的 Conda 环境定义。 |
| `requirements.txt` | 简化版 pip 依赖清单。 |
| `.env.example` | 实机 BLE 和回放模式的环境变量模板。 |

## 测试与验证脚本

| 文件 | 作用 |
| --- | --- |
| `test_astar_failure_recovery.py` | 验证 A* 失败后的前沿恢复策略。 |
| `test_frontier_refresh.py` | 验证前沿增量刷新与全量刷新。 |
| `test_icp_negative_values.py` | 验证 ICP 和带符号位移处理。 |
| `test_negative_delta.py` | 验证编码器负增量处理。 |
| `test_unwrap_16bit.py` | 验证 16 位编码器回绕展开逻辑。 |
| `verify_icp_negative_handling.py` | 进一步检查 ICP 对负位移和累计量的处理。 |

## 杂项

| 文件 | 作用 |
| --- | --- |
| `scale_maze.py` | 对地图配置做缩放变换。 |
| `sitecustomize.py` | 预留给本地 Python 启动定制，目前为空。 |

## 历史文档

课程迭代过程中产出的优化分析和调参记录已经移动到 [`docs/notes/`](notes/README.md)。
