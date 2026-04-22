# SLAM 小车软件栈

面向小学期 SLAM 小车项目的软件部分，包含：

- 仿真与真实 BLE 小车两套输入链路
- 基于激光雷达和里程计的 ICP SLAM
- 基于 Frontier 的自主探索与回退
- 基于 DWA 的局部避障与路径跟随
- 面向激光序列的时序去噪训练与评估脚本

仓库已经按公开仓库需求做了基础整理：默认使用仿真模式，不再内置真实设备地址，本机路径和运行日志已从代码/配置中移除。

## 快速开始

推荐直接使用 Conda 环境文件：

```bash
conda env create -f conda-environment.yml
conda activate SmallSemester_Slim
```

或使用 `pip`：

```bash
pip install -r requirements.txt
```

默认以仿真模式运行：

```bash
python main.py
```

如果需要接真实 BLE 小车，先复制环境变量模板并填入自己的设备参数：

```bash
cp .env.example .env
```

然后自行导出变量，或手动在当前 shell 中设置：

```bash
export USE_REAL_BLE_DATA=1
export BLE_DEVICE_ADDRESS=YOUR_DEVICE_ADDRESS
export BLE_NOTIFY_CHAR=0000ffe1-0000-1000-8000-00805f9b34fb
export BLE_WRITE_CHAR=0000ffe1-0000-1000-8000-00805f9b34fb
python main.py
```

## 运行模式

- 仿真模式：默认模式，不依赖真实硬件，适合阅读代码、复现实验和调试算法。
- BLE 实机模式：通过 `real_robot_bridge.py` 和 `bluetooth_connection.py` 接收激光/编码器数据并下发控制命令。
- 离线回放模式：读取已经录制的 BLE 日志，便于排查 SLAM 和控制问题。

## 核心算法流程

主流程位于 `main.py`，核心链路如下：

1. 读取地图配置，初始化机器人、雷达、栅格地图和可视化。
2. 从仿真、BLE 实机或离线回放获取一帧激光扫描和运动增量。
3. 可选地对雷达或里程计做滤波。
4. 用 `icp_slam.py` 将当前扫描与历史点云对齐，更新位姿和占据栅格。
5. 用 `frontier_explorer.py` 在已知空闲区边界寻找前沿目标。
6. 用 `fast_path_planner.py` / `dwa.py` 生成可执行路径和局部控制。
7. 通过可视化模块展示地图、路径、前沿、轨迹和控制状态。
8. 探索结束后保存地图结果，或在实机模式下执行回退和停车。

## 文档导航

- [架构说明](docs/ARCHITECTURE.md)
- [文件索引](docs/FILE_GUIDE.md)
- [历史优化笔记](docs/notes/README.md)

## 仓库结构

```text
.
├── main.py                    # 主入口：SLAM + 探索 + 控制闭环
├── icp_slam.py                # ICP 建图与定位
├── frontier_explorer.py       # Frontier 搜索与目标选择
├── dwa.py                     # 主用局部规划器
├── visualizer.py              # Matplotlib 可视化
├── bluetooth_connection.py    # BLE 数据解析与监听
├── real_robot_bridge.py       # BLE 数据适配到导航栈
├── dncnn2d_temporal.py        # 时序去噪网络与推理接口
├── weights/                   # 训练好的示例权重
├── docs/                      # 说明文档与历史优化记录
└── test_*.py                  # 核心功能验证脚本
```

## 说明

- 根目录仍然保持相对扁平，目的是尽量不打断原有脚本的相对路径和使用方式。
- 历史运行产物、缓存、日志和本机路径相关内容不再作为仓库内容的一部分。
- `weights/` 中保留的是项目训练得到的示例模型；如果后续模型继续增大，建议迁移到 Release 或 Git LFS。

## 开源协议

本仓库采用 [MIT License](LICENSE)。
