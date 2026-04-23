# 历史优化笔记

这里保留项目开发过程中形成的调参与问题分析记录。它们不是首次阅读仓库必须先看的内容，但对理解项目演进过程和某些参数为什么这样设置很有帮助。

## 索引

- `config_parameters.md`：系统参数与调参建议。
- `dynamic_lookahead_fix.md`：探索阶段动态前瞻索引修复说明。
- `dynamic_lookahead_parameters_fix.md`：动态前瞻参数使用方式修复说明。
- `frontier_optimization.md`：前沿搜索性能优化记录。
- `frontier_refresh_optimization.md`：A* 失败后前沿刷新与恢复策略。
- `icp_iterations_analysis.md`：ICP 迭代次数波动原因分析。
- `keyboard_shortcuts.md`：可视化窗口快捷键说明。
- `noise_filter_usage.md`：激光和里程计滤波开关用法。
- `robustness.md`：去噪模型在不同噪声下的鲁棒性结果。

## 阅读顺序建议

1. 先看 `README.md` 和 `docs/ARCHITECTURE.md` 建立整体概念。
2. 需要调参时再查 `config_parameters.md`。
3. 需要定位历史设计决策时，按问题域查看对应笔记。
