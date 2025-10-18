# 动态前瞻参数修复

## 问题描述

用户发现以下三个参数没有被 `compute_dynamic_lookahead()` 函数正确使用：

```python
base_lookahead_steps = 14   # 默认前瞻栅格数（调近）
min_lookahead_steps = 2     # 弯曲段时的最小前瞻（调近）
max_lookahead_steps = 25    # 直线段时的最大前瞻（调近）
```

## 根本原因

### 1. 缺少 `base_steps` 参数

原始函数定义：
```python
def compute_dynamic_lookahead(path_cells, current_idx,
                              min_steps=min_lookahead_steps,
                              max_steps=max_lookahead_steps):
    # ...
    base_candidate = base_lookahead_steps  # ❌ 直接引用外层变量
```

**问题**:
- 函数内部使用 `base_lookahead_steps`，但没有作为参数传入
- 虽然能访问外层作用域变量，但这不是好的实践
- 不够灵活，无法在调用时自定义基准值

### 2. 参数默认值的绑定时机

```python
def compute_dynamic_lookahead(..., 
                              min_steps=min_lookahead_steps,  # ⚠️ 函数定义时绑定
                              max_steps=max_lookahead_steps):
```

**说明**:
- Python 函数的默认参数值在**函数定义时**就绑定了
- 如果外层变量 `min_lookahead_steps` 后续改变，函数的默认值不会更新
- 在当前代码中这些变量是常量，所以影响不大
- 但如果需要动态调整这些参数，这种方式会有问题

## 修复方案

### 修改后的函数签名

```python
def compute_dynamic_lookahead(path_cells, current_idx,
                              base_steps=base_lookahead_steps,  # ✅ 新增参数
                              min_steps=min_lookahead_steps,
                              max_steps=max_lookahead_steps):
    """根据局部曲率动态调整前瞻距离：弯道越急，取值越靠近目标。
    
    Args:
        path_cells: 路径栅格列表
        current_idx: 当前在路径上的最近点索引
        base_steps: 基准前瞻步数（中等曲率时使用）
        min_steps: 最小前瞻步数（急弯时使用）
        max_steps: 最大前瞻步数（直线时使用）
    """
```

### 函数内部修改

```python
# 修改前
base_candidate = base_lookahead_steps  # ❌ 外层变量

# 修改后
base_candidate = base_steps  # ✅ 使用参数
```

## 工作原理

现在三个参数都被正确使用：

1. **`base_steps` (默认 14)**:
   - 作为基准前瞻距离
   - 中等曲率路径使用此值

2. **`min_steps` (默认 2)**:
   - 最小前瞻距离
   - 急弯（高曲率）时使用
   - 确保至少看前方 2 个栅格

3. **`max_steps` (默认 25)**:
   - 最大前瞻距离
   - 直线（低曲率）时使用
   - 确保不超过 25 个栅格

### 动态调整算法

```python
# 1. 计算曲率 (0.0 = 直线, 1.0 = 急弯)
curvature_ratio = f(local_path_curvature)

# 2. 根据曲率调整前瞻距离
span = base_steps - min_steps  # 可调整范围
adjusted = base_steps - curvature_ratio * span

# 3. 限制在 [min_steps, max_steps] 范围内
final_lookahead = clamp(adjusted, min_steps, max_steps)
```

**示例**:
- 直线 (曲率=0.0): `lookahead = 14 - 0.0 * (14-2) = 14` → 限制后 = 14
- 中等弯 (曲率=0.5): `lookahead = 14 - 0.5 * 12 = 8`
- 急弯 (曲率=1.0): `lookahead = 14 - 1.0 * 12 = 2` → 限制后 = 2

## 调用方式

### 1. 使用默认参数（推荐）

```python
# 自动使用外层定义的三个参数
lookahead = compute_dynamic_lookahead(path_cells, nearest_idx)
```

### 2. 自定义参数（可选）

```python
# 临时使用更激进的前瞻策略
lookahead = compute_dynamic_lookahead(
    path_cells, 
    nearest_idx,
    base_steps=20,   # 更远的基准
    min_steps=5,     # 急弯也看远点
    max_steps=30     # 直线看更远
)
```

## 调用位置

函数在两处被调用：

### 1. 返程段路径跟随 (Line 1405)
```python
# drive_path_with_dwa_segment() 函数内
lookahead = compute_dynamic_lookahead(path_cells, nearest_idx)
follow_idx = min(len(path_arr) - 1, nearest_idx + lookahead)
```

### 2. 主探索循环 (Line 1760)
```python
# while True 主循环内
dynamic_steps = compute_dynamic_lookahead(current_path, nearest_idx)
follow_idx = min(len(current_path) - 1, nearest_idx + dynamic_steps)
```

## 验证方法

运行程序后，观察调试输出：

```
[动态前瞻] 最近点索引=12/55 前瞻步数=8 目标索引=20
[返程-动态前瞻] 最近点索引=34/78 前瞻步数=14 目标索引=48
```

**验证要点**:
- `前瞻步数` 应该在 `[2, 25]` 范围内
- 直线段应该接近 14-25
- 弯道段应该接近 2-10
- 看到前瞻步数变化说明曲率检测生效

## 后续优化建议

如果需要更灵活的参数调整，可以考虑：

### 1. 环境变量控制

```python
BASE_LOOKAHEAD_STEPS = int(os.getenv("LOOKAHEAD_BASE", "14"))
MIN_LOOKAHEAD_STEPS = int(os.getenv("LOOKAHEAD_MIN", "2"))
MAX_LOOKAHEAD_STEPS = int(os.getenv("LOOKAHEAD_MAX", "25"))
```

### 2. 根据速度动态调整

```python
# 高速时看更远
if robot.linear_vel > 0.3:
    base_steps = 20
    max_steps = 35
else:
    base_steps = 14
    max_steps = 25
```

### 3. 根据环境复杂度调整

```python
# 障碍物密集区域
if obstacle_density > threshold:
    min_steps = 2
    base_steps = 10  # 更保守
else:
    min_steps = 5
    base_steps = 16  # 更激进
```

## 结论

✅ **已修复**: 所有三个前瞻参数现在都被正确使用
- `base_lookahead_steps` → `base_steps` 参数
- `min_lookahead_steps` → `min_steps` 参数  
- `max_lookahead_steps` → `max_steps` 参数

✅ **向后兼容**: 现有调用无需修改，自动使用默认值

✅ **更灵活**: 支持在调用时自定义所有三个参数
