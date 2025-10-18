# Motion Update 延迟优化

## 问题现象

从日志中观察到 `motion_update` 延迟不稳定:
- **正常情况**: 40-90ms
- **高延迟情况**: 440-500ms (约 10 倍延迟)

## 根本原因

### 1. 阻塞性蓝牙轮询
在 `real_robot_bridge.py` 的 `MotionDataAdapter.poll_motion()` 中:

```python
def poll_motion(self):
    deadline = time.monotonic() + self._timeout  # 默认 0.25s = 250ms
    while True:
        status = get_mpu_status()
        if status is not None:
            result = self._process_status(status)
            if result is not None:
                return result
        if time.monotonic() >= deadline:
            return 0.0, 0.0, None  # 超时返回零运动
        time.sleep(self._poll_interval)  # 每 10ms 轮询一次
```

**问题**:
- `poll_motion()` 会**阻塞等待**机器人返回新的编码器数据
- 如果蓝牙通信延迟/数据丢失,会一直等待直到 **250ms 超时**
- 超时后返回 `(0.0, 0.0, None)`,相当于机器人停止,与实际不符

### 2. 与 ICP 失败的关联

从日志规律可见:
- **ICP iters=200** (收敛失败) ⇒ `motion_update=440-500ms`
- **ICP iters 小**(4-20) ⇒ `motion_update=40-90ms`

**原因**:
- ICP 迭代 200 次说明机器人在快速运动/剧烈转向
- 此时编码器数据可能延迟/丢包
- `poll_motion()` 等待超时 250ms,导致延迟激增

## 解决方案

### 1. 缩短超时时间

```python
# main.py line 91
BLE_ENCODER_TIMEOUT = float(os.getenv("BLE_ENCODER_TIMEOUT", "0.08"))  # 250ms → 80ms
```

**效果**: 超时延迟从 250ms 降到 80ms

### 2. 添加运动数据缓存

```python
# real_robot_bridge.py MotionDataAdapter
def __init__(self, ...):
    # ...
    # Cache last valid motion to avoid blocking on timeout
    self._last_valid_motion: Tuple[float, float, Optional[Tuple[float, float]]] = (0.0, 0.0, None)

def poll_motion(self):
    deadline = time.monotonic() + self._timeout
    while True:
        status = get_mpu_status()
        if status is not None:
            result = self._process_status(status)
            if result is not None:
                self._last_valid_motion = result  # ✅ 缓存有效数据
                return result
        if time.monotonic() >= deadline:
            return self._last_valid_motion  # ✅ 返回缓存而非零运动
        time.sleep(self._poll_interval)
```

**效果**:
- 超时时返回**上一次有效的运动数据**,而非 `(0.0, 0.0, None)`
- 保持机器人运动连续性,避免突然"冻结"
- 即使蓝牙短暂延迟,系统仍能基于最近状态继续规划

### 3. 添加超时警告

```python
def poll_motion(self):
    start_time = time.monotonic()
    # ...
    if time.monotonic() >= deadline:
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        if elapsed_ms > 50.0:  # 超过 50ms 就警告
            print(f"[WARN] poll_motion timeout after {elapsed_ms:.1f}ms, using cached data")
        return self._last_valid_motion
```

**效果**: 方便调试,知道何时发生蓝牙延迟

## 预期改善

### Before (旧超时 250ms + 无缓存)
- 蓝牙延迟 ⇒ 阻塞 250ms ⇒ 返回零运动 ⇒ 规划混乱
- `motion_update` 延迟: 440-500ms

### After (新超时 80ms + 有缓存)
- 蓝牙延迟 ⇒ 阻塞最多 80ms ⇒ 返回缓存运动 ⇒ 规划连续
- `motion_update` 延迟预计: **60-120ms** (降低 70-80%)

## 测试验证

运行系统后观察:
1. `motion_update` 延迟是否稳定在 100ms 以内
2. 是否出现 `[WARN] poll_motion timeout` 警告
3. 如果频繁超时,可能需要:
   - 优化蓝牙通信频率
   - 检查机器人编码器数据发送频率
   - 进一步降低 `timeout` (如降到 50ms)

## 权衡考虑

**优点**:
- ✅ 显著降低 `motion_update` 延迟峰值
- ✅ 保持运动连续性,避免突然停止
- ✅ 系统对蓝牙延迟更鲁棒

**潜在风险**:
- ⚠️ 如果机器人**真的停止**但数据延迟,会使用缓存的"运动中"数据
- ⚠️ 需要确保 80ms 超时足够接收正常数据(取决于蓝牙通信频率)

**建议**:
- 监控超时警告频率,如果每步都超时,说明 80ms 太短
- 如果机器人停止检测有问题,可以增加"零运动"检测逻辑
