"""
基于共享内存的多进程可视化模块
使用 SharedMemory 实现零拷贝数据传输，显著降低可视化延迟
"""
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
from typing import Optional, Any, Dict, Tuple
import sys


class SharedMemoryVisualizer:
    """
    使用共享内存的可视化器
    大型数据（occupancy, unsafe_mask）通过共享内存传输
    小型数据（pose, scan, paths等）通过队列传输
    """
    
    def __init__(self, maze, robot=None, slam=None, max_queue_size: int = 3):
        # 命令队列（用于小型数据和控制命令）
        self.command_queue = mp.Queue(maxsize=max_queue_size)
        self.response_queue = mp.Queue(maxsize=10)
        self.process: Optional[mp.Process] = None
        
        # 估算共享内存大小（基于栅格地图）
        # 假设最大地图尺寸为 1000x1000
        self.max_grid_size = 1000
        
        # occupancy: int8 (1 byte per cell, values: -1, 0, 1)
        self.occupancy_size = self.max_grid_size * self.max_grid_size * np.dtype(np.int8).itemsize
        
        # unsafe_mask: bool (1 byte per cell)
        self.unsafe_mask_size = self.max_grid_size * self.max_grid_size * np.dtype(np.bool_).itemsize
        
        # 创建共享内存块
        try:
            self.shm_occupancy = shared_memory.SharedMemory(
                create=True, 
                size=self.occupancy_size,
                name=f"viz_occupancy_{id(self)}"
            )
            self.shm_unsafe_mask = shared_memory.SharedMemory(
                create=True,
                size=self.unsafe_mask_size,
                name=f"viz_unsafe_{id(self)}"
            )
            print(f"[SHM VIZ] Created shared memory: occupancy={self.occupancy_size/1024:.1f}KB, "
                  f"unsafe_mask={self.unsafe_mask_size/1024:.1f}KB")
        except Exception as e:
            print(f"[SHM VIZ ERROR] Failed to create shared memory: {e}")
            raise
        
        # 初始化共享内存（填充-1表示未知）
        occupancy_array = np.ndarray(
            (self.max_grid_size, self.max_grid_size),
            dtype=np.int8,
            buffer=self.shm_occupancy.buf
        )
        occupancy_array.fill(-1)
        
        unsafe_array = np.ndarray(
            (self.max_grid_size, self.max_grid_size),
            dtype=np.bool_,
            buffer=self.shm_unsafe_mask.buf
        )
        unsafe_array.fill(False)
        
        # 存储共享内存名称，传递给子进程
        self.init_data = {
            'maze': maze,
            'shm_occupancy_name': self.shm_occupancy.name,
            'shm_unsafe_mask_name': self.shm_unsafe_mask.name,
            'max_grid_size': self.max_grid_size,
        }
        
        # 当前地图尺寸（会在update时更新）
        self.current_shape = (0, 0)
        
        # 类型存根
        self.paused: bool = False
        self.fig: Any = None
        
    def start(self):
        """启动可视化进程"""
        if self.process is not None and self.process.is_alive():
            return
            
        self.process = mp.Process(
            target=self._visualization_process_main,
            args=(self.command_queue, self.response_queue, self.init_data),
            daemon=False
        )
        self.process.start()
        print("[SHM VIZ] Visualization process started")
        
    def stop(self):
        """停止可视化进程并清理共享内存"""
        if self.process is not None:
            try:
                # 发送停止命令
                self.command_queue.put(('stop', None), timeout=1.0)
                # 等待进程结束
                self.process.join(timeout=3.0)
                
                if self.process.is_alive():
                    print("[SHM VIZ] Force terminating visualization process")
                    self.process.terminate()
                    self.process.join(timeout=1.0)
                    
            except Exception as e:
                print(f"[SHM VIZ ERROR] Error stopping process: {e}")
                if self.process.is_alive():
                    self.process.kill()
                    
            self.process = None
        
        # 清理共享内存
        try:
            self.shm_occupancy.close()
            self.shm_occupancy.unlink()
            print("[SHM VIZ] Occupancy shared memory cleaned up")
        except Exception as e:
            print(f"[SHM VIZ WARN] Error cleaning occupancy memory: {e}")
            
        try:
            self.shm_unsafe_mask.close()
            self.shm_unsafe_mask.unlink()
            print("[SHM VIZ] Unsafe mask shared memory cleaned up")
        except Exception as e:
            print(f"[SHM VIZ WARN] Error cleaning unsafe mask memory: {e}")
            
        print("[SHM VIZ] Visualization process stopped")
        
    def update(self, robot_pose, scan, frontiers=None, target=None, path=None, occupancy=None,
               predicted_traj=None, robot_radius=None, safety_radius=None,
               actual_traj=None, actual_traj_style=None, extra_trajs=None,
               scan_angles=None, unsafe_mask=None, dwa_eval_paths=None):
        """
        更新可视化
        大型数据通过共享内存传输，小型数据通过队列传输
        """
        import time
        t_start = time.perf_counter()
        
        if self.process is None or not self.process.is_alive():
            return
        
        # 将大型数据写入共享内存
        t_shm_write = time.perf_counter()
        shape = (0, 0)
        if occupancy is not None:
            shape = occupancy.shape
            if shape[0] <= self.max_grid_size and shape[1] <= self.max_grid_size:
                # 写入占用网格（使用切片赋值，避免创建临时数组）
                shm_array = np.ndarray(
                    (self.max_grid_size, self.max_grid_size),
                    dtype=np.int8,
                    buffer=self.shm_occupancy.buf
                )
                # 只在数据类型不同时才转换
                if occupancy.dtype == np.int8:
                    shm_array[:shape[0], :shape[1]] = occupancy
                else:
                    shm_array[:shape[0], :shape[1]] = occupancy.astype(np.int8)
                self.current_shape = shape
            else:
                print(f"[SHM VIZ WARN] Occupancy shape {shape} exceeds max size {self.max_grid_size}")
                shape = (0, 0)
        
        if unsafe_mask is not None and shape != (0, 0):
            if unsafe_mask.shape == shape:
                # 写入不安全区域掩码
                shm_array = np.ndarray(
                    (self.max_grid_size, self.max_grid_size),
                    dtype=np.bool_,
                    buffer=self.shm_unsafe_mask.buf
                )
                shm_array[:shape[0], :shape[1]] = unsafe_mask
        
        shm_write_time = (time.perf_counter() - t_shm_write) * 1000.0
        
        # 小型数据通过队列传输
        t_queue = time.perf_counter()
        data = {
            'robot_pose': robot_pose,
            'scan': scan,
            'frontiers': frontiers,
            'target': target,
            'path': path,
            'occupancy_shape': shape,  # 告诉子进程共享内存中的有效数据大小
            'predicted_traj': predicted_traj,
            'robot_radius': robot_radius,
            'safety_radius': safety_radius,
            'actual_traj': actual_traj,
            'actual_traj_style': actual_traj_style,
            'extra_trajs': extra_trajs,
            'scan_angles': scan_angles,
            'has_unsafe_mask': unsafe_mask is not None,
            'dwa_eval_paths': dwa_eval_paths,
        }
        
        # 如果队列满了，移除旧数据
        queue_dropped = False
        if self.command_queue.full():
            try:
                self.command_queue.get_nowait()
                queue_dropped = True
            except:
                pass
        
        queue_blocked = False
        try:
            self.command_queue.put_nowait(('update', data))
        except:
            # 队列满了就跳过这一帧
            queue_blocked = True
        
        queue_time = (time.perf_counter() - t_queue) * 1000.0
        total_time = (time.perf_counter() - t_start) * 1000.0
        
        # 如果更新耗时超过阈值，打印警告
        if total_time > 10.0 or queue_dropped or queue_blocked:
            print(f"[SHM VIZ] update: {total_time:.2f}ms (shm_write={shm_write_time:.2f}ms, "
                  f"queue={queue_time:.2f}ms, dropped={queue_dropped}, blocked={queue_blocked})")
    
    def __getattr__(self, name: str) -> Any:
        """
        动态代理其他方法调用到子进程
        """
        def method_proxy(*args, **kwargs):
            if self.process is None or not self.process.is_alive():
                return None
            
            try:
                self.command_queue.put(
                    ('call_method', {'method': name, 'args': args, 'kwargs': kwargs}),
                    timeout=0.5
                )
                
                # 等待响应
                try:
                    status, result = self.response_queue.get(timeout=1.0)
                    if status == 'success':
                        return result
                    else:
                        print(f"[SHM VIZ ERROR] Method {name} failed: {result}")
                        return None
                except:
                    return None
            except:
                return None
        
        return method_proxy
    
    def _call_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        调用子进程中visualizer的方法（内部辅助方法）
        """
        if self.process is None or not self.process.is_alive():
            return None
        
        try:
            self.command_queue.put(
                ('call_method', {'method': method_name, 'args': args, 'kwargs': kwargs}),
                timeout=0.5
            )
            
            # 等待响应
            try:
                status, result = self.response_queue.get(timeout=1.0)
                if status == 'success':
                    return result
                else:
                    print(f"[SHM VIZ ERROR] Method {method_name} failed: {result}")
                    return None
            except:
                return None
        except:
            return None
    
    # 显式定义常用方法（实际调用_call_method）
    def set_grid_system(self, grid_system, target_cell_id):
        """设置网格系统"""
        return self._call_method('set_grid_system', grid_system, target_cell_id)
    
    def save_map(self, filename):
        """保存地图"""
        return self._call_method('save_map', filename)
    
    def check_and_clear_force_stop(self) -> bool:
        """检查并清除强制停止标志"""
        result = self._call_method('check_and_clear_force_stop')
        return result if result is not None else False
    
    def set_bfs_debug_points(self, points, color='cyan'):
        """设置BFS调试点"""
        return self._call_method('set_bfs_debug_points', points, color)
    
    def set_emergency_path(self, path):
        """设置紧急路径"""
        return self._call_method('set_emergency_path', path)
    
    def set_obstacle_search_region(self, region):
        """设置障碍物搜索区域"""
        return self._call_method('set_obstacle_search_region', region)
    
    @staticmethod
    def _visualization_process_main(command_queue: mp.Queue, response_queue: mp.Queue, init_data: Dict):
        """可视化进程的主函数（在子进程中运行）"""
        shm_occupancy = None
        shm_unsafe_mask = None
        
        try:
            # 配置matplotlib
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            from visualizer import Visualizer
            
            # 在子进程中附加到共享内存
            shm_occupancy = shared_memory.SharedMemory(
                name=init_data['shm_occupancy_name']
            )
            shm_unsafe_mask = shared_memory.SharedMemory(
                name=init_data['shm_unsafe_mask_name']
            )
            
            max_grid_size = init_data['max_grid_size']
            
            print(f"[SHM VIZ PROCESS] Attached to shared memory: "
                  f"occupancy={shm_occupancy.name}, unsafe_mask={shm_unsafe_mask.name}")
            
            # 创建Visualizer
            maze = init_data['maze']
            viz = Visualizer(maze, robot=None, slam=None)
            print("[SHM VIZ PROCESS] Visualizer initialized")
            
            # 主循环：处理命令
            while True:
                try:
                    # 等待命令
                    try:
                        command, data = command_queue.get(timeout=0.1)
                    except:
                        # 超时，检查窗口是否关闭
                        if not plt.fignum_exists(viz.fig.number):
                            print("[SHM VIZ PROCESS] Window closed, exiting")
                            break
                        continue
                    
                    # 处理停止命令
                    if command == 'stop':
                        print("[SHM VIZ PROCESS] Received stop command")
                        break
                    
                    # 处理更新命令
                    elif command == 'update':
                        # 从共享内存读取大型数据
                        occupancy_shape = data.get('occupancy_shape', (0, 0))
                        
                        occupancy = None
                        unsafe_mask = None
                        
                        if occupancy_shape != (0, 0):
                            # 从共享内存读取occupancy
                            shm_array = np.ndarray(
                                (max_grid_size, max_grid_size),
                                dtype=np.int8,
                                buffer=shm_occupancy.buf
                            )
                            # 只复制有效部分（避免复制整个缓冲区）
                            occupancy = shm_array[:occupancy_shape[0], :occupancy_shape[1]].copy()
                            
                            # 从共享内存读取unsafe_mask（如果有）
                            if data.get('has_unsafe_mask', False):
                                shm_array = np.ndarray(
                                    (max_grid_size, max_grid_size),
                                    dtype=np.bool_,
                                    buffer=shm_unsafe_mask.buf
                                )
                                unsafe_mask = shm_array[:occupancy_shape[0], :occupancy_shape[1]].copy()
                        
                        # 从队列获取小型数据
                        viz.update(
                            robot_pose=data['robot_pose'],
                            scan=data['scan'],
                            frontiers=data.get('frontiers'),
                            target=data.get('target'),
                            path=data.get('path'),
                            occupancy=occupancy,
                            predicted_traj=data.get('predicted_traj'),
                            robot_radius=data.get('robot_radius'),
                            safety_radius=data.get('safety_radius'),
                            actual_traj=data.get('actual_traj'),
                            actual_traj_style=data.get('actual_traj_style'),
                            extra_trajs=data.get('extra_trajs'),
                            scan_angles=data.get('scan_angles'),
                            unsafe_mask=unsafe_mask,
                            dwa_eval_paths=data.get('dwa_eval_paths'),
                        )
                        
                    # 处理方法调用
                    elif command == 'call_method':
                        method_name = data['method']
                        args = data.get('args', ())
                        kwargs = data.get('kwargs', {})
                        
                        if hasattr(viz, method_name):
                            method = getattr(viz, method_name)
                            result = method(*args, **kwargs)
                            response_queue.put(('success', result))
                        else:
                            response_queue.put(('error', f"Method {method_name} not found"))
                    
                    # 处理属性获取
                    elif command == 'get_attr':
                        attr_name = data['attr']
                        if hasattr(viz, attr_name):
                            value = getattr(viz, attr_name)
                            response_queue.put(('success', value))
                        else:
                            response_queue.put(('error', f"Attribute {attr_name} not found"))
                    
                except KeyboardInterrupt:
                    print("[SHM VIZ PROCESS] Keyboard interrupt")
                    break
                except Exception as e:
                    print(f"[SHM VIZ PROCESS ERROR] {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"[SHM VIZ PROCESS FATAL] {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理：关闭共享内存（但不unlink，由主进程负责）
            if shm_occupancy is not None:
                try:
                    shm_occupancy.close()
                    print("[SHM VIZ PROCESS] Closed occupancy shared memory")
                except Exception as e:
                    print(f"[SHM VIZ PROCESS ERROR] Error closing occupancy memory: {e}")
                    
            if shm_unsafe_mask is not None:
                try:
                    shm_unsafe_mask.close()
                    print("[SHM VIZ PROCESS] Closed unsafe mask shared memory")
                except Exception as e:
                    print(f"[SHM VIZ PROCESS ERROR] Error closing unsafe mask memory: {e}")
            
            try:
                plt.close('all')
            except:
                pass
            print("[SHM VIZ PROCESS] Visualization process exiting")


# 为了兼容性，创建一个工厂函数
def create_visualizer(maze, robot=None, slam=None, max_queue_size: int = 3, use_shm: bool = True):
    """
    工厂函数：根据配置创建合适的可视化器
    
    Args:
        use_shm: 是否使用共享内存（默认True，显著降低延迟）
    """
    if use_shm:
        try:
            return SharedMemoryVisualizer(maze, robot=robot, slam=slam, max_queue_size=max_queue_size)
        except Exception as e:
            print(f"[VIZ] Failed to create SharedMemoryVisualizer: {e}")
            print("[VIZ] Falling back to MultiprocessVisualizer")
            from multiprocess_visualizer import MultiprocessVisualizer
            return MultiprocessVisualizer(maze, robot=robot, slam=slam, max_queue_size=max_queue_size)
    else:
        from multiprocess_visualizer import MultiprocessVisualizer
        return MultiprocessVisualizer(maze, robot=robot, slam=slam, max_queue_size=max_queue_size)
