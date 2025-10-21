"""
多进程可视化模块
在独立进程中运行可视化，完全隔离主计算进程
解决 Matplotlib 不支持在子线程中运行的问题
"""
import multiprocessing as mp
import pickle
import time
from typing import Optional, Any, Dict, Callable


class VisualizationProcess:
    """
    可视化进程包装器
    在独立进程中运行Matplotlib，避免主线程冲突
    """
    def __init__(self, maze, robot=None, slam=None, max_queue_size: int = 3):
        self.command_queue = mp.Queue(maxsize=max_queue_size)
        self.response_queue = mp.Queue(maxsize=10)
        self.process: Optional[mp.Process] = None
        
        # 只传递可序列化的数据，不传递robot和slam对象（它们包含线程锁）
        self.init_data = {
            'maze': maze,
            # robot 和 slam 不传递，在子进程中创建简化版本或置为None
        }
        
        # 添加类型存根，帮助类型检查器理解这些方法
        # 这些会在运行时被 __getattr__ 动态提供
        self.paused: bool = False
        self.fig: Any = None
        
    def start(self):
        """启动可视化进程"""
        if self.process is not None and self.process.is_alive():
            return
            
        self.process = mp.Process(
            target=self._visualization_process_main,
            args=(self.command_queue, self.response_queue, self.init_data),
            daemon=False  # 不是守护进程，确保可以正常退出
        )
        self.process.start()
        print("[VIZ PROCESS] Visualization process started")
        
    def stop(self):
        """停止可视化进程"""
        if self.process is None:
            return
            
        try:
            # 发送停止命令
            self.command_queue.put(('stop', None), timeout=1.0)
            # 等待进程结束
            self.process.join(timeout=3.0)
            
            if self.process.is_alive():
                print("[VIZ PROCESS] Force terminating visualization process")
                self.process.terminate()
                self.process.join(timeout=1.0)
                
        except Exception as e:
            print(f"[VIZ PROCESS ERROR] Error stopping process: {e}")
            if self.process.is_alive():
                self.process.kill()
                
        self.process = None
        print("[VIZ PROCESS] Visualization process stopped")
        
    @staticmethod
    def _visualization_process_main(command_queue: mp.Queue, response_queue: mp.Queue, init_data: Dict):
        """可视化进程的主函数（在子进程中运行）"""
        try:
            # 在子进程的开始处配置matplotlib，避免警告
            import matplotlib
            matplotlib.use('TkAgg')  # 或 'Qt5Agg'
            
            # 在子进程中导入这些模块，避免主进程中的警告
            import matplotlib.pyplot as plt
            from visualizer import Visualizer
            
            # 在子进程中创建Visualizer
            maze = init_data['maze']
            # robot 和 slam 设为 None，因为它们不可序列化
            # Visualizer 可以在没有 robot/slam 的情况下工作
            robot = None
            slam = None
            
            viz = Visualizer(maze, robot=robot, slam=slam)
            print("[VIZ PROCESS] Visualizer initialized in subprocess")
            
            # 主循环：处理命令
            while True:
                try:
                    # 阻塞等待命令，但设置超时以便检查窗口状态
                    try:
                        command, data = command_queue.get(timeout=0.1)
                    except:
                        # 超时，检查窗口是否关闭
                        if not plt.fignum_exists(viz.fig.number):
                            print("[VIZ PROCESS] Window closed, exiting")
                            break
                        continue
                    
                    # 处理停止命令
                    if command == 'stop':
                        print("[VIZ PROCESS] Received stop command")
                        break
                    
                    # 处理更新命令
                    elif command == 'update':
                        args = data.get('args', ())
                        kwargs = data.get('kwargs', {})
                        viz.update(*args, **kwargs)
                        
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
                    
                    # 处理属性设置
                    elif command == 'set_attr':
                        attr_name = data['attr']
                        value = data['value']
                        setattr(viz, attr_name, value)
                        response_queue.put(('success', None))
                        
                except Exception as e:
                    print(f"[VIZ PROCESS ERROR] Error processing command: {e}")
                    continue
                    
        except Exception as e:
            print(f"[VIZ PROCESS ERROR] Fatal error in visualization process: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理资源
            try:
                plt.close('all')
            except:
                pass
            print("[VIZ PROCESS] Visualization process exiting")
    
    def update(self, *args, **kwargs):
        """
        非阻塞更新可视化
        """
        if self.process is None or not self.process.is_alive():
            return
            
        data = {
            'args': args,
            'kwargs': kwargs
        }
        
        # 如果队列满了，移除旧数据
        if self.command_queue.full():
            try:
                self.command_queue.get_nowait()
            except:
                pass
        
        try:
            self.command_queue.put_nowait(('update', data))
        except:
            # 队列满了就跳过这一帧
            pass
    
    def call_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        调用visualizer的方法（阻塞调用）
        """
        if self.process is None or not self.process.is_alive():
            return None
            
        data = {
            'method': method_name,
            'args': args,
            'kwargs': kwargs
        }
        
        try:
            self.command_queue.put(('call_method', data), timeout=1.0)
            status, result = self.response_queue.get(timeout=2.0)
            if status == 'success':
                return result
            else:
                print(f"[VIZ PROCESS ERROR] {result}")
                return None
        except Exception as e:
            print(f"[VIZ PROCESS ERROR] Method call failed: {e}")
            return None
    
    # 为常用方法添加类型提示和存根
    def set_grid_system(self, grid_system: Any, target_cell_id: Optional[int]) -> Any:
        """设置网格系统"""
        return self.call_method('set_grid_system', grid_system, target_cell_id)
    
    def set_obstacle_search_region(self, region: Optional[tuple]) -> Any:
        """设置障碍物搜索区域"""
        return self.call_method('set_obstacle_search_region', region)
    
    def set_bfs_debug_points(self, points: Optional[list], color: str = 'cyan') -> Any:
        """设置BFS调试点"""
        return self.call_method('set_bfs_debug_points', points, color=color)
    
    def set_emergency_path(self, path: Optional[list]) -> Any:
        """设置紧急路径"""
        return self.call_method('set_emergency_path', path)
    
    def save_map(self, filename: str) -> Any:
        """保存地图"""
        return self.call_method('save_map', filename)
    
    def save_path(self, filename: str) -> Any:
        """保存路径"""
        return self.call_method('save_path', filename)
    
    def check_and_clear_force_stop(self) -> bool:
        """检查并清除强制停止标志"""
        result = self.call_method('check_and_clear_force_stop')
        return bool(result) if result is not None else False
    
    def is_auto_control_enabled(self) -> bool:
        """检查是否启用自动控制"""
        result = self.call_method('is_auto_control_enabled')
        return bool(result) if result is not None else False
    
    def __getattr__(self, name):
        """
        代理属性访问（同步调用，可能有延迟）
        """
        # 避免递归
        if name in ('command_queue', 'response_queue', 'process', 'init_data', 'paused', 'fig'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 特殊属性：paused (默认返回False，避免阻塞)
        if name == 'paused':
            return False
        
        # 特殊属性：fig (返回一个模拟对象，避免检查fig.number时出错)
        if name == 'fig':
            class FakeFig:
                number = 1
            return FakeFig()
        
        # 返回一个lambda函数用于方法调用
        return lambda *args, **kwargs: self.call_method(name, *args, **kwargs)


class MultiprocessVisualizer(VisualizationProcess):
    """
    多进程可视化器的便捷别名
    """
    pass
