"""
真正的资源限制器
实现对程序资源使用的实际限制和控制
"""

import os
import sys
import time
import threading
import gc
import psutil
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

# Windows特定导入
WIN32_AVAILABLE = False
if os.name == 'nt':
    try:
        import win32api
        import win32job
        import win32process
        WIN32_AVAILABLE = True
    except ImportError as e:
        print("ℹ️ Win32模块不可用，Windows强制内存限制功能将被禁用")
        print("   如需启用，请安装: pip install pywin32")
        WIN32_AVAILABLE = False

# Unix/Linux特定导入
RESOURCE_AVAILABLE = False
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ResourceLimits:
    """资源限制配置"""
    max_memory_gb: float = 8.0
    max_cpu_percent: float = 80.0
    max_disk_usage_gb: float = 10.0
    max_processes: int = 4
    max_threads: int = 8
    check_interval: float = 1.0
    enforce_limits: bool = True
    auto_cleanup: bool = True


class ResourceLimitException(Exception):
    """资源限制异常"""
    def __init__(self, resource_type: str, current_value: float, limit_value: float):
        self.resource_type = resource_type
        self.current_value = current_value
        self.limit_value = limit_value
        super().__init__(f"{resource_type}资源超限: {current_value:.2f} > {limit_value:.2f}")


class ResourceLimiter:
    """真正的资源限制器"""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.current_process = psutil.Process(os.getpid())
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = {
            'memory_limit': [],
            'cpu_limit': [],
            'disk_limit': [],
            'process_limit': []
        }
        
        # Windows Job Object
        self.job_object = None
        self._setup_job_object()
        
        # 临时文件跟踪
        self.temp_files = set()
        self.temp_dirs = set()
        
        # 线程控制
        self.active_threads = set()
        self.max_threads_semaphore = threading.Semaphore(limits.max_threads)
        
        # 停止标志
        self.stop_requested = False
        
    def _setup_job_object(self):
        """设置Windows Job Object进行内存限制"""
        if not WIN32_AVAILABLE or not self.limits.enforce_limits:
            return
            
        try:
            # 创建Job Object
            self.job_object = win32job.CreateJobObject(None, None)
            
            # 设置内存限制
            job_info = win32job.QueryInformationJobObject(
                self.job_object, 
                win32job.JobObjectExtendedLimitInformation
            )
            
            # 设置进程内存限制
            job_info['ProcessMemoryLimit'] = int(self.limits.max_memory_gb * 1024 * 1024 * 1024)
            job_info['JobMemoryLimit'] = int(self.limits.max_memory_gb * 1024 * 1024 * 1024)
            job_info['LimitFlags'] = (
                win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY |
                win32job.JOB_OBJECT_LIMIT_JOB_MEMORY |
                win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            )
            job_info['ActiveProcessLimit'] = self.limits.max_processes
            
            win32job.SetInformationJobObject(
                self.job_object,
                win32job.JobObjectExtendedLimitInformation,
                job_info
            )
            
            # 将当前进程添加到Job Object
            current_process_handle = win32api.GetCurrentProcess()
            win32job.AssignProcessToJobObject(self.job_object, current_process_handle)
            
            print(f"✅ Windows Job Object设置成功: 内存限制 {self.limits.max_memory_gb}GB")
            
        except Exception as e:
            print(f"⚠️ Windows Job Object设置失败: {e}")
            self.job_object = None
    
    def start_monitoring(self):
        """开始资源监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.stop_requested = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 资源监控已启动")
    
    def stop_monitoring(self):
        """停止资源监控"""
        self.monitoring = False
        self.stop_requested = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3.0)
        print("⏹️ 资源监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring and not self.stop_requested:
            try:
                self._check_memory_limit()
                self._check_cpu_limit()
                self._check_disk_limit()
                self._check_process_limit()
                
                if self.limits.auto_cleanup:
                    self._auto_cleanup()
                    
                time.sleep(self.limits.check_interval)
                
            except Exception as e:
                print(f"⚠️ 资源监控错误: {e}")
                time.sleep(1.0)
    
    def _check_memory_limit(self):
        """检查内存限制"""
        try:
            # 获取当前进程内存使用
            memory_info = self.current_process.memory_info()
            current_memory_gb = memory_info.rss / (1024**3)
            
            if current_memory_gb > self.limits.max_memory_gb:
                self._trigger_callbacks('memory_limit', current_memory_gb, self.limits.max_memory_gb)
                
                if self.limits.enforce_limits:
                    # 强制垃圾回收
                    gc.collect()
                    
                    # 再次检查
                    memory_info = self.current_process.memory_info()
                    current_memory_gb = memory_info.rss / (1024**3)
                    
                    if current_memory_gb > self.limits.max_memory_gb:
                        raise ResourceLimitException(
                            "内存", current_memory_gb, self.limits.max_memory_gb
                        )
                        
        except psutil.Error:
            pass
    
    def _check_cpu_limit(self):
        """检查CPU限制"""
        try:
            cpu_percent = self.current_process.cpu_percent()
            
            if cpu_percent > self.limits.max_cpu_percent:
                self._trigger_callbacks('cpu_limit', cpu_percent, self.limits.max_cpu_percent)
                
                if self.limits.enforce_limits:
                    # 降低进程优先级
                    if os.name == 'nt':
                        try:
                            self.current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                        except:
                            pass
                    else:
                        try:
                            self.current_process.nice(10)  # 降低优先级
                        except:
                            pass
                        
        except psutil.Error:
            pass
    
    def _check_disk_limit(self):
        """检查磁盘使用限制"""
        try:
            total_size = 0
            for temp_file in list(self.temp_files):
                if os.path.exists(temp_file):
                    total_size += os.path.getsize(temp_file)
                else:
                    self.temp_files.discard(temp_file)
            
            for temp_dir in list(self.temp_dirs):
                if os.path.exists(temp_dir):
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                total_size += os.path.getsize(file_path)
                            except OSError:
                                continue
                else:
                    self.temp_dirs.discard(temp_dir)
            
            current_disk_gb = total_size / (1024**3)
            
            if current_disk_gb > self.limits.max_disk_usage_gb:
                self._trigger_callbacks('disk_limit', current_disk_gb, self.limits.max_disk_usage_gb)
                
                if self.limits.enforce_limits:
                    self._cleanup_temp_files()
                    
        except Exception:
            pass
    
    def _check_process_limit(self):
        """检查进程数限制"""
        try:
            children = self.current_process.children(recursive=True)
            process_count = len(children) + 1  # +1 for current process
            
            if process_count > self.limits.max_processes:
                self._trigger_callbacks('process_limit', process_count, self.limits.max_processes)
                
                if self.limits.enforce_limits:
                    # 终止多余的子进程
                    for child in children[self.limits.max_processes-1:]:
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                            
        except psutil.Error:
            pass
    
    def _trigger_callbacks(self, event_type: str, current_value: float, limit_value: float):
        """触发回调函数"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(event_type, current_value, limit_value)
            except Exception as e:
                print(f"⚠️ 回调函数执行错误: {e}")
    
    def add_callback(self, event_type: str, callback: Callable):
        """添加事件回调"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def _auto_cleanup(self):
        """自动清理资源"""
        try:
            # 强制垃圾回收
            gc.collect()
            
            # 清理临时文件
            self._cleanup_temp_files()
            
            # 清理缓存
            self._cleanup_caches()
            
        except Exception as e:
            print(f"⚠️ 自动清理错误: {e}")
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        cleaned_files = 0
        cleaned_size = 0
        
        # 清理跟踪的临时文件
        for temp_file in list(self.temp_files):
            try:
                if os.path.exists(temp_file):
                    size = os.path.getsize(temp_file)
                    os.remove(temp_file)
                    cleaned_files += 1
                    cleaned_size += size
                self.temp_files.discard(temp_file)
            except OSError:
                pass
        
        # 清理临时目录中的老文件
        for temp_dir in self.temp_dirs:
            if os.path.isdir(temp_dir):
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            # 删除超过1小时的文件
                            if time.time() - os.path.getmtime(file_path) > 3600:
                                size = os.path.getsize(file_path)
                                os.remove(file_path)
                                cleaned_files += 1
                                cleaned_size += size
                        except OSError:
                            continue
        
        if cleaned_files > 0:
            print(f"🧹 清理了 {cleaned_files} 个临时文件，释放 {cleaned_size / (1024**2):.1f} MB")
    
    def _cleanup_caches(self):
        """清理各种缓存"""
        try:
            # 清理matplotlib缓存
            try:
                import matplotlib
                if hasattr(matplotlib, 'get_cachedir'):
                    cache_dir = matplotlib.get_cachedir()
                    if os.path.exists(cache_dir):
                        import shutil
                        shutil.rmtree(cache_dir, ignore_errors=True)
            except:
                pass
            
            # 清理PIL缓存
            try:
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None  # 重置限制
            except ImportError:
                pass
                
        except Exception:
            pass
    
    def register_temp_file(self, file_path: str):
        """注册临时文件以供跟踪"""
        self.temp_files.add(file_path)
    
    def register_temp_dir(self, dir_path: str):
        """注册临时目录以供跟踪"""
        self.temp_dirs.add(dir_path)
    
    def unregister_temp_file(self, file_path: str):
        """取消注册临时文件"""
        self.temp_files.discard(file_path)
    
    def create_limited_thread(self, target, args=(), kwargs=None, name=None):
        """创建受限制的线程"""
        if kwargs is None:
            kwargs = {}
        
        def wrapped_target(*args, **kwargs):
            # 获取线程信号量
            with self.max_threads_semaphore:
                thread_id = threading.get_ident()
                self.active_threads.add(thread_id)
                try:
                    return target(*args, **kwargs)
                finally:
                    self.active_threads.discard(thread_id)
        
        thread = threading.Thread(target=wrapped_target, args=args, kwargs=kwargs, name=name)
        return thread
    
    def check_resource_before_operation(self, operation_name: str = "操作"):
        """在执行重要操作前检查资源"""
        try:
            # 检查内存
            memory_info = self.current_process.memory_info()
            current_memory_gb = memory_info.rss / (1024**3)
            
            if current_memory_gb > self.limits.max_memory_gb * 0.9:  # 90% 阈值
                raise ResourceLimitException(
                    "内存预检查", current_memory_gb, self.limits.max_memory_gb * 0.9
                )
            
            # 检查磁盘空间
            temp_dir = Path.cwd()
            disk_usage = psutil.disk_usage(str(temp_dir))
            available_gb = disk_usage.free / (1024**3)
            
            if available_gb < 1.0:  # 至少需要1GB可用空间
                raise ResourceLimitException(
                    "磁盘空间预检查", 1.0 - available_gb, 1.0
                )
                
            return True
            
        except ResourceLimitException:
            raise
        except Exception as e:
            print(f"⚠️ 资源预检查错误: {e}")
            return True
    
    def get_resource_status(self) -> Dict[str, Any]:
        """获取当前资源状态"""
        try:
            memory_info = self.current_process.memory_info()
            current_memory_gb = memory_info.rss / (1024**3)
            
            cpu_percent = self.current_process.cpu_percent()
            
            temp_size = sum(
                os.path.getsize(f) if os.path.exists(f) else 0 
                for f in self.temp_files
            )
            temp_size_gb = temp_size / (1024**3)
            
            children_count = len(self.current_process.children(recursive=True))
            
            return {
                'memory_gb': current_memory_gb,
                'memory_percent': (current_memory_gb / self.limits.max_memory_gb) * 100,
                'cpu_percent': cpu_percent,
                'temp_disk_gb': temp_size_gb,
                'temp_disk_percent': (temp_size_gb / self.limits.max_disk_usage_gb) * 100,
                'process_count': children_count + 1,
                'thread_count': len(self.active_threads),
                'monitoring': self.monitoring
            }
            
        except Exception as e:
            print(f"⚠️ 获取资源状态错误: {e}")
            return {}
    
    def emergency_cleanup(self):
        """紧急资源清理"""
        print("🚨 执行紧急资源清理...")
        
        try:
            # 强制垃圾回收
            for _ in range(3):
                collected = gc.collect()
                print(f"   垃圾回收: {collected} 个对象")
            
            # 清理所有临时文件
            self._cleanup_temp_files()
            
            # 清理缓存
            self._cleanup_caches()
            
            # 降低进程优先级
            try:
                if os.name == 'nt':
                    self.current_process.nice(psutil.IDLE_PRIORITY_CLASS)
                else:
                    self.current_process.nice(19)
            except:
                pass
            
            # 终止额外的子进程
            children = self.current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            print("✅ 紧急清理完成")
            
        except Exception as e:
            print(f"❌ 紧急清理失败: {e}")
    
    def request_stop(self):
        """请求停止所有操作"""
        self.stop_requested = True
        print("🛑 已请求停止所有操作")
    
    def is_stop_requested(self) -> bool:
        """检查是否请求停止"""
        return self.stop_requested
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
        
        # 清理临时文件
        self._cleanup_temp_files()
        
        # 关闭Job Object
        if self.job_object:
            try:
                if WIN32_AVAILABLE:
                    win32api.CloseHandle(self.job_object)
            except:
                pass


# 全局资源限制器实例
_global_limiter = None

def get_resource_limiter() -> Optional[ResourceLimiter]:
    """获取全局资源限制器"""
    return _global_limiter

def initialize_resource_limiter(limits: ResourceLimits) -> ResourceLimiter:
    """初始化全局资源限制器"""
    global _global_limiter
    if _global_limiter:
        _global_limiter.stop_monitoring()
    
    _global_limiter = ResourceLimiter(limits)
    return _global_limiter

def resource_limited_operation(operation_name: str = "操作"):
    """装饰器：为操作添加资源限制检查"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            limiter = get_resource_limiter()
            if limiter:
                limiter.check_resource_before_operation(operation_name)
                if limiter.is_stop_requested():
                    raise ResourceLimitException("操作中断", 1, 0)
            return func(*args, **kwargs)
        return wrapper
    return decorator 