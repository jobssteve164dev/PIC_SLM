"""
统一日志管理系统

功能特性：
- 统一的日志配置和管理
- 支持多种日志级别和输出目标
- 自动日志文件轮转
- 结构化日志记录
- 性能监控日志
- 错误追踪和报告
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any
import json
import traceback
import threading
from functools import wraps
import time
import psutil


class LoggerConfig:
    """日志配置类"""
    
    def __init__(self):
        self.log_dir = "logs"
        self.log_level = logging.INFO
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5
        self.console_output = True
        self.file_output = True
        self.structured_logging = True
        self.performance_logging = True
        self.error_tracking = True


class StructuredFormatter(logging.Formatter):
    """结构化日志格式器"""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record):
        # 创建结构化日志记录
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加额外信息
        if hasattr(record, 'component'):
            log_entry['component'] = record.component
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_entry['duration_ms'] = record.duration
        if hasattr(record, 'memory_usage'):
            log_entry['memory_mb'] = record.memory_usage
        if hasattr(record, 'error_details'):
            log_entry['error_details'] = record.error_details
        if hasattr(record, 'stack_trace'):
            log_entry['stack_trace'] = record.stack_trace
            
        return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'))


class ConsoleFormatter(logging.Formatter):
    """控制台友好的日志格式器"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def format(self, record):
        # 为不同级别的日志添加颜色（如果支持）
        if hasattr(record, 'component'):
            record.name = f"{record.name}({record.component})"
        return super().format(record)


class LoggerManager:
    """日志管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.config = LoggerConfig()
        self.loggers = {}
        self.performance_data = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志系统"""
        # 创建日志目录
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # 设置根日志级别
        logging.getLogger().setLevel(self.config.log_level)
        
        # 创建不同类型的日志文件
        self._setup_main_logger()
        self._setup_error_logger()
        self._setup_performance_logger()
        
    def _setup_main_logger(self):
        """设置主日志记录器"""
        main_log_file = os.path.join(self.config.log_dir, 'main.log')
        
        # 文件处理器（带轮转）
        if self.config.file_output:
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            
            if self.config.structured_logging:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(ConsoleFormatter())
            
            logging.getLogger().addHandler(file_handler)
        
        # 控制台处理器
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleFormatter())
            logging.getLogger().addHandler(console_handler)
    
    def _setup_error_logger(self):
        """设置错误日志记录器"""
        if not self.config.error_tracking:
            return
            
        error_log_file = os.path.join(self.config.log_dir, 'errors.log')
        
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        
        # 创建专门的错误日志记录器
        error_logger = logging.getLogger('errors')
        error_logger.addHandler(error_handler)
        error_logger.setLevel(logging.ERROR)
        
    def _setup_performance_logger(self):
        """设置性能日志记录器"""
        if not self.config.performance_logging:
            return
            
        perf_log_file = os.path.join(self.config.log_dir, 'performance.log')
        
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        perf_handler.setFormatter(StructuredFormatter())
        
        # 创建专门的性能日志记录器
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
    
    def get_logger(self, name: str, component: Optional[str] = None) -> logging.Logger:
        """获取日志记录器"""
        logger_key = f"{name}:{component}" if component else name
        
        if logger_key not in self.loggers:
            logger = logging.getLogger(name)
            
            # 创建适配器来添加组件信息
            if component:
                logger = ComponentLoggerAdapter(logger, {'component': component})
                
            self.loggers[logger_key] = logger
            
        return self.loggers[logger_key]
    
    def log_error(self, logger_name: str, error: Exception, 
                  context: Optional[Dict[str, Any]] = None,
                  component: Optional[str] = None):
        """记录错误信息"""
        error_logger = logging.getLogger('errors')
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': traceback.format_exc(),
            'logger_name': logger_name,
            'context': context or {}
        }
        
        if component:
            error_info['component'] = component
            
        error_logger.error(
            f"Error in {logger_name}: {str(error)}",
            extra={'error_details': error_info, 'stack_trace': traceback.format_exc()}
        )
    
    def log_performance(self, operation: str, duration: float, 
                       memory_usage: Optional[float] = None,
                       component: Optional[str] = None,
                       additional_metrics: Optional[Dict[str, Any]] = None):
        """记录性能信息"""
        if not self.config.performance_logging:
            return
            
        perf_logger = logging.getLogger('performance')
        
        metrics = {
            'operation': operation,
            'duration_ms': round(duration * 1000, 2),
            'memory_mb': round(memory_usage, 2) if memory_usage else None
        }
        
        if component:
            metrics['component'] = component
            
        if additional_metrics:
            metrics.update(additional_metrics)
            
        perf_logger.info(
            f"Performance: {operation} took {metrics['duration_ms']}ms",
            extra=metrics
        )
    
    def set_log_level(self, level: int):
        """设置日志级别"""
        self.config.log_level = level
        logging.getLogger().setLevel(level)
    
    def enable_debug_mode(self):
        """启用调试模式"""
        self.set_log_level(logging.DEBUG)
        
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        stats = {
            'log_directory': self.config.log_dir,
            'log_files': [],
            'total_size_mb': 0
        }
        
        if os.path.exists(self.config.log_dir):
            for filename in os.listdir(self.config.log_dir):
                filepath = os.path.join(self.config.log_dir, filename)
                if os.path.isfile(filepath) and filename.endswith('.log'):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    stats['log_files'].append({
                        'name': filename,
                        'size_mb': round(size_mb, 2),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                    })
                    stats['total_size_mb'] += size_mb
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        return stats


class ComponentLoggerAdapter(logging.LoggerAdapter):
    """组件日志适配器"""
    
    def process(self, msg, kwargs):
        return msg, kwargs


class PerformanceMonitor:
    """性能监控装饰器和上下文管理器"""
    
    def __init__(self, operation: str, component: Optional[str] = None,
                 logger_name: Optional[str] = None):
        self.operation = operation
        self.component = component
        self.logger_name = logger_name or __name__
        self.start_time = None
        self.start_memory = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_delta = current_memory - self.start_memory
        
        logger_manager = LoggerManager()
        logger_manager.log_performance(
            self.operation, duration, memory_delta, self.component
        )
        
        if exc_type:
            logger_manager.log_error(
                self.logger_name, exc_val, 
                {'operation': self.operation}, self.component
            )


def performance_monitor(operation: str, component: Optional[str] = None):
    """性能监控装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceMonitor(operation, component, func.__module__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return LoggerManager().get_logger(name, component)


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None,
              component: Optional[str] = None, logger_name: Optional[str] = None):
    """记录错误的便捷函数"""
    caller_name = logger_name or sys._getframe(1).f_globals.get('__name__', 'unknown')
    LoggerManager().log_error(caller_name, error, context, component)


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO,
                  console_output: bool = True, file_output: bool = True,
                  structured_logging: bool = True):
    """设置日志系统的便捷函数"""
    manager = LoggerManager()
    manager.config.log_dir = log_dir
    manager.config.log_level = log_level
    manager.config.console_output = console_output
    manager.config.file_output = file_output
    manager.config.structured_logging = structured_logging
    manager._setup_logging()


# 导出主要接口
__all__ = [
    'LoggerManager', 'PerformanceMonitor', 'performance_monitor',
    'get_logger', 'log_error', 'setup_logging'
] 