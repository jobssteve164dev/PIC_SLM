"""
智能训练管理器 - 协调智能训练控制器与现有训练系统

主要功能：
- 管理智能训练控制器的生命周期
- 协调训练系统的启动、停止、重启
- 处理参数更新和配置同步
- 提供统一的接口给UI层
"""

import os
import json
import time
import threading
from typing import Dict, Any, Optional, List
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QMessageBox

from .intelligent_training_controller import IntelligentTrainingController
from .model_trainer import ModelTrainer
from .training_thread import TrainingThread


class IntelligentTrainingManager(QObject):
    """智能训练管理器"""
    
    # 信号定义
    training_started = pyqtSignal(dict)  # 训练开始信号
    training_stopped = pyqtSignal(dict)  # 训练停止信号
    training_restarted = pyqtSignal(dict)  # 训练重启信号
    intervention_occurred = pyqtSignal(dict)  # 干预发生信号
    analysis_completed = pyqtSignal(dict)  # 分析完成信号
    status_updated = pyqtSignal(str)  # 状态更新信号
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.intelligent_controller = None
        self.model_trainer = None
        self.current_training_config = None
        self.is_intelligent_mode = False
        
        # 训练状态管理
        self.training_status = 'idle'  # idle, running, stopped, restarting
        self.intervention_count = 0
        self.best_checkpoint_path = None
        
        # 配置管理
        self.config_file = "setting/intelligent_training_config.json"
        self.config = self._load_default_config()
        
        # 初始化组件
        self._initialize_components()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        default_config = {
            'auto_intervention_enabled': True,
            'analysis_interval': 10,
            'max_interventions_per_session': 3,
            'parameter_tuning_strategy': 'conservative',
            'intervention_thresholds': {
                'overfitting_risk': 0.8,
                'underfitting_risk': 0.7,
                'stagnation_epochs': 5,
                'divergence_threshold': 2.0,
                'min_training_epochs': 3
            }
        }
        
        # 尝试从文件加载配置
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
        
        return default_config
    
    def _initialize_components(self):
        """初始化相关组件"""
        try:
            # 初始化智能训练控制器
            self.intelligent_controller = IntelligentTrainingController()
            
            # 连接信号
            self.intelligent_controller.intervention_triggered.connect(self._on_intervention_triggered)
            self.intelligent_controller.training_restarted.connect(self._on_training_restart_requested)
            self.intelligent_controller.analysis_completed.connect(self._on_analysis_completed)
            self.intelligent_controller.status_updated.connect(self._on_controller_status_updated)
            self.intelligent_controller.error_occurred.connect(self._on_controller_error)
            
            self.status_updated.emit("智能训练管理器初始化完成")
            
        except Exception as e:
            self.error_occurred.emit(f"初始化组件失败: {str(e)}")
    
    def set_model_trainer(self, model_trainer: ModelTrainer):
        """设置模型训练器"""
        self.model_trainer = model_trainer
        
        # 将训练器传递给智能控制器
        if self.intelligent_controller:
            self.intelligent_controller.training_system = self.model_trainer
    
    def start_intelligent_training(self, training_config: Dict[str, Any]):
        """启动智能训练"""
        try:
            if self.training_status == 'running':
                self.error_occurred.emit("训练已在运行中")
                return
            
            # 保存训练配置
            self.current_training_config = training_config.copy()
            
            # 更新配置
            if self.intelligent_controller:
                self.intelligent_controller.config.update(self.config)
            
            # 启动智能监控
            if self.intelligent_controller:
                self.intelligent_controller.start_monitoring(training_config)
                self.is_intelligent_mode = True
            
            # 启动训练
            self._start_training(training_config)
            
            self.training_status = 'running'
            self.status_updated.emit("智能训练已启动")
            
        except Exception as e:
            self.error_occurred.emit(f"启动智能训练失败: {str(e)}")
    
    def stop_intelligent_training(self):
        """停止智能训练"""
        try:
            # 停止智能监控
            if self.intelligent_controller:
                self.intelligent_controller.stop_monitoring()
                self.is_intelligent_mode = False
            
            # 停止训练
            self._stop_training()
            
            self.training_status = 'stopped'
            self.status_updated.emit("智能训练已停止")
            
        except Exception as e:
            self.error_occurred.emit(f"停止智能训练失败: {str(e)}")
    
    def _start_training(self, config: Dict[str, Any]):
        """启动训练"""
        try:
            if not self.model_trainer:
                raise Exception("模型训练器未设置")
            
            # 启动训练
            self.model_trainer.train_model_with_config(config)
            
            # 发射训练开始信号
            self.training_started.emit({
                'config': config,
                'timestamp': time.time(),
                'intelligent_mode': True
            })
            
        except Exception as e:
            self.error_occurred.emit(f"启动训练失败: {str(e)}")
    
    def _stop_training(self):
        """停止训练"""
        try:
            if self.model_trainer:
                self.model_trainer.stop()
                
                # 发射训练停止信号
                self.training_stopped.emit({
                    'timestamp': time.time(),
                    'intervention_count': self.intervention_count
                })
                
        except Exception as e:
            self.error_occurred.emit(f"停止训练失败: {str(e)}")
    
    def _on_intervention_triggered(self, intervention_data: Dict[str, Any]):
        """处理干预触发事件"""
        try:
            self.intervention_count += 1
            
            # 发射干预信号
            self.intervention_occurred.emit(intervention_data)
            
            # 自动停止训练
            if self.config.get('auto_intervention_enabled', True):
                self._stop_training()
                self.training_status = 'stopped'
                
                # 保存最佳检查点
                if self.config.get('training_restart', {}).get('preserve_best_checkpoint', True):
                    self._save_best_checkpoint()
                
                self.status_updated.emit(f"检测到训练问题，已自动停止训练 (干预 #{self.intervention_count})")
            
        except Exception as e:
            self.error_occurred.emit(f"处理干预事件失败: {str(e)}")
    
    def _on_training_restart_requested(self, restart_data: Dict[str, Any]):
        """处理训练重启请求"""
        try:
            if self.training_status == 'running':
                self.error_occurred.emit("训练仍在运行中，无法重启")
                return
            
            # 检查重启次数限制
            max_attempts = self.config.get('training_restart', {}).get('max_restart_attempts', 3)
            if self.intervention_count >= max_attempts:
                self.error_occurred.emit(f"已达到最大重启次数限制 ({max_attempts})")
                return
            
            # 获取建议的参数
            suggested_params = restart_data.get('new_params', {})
            if not suggested_params:
                self.error_occurred.emit("没有可用的参数建议")
                return
            
            # 更新训练配置
            updated_config = self._update_training_config(suggested_params)
            
            # 延迟重启
            restart_delay = self.config.get('training_restart', {}).get('restart_delay', 5)
            self.status_updated.emit(f"将在 {restart_delay} 秒后使用优化参数重启训练...")
            
            # 使用定时器延迟重启
            QTimer.singleShot(restart_delay * 1000, lambda: self._restart_training(updated_config))
            
        except Exception as e:
            self.error_occurred.emit(f"处理重启请求失败: {str(e)}")
    
    def _restart_training(self, updated_config: Dict[str, Any]):
        """重启训练"""
        try:
            self.training_status = 'restarting'
            self.status_updated.emit("正在重启训练...")
            
            # 更新当前配置
            self.current_training_config = updated_config
            
            # 重新启动智能监控
            if self.intelligent_controller:
                self.intelligent_controller.start_monitoring(updated_config)
                self.is_intelligent_mode = True
            
            # 启动训练
            self._start_training(updated_config)
            
            self.training_status = 'running'
            
            # 发射重启信号
            self.training_restarted.emit({
                'config': updated_config,
                'timestamp': time.time(),
                'intervention_count': self.intervention_count,
                'restart_number': self.intervention_count
            })
            
            self.status_updated.emit("训练已使用优化参数重启")
            
        except Exception as e:
            self.error_occurred.emit(f"重启训练失败: {str(e)}")
            self.training_status = 'stopped'
    
    def _update_training_config(self, suggested_params: Dict[str, Any]) -> Dict[str, Any]:
        """更新训练配置"""
        if not self.current_training_config:
            return suggested_params
        
        updated_config = self.current_training_config.copy()
        
        # 获取调优策略
        strategy = self.config.get('parameter_tuning_strategy', 'conservative')
        strategy_config = self.config.get('intervention_strategies', {}).get(strategy, {})
        
        # 应用参数调整
        for param_name, new_value in suggested_params.items():
            if param_name in updated_config:
                # 根据策略调整参数
                if param_name == 'learning_rate':
                    adjustment = strategy_config.get('learning_rate_adjustment', 0.5)
                    updated_config[param_name] = updated_config[param_name] * adjustment
                elif param_name == 'batch_size':
                    adjustment = strategy_config.get('batch_size_adjustment', 0.8)
                    updated_config[param_name] = int(updated_config[param_name] * adjustment)
                elif param_name == 'dropout_rate':
                    increase = strategy_config.get('dropout_increase', 0.1)
                    updated_config[param_name] = min(0.9, updated_config[param_name] + increase)
                elif param_name == 'weight_decay':
                    increase = strategy_config.get('weight_decay_increase', 1.5)
                    updated_config[param_name] = updated_config[param_name] * increase
                else:
                    # 直接使用建议值
                    updated_config[param_name] = new_value
        
        return updated_config
    
    def _save_best_checkpoint(self):
        """保存最佳检查点"""
        try:
            # 这里需要根据实际的检查点保存逻辑来实现
            # 暂时记录路径
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.best_checkpoint_path = f"models/checkpoints/best_checkpoint_{timestamp}.pth"
            
            self.status_updated.emit(f"最佳检查点已保存: {self.best_checkpoint_path}")
            
        except Exception as e:
            self.error_occurred.emit(f"保存检查点失败: {str(e)}")
    
    def _on_analysis_completed(self, analysis_data: Dict[str, Any]):
        """处理分析完成事件"""
        try:
            # 发射分析完成信号
            self.analysis_completed.emit(analysis_data)
            
        except Exception as e:
            self.error_occurred.emit(f"处理分析完成事件失败: {str(e)}")
    
    def _on_controller_status_updated(self, status: str):
        """处理控制器状态更新"""
        self.status_updated.emit(f"智能控制器: {status}")
    
    def _on_controller_error(self, error: str):
        """处理控制器错误"""
        self.error_occurred.emit(f"智能控制器错误: {error}")
    
    def update_training_progress(self, metrics: Dict[str, Any]):
        """更新训练进度"""
        try:
            # 传递给智能控制器
            if self.intelligent_controller:
                self.intelligent_controller.update_training_progress(metrics)
            
        except Exception as e:
            self.error_occurred.emit(f"更新训练进度失败: {str(e)}")
    
    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """获取干预历史"""
        if self.intelligent_controller:
            return self.intelligent_controller.get_intervention_history()
        return []
    
    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """获取当前会话信息"""
        if self.intelligent_controller:
            return self.intelligent_controller.get_current_session_info()
        return None
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            'status': self.training_status,
            'intelligent_mode': self.is_intelligent_mode,
            'intervention_count': self.intervention_count,
            'best_checkpoint': self.best_checkpoint_path,
            'config': self.current_training_config
        }
    
    def load_config(self, config_file: str = None):
        """加载配置"""
        try:
            if config_file:
                self.config_file = config_file
            
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # 合并配置
                self.config.update(loaded_config)
                
                # 更新智能控制器配置
                if self.intelligent_controller:
                    self.intelligent_controller.config.update(self.config)
                
                self.status_updated.emit(f"配置已从 {self.config_file} 加载")
            
        except Exception as e:
            self.error_occurred.emit(f"加载配置失败: {str(e)}")
    
    def save_config(self, config_file: str = None):
        """保存配置"""
        try:
            if config_file:
                self.config_file = config_file
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            self.status_updated.emit(f"配置已保存到: {self.config_file}")
            
        except Exception as e:
            self.error_occurred.emit(f"保存配置失败: {str(e)}")
    
    def reset_config(self):
        """重置为默认配置"""
        try:
            self.config = self._load_default_config()
            
            # 更新智能控制器配置
            if self.intelligent_controller:
                self.intelligent_controller.config.update(self.config)
            
            self.status_updated.emit("配置已重置为默认值")
            
        except Exception as e:
            self.error_occurred.emit(f"重置配置失败: {str(e)}")
    
    def generate_session_report(self, file_path: str = None):
        """生成会话报告"""
        try:
            if not self.intelligent_controller:
                return
            
            if not file_path:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_path = f"logs/intelligent_training_report_{timestamp}.json"
            
            self.intelligent_controller.save_session_report(file_path)
            
        except Exception as e:
            self.error_occurred.emit(f"生成会话报告失败: {str(e)}")
    
    def is_training_active(self) -> bool:
        """检查训练是否活跃"""
        return self.training_status == 'running'
    
    def get_intelligent_mode_status(self) -> bool:
        """获取智能模式状态"""
        return self.is_intelligent_mode 