"""
智能训练编排器

协调智能配置生成器与现有训练系统，实现完整的智能训练流程
主要功能：
- 管理智能训练的生命周期
- 协调配置生成和训练执行
- 提供统一的接口给UI层
- 处理训练重启和参数迭代
"""

import os
import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread
from PyQt5.QtWidgets import QMessageBox

from .intelligent_config_generator import IntelligentConfigGenerator, ConfigAdjustment
from .model_trainer import ModelTrainer
from .real_time_metrics_collector import get_global_metrics_collector
from .intelligent_training_state_manager import IntelligentTrainingStateManager, TrainingState


@dataclass
class IntelligentTrainingSession:
    """智能训练会话"""
    session_id: str
    start_time: float
    original_config: Dict[str, Any]
    current_config: Dict[str, Any]
    training_iterations: List[Dict[str, Any]]  # 每次训练的记录
    total_iterations: int
    max_iterations: int
    status: str  # 'running', 'completed', 'failed', 'stopped'
    best_metrics: Optional[Dict[str, Any]]
    best_config: Optional[Dict[str, Any]]


class IntelligentTrainingOrchestrator(QObject):
    """智能训练编排器"""
    
    # 信号定义
    training_started = pyqtSignal(dict)      # 训练开始信号
    training_completed = pyqtSignal(dict)    # 训练完成信号
    training_failed = pyqtSignal(dict)       # 训练失败信号
    config_generated = pyqtSignal(dict)      # 配置生成信号
    config_applied = pyqtSignal(dict)        # 配置应用信号
    iteration_completed = pyqtSignal(dict)   # 迭代完成信号
    status_updated = pyqtSignal(str)         # 状态更新信号
    error_occurred = pyqtSignal(str)         # 错误信号
    apply_config_requested = pyqtSignal(dict)  # 请求应用配置信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 核心组件
        self.config_generator = IntelligentConfigGenerator()
        self.model_trainer = None
        self.metrics_collector = get_global_metrics_collector()
        self.state_manager = IntelligentTrainingStateManager()
        
        # 训练会话管理
        self.current_session: Optional[IntelligentTrainingSession] = None
        self.training_tab = None  # 训练标签页引用
        
        # 配置参数 - 默认值
        self.config = {
            'max_iterations': 5,           # 最大迭代次数
            'min_iteration_epochs': 2,     # 每次迭代最小训练轮数（调试用）
            'analysis_interval': 2,        # 分析间隔（epoch）（调试用）
            'convergence_threshold': 0.01, # 收敛阈值
            'improvement_threshold': 0.02, # 改进阈值
            'auto_restart': True,          # 自动重启训练
            'preserve_best_model': True,   # 保留最佳模型
        }
        
        # 加载外部配置
        self._load_external_config()
        
        # 状态管理
        self.is_running = False
        self.current_iteration = 0
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # 初始化组件
        self._initialize_components()
        
        # 连接状态管理器信号
        self.state_manager.state_changed.connect(self._on_state_changed)
    
    def _load_external_config(self):
        """加载外部配置文件"""
        try:
            # 首先尝试从主配置文件加载
            main_config_file = "config.json"
            if os.path.exists(main_config_file):
                with open(main_config_file, 'r', encoding='utf-8') as f:
                    main_config = json.load(f)
                    intelligent_config = main_config.get('intelligent_training', {})
                    if intelligent_config:
                        self._update_config_from_dict(intelligent_config)
                        print(f"[INFO] 从主配置文件加载智能训练配置: {intelligent_config}")
                        return
            
            # 然后尝试从智能训练专用配置文件加载
            intelligent_config_file = "setting/intelligent_training_config.json"
            if os.path.exists(intelligent_config_file):
                with open(intelligent_config_file, 'r', encoding='utf-8') as f:
                    intelligent_config = json.load(f)
                    self._update_config_from_dict(intelligent_config)
                    print(f"[INFO] 从智能训练配置文件加载配置: {intelligent_config}")
                    return
            
            print("[INFO] 未找到外部配置文件，使用默认配置")
            
        except Exception as e:
            print(f"[WARNING] 加载外部配置失败: {str(e)}，使用默认配置")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        try:
            # 更新编排器相关配置
            orchestrator_keys = [
                'max_iterations', 'min_iteration_epochs', 'analysis_interval',
                'convergence_threshold', 'improvement_threshold', 'auto_restart', 'preserve_best_model'
            ]
            
            for key in orchestrator_keys:
                if key in config_dict:
                    self.config[key] = config_dict[key]
                    print(f"[DEBUG] 更新配置 {key}: {config_dict[key]}")
            
            # 更新配置生成器的配置
            if self.config_generator:
                self.config_generator.update_config(config_dict)
                
        except Exception as e:
            print(f"[ERROR] 更新配置失败: {str(e)}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置（供外部调用）"""
        try:
            self._update_config_from_dict(new_config)
            self.status_updated.emit("智能训练配置已更新")
            print(f"[INFO] 智能训练编排器配置已更新: {new_config}")
        except Exception as e:
            self.error_occurred.emit(f"更新配置失败: {str(e)}")
            print(f"[ERROR] 更新配置失败: {str(e)}")
        
    def _initialize_components(self):
        """初始化组件"""
        try:
            # 连接配置生成器信号
            self.config_generator.config_generated.connect(self._on_config_generated)
            self.config_generator.config_applied.connect(self._on_config_applied)
            self.config_generator.adjustment_recorded.connect(self._on_adjustment_recorded)
            self.config_generator.status_updated.connect(self.status_updated)
            self.config_generator.error_occurred.connect(self.error_occurred)
            
            # 检查LLM配置
            self._check_llm_configuration()
            
            print("智能训练编排器初始化完成")
            
        except Exception as e:
            self.error_occurred.emit(f"初始化组件失败: {str(e)}")
    
    def _check_llm_configuration(self):
        """检查LLM配置"""
        try:
            from ..utils.llm_config_checker import LLMConfigChecker
            checker = LLMConfigChecker()
            result = checker.check_all_configs()
            
            if result['overall_status'] == 'error':
                self.error_occurred.emit("❌ LLM配置检查失败，请检查配置")
                print("[ERROR] LLM配置检查失败:")
                for error in result['errors']:
                    print(f"  ❌ {error}")
            elif result['overall_status'] == 'warning':
                self.status_updated.emit("⚠️ LLM配置有警告，请检查")
                print("[WARNING] LLM配置有警告:")
                for warning in result['warnings']:
                    print(f"  ⚠️ {warning}")
            else:
                self.status_updated.emit("✅ LLM配置检查通过")
                print("[INFO] LLM配置检查通过")
                
        except Exception as e:
            print(f"[WARNING] LLM配置检查失败: {str(e)}")
            self.status_updated.emit("⚠️ LLM配置检查失败，请手动验证")
    
    def set_model_trainer(self, model_trainer: ModelTrainer):
        """设置模型训练器"""
        self.model_trainer = model_trainer
        
        # 连接训练器信号
        if self.model_trainer:
            # 使用ModelTrainer实际存在的信号
            # training_finished信号不传递参数，所以使用lambda包装
            self.model_trainer.training_finished.connect(lambda: self._on_training_completed({}))
            self.model_trainer.training_error.connect(self._on_training_failed)
            self.model_trainer.status_updated.connect(self.status_updated)
    
    def set_training_tab(self, training_tab):
        """设置训练标签页引用"""
        self.training_tab = training_tab
    
    def start_intelligent_training(self, initial_config: Dict[str, Any]) -> bool:
        """开始智能训练"""
        try:
            if self.is_running:
                self.error_occurred.emit("智能训练已在运行中")
                return False
            
            # 验证必需组件
            if not self.model_trainer:
                self.error_occurred.emit("模型训练器未设置")
                return False
            
            if not self.training_tab:
                self.error_occurred.emit("训练标签页未设置")
                return False
            
            # 开始新的训练会话
            session_id = self.config_generator.start_training_session(initial_config)
            if not session_id:
                self.error_occurred.emit("无法开始训练会话")
                return False
            
            # 创建智能训练会话
            self.current_session = IntelligentTrainingSession(
                session_id=session_id,
                start_time=time.time(),
                original_config=initial_config.copy(),
                current_config=initial_config.copy(),
                training_iterations=[],
                total_iterations=0,
                max_iterations=self.config['max_iterations'],
                status='running',
                best_metrics=None,
                best_config=None
            )
            
            self.is_running = True
            self.current_iteration = 0
            
            # 开始第一次训练
            self._start_training_iteration()
            
            # 开始监控
            self._start_monitoring()
            
            self.status_updated.emit(f"智能训练已启动，会话ID: {session_id}")
            self.training_started.emit({
                'session_id': session_id,
                'config': initial_config,
                'timestamp': time.time()
            })
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"启动智能训练失败: {str(e)}")
            return False
    
    def _start_training_iteration(self):
        """开始训练迭代"""
        try:
            self.current_iteration += 1
            
            if self.current_iteration > self.config['max_iterations']:
                self._complete_training_session()
                return
            
            self.status_updated.emit(f"开始第 {self.current_iteration} 次训练迭代")
            
            # 记录迭代开始
            iteration_record = {
                'iteration': self.current_iteration,
                'start_time': time.time(),
                'config': self.current_session.current_config.copy(),
                'status': 'running'
            }
            self.current_session.training_iterations.append(iteration_record)
            
            # 启动训练
            self.state_manager.start_training()
            self.model_trainer.train_model_with_config(self.current_session.current_config)
            
        except Exception as e:
            self.error_occurred.emit(f"启动训练迭代失败: {str(e)}")
            self._on_training_failed({'error': str(e)})
    
    def _start_monitoring(self):
        """开始监控训练"""
        try:
            self.stop_monitoring = False
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
        except Exception as e:
            self.error_occurred.emit(f"启动监控失败: {str(e)}")
    
    def _monitoring_loop(self):
        """监控循环"""
        try:
            while not self.stop_monitoring and self.is_running:
                # 检查训练状态
                if self._should_analyze_and_optimize():
                    # 再次检查停止标志，防止在分析过程中被停止
                    if not self.stop_monitoring and self.is_running:
                        self._analyze_and_optimize()
                
                # 使用较长的睡眠时间，避免过于频繁的检查
                # 每30秒检查一次，而不是每秒检查
                for _ in range(30):  # 30秒检查一次
                    if self.stop_monitoring or not self.is_running:
                        break
                    time.sleep(1)  # 每秒检查一次停止标志
                
        except Exception as e:
            self.error_occurred.emit(f"监控循环出错: {str(e)}")
    
    def _should_analyze_and_optimize(self) -> bool:
        """判断是否应该进行分析和优化"""
        try:
            # 获取当前训练数据
            real_data = self.metrics_collector.get_current_training_data_for_ai()
            if 'error' in real_data:
                return False
            
            current_metrics = real_data.get('current_metrics', {})
            epoch = current_metrics.get('epoch', 0)
            
            # 添加调试日志
            self.status_updated.emit(f"🔍 检查分析条件: epoch={epoch}, min_iteration_epochs={self.config['min_iteration_epochs']}, analysis_interval={self.config['analysis_interval']}")
            
            # 检查是否达到最小训练轮数
            if epoch < self.config['min_iteration_epochs']:
                self.status_updated.emit(f"⏳ 未达到最小训练轮数: {epoch} < {self.config['min_iteration_epochs']}")
                return False
            
            # 检查是否达到分析间隔
            if epoch % self.config['analysis_interval'] != 0:
                self.status_updated.emit(f"⏳ 未达到分析间隔: {epoch} % {self.config['analysis_interval']} != 0")
                return False
            
            self.status_updated.emit(f"✅ 满足分析条件: epoch={epoch}")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"检查分析条件失败: {str(e)}")
            return False
    
    def _analyze_and_optimize(self):
        """分析和优化"""
        try:
            self.status_updated.emit("正在进行智能分析和优化...")
            
            # 生成优化配置
            optimized_config = self.config_generator.generate_optimized_config(
                self.current_session.current_config
            )
            
            # 检查配置是否有变化
            if optimized_config != self.current_session.current_config:
                self.status_updated.emit("检测到配置优化机会，准备重启训练...")
                
                # 开始智能重启过程
                restart_context = {
                    'config': optimized_config,
                    'session_id': self.current_session.session_id if self.current_session else None,
                    'reason': 'parameter_optimization'
                }
                self.state_manager.start_intelligent_restart(restart_context)
                
                # 停止当前训练
                if self.model_trainer:
                    self.model_trainer.stop()
                
                # 等待训练停止
                time.sleep(2)
                
                # 通过信号请求应用配置（避免在后台线程中直接操作UI）
                self.apply_config_requested.emit(restart_context)
            
        except Exception as e:
            self.error_occurred.emit(f"分析和优化失败: {str(e)}")
    
    def _on_training_completed(self, result: Dict[str, Any]):
        """训练完成回调"""
        try:
            if not self.current_session:
                return
            
            # 更新迭代记录
            if self.current_session.training_iterations:
                current_iteration = self.current_session.training_iterations[-1]
                current_iteration['end_time'] = time.time()
                current_iteration['status'] = 'completed'
                current_iteration['final_metrics'] = result.get('metrics', {})
            
            # 更新最佳结果
            metrics = result.get('metrics', {})
            if self._is_better_than_current_best(metrics):
                self.current_session.best_metrics = metrics
                self.current_session.best_config = self.current_session.current_config.copy()
            
            # 发射迭代完成信号
            self.iteration_completed.emit({
                'iteration': self.current_iteration,
                'metrics': metrics,
                'config': self.current_session.current_config,
                'timestamp': time.time()
            })
            
            # 检查是否仍在运行状态
            if not self.is_running or self.stop_monitoring:
                self.status_updated.emit("训练已停止，不继续下一轮")
                return
            
            # 检查是否继续下一轮
            if self._should_continue_training(metrics):
                self.status_updated.emit("训练完成，准备下一轮优化...")
                time.sleep(3)  # 等待一段时间
                
                # 再次检查运行状态，防止在等待期间被停止
                if self.is_running and not self.stop_monitoring:
                    self._start_training_iteration()
                else:
                    self.status_updated.emit("训练已停止，取消下一轮")
            else:
                self._complete_training_session()
            
        except Exception as e:
            self.error_occurred.emit(f"处理训练完成事件失败: {str(e)}")
    
    def _on_training_failed(self, error_message: str):
        """训练失败回调"""
        try:
            if not self.current_session:
                return
            
            # 更新迭代记录
            if self.current_session.training_iterations:
                current_iteration = self.current_session.training_iterations[-1]
                current_iteration['end_time'] = time.time()
                current_iteration['status'] = 'failed'
                current_iteration['error'] = error_message
            
            self.status_updated.emit(f"训练失败: {error_message}")
            
            # 检查是否仍在运行状态
            if not self.is_running or self.stop_monitoring:
                self.status_updated.emit("训练已停止，不进行重试")
                return
            
            # 检查是否重试
            if self.current_iteration < self.config['max_iterations']:
                self.status_updated.emit("准备重试训练...")
                time.sleep(5)  # 等待一段时间
                
                # 再次检查运行状态，防止在等待期间被停止
                if self.is_running and not self.stop_monitoring:
                    self._start_training_iteration()
                else:
                    self.status_updated.emit("训练已停止，取消重试")
            else:
                self._complete_training_session()
            
        except Exception as e:
            self.error_occurred.emit(f"处理训练失败事件失败: {str(e)}")
    
    def _is_better_than_current_best(self, metrics: Dict[str, Any]) -> bool:
        """判断当前结果是否比最佳结果更好"""
        try:
            if not self.current_session.best_metrics:
                return True
            
            current_val_acc = metrics.get('val_accuracy', 0)
            best_val_acc = self.current_session.best_metrics.get('val_accuracy', 0)
            
            # 如果验证准确率提升超过阈值，认为更好
            return current_val_acc > best_val_acc + self.config['improvement_threshold']
            
        except Exception as e:
            self.error_occurred.emit(f"比较结果失败: {str(e)}")
            return False
    
    def _should_continue_training(self, metrics: Dict[str, Any]) -> bool:
        """判断是否应该继续训练"""
        try:
            # 检查是否达到最大迭代次数
            if self.current_iteration >= self.config['max_iterations']:
                return False
            
            # 检查是否收敛
            if self._is_converged(metrics):
                return False
            
            # 检查是否有改进空间
            if not self._has_improvement_potential(metrics):
                return False
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"判断是否继续训练失败: {str(e)}")
            return False
    
    def _is_converged(self, metrics: Dict[str, Any]) -> bool:
        """判断是否已收敛"""
        try:
            val_acc = metrics.get('val_accuracy', 0)
            
            # 如果验证准确率很高，认为已收敛
            if val_acc > 0.95:
                return True
            
            # 检查最近几次迭代的改进
            if len(self.current_session.training_iterations) >= 3:
                recent_improvements = []
                for i in range(-3, 0):
                    if i < 0:
                        iteration = self.current_session.training_iterations[i]
                        if 'final_metrics' in iteration:
                            recent_improvements.append(iteration['final_metrics'].get('val_accuracy', 0))
                
                if len(recent_improvements) >= 3:
                    # 计算改进幅度
                    improvement = max(recent_improvements) - min(recent_improvements)
                    if improvement < self.config['convergence_threshold']:
                        return True
            
            return False
            
        except Exception as e:
            self.error_occurred.emit(f"判断收敛状态失败: {str(e)}")
            return False
    
    def _has_improvement_potential(self, metrics: Dict[str, Any]) -> bool:
        """判断是否还有改进空间"""
        try:
            val_acc = metrics.get('val_accuracy', 0)
            val_loss = metrics.get('val_loss', 1.0)
            
            # 如果准确率较低或损失较高，还有改进空间
            return val_acc < 0.9 or val_loss > 0.1
            
        except Exception as e:
            self.error_occurred.emit(f"判断改进空间失败: {str(e)}")
            return True
    
    def _complete_training_session(self):
        """完成训练会话"""
        try:
            if not self.current_session:
                return
            
            self.current_session.status = 'completed'
            self.is_running = False
            self.stop_monitoring = True
            
            # 停止配置生成器会话
            self.config_generator.stop_training_session()
            
            # 发射完成信号
            self.training_completed.emit({
                'session_id': self.current_session.session_id,
                'total_iterations': self.current_iteration,
                'best_metrics': self.current_session.best_metrics,
                'best_config': self.current_session.best_config,
                'timestamp': time.time()
            })
            
            self.status_updated.emit(f"智能训练完成，共进行了 {self.current_iteration} 次迭代")
            
        except Exception as e:
            self.error_occurred.emit(f"完成训练会话失败: {str(e)}")
    
    def stop_intelligent_training(self):
        """停止智能训练"""
        try:
            if not self.is_running:
                return
            
            # 立即设置停止标志
            self.is_running = False
            self.stop_monitoring = True
            
            # 停止当前训练
            if self.model_trainer:
                self.model_trainer.stop()
            
            # 更新会话状态
            if self.current_session:
                self.current_session.status = 'stopped'
                self.config_generator.stop_training_session()
            
            # 等待监控线程结束
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
                if self.monitoring_thread.is_alive():
                    print("⚠️ 监控线程未能在10秒内结束，强制继续")
            
            # 设置状态为真正停止
            self.state_manager.stop_training()
            self.status_updated.emit("智能训练已停止")
            
        except Exception as e:
            self.error_occurred.emit(f"停止智能训练失败: {str(e)}")
    
    def _on_config_generated(self, result: Dict[str, Any]):
        """配置生成回调"""
        self.config_generated.emit(result)
    
    def _on_config_applied(self, result: Dict[str, Any]):
        """配置应用回调"""
        self.config_applied.emit(result)
    
    def _on_adjustment_recorded(self, adjustment: Dict[str, Any]):
        """调整记录回调"""
        self.status_updated.emit(f"配置调整已记录: {adjustment.get('adjustment_id', 'unknown')}")
    
    def apply_config_request(self, request_data: Dict[str, Any]):
        """处理配置应用请求（在主线程中执行）"""
        try:
            config = request_data.get('config', {})
            session_id = request_data.get('session_id')
            
            if self.training_tab is None:
                self.error_occurred.emit("训练标签页引用为空，无法应用配置")
                return
            
            self.status_updated.emit(f"正在应用配置到训练标签页: {type(self.training_tab)}")
            success = self.config_generator.apply_config_to_training_system(
                config, self.training_tab
            )
            
            if success:
                # 更新当前配置
                if self.current_session:
                    self.current_session.current_config = config
                
                # 完成智能重启过程
                self.state_manager.complete_intelligent_restart()
                
                # 开始新的训练迭代
                self._start_training_iteration()
            else:
                self.error_occurred.emit("应用优化配置失败")
                
        except Exception as e:
            self.error_occurred.emit(f"应用配置请求失败: {str(e)}")
    
    def get_current_session_info(self) -> Dict[str, Any]:
        """获取当前会话信息"""
        if not self.current_session:
            return {}
        
        return {
            'session_id': self.current_session.session_id,
            'status': self.current_session.status,
            'current_iteration': self.current_iteration,
            'max_iterations': self.current_session.max_iterations,
            'total_iterations': len(self.current_session.training_iterations),
            'best_metrics': self.current_session.best_metrics,
            'is_running': self.is_running
        }
    
    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """获取调整历史"""
        return self.config_generator.get_adjustment_history()
    
    def export_training_report(self) -> Dict[str, Any]:
        """导出训练报告"""
        if not self.current_session:
            return {}
        
        return {
            'session_info': self.get_current_session_info(),
            'training_iterations': [asdict(iter) for iter in self.current_session.training_iterations],
            'adjustment_history': self.get_adjustment_history(),
            'config_generator_report': self.config_generator.export_adjustment_report(),
            'export_time': time.time()
        }
    
    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """获取干预历史"""
        if not self.current_session:
            return []
        
        # 从训练迭代中提取干预记录
        interventions = []
        for iteration in self.current_session.training_iterations:
            if 'intervention' in iteration:
                interventions.append(iteration['intervention'])
        
        return interventions
    
    def update_training_progress(self, metrics: Dict[str, Any]):
        """更新训练进度"""
        try:
            # 这个方法主要用于兼容性，智能训练编排器通过metrics_collector获取训练数据
            # 这里可以添加一些额外的进度处理逻辑
            if self.current_session:
                # 更新当前会话的训练进度信息
                if self.current_session.training_iterations:
                    current_iteration = self.current_session.training_iterations[-1]
                    current_iteration['latest_metrics'] = metrics
                    current_iteration['last_update_time'] = time.time()
            
            # 可以在这里添加进度更新的事件处理
            # 例如：检查是否达到分析条件、更新UI显示等
            
        except Exception as e:
            self.error_occurred.emit(f"更新训练进度失败: {str(e)}")
    
    def save_session_report(self, file_path: str):
        """保存会话报告"""
        try:
            if not self.current_session:
                self.error_occurred.emit("没有活跃的会话可以保存")
                return
            
            # 准备报告数据
            report_data = {
                'session_info': self.get_current_session_info(),
                'training_iterations': self.current_session.training_iterations,
                'best_metrics': self.current_session.best_metrics,
                'best_config': self.current_session.best_config,
                'intervention_history': self.get_intervention_history(),
                'adjustment_history': self.get_adjustment_history(),
                'timestamp': time.time()
            }
            
            # 保存到文件
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.status_updated.emit(f"会话报告已保存到: {file_path}")
            
        except Exception as e:
            self.error_occurred.emit(f"保存会话报告失败: {str(e)}")
    
    def _on_state_changed(self, new_state: TrainingState, message: str):
        """处理状态改变事件"""
        try:
            # 根据状态类型发射相应的信号
            if new_state == TrainingState.INTELLIGENT_RESTARTING:
                self.status_updated.emit("🔄 智能训练正在重启...")
            elif new_state == TrainingState.RUNNING:
                if self.state_manager.is_intelligent_restarting():
                    self.status_updated.emit("✅ 智能训练重启完成")
                else:
                    self.status_updated.emit("🚀 训练已开始")
            elif new_state == TrainingState.STOPPED:
                self.status_updated.emit("⏹️ 训练已停止")
            elif new_state == TrainingState.COMPLETED:
                self.status_updated.emit("✅ 训练已完成")
            elif new_state == TrainingState.ERROR:
                self.status_updated.emit(f"❌ {message}")
        except Exception as e:
            self.error_occurred.emit(f"处理状态改变事件失败: {str(e)}")
    
    def get_state_manager(self) -> IntelligentTrainingStateManager:
        """获取状态管理器"""
        return self.state_manager
