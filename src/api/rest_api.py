"""
REST API - HTTP REST接口服务

提供训练系统的HTTP REST API接口，支持指标查询、训练控制等功能。
"""

import json
import time
from typing import Dict, Any, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging


class TrainingAPI:
    """训练系统REST API服务"""
    
    def __init__(self, training_system=None):
        """
        初始化REST API服务
        
        Args:
            training_system: 训练系统实例，用于控制训练过程
        """
        self.app = Flask(__name__)
        CORS(self.app)
        self.training_system = training_system
        self.metrics_history = []
        self.current_metrics = {}
        self.logger = logging.getLogger(__name__)
        
        # 设置路由
        self._setup_routes()
        
        # API统计信息
        self.api_stats = {
            'requests_count': 0,
            'start_time': time.time(),
            'endpoints_accessed': {}
        }
    
    def _setup_routes(self):
        """设置API路由"""
        
        # 指标相关接口
        @self.app.route('/api/metrics/current', methods=['GET'])
        def get_current_metrics():
            """获取当前训练指标"""
            self._update_stats('/api/metrics/current')
            return jsonify({
                'status': 'success',
                'data': self.current_metrics,
                'timestamp': time.time()
            })
        
        @self.app.route('/api/metrics/history', methods=['GET'])
        def get_metrics_history():
            """获取历史训练指标"""
            self._update_stats('/api/metrics/history')
            
            # 获取查询参数
            limit = request.args.get('limit', 100, type=int)
            offset = request.args.get('offset', 0, type=int)
            metric_type = request.args.get('type', None)
            
            # 过滤和分页
            filtered_metrics = self.metrics_history
            if metric_type:
                filtered_metrics = [m for m in filtered_metrics 
                                  if m.get('event_type') == metric_type]
            
            total_count = len(filtered_metrics)
            paginated_metrics = filtered_metrics[offset:offset + limit]
            
            return jsonify({
                'status': 'success',
                'data': {
                    'metrics': paginated_metrics,
                    'pagination': {
                        'limit': limit,
                        'offset': offset,
                        'total': total_count,
                        'has_more': offset + limit < total_count
                    }
                },
                'timestamp': time.time()
            })
        
        @self.app.route('/api/metrics/summary', methods=['GET'])
        def get_metrics_summary():
            """获取指标摘要统计"""
            self._update_stats('/api/metrics/summary')
            
            if not self.metrics_history:
                return jsonify({
                    'status': 'success',
                    'data': {'message': 'No metrics available'},
                    'timestamp': time.time()
                })
            
            # 计算摘要统计
            recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
            
            summary = {
                'total_metrics': len(self.metrics_history),
                'latest_epoch': self.current_metrics.get('epoch', 0),
                'training_duration': time.time() - self.api_stats['start_time'],
                'recent_performance': self._calculate_recent_performance(recent_metrics)
            }
            
            return jsonify({
                'status': 'success',
                'data': summary,
                'timestamp': time.time()
            })
        
        # 训练控制接口
        @self.app.route('/api/training/control', methods=['POST'])
        def control_training():
            """训练控制接口"""
            self._update_stats('/api/training/control')
            
            try:
                data = request.get_json()
                if not data:
                    return jsonify({
                        'status': 'error',
                        'message': 'No JSON data provided'
                    }), 400
                
                action = data.get('action')
                if not action:
                    return jsonify({
                        'status': 'error',
                        'message': 'No action specified'
                    }), 400
                
                # 执行训练控制操作
                result = self._execute_training_control(action, data)
                
                return jsonify({
                    'status': 'success',
                    'data': result,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                self.logger.error(f"训练控制操作失败: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/training/status', methods=['GET'])
        def get_training_status():
            """获取训练状态"""
            self._update_stats('/api/training/status')
            
            status_info = {
                'is_training': self._is_training_active(),
                'current_epoch': self.current_metrics.get('epoch', 0),
                'total_epochs': self.current_metrics.get('total_epochs', 0),
                'model_name': self.current_metrics.get('model_name', 'Unknown'),
                'training_speed': self.current_metrics.get('training_speed', 0),
                'gpu_memory_usage': self.current_metrics.get('gpu_memory_used', 0)
            }
            
            return jsonify({
                'status': 'success',
                'data': status_info,
                'timestamp': time.time()
            })
        
        # 系统信息接口
        @self.app.route('/api/system/info', methods=['GET'])
        def get_system_info():
            """获取系统信息"""
            self._update_stats('/api/system/info')
            
            system_info = {
                'api_version': '1.0.0',
                'uptime': time.time() - self.api_stats['start_time'],
                'total_requests': self.api_stats['requests_count'],
                'endpoints_stats': self.api_stats['endpoints_accessed'],
                'metrics_count': len(self.metrics_history)
            }
            
            return jsonify({
                'status': 'success',
                'data': system_info,
                'timestamp': time.time()
            })
        
        @self.app.route('/api/system/health', methods=['GET'])
        def health_check():
            """健康检查接口"""
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'uptime': time.time() - self.api_stats['start_time']
            })
        
        # 错误处理
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'status': 'error',
                'message': 'Endpoint not found',
                'timestamp': time.time()
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'status': 'error',
                'message': 'Internal server error',
                'timestamp': time.time()
            }), 500
    
    def _update_stats(self, endpoint: str):
        """更新API统计信息"""
        self.api_stats['requests_count'] += 1
        if endpoint not in self.api_stats['endpoints_accessed']:
            self.api_stats['endpoints_accessed'][endpoint] = 0
        self.api_stats['endpoints_accessed'][endpoint] += 1
    
    def _calculate_recent_performance(self, recent_metrics: list) -> Dict[str, Any]:
        """计算最近的性能指标"""
        if not recent_metrics:
            return {}
        
        # 提取训练损失和准确率
        train_losses = []
        val_accuracies = []
        
        for metric in recent_metrics:
            data = metric.get('data', {})
            if 'train_loss' in data:
                train_losses.append(data['train_loss'])
            if 'val_accuracy' in data:
                val_accuracies.append(data['val_accuracy'])
        
        performance = {}
        if train_losses:
            performance['avg_train_loss'] = sum(train_losses) / len(train_losses)
            performance['loss_trend'] = 'decreasing' if len(train_losses) > 1 and train_losses[-1] < train_losses[0] else 'stable'
        
        if val_accuracies:
            performance['avg_val_accuracy'] = sum(val_accuracies) / len(val_accuracies)
            performance['accuracy_trend'] = 'increasing' if len(val_accuracies) > 1 and val_accuracies[-1] > val_accuracies[0] else 'stable'
        
        return performance
    
    def _execute_training_control(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行训练控制操作"""
        if not self.training_system:
            return {'message': 'Training system not available'}
        
        if action == 'pause':
            # 暂停训练
            return {'message': 'Training paused', 'action': 'pause'}
        elif action == 'resume':
            # 恢复训练
            return {'message': 'Training resumed', 'action': 'resume'}
        elif action == 'stop':
            # 停止训练
            if hasattr(self.training_system, 'stop'):
                self.training_system.stop()
            return {'message': 'Training stopped', 'action': 'stop'}
        elif action == 'get_config':
            # 获取训练配置
            return {'message': 'Training configuration', 'config': {}}
        else:
            raise ValueError(f'Unknown action: {action}')
    
    def _is_training_active(self) -> bool:
        """检查训练是否正在进行"""
        if not self.training_system:
            return False
        
        # 检查训练系统状态
        if hasattr(self.training_system, 'training_thread'):
            return (self.training_system.training_thread and 
                   self.training_system.training_thread.isRunning())
        
        return False
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """更新训练指标"""
        # 格式化指标数据
        formatted_metrics = {
            'timestamp': time.time(),
            'event_type': 'metrics_update',
            'data': metrics
        }
        
        # 更新当前指标和历史记录
        self.current_metrics = metrics
        self.metrics_history.append(formatted_metrics)
        
        # 保持历史记录在合理范围内
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
    
    def start_server(self, host='127.0.0.1', port=8890, debug=False):
        """启动REST API服务器"""
        self.logger.info(f"启动REST API服务器: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)
    
    def get_app(self):
        """获取Flask应用实例"""
        return self.app 