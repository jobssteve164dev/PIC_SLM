import logging
import json
import os
import time


class MetricsDataManager:
    """训练指标数据管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_metrics()
        self.update_frequency = 1.0  # 默认更新频率为1Hz
        self.last_update_time = 0
        self.task_type = None  # 任务类型：'classification' 或 'detection'
        self.last_metrics = {}  # 保存最后一次更新的指标
        
        # 添加记录历史极值的变量，用于缩放坐标轴
        self.loss_min_history = float('inf')
        self.loss_max_history = 0.0
        self.acc_min_history = float('inf')
        self.acc_max_history = 0.0
        
    def setup_metrics(self):
        """初始化训练指标"""
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.maps = []
        self.train_maps = []  # 添加训练准确率/mAP存储
        self.learning_rates = []
        # 添加更多指标存储
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.map50s = []
        self.map75s = []
        self.class_losses = []
        self.obj_losses = []
        self.box_losses = []
        # 添加分类任务特有指标
        self.roc_aucs = []
        self.average_precisions = []
        self.top_k_accuracies = []
        self.balanced_accuracies = []
        
    def set_update_frequency(self, frequency):
        """设置更新频率"""
        try:
            self.update_frequency = max(1, min(frequency, 100))  # 限制频率范围
            self.logger.info(f"更新频率设置为: {self.update_frequency}")
        except Exception as e:
            self.logger.error(f"设置更新频率时出错: {str(e)}")
            
    def can_update(self):
        """检查是否可以更新（基于更新频率）"""
        current_time = time.time()
        if current_time - self.last_update_time < 1.0 / self.update_frequency:
            return False
        self.last_update_time = current_time
        return True
        
    def determine_task_type(self, metrics):
        """确定任务类型并持久化"""
        if self.task_type is None:
            if 'val_accuracy' in metrics:
                self.task_type = 'classification'
            elif 'val_map' in metrics:
                self.task_type = 'detection'
            if self.task_type:
                self.logger.info(f"任务类型设置为: {self.task_type}")
                
    def fill_missing_epochs(self, epoch):
        """填充中间缺失的epochs，确保连续性"""
        if epoch > 0 and self.epochs and epoch > max(self.epochs) + 1:
            # 获取上一个epoch的索引
            last_idx = self.epochs.index(max(self.epochs))
            
            # 填充中间缺失的epochs
            for missing_epoch in range(max(self.epochs) + 1, epoch):
                self.epochs.append(missing_epoch)
                
                # 继承上一个epoch的值来确保连续性
                self.train_losses.append(self.train_losses[last_idx])
                self.val_losses.append(self.val_losses[last_idx])
                self.maps.append(self.maps[last_idx])
                self.train_maps.append(self.train_maps[last_idx] if hasattr(self, 'train_maps') and self.train_maps else 0.0)
                self.learning_rates.append(self.learning_rates[last_idx])
                
                # 继承其他指标
                self.precisions.append(self.precisions[last_idx] if self.precisions else 0.0)
                self.recalls.append(self.recalls[last_idx] if self.recalls else 0.0)
                self.f1_scores.append(self.f1_scores[last_idx] if self.f1_scores else 0.0)
                self.map50s.append(self.map50s[last_idx] if self.map50s else 0.0)
                self.map75s.append(self.map75s[last_idx] if self.map75s else 0.0)
                self.class_losses.append(self.class_losses[last_idx] if self.class_losses else 0.0)
                self.obj_losses.append(self.obj_losses[last_idx] if self.obj_losses else 0.0)
                self.box_losses.append(self.box_losses[last_idx] if self.box_losses else 0.0)
                
                # 继承分类任务特有指标
                self.roc_aucs.append(self.roc_aucs[last_idx] if self.roc_aucs else 0.0)
                self.average_precisions.append(self.average_precisions[last_idx] if self.average_precisions else 0.0)
                self.top_k_accuracies.append(self.top_k_accuracies[last_idx] if self.top_k_accuracies else 0.0)
                self.balanced_accuracies.append(self.balanced_accuracies[last_idx] if self.balanced_accuracies else 0.0)
                
    def add_new_epoch(self, epoch, metrics):
        """添加新的epoch数据"""
        self.epochs.append(epoch)
        self.train_losses.append(metrics.get('train_loss', 0.0))
        self.val_losses.append(metrics.get('val_loss', 0.0))
        self.learning_rates.append(metrics.get('learning_rate', 0.0))
        
        # 根据任务类型获取准确率/mAP (兼容分类和检测任务)
        if 'val_map' in metrics:  # 检测任务
            self.maps.append(metrics.get('val_map', 0.0))
            self.train_maps.append(metrics.get('train_map', 0.0))  # 添加训练mAP
        elif 'val_accuracy' in metrics:  # 分类任务
            self.maps.append(metrics.get('val_accuracy', 0.0))
            self.train_maps.append(metrics.get('train_accuracy', 0.0))  # 添加训练准确率
        else:
            self.maps.append(0.0)
            self.train_maps.append(0.0)
        
        # 更新其他指标
        self.precisions.append(metrics.get('precision', 0.0))
        self.recalls.append(metrics.get('recall', 0.0))
        self.f1_scores.append(metrics.get('f1_score', 0.0))
        self.map50s.append(metrics.get('mAP50', 0.0))
        self.map75s.append(metrics.get('mAP75', 0.0))
        self.class_losses.append(metrics.get('class_loss', 0.0))
        self.obj_losses.append(metrics.get('obj_loss', 0.0))
        self.box_losses.append(metrics.get('box_loss', 0.0))
        
        # 更新分类任务特有指标
        self.roc_aucs.append(metrics.get('roc_auc', 0.0))
        self.average_precisions.append(metrics.get('average_precision', 0.0))
        self.top_k_accuracies.append(metrics.get('top_k_accuracy', 0.0))
        self.balanced_accuracies.append(metrics.get('balanced_accuracy', 0.0))
        
    def update_existing_epoch(self, epoch, metrics):
        """更新现有epoch的数据"""
        idx = self.epochs.index(epoch)
        self.train_losses[idx] = metrics['train_loss']
        self.val_losses[idx] = metrics['val_loss']
        self.learning_rates[idx] = metrics['learning_rate']
        
        # 更新准确率/mAP
        if 'val_map' in metrics:
            self.maps[idx] = metrics['val_map']
            self.train_maps[idx] = metrics.get('train_map', 0.0)  # 更新训练mAP
        elif 'val_accuracy' in metrics:
            self.maps[idx] = metrics['val_accuracy']
            self.train_maps[idx] = metrics.get('train_accuracy', 0.0)  # 更新训练准确率
        
        # 更新其他指标
        if 'precision' in metrics:
            self.precisions[idx] = metrics['precision']
        if 'recall' in metrics:
            self.recalls[idx] = metrics['recall']
        if 'f1_score' in metrics:
            self.f1_scores[idx] = metrics['f1_score']
        if 'mAP50' in metrics:
            self.map50s[idx] = metrics['mAP50']
        if 'mAP75' in metrics:
            self.map75s[idx] = metrics['mAP75']
        if 'class_loss' in metrics:
            self.class_losses[idx] = metrics['class_loss']
        if 'obj_loss' in metrics:
            self.obj_losses[idx] = metrics['obj_loss']
        if 'box_loss' in metrics:
            self.box_losses[idx] = metrics['box_loss']
        
        # 更新分类任务特有指标
        if 'roc_auc' in metrics:
            self.roc_aucs[idx] = metrics['roc_auc']
        if 'average_precision' in metrics:
            self.average_precisions[idx] = metrics['average_precision']
        if 'top_k_accuracy' in metrics:
            self.top_k_accuracies[idx] = metrics['top_k_accuracy']
        if 'balanced_accuracy' in metrics:
            self.balanced_accuracies[idx] = metrics['balanced_accuracy']
            
    def update_metrics(self, metrics):
        """更新训练指标"""
        try:
            # 获取epoch
            epoch = metrics.get('epoch', 0)
            
            # 确定任务类型并持久化
            self.determine_task_type(metrics)
            
            # 检查更新频率
            if not self.can_update():
                return
            
            # 确保连续性：如果有跳跃的epoch，填充中间缺失的数据点
            self.fill_missing_epochs(epoch)
            
            # 处理新的epoch或更新现有epoch的数据
            if epoch not in self.epochs:
                self.add_new_epoch(epoch, metrics)
            else:
                self.update_existing_epoch(epoch, metrics)
            
            # 保存最后一次更新的指标以便后续使用
            self.last_metrics = metrics.copy()
            
            # 更新历史极值
            self.update_historical_extremes()
            
            return True  # 成功更新
            
        except Exception as e:
            self.logger.error(f"更新训练指标时出错: {str(e)}")
            return False
            
    def update_historical_extremes(self):
        """更新历史极值记录"""
        try:
            # 更新损失极值
            all_losses = []
            if self.train_losses:
                all_losses.extend([l for l in self.train_losses if l is not None and l > 0])
            if self.val_losses:
                all_losses.extend([l for l in self.val_losses if l is not None and l > 0])
            
            if all_losses:
                current_min = min(all_losses)
                current_max = max(all_losses)
                self.loss_min_history = min(self.loss_min_history, current_min)
                self.loss_max_history = max(self.loss_max_history, current_max)
            
            # 更新准确率/mAP极值
            all_metrics = []
            if self.maps:
                all_metrics.extend([m for m in self.maps if m is not None and m > 0])
            if self.train_maps:
                all_metrics.extend([m for m in self.train_maps if m is not None and m > 0])
            
            if all_metrics:
                current_min = min(all_metrics)
                current_max = max(all_metrics)
                if current_min < self.acc_min_history:
                    self.acc_min_history = current_min
                if current_max > self.acc_max_history:
                    self.acc_max_history = current_max
                    
        except Exception as e:
            self.logger.error(f"更新历史极值时出错: {str(e)}")
            
    def get_data_for_display(self):
        """获取用于显示的数据，确保所有数据列表长度一致"""
        min_len = min(len(self.epochs), len(self.train_losses), len(self.val_losses), len(self.maps))
        
        if min_len == 0:
            return None  # 无数据
            
        # 基础数据
        data = {
            'epochs': self.epochs[:min_len],
            'train_losses': self.train_losses[:min_len],
            'val_losses': self.val_losses[:min_len],
            'maps': self.maps[:min_len],
            'train_maps': self.train_maps[:min_len] if hasattr(self, 'train_maps') and len(self.train_maps) > 0 else [],
            'learning_rates': self.learning_rates[:min_len] if len(self.learning_rates) >= min_len else self.learning_rates + [0.0] * (min_len - len(self.learning_rates))
        }
        
        # 确保附加指标列表长度一致
        data.update({
            'precisions': self.precisions[:min_len] if len(self.precisions) >= min_len else self.precisions + [0.0] * (min_len - len(self.precisions)),
            'recalls': self.recalls[:min_len] if len(self.recalls) >= min_len else self.recalls + [0.0] * (min_len - len(self.recalls)),
            'f1_scores': self.f1_scores[:min_len] if len(self.f1_scores) >= min_len else self.f1_scores + [0.0] * (min_len - len(self.f1_scores)),
            'map50s': self.map50s[:min_len] if len(self.map50s) >= min_len else self.map50s + [0.0] * (min_len - len(self.map50s)),
            'map75s': self.map75s[:min_len] if len(self.map75s) >= min_len else self.map75s + [0.0] * (min_len - len(self.map75s)),
            'class_losses': self.class_losses[:min_len] if len(self.class_losses) >= min_len else self.class_losses + [0.0] * (min_len - len(self.class_losses)),
            'obj_losses': self.obj_losses[:min_len] if len(self.obj_losses) >= min_len else self.obj_losses + [0.0] * (min_len - len(self.obj_losses)),
            'box_losses': self.box_losses[:min_len] if len(self.box_losses) >= min_len else self.box_losses + [0.0] * (min_len - len(self.box_losses)),
            'roc_aucs': self.roc_aucs[:min_len] if len(self.roc_aucs) >= min_len else self.roc_aucs + [0.0] * (min_len - len(self.roc_aucs)),
            'average_precisions': self.average_precisions[:min_len] if len(self.average_precisions) >= min_len else self.average_precisions + [0.0] * (min_len - len(self.average_precisions)),
            'top_k_accuracies': self.top_k_accuracies[:min_len] if len(self.top_k_accuracies) >= min_len else self.top_k_accuracies + [0.0] * (min_len - len(self.top_k_accuracies)),
            'balanced_accuracies': self.balanced_accuracies[:min_len] if len(self.balanced_accuracies) >= min_len else self.balanced_accuracies + [0.0] * (min_len - len(self.balanced_accuracies))
        })
        
        return data
        
    def get_historical_ranges(self):
        """获取历史极值范围"""
        return {
            'loss_min': self.loss_min_history,
            'loss_max': self.loss_max_history,
            'acc_min': self.acc_min_history,
            'acc_max': self.acc_max_history
        }
        
    def reset_data(self):
        """重置所有数据"""
        self.setup_metrics()
        self.task_type = None
        self.last_metrics = {}
        
        # 重置历史极值记录
        self.loss_min_history = float('inf')
        self.loss_max_history = 0.0
        self.acc_min_history = float('inf')
        self.acc_max_history = 0.0
        
        self.logger.info("指标数据已重置")
        
    def save_training_data(self, save_path=None):
        """保存训练数据"""
        try:
            if save_path is None:
                save_path = os.path.join('models', 'training_data.json')
                
            data = {
                'epochs': self.epochs,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'maps': self.maps,
                'train_maps': self.train_maps,
                'learning_rates': self.learning_rates,
                'precisions': self.precisions,
                'recalls': self.recalls,
                'f1_scores': self.f1_scores,
                'map50s': self.map50s,
                'map75s': self.map75s,
                'class_losses': self.class_losses,
                'obj_losses': self.obj_losses,
                'box_losses': self.box_losses,
                'roc_aucs': self.roc_aucs,
                'average_precisions': self.average_precisions,
                'top_k_accuracies': self.top_k_accuracies,
                'balanced_accuracies': self.balanced_accuracies,
                'task_type': self.task_type
            }
            
            # 保存为JSON文件
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"训练数据已保存到: {save_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存训练数据时出错: {str(e)}")
            return False 