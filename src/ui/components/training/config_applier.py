"""
配置应用器

该模块提供将训练配置应用到不同训练组件的功能
"""

from PyQt5.QtWidgets import QMessageBox
import os


class ConfigApplier:
    """配置应用器类，用于将配置应用到训练组件"""
    
    @staticmethod
    def apply_to_classification_widget(config, classification_widget):
        """将配置应用到图像分类训练组件"""
        try:
            # 基础参数
            if 'model_name' in config:
                model_combo = classification_widget.model_combo
                model_name = config['model_name']
                index = model_combo.findText(model_name)
                if index >= 0:
                    model_combo.setCurrentIndex(index)
            
            if 'batch_size' in config:
                classification_widget.batch_size_spin.setValue(config['batch_size'])
            
            if 'num_epochs' in config:
                classification_widget.epochs_spin.setValue(config['num_epochs'])
            
            if 'learning_rate' in config:
                classification_widget.lr_spin.setValue(config['learning_rate'])
            
            if 'optimizer' in config:
                optimizer_combo = classification_widget.optimizer_combo
                optimizer = config['optimizer']
                index = optimizer_combo.findText(optimizer)
                if index >= 0:
                    optimizer_combo.setCurrentIndex(index)
            
            if 'lr_scheduler' in config:
                lr_scheduler_combo = classification_widget.lr_scheduler_combo
                lr_scheduler = config['lr_scheduler']
                index = lr_scheduler_combo.findText(lr_scheduler)
                if index >= 0:
                    lr_scheduler_combo.setCurrentIndex(index)
            
            if 'weight_decay' in config:
                classification_widget.weight_decay_spin.setValue(config['weight_decay'])
            
            if 'activation_function' in config and config['activation_function'] != 'None':
                activation_combo = classification_widget.activation_combo
                activation = config['activation_function']
                index = activation_combo.findText(activation)
                if index >= 0:
                    activation_combo.setCurrentIndex(index)
            
            # 高级参数
            if hasattr(classification_widget, 'use_augmentation_checkbox') and 'use_augmentation' in config:
                classification_widget.use_augmentation_checkbox.setChecked(config['use_augmentation'])
            
            if hasattr(classification_widget, 'augmentation_checkbox') and 'use_augmentation' in config:
                classification_widget.augmentation_checkbox.setChecked(config['use_augmentation'])
            
            if hasattr(classification_widget, 'early_stopping_checkbox') and 'early_stopping' in config:
                classification_widget.early_stopping_checkbox.setChecked(config['early_stopping'])
            
            if hasattr(classification_widget, 'early_stopping_patience_spin') and 'early_stopping_patience' in config:
                classification_widget.early_stopping_patience_spin.setValue(config['early_stopping_patience'])
            
            if hasattr(classification_widget, 'mixed_precision_checkbox') and 'mixed_precision' in config:
                classification_widget.mixed_precision_checkbox.setChecked(config['mixed_precision'])
            
            if hasattr(classification_widget, 'gradient_clipping_checkbox') and 'gradient_clipping' in config:
                classification_widget.gradient_clipping_checkbox.setChecked(config['gradient_clipping'])
            
            if hasattr(classification_widget, 'gradient_clipping_value_spin') and 'gradient_clipping_value' in config:
                classification_widget.gradient_clipping_value_spin.setValue(config['gradient_clipping_value'])
            
            if hasattr(classification_widget, 'dropout_rate_spin') and 'dropout_rate' in config:
                classification_widget.dropout_rate_spin.setValue(config['dropout_rate'])
            
            # 预训练模型
            if 'use_pretrained' in config:
                if hasattr(classification_widget, 'pretrained_checkbox'):
                    classification_widget.pretrained_checkbox.setChecked(config['use_pretrained'])
            
            if 'pretrained_path' in config and config['pretrained_path']:
                if hasattr(classification_widget, 'use_local_pretrained_checkbox'):
                    classification_widget.use_local_pretrained_checkbox.setChecked(True)
                    classification_widget.pretrained_path_edit.setText(config['pretrained_path'])
            
            # 数据路径
            if 'data_dir' in config and config['data_dir']:
                if os.path.exists(config['data_dir']):
                    classification_widget.set_folder_path(config['data_dir'])
            
            # 应用高级超参数（直接使用配置，不进行智能推断）
            if hasattr(classification_widget, 'advanced_hyperparams_widget'):
                # 直接使用配置文件中的启用状态字段，不进行任何推断
                classification_widget.advanced_hyperparams_widget.set_config(config)
            
            return True
            
        except Exception as e:
            print(f"应用配置到分类组件失败: {str(e)}")
            return False
    
    @staticmethod
    def apply_to_detection_widget(config, detection_widget):
        """将配置应用到目标检测训练组件"""
        try:
            # 基础参数 - 检测模型的参数可能与分类不同
            if 'model_name' in config:
                # 检测任务可能使用不同的模型名称映射
                model_combo = detection_widget.model_combo
                model_name = config['model_name']
                
                # 尝试直接匹配
                index = model_combo.findText(model_name)
                if index >= 0:
                    model_combo.setCurrentIndex(index)
                else:
                    # 如果是分类模型，尝试映射到检测模型
                    detection_model_mapping = {
                        'MobileNetV2': 'YOLOv5s',
                        'ResNet18': 'YOLOv5m',
                        'ResNet50': 'YOLOv5l',
                        'EfficientNetB0': 'YOLOv8s',
                        'EfficientNetB4': 'YOLOv8m'
                    }
                    mapped_model = detection_model_mapping.get(model_name)
                    if mapped_model:
                        index = model_combo.findText(mapped_model)
                        if index >= 0:
                            model_combo.setCurrentIndex(index)
            
            if 'batch_size' in config:
                detection_widget.batch_size_spin.setValue(config['batch_size'])
            
            if 'num_epochs' in config:
                detection_widget.epochs_spin.setValue(config['num_epochs'])
            
            if 'learning_rate' in config:
                detection_widget.lr_spin.setValue(config['learning_rate'])
            
            if 'optimizer' in config:
                optimizer_combo = detection_widget.optimizer_combo
                optimizer = config['optimizer']
                index = optimizer_combo.findText(optimizer)
                if index >= 0:
                    optimizer_combo.setCurrentIndex(index)
            
            # 学习率调度器
            if 'lr_scheduler' in config:
                lr_scheduler_combo = detection_widget.lr_scheduler_combo
                lr_scheduler = config['lr_scheduler']
                index = lr_scheduler_combo.findText(lr_scheduler)
                if index >= 0:
                    lr_scheduler_combo.setCurrentIndex(index)
            
            # 权重衰减
            if 'weight_decay' in config:
                detection_widget.weight_decay_spin.setValue(config['weight_decay'])
            
            # 激活函数
            if 'activation_function' in config and config['activation_function'] != 'None':
                activation_combo = detection_widget.activation_combo
                activation = config['activation_function']
                index = activation_combo.findText(activation)
                if index >= 0:
                    activation_combo.setCurrentIndex(index)
            
            # 高级参数
            if hasattr(detection_widget, 'augmentation_checkbox') and 'use_augmentation' in config:
                detection_widget.augmentation_checkbox.setChecked(config['use_augmentation'])
            
            if hasattr(detection_widget, 'early_stopping_checkbox') and 'early_stopping' in config:
                detection_widget.early_stopping_checkbox.setChecked(config['early_stopping'])
            
            if hasattr(detection_widget, 'early_stopping_patience_spin') and 'early_stopping_patience' in config:
                detection_widget.early_stopping_patience_spin.setValue(config['early_stopping_patience'])
            
            if hasattr(detection_widget, 'mixed_precision_checkbox') and 'mixed_precision' in config:
                detection_widget.mixed_precision_checkbox.setChecked(config['mixed_precision'])
            
            if hasattr(detection_widget, 'gradient_clipping_checkbox') and 'gradient_clipping' in config:
                detection_widget.gradient_clipping_checkbox.setChecked(config['gradient_clipping'])
            
            if hasattr(detection_widget, 'gradient_clipping_value_spin') and 'gradient_clipping_value' in config:
                detection_widget.gradient_clipping_value_spin.setValue(config['gradient_clipping_value'])
            
            if hasattr(detection_widget, 'dropout_spin') and 'dropout_rate' in config:
                detection_widget.dropout_spin.setValue(config['dropout_rate'])
            
            if hasattr(detection_widget, 'ema_checkbox') and 'use_ema' in config:
                detection_widget.ema_checkbox.setChecked(config['use_ema'])
            
            # 预训练模型
            if 'use_pretrained' in config:
                if hasattr(detection_widget, 'pretrained_checkbox'):
                    detection_widget.pretrained_checkbox.setChecked(config['use_pretrained'])
            
            if 'pretrained_path' in config and config['pretrained_path']:
                if hasattr(detection_widget, 'use_local_pretrained_checkbox'):
                    detection_widget.use_local_pretrained_checkbox.setChecked(True)
                    detection_widget.pretrained_path_edit.setText(config['pretrained_path'])
            
            # 数据路径
            if 'data_dir' in config and config['data_dir']:
                # 检测数据目录可能需要特殊处理
                detection_data_dir = config['data_dir'].replace('dataset', 'detection_data')
                if os.path.exists(detection_data_dir):
                    detection_widget.set_folder_path(detection_data_dir)
                elif os.path.exists(config['data_dir']):
                    detection_widget.set_folder_path(config['data_dir'])
            
            # 应用高级超参数（直接使用配置，不进行智能推断）
            if hasattr(detection_widget, 'advanced_hyperparams_widget'):
                # 直接使用配置文件中的启用状态字段，不进行任何推断
                detection_widget.advanced_hyperparams_widget.set_config(config)
            
            return True
            
        except Exception as e:
            print(f"应用配置到检测组件失败: {str(e)}")
            return False
    
    @staticmethod
    def apply_to_training_tab(config, training_tab):
        """将配置应用到训练标签页"""
        try:
            # 根据任务类型选择相应的组件
            task_type = config.get('task_type', 'classification')
            
            if task_type == 'classification':
                training_tab.classification_radio.setChecked(True)
                training_tab.stacked_widget.setCurrentIndex(0)
                ConfigApplier.apply_to_classification_widget(config, training_tab.classification_widget)
            elif task_type == 'detection':
                training_tab.detection_radio.setChecked(True)
                training_tab.stacked_widget.setCurrentIndex(1)
                ConfigApplier.apply_to_detection_widget(config, training_tab.detection_widget)
            
            return True
            
        except Exception as e:
            print(f"应用配置到训练标签页失败: {str(e)}")
            return False
    
    @staticmethod
    def show_apply_result(success, error_msg=None):
        """显示应用结果"""
        if success:
            QMessageBox.information(None, "成功", "训练配置已成功应用到界面！")
        else:
            error_text = f"应用配置失败"
            if error_msg:
                error_text += f": {error_msg}"
            QMessageBox.warning(None, "警告", error_text) 