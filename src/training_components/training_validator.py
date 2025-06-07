"""
训练配置验证器 - 负责验证训练配置的有效性

主要功能：
- 验证数据集路径是否存在
- 验证训练参数的合理性
- 验证模型配置的正确性
- 验证保存路径的可写性
"""

import os
from PyQt5.QtCore import QObject, pyqtSignal


class TrainingValidator(QObject):
    """训练配置验证器"""
    
    # 信号定义
    status_updated = pyqtSignal(str)
    validation_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
    
    def validate_config(self, config):
        """
        验证训练配置
        
        Args:
            config: 训练配置字典
            
        Returns:
            bool: 验证是否通过
        """
        self.status_updated.emit("开始验证训练配置...")
        
        try:
            # 验证数据集路径
            if not self._validate_dataset_paths(config):
                return False
            
            # 验证训练参数
            if not self._validate_training_parameters(config):
                return False
            
            # 验证模型配置
            if not self._validate_model_config(config):
                return False
            
            # 验证保存路径
            if not self._validate_save_paths(config):
                return False
            
            self.status_updated.emit("配置验证通过")
            return True
            
        except Exception as e:
            self.validation_error.emit(f"配置验证时发生错误: {str(e)}")
            return False
    
    def _validate_dataset_paths(self, config):
        """验证数据集路径"""
        data_dir = config.get('data_dir', '')
        
        if not data_dir:
            self.validation_error.emit("数据集路径不能为空")
            return False
        
        if not os.path.exists(data_dir):
            self.validation_error.emit(f"数据集路径不存在: {data_dir}")
            return False
        
        # 检查训练和验证数据集目录
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        
        if not os.path.exists(train_dir):
            self.validation_error.emit(f"训练数据集目录不存在: {train_dir}")
            return False
        
        if not os.path.exists(val_dir):
            self.validation_error.emit(f"验证数据集目录不存在: {val_dir}")
            return False
        
        # 检查数据集是否为空
        if not self._check_directory_has_data(train_dir):
            self.validation_error.emit("训练数据集目录为空")
            return False
        
        if not self._check_directory_has_data(val_dir):
            self.validation_error.emit("验证数据集目录为空")
            return False
        
        self.status_updated.emit("数据集路径验证通过")
        return True
    
    def _validate_training_parameters(self, config):
        """验证训练参数"""
        # 验证num_epochs
        num_epochs = config.get('num_epochs', 20)
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            self.validation_error.emit("训练轮数必须为正整数")
            return False
        
        if num_epochs > 1000:
            self.status_updated.emit("警告: 训练轮数过大，可能需要很长时间")
        
        # 验证batch_size
        batch_size = config.get('batch_size', 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            self.validation_error.emit("批次大小必须为正整数")
            return False
        
        if batch_size > 256:
            self.status_updated.emit("警告: 批次大小较大，可能会消耗大量内存")
        
        # 验证learning_rate
        learning_rate = config.get('learning_rate', 0.001)
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            self.validation_error.emit("学习率必须为正数")
            return False
        
        if learning_rate > 1.0:
            self.status_updated.emit("警告: 学习率较大，可能导致训练不稳定")
        elif learning_rate < 1e-6:
            self.status_updated.emit("警告: 学习率较小，可能导致训练缓慢")
        
        # 验证dropout_rate
        dropout_rate = config.get('dropout_rate', 0.0)
        if not isinstance(dropout_rate, (int, float)) or dropout_rate < 0 or dropout_rate >= 1:
            self.validation_error.emit("Dropout率必须在[0, 1)范围内")
            return False
        
        self.status_updated.emit("训练参数验证通过")
        return True
    
    def _validate_model_config(self, config):
        """验证模型配置"""
        # 验证model_name
        model_name = config.get('model_name', '')
        if not model_name:
            self.validation_error.emit("模型名称不能为空")
            return False
        
        # 支持的模型列表
        supported_models = [
            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
            'MobileNetV2', 'MobileNetV3',
            'VGG16', 'VGG19',
            'DenseNet121', 'DenseNet169', 'DenseNet201',
            'InceptionV3',
            'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
            'Xception'
        ]
        
        if model_name not in supported_models:
            self.validation_error.emit(f"不支持的模型: {model_name}")
            return False
        
        # 验证task_type
        task_type = config.get('task_type', 'classification')
        if task_type not in ['classification', 'detection']:
            self.validation_error.emit(f"不支持的任务类型: {task_type}")
            return False
        
        # 验证activation_function
        activation_function = config.get('activation_function')
        if activation_function:
            supported_activations = [
                'None', 'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'GELU', 'Mish', 'Swish', 'SiLU'
            ]
            if activation_function not in supported_activations:
                self.validation_error.emit(f"不支持的激活函数: {activation_function}")
                return False
        
        # 验证weight_strategy
        weight_strategy = config.get('weight_strategy', 'balanced')
        if weight_strategy not in ['balanced', 'inverse', 'log_inverse', 'custom']:
            self.validation_error.emit(f"不支持的权重策略: {weight_strategy}")
            return False
        
        self.status_updated.emit("模型配置验证通过")
        return True
    
    def _validate_save_paths(self, config):
        """验证保存路径"""
        # 验证model_save_dir
        model_save_dir = config.get('model_save_dir', 'models/saved_models')
        
        try:
            # 尝试创建目录
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 检查目录是否可写
            test_file = os.path.join(model_save_dir, 'test_write.tmp')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception:
                self.validation_error.emit(f"模型保存目录不可写: {model_save_dir}")
                return False
                
        except Exception as e:
            self.validation_error.emit(f"无法创建模型保存目录: {model_save_dir}, 错误: {str(e)}")
            return False
        
        # 验证参数保存目录（如果指定）
        param_save_dir = config.get('default_param_save_dir')
        if param_save_dir:
            try:
                os.makedirs(param_save_dir, exist_ok=True)
                
                # 检查目录是否可写
                test_file = os.path.join(param_save_dir, 'test_write.tmp')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception:
                    self.validation_error.emit(f"参数保存目录不可写: {param_save_dir}")
                    return False
                    
            except Exception as e:
                self.validation_error.emit(f"无法创建参数保存目录: {param_save_dir}, 错误: {str(e)}")
                return False
        
        self.status_updated.emit("保存路径验证通过")
        return True
    
    def _check_directory_has_data(self, directory):
        """检查目录是否包含数据"""
        try:
            # 检查是否有子目录（类别目录）
            subdirs = [d for d in os.listdir(directory) 
                      if os.path.isdir(os.path.join(directory, d))]
            
            if not subdirs:
                return False
            
            # 检查子目录中是否有文件
            for subdir in subdirs:
                subdir_path = os.path.join(directory, subdir)
                files = [f for f in os.listdir(subdir_path) 
                        if os.path.isfile(os.path.join(subdir_path, f))]
                
                # 过滤图像文件
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                image_files = [f for f in files 
                              if any(f.lower().endswith(ext) for ext in image_extensions)]
                
                if image_files:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def validate_runtime_config(self, config, class_names):
        """
        验证运行时配置
        
        Args:
            config: 训练配置
            class_names: 类别名称列表
            
        Returns:
            bool: 验证是否通过
        """
        # 验证类别权重配置（如果使用自定义权重）
        if config.get('weight_strategy') == 'custom':
            return self._validate_custom_weights(config, class_names)
        
        return True
    
    def _validate_custom_weights(self, config, class_names):
        """验证自定义权重配置"""
        custom_weights = {}
        
        # 从配置中获取权重
        if 'class_weights' in config:
            custom_weights = config.get('class_weights', {})
        elif 'custom_class_weights' in config:
            custom_weights = config.get('custom_class_weights', {})
        elif 'weight_config_file' in config:
            weight_config_file = config.get('weight_config_file')
            if not weight_config_file or not os.path.exists(weight_config_file):
                self.validation_error.emit("权重配置文件不存在")
                return False
        
        # 检查权重是否覆盖所有类别
        if custom_weights:
            missing_classes = set(class_names) - set(custom_weights.keys())
            if missing_classes:
                self.status_updated.emit(f"警告: 以下类别缺少权重配置: {list(missing_classes)[:5]}...")
        
        return True 