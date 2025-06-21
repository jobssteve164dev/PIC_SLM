import os
import logging
import traceback
from typing import Dict, Tuple, List, Optional, Union, Any
import time
from contextlib import contextmanager
import threading
import json
import shutil

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from PyQt5.QtCore import QObject, pyqtSignal, QThread

# 常量定义
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CONFIDENCE_THRESHOLD = 50.0
DEFAULT_TIMEOUT_MS = 5000
THREAD_WAIT_TIMEOUT_MS = 3000
VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
LOG_SEPARATOR = "=" * 60

# 模型架构映射
RESNET_MODELS = {
    'ResNet18': 'resnet18',
    'ResNet34': 'resnet34', 
    'ResNet50': 'resnet50',
    'ResNet101': 'resnet101',
    'ResNet152': 'resnet152'
}

MOBILENET_MODELS = {
    'MobileNetV2': 'mobilenet_v2',
    'MobileNetV3': 'mobilenet_v3_large',
    'MobileNetV3Large': 'mobilenet_v3_large',
    'MobileNetV3Small': 'mobilenet_v3_small'
}

EFFICIENTNET_MODELS = {
    'EfficientNetB0': 'efficientnet_b0',
    'EfficientNetB1': 'efficientnet_b1', 
    'EfficientNetB2': 'efficientnet_b2',
    'EfficientNetB3': 'efficientnet_b3',
    'EfficientNetB4': 'efficientnet_b4'
}

VGG_MODELS = {
    'VGG16': 'vgg16',
    'VGG19': 'vgg19'
}

DENSENET_MODELS = {
    'DenseNet121': 'densenet121',
    'DenseNet169': 'densenet169',
    'DenseNet201': 'densenet201'
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """模型加载错误"""
    pass


class PredictionError(Exception):
    """预测错误"""
    pass


class BatchPredictionThread(QThread):
    """批量预测独立线程"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    prediction_finished = pyqtSignal(dict)
    prediction_error = pyqtSignal(str)
    
    def __init__(self, predictor: 'Predictor', params: Dict[str, Any]):
        super().__init__()
        self.predictor = predictor
        self.params = params
        self._stop_processing = False
        self._lock = threading.Lock()
        
    def run(self) -> None:
        """线程运行入口"""
        try:
            self._batch_predict()
        except Exception as e:
            error_msg = f"批量预测线程出错: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.prediction_error.emit(error_msg)
    
    def stop_processing(self) -> None:
        """停止处理"""
        with self._lock:
            self._stop_processing = True
        
    def _is_stopped(self) -> bool:
        """检查是否已停止"""
        with self._lock:
            return self._stop_processing
            
    def _batch_predict(self) -> None:
        """执行批量预测"""
        batch_start_time = time.time()
        logger.info(LOG_SEPARATOR)
        logger.info("开始批量预测（独立线程）")
        logger.info("批量预测开始时间: %s", time.strftime('%Y-%m-%d %H:%M:%S'))
        
        try:
            # 验证模型状态
            if not self._validate_model_state():
                return
                
            # 验证参数
            source_folder, target_folder, config = self._validate_and_extract_params()
            if not source_folder or not target_folder:
                return
                
            # 获取图片文件
            image_files = self._get_image_files(source_folder)
            if not image_files:
                logger.warning("未找到图片文件")
                self.status_updated.emit('未找到图片文件')
                return
                
            logger.info("找到 %d 张图片", len(image_files))
            self._log_image_format_stats(image_files)
                
            # 创建目标文件夹结构
            self._create_target_folders(target_folder, config['create_subfolders'])
            
            # 执行批量处理
            results = self._process_images(
                image_files, source_folder, target_folder, config
            )
            
            # 记录统计信息
            self._log_batch_statistics(results, batch_start_time, len(image_files))
            
            # 发送完成信号
            self.prediction_finished.emit(results)
            self.status_updated.emit('批量处理完成')
            
        except Exception as e:
            error_msg = f"批量预测过程中出错: {str(e)}"
            logger.error(error_msg)
            self.prediction_error.emit(error_msg)
            
    def _validate_model_state(self) -> bool:
        """验证模型状态"""
        if self.predictor.model is None:
            logger.error("模型未加载")
            self.prediction_error.emit('请先加载模型')
            return False
            
        logger.info("模型已加载: %s", type(self.predictor.model).__name__)
        logger.info("设备: %s", self.predictor.device)
        logger.info("模型状态: %s", '训练模式' if self.predictor.model.training else '评估模式')
        
        if self.predictor.class_names is None or len(self.predictor.class_names) == 0:
            logger.error("类别信息未加载")
            self.prediction_error.emit('类别信息未加载，请先加载模型')
            return False
            
        logger.info("类别信息已加载: %d 个类别", len(self.predictor.class_names))
        logger.info("类别列表: %s", self.predictor.class_names)
        return True
        
    def _validate_and_extract_params(self) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """验证并提取参数"""
        source_folder = self.params.get('source_folder')
        target_folder = self.params.get('target_folder')
        
        logger.info("源文件夹: %s", source_folder)
        logger.info("目标文件夹: %s", target_folder)
        
        # 验证必要参数
        if not source_folder:
            self.prediction_error.emit('源文件夹路径不能为空')
            return None, None, {}
        
        if not target_folder:
            self.prediction_error.emit('目标文件夹路径不能为空')
            return None, None, {}
        
        if not os.path.exists(source_folder):
            self.prediction_error.emit(f'源文件夹不存在: {source_folder}')
            return None, None, {}
        
        if not os.path.isdir(source_folder):
            self.prediction_error.emit(f'源路径不是文件夹: {source_folder}')
            return None, None, {}
            
        config = {
            'confidence_threshold': self.params.get('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD),
            'copy_mode': self.params.get('copy_mode', 'copy'),
            'create_subfolders': self.params.get('create_subfolders', True)
        }
        
        logger.info("置信度阈值: %.1f%%", config['confidence_threshold'])
        logger.info("文件操作模式: %s", config['copy_mode'])
        logger.info("创建子文件夹: %s", config['create_subfolders'])
        
        return source_folder, target_folder, config
        
    def _get_image_files(self, source_folder: str) -> List[str]:
        """获取图片文件列表"""
        return [f for f in os.listdir(source_folder) 
                if os.path.isfile(os.path.join(source_folder, f)) and 
                os.path.splitext(f.lower())[1] in VALID_IMAGE_EXTENSIONS]
                
    def _log_image_format_stats(self, image_files: List[str]) -> None:
        """记录图片格式统计"""
        format_stats = {}
        for ext in VALID_IMAGE_EXTENSIONS:
            count = sum(1 for f in image_files if f.lower().endswith(ext))
            if count > 0:
                format_stats[ext] = count
        logger.info("图片格式统计: %s", format_stats)
        
    def _create_target_folders(self, target_folder: str, create_subfolders: bool) -> None:
        """创建目标文件夹结构"""
        os.makedirs(target_folder, exist_ok=True)
        
        if create_subfolders:
            for class_name in self.predictor.class_names:
                os.makedirs(os.path.join(target_folder, class_name), exist_ok=True)
            logger.info("已创建 %d 个类别子文件夹", len(self.predictor.class_names))
            
    def _process_images(self, image_files: List[str], source_folder: str, 
                       target_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理图片"""
        results = {
            'total': len(image_files),
            'processed': 0,
            'classified': 0,
            'unclassified': 0,
            'class_counts': {class_name: 0 for class_name in self.predictor.class_names}
        }
        
        prediction_times = []
        
        logger.info("开始处理图片...")
        for i, image_file in enumerate(image_files):
            if self._is_stopped():
                logger.info("批量处理已停止")
                self.status_updated.emit('批量处理已停止')
                break
                
            image_path = os.path.join(source_folder, image_file)
            
            # 预测图片
            single_start = time.time()
            result = self.predictor.predict_image(image_path)
            single_time = time.time() - single_start
            prediction_times.append(single_time)
            
            if result:
                self._process_single_image_result(
                    result, image_file, image_path, target_folder, config, results
                )
            else:
                logger.error("图片 %s 预测失败", image_file)
                
            # 更新进度
            progress = int((i + 1) / len(image_files) * 100)
            self.progress_updated.emit(progress)
            self.status_updated.emit(f'处理图片 {i+1}/{len(image_files)}: {image_file}')
            results['processed'] += 1
            
        return results
        
    def _process_single_image_result(self, result: Dict[str, Any], image_file: str,
                                   image_path: str, target_folder: str,
                                   config: Dict[str, Any], results: Dict[str, Any]) -> None:
        """处理单张图片的预测结果"""
        top_prediction = result['predictions'][0]
        class_name = top_prediction['class_name']
        probability = top_prediction['probability']
        
        logger.debug("置信度比较: %s", image_file)
        logger.debug("预测置信度: %.2f%%", probability)
        logger.debug("设定阈值: %.2f%%", config['confidence_threshold'])
        
        if probability >= config['confidence_threshold']:
            logger.debug("置信度达标，将分类到: %s", class_name)
            
            try:
                target_path = self._get_target_path(
                    target_folder, class_name, image_file, config['create_subfolders']
                )
                
                # 复制或移动文件
                if config['copy_mode'] == 'copy':
                    shutil.copy2(image_path, target_path)
                else:  # move
                    shutil.move(image_path, target_path)
                    
                results['classified'] += 1
                results['class_counts'][class_name] += 1
                logger.debug("文件已%s到: %s", 
                           '复制' if config['copy_mode'] == 'copy' else '移动', 
                           target_path)
            except Exception as e:
                logger.error("处理文件 %s 时出错: %s", image_file, str(e))
                self.status_updated.emit(f'处理文件 {image_file} 时出错: {str(e)}')
        else:
            results['unclassified'] += 1
            logger.debug("置信度不达标，未分类")
            logger.warning("图片 %s 置信度过低 (%.2f%% < %.2f%%)，未分类", 
                         image_file, probability, config['confidence_threshold'])
                         
    def _get_target_path(self, target_folder: str, class_name: str, 
                        image_file: str, create_subfolders: bool) -> str:
        """获取目标路径"""
        if create_subfolders:
            return os.path.join(target_folder, class_name, image_file)
        else:
            base_name, ext = os.path.splitext(image_file)
            return os.path.join(target_folder, f"{class_name}_{base_name}{ext}")
            
    def _log_batch_statistics(self, results: Dict[str, Any], 
                             batch_start_time: float, total_images: int) -> None:
        """记录批量处理统计信息"""
        batch_total_time = time.time() - batch_start_time
        
        logger.info(LOG_SEPARATOR)
        logger.info("批量预测完成统计:")
        logger.info("总耗时: %.2f秒", batch_total_time)
        logger.info("预测速度: %.2f 张/秒", total_images / batch_total_time)
        logger.info("总图片数: %d", results['total'])
        logger.info("已处理: %d", results['processed'])
        logger.info("已分类: %d", results['classified'])
        logger.info("未分类: %d", results['unclassified'])
        logger.info("各类别统计:")
        for class_name, count in results['class_counts'].items():
            if count > 0:
                logger.info("  %s: %d 张", class_name, count)
        logger.info(LOG_SEPARATOR)


class Predictor(QObject):
    """图像预测器"""
    
    # 定义信号
    prediction_finished = pyqtSignal(dict)
    prediction_error = pyqtSignal(str)
    batch_prediction_progress = pyqtSignal(int)
    batch_prediction_status = pyqtSignal(str)
    batch_prediction_finished = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.device = self._get_device()
        self.model: Optional[nn.Module] = None
        self.class_names: Optional[List[str]] = None
        self.transform = self._create_default_transform()
        self._stop_batch_processing = False
        self.batch_prediction_thread: Optional[BatchPredictionThread] = None
        self._lock = threading.Lock()
        
        logger.info("预测器初始化完成，使用设备: %s", self.device)
        
    def _get_device(self) -> torch.device:
        """获取计算设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("CUDA可用，使用GPU: %s", torch.cuda.get_device_name(0))
        else:
            device = torch.device('cpu')
            logger.info("CUDA不可用，使用CPU")
        return device
        
    def _create_default_transform(self) -> transforms.Compose:
        """创建默认的图像变换"""
        return transforms.Compose([
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @contextmanager
    def _model_context(self):
        """模型上下文管理器，确保资源正确清理"""
        try:
            yield
        finally:
            if hasattr(self, 'model') and self.model is not None:
                # 如果模型在GPU上，清理GPU内存
                if next(self.model.parameters()).is_cuda:
                    torch.cuda.empty_cache()

    def _clear_previous_model(self) -> None:
        """清理之前的模型"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("已清理之前的模型")

    def load_model(self, model_path: str, class_info_path: str) -> None:
        """加载模型和类别信息（简化版，用于向后兼容）"""
        model_info = {
            'model_path': model_path,
            'class_info_path': class_info_path,
            'model_type': "分类模型",
            'model_arch': "ResNet50"  # 默认架构
        }
        self.load_model_with_info(model_info)

    def load_model_with_info(self, model_info: Dict[str, Any]) -> None:
        """根据模型信息加载模型和类别信息"""
        try:
            with self._model_context():
                self._clear_previous_model()
                
                # 提取模型信息
                model_path = model_info.get('model_path')
                class_info_path = model_info.get('class_info_path')
                model_type = model_info.get('model_type')
                model_arch = model_info.get('model_arch')
                
                logger.info("加载模型信息: %s - %s", model_type, model_arch)
                logger.info("模型路径: %s", model_path)
                logger.info("类别信息路径: %s", class_info_path)
                
                # 验证路径
                if not os.path.exists(model_path):
                    raise ModelLoadError(f"模型文件不存在: {model_path}")
                if not os.path.exists(class_info_path):
                    raise ModelLoadError(f"类别信息文件不存在: {class_info_path}")
                
                # 加载类别信息
                self._load_class_info(class_info_path)
                
                # 根据模型类型加载模型
                if model_type == "分类模型":
                    self._load_classification_model(model_path, model_arch)
                elif model_type == "检测模型":
                    self._load_detection_model(model_path, model_arch)
                else:
                    raise ModelLoadError(f"不支持的模型类型: {model_type}")
                    
                logger.info("模型加载成功: %s - %s", model_type, model_arch)
                
        except Exception as e:
            error_msg = f'加载模型时出错: {str(e)}\n{traceback.format_exc()}'
            logger.error(error_msg)
            self.prediction_error.emit(error_msg)
            
    def _load_class_info(self, class_info_path: str) -> None:
        """加载类别信息"""
        try:
            with open(class_info_path, 'r', encoding='utf-8') as f:
                class_info = json.load(f)
            self.class_names = class_info['class_names']
            logger.info("类别信息加载成功，类别: %s", self.class_names)
        except Exception as e:
            raise ModelLoadError(f"加载类别信息失败: {str(e)}")
            
    def _load_classification_model(self, model_path: str, model_arch: str) -> None:
        """加载分类模型"""
        try:
            num_classes = len(self.class_names)
            
            # 创建模型
            self.model = self._create_classification_model(model_arch, num_classes)
            logger.info("成功创建模型架构: %s", model_arch)
                
            # 加载模型权重
            self._load_model_weights(model_path)
            
            # 设置模型状态
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("成功加载模型权重: %s", model_path)
            
        except Exception as e:
            error_msg = f"加载分类模型失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
            
    def _create_classification_model(self, model_arch: str, num_classes: int) -> nn.Module:
        """创建分类模型"""
        if model_arch in RESNET_MODELS:
            return self._create_resnet_model(model_arch, num_classes)
        elif model_arch in MOBILENET_MODELS:
            return self._create_mobilenet_model(model_arch, num_classes)
        elif model_arch in EFFICIENTNET_MODELS:
            return self._create_efficientnet_model(model_arch, num_classes)
        elif model_arch in VGG_MODELS:
            return self._create_vgg_model(model_arch, num_classes)
        elif model_arch in DENSENET_MODELS:
            return self._create_densenet_model(model_arch, num_classes)
        elif model_arch in ["InceptionV3", "Xception"]:
            return self._create_other_model(model_arch, num_classes)
        else:
            raise ValueError(f"不支持的分类模型架构: {model_arch}")
            
    def _create_resnet_model(self, model_arch: str, num_classes: int) -> nn.Module:
        """创建ResNet模型"""
        from torchvision import models
        model_fn = getattr(models, RESNET_MODELS[model_arch])
        model = model_fn(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
        
    def _create_mobilenet_model(self, model_arch: str, num_classes: int) -> nn.Module:
        """创建MobileNet模型"""
        from torchvision import models
        model_fn = getattr(models, MOBILENET_MODELS[model_arch])
        model = model_fn(pretrained=False)
        
        if model_arch == "MobileNetV2":
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:  # MobileNetV3 variants
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
        
    def _create_efficientnet_model(self, model_arch: str, num_classes: int) -> nn.Module:
        """创建EfficientNet模型"""
        from torchvision import models
        model_fn = getattr(models, EFFICIENTNET_MODELS[model_arch])
        model = model_fn(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
        
    def _create_vgg_model(self, model_arch: str, num_classes: int) -> nn.Module:
        """创建VGG模型"""
        from torchvision import models
        model_fn = getattr(models, VGG_MODELS[model_arch])
        model = model_fn(pretrained=False)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model
        
    def _create_densenet_model(self, model_arch: str, num_classes: int) -> nn.Module:
        """创建DenseNet模型"""
        from torchvision import models
        model_fn = getattr(models, DENSENET_MODELS[model_arch])
        model = model_fn(pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
        
    def _create_other_model(self, model_arch: str, num_classes: int) -> nn.Module:
        """创建其他特殊模型"""
        logger.info("注意: %s模型需要特殊的预处理，可能需要调整transform", model_arch)
        
        if model_arch == "InceptionV3":
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=False)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model
        elif model_arch == "Xception":
            try:
                from pretrainedmodels import xception
                model = xception(num_classes=1000, pretrained=None)
                model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
                return model
            except ImportError:
                raise ImportError("加载Xception模型需要安装pretrainedmodels库: pip install pretrainedmodels")
        else:
            raise ValueError(f"不支持的模型架构: {model_arch}")
            
    def _load_model_weights(self, model_path: str) -> None:
        """加载模型权重"""
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 处理不同格式的模型文件
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
        # 加载权重（使用非严格模式，允许部分权重不匹配）
        self.model.load_state_dict(state_dict, strict=False)
        
    def _load_detection_model(self, model_path: str, model_arch: str) -> None:
        """加载检测模型"""
        try:
            # YOLO系列模型
            if model_arch.startswith("YOLO"):
                self._load_yolo_model(model_path, model_arch)
            # SSD系列模型
            elif model_arch.startswith("SSD"):
                self._load_ssd_model(model_path, model_arch)
            # Faster R-CNN和Mask R-CNN系列
            elif model_arch in ["Faster R-CNN", "Mask R-CNN"]:
                self._load_rcnn_model(model_path, model_arch)
            # RetinaNet模型
            elif model_arch == "RetinaNet":
                self._load_retinanet_model(model_path)
            # DETR模型
            elif model_arch == "DETR":
                self._load_detr_model(model_path)
            else:
                raise ValueError(f"不支持的检测模型架构: {model_arch}")
                
        except Exception as e:
            error_msg = f"加载检测模型失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
            
    def _load_yolo_model(self, model_path: str, model_arch: str) -> None:
        """加载YOLO模型"""
        if model_arch == "YOLOv5":
            try:
                import yolov5
                self.model = yolov5.load(model_path)
                logger.info("使用yolov5库加载模型: %s", model_path)
            except ImportError:
                raise ImportError("需要安装YOLOv5库: pip install yolov5")
        elif model_arch == "YOLOv8":
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                logger.info("使用ultralytics库加载模型: %s", model_path)
            except ImportError:
                raise ImportError("需要安装Ultralytics库: pip install ultralytics")
        elif model_arch in ["YOLOv7", "YOLOv6", "YOLOv4", "YOLOv3"]:
            self._load_legacy_yolo_model(model_path, model_arch)
        else:
            raise ValueError(f"不支持的YOLO架构: {model_arch}")
            
    def _load_legacy_yolo_model(self, model_path: str, model_arch: str) -> None:
        """加载遗留版本YOLO模型"""
        try:
            if model_arch == "YOLOv7":
                from models.experimental import attempt_load
                self.model = attempt_load(model_path, map_location=self.device)
                logger.info("使用YOLOv7专用加载器加载模型: %s", model_path)
            elif model_arch == "YOLOv6":
                from yolov6.core.inferer import Inferer
                self.model = Inferer(model_path, device=self.device)
                logger.info("使用YOLOv6库加载模型: %s", model_path)
            elif model_arch in ["YOLOv4", "YOLOv3"]:
                import cv2
                self.model = cv2.dnn.readNetFromDarknet(
                    model_path, model_path.replace(".weights", ".cfg")
                )
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("使用OpenCV DNN加载%s模型: %s", model_arch, model_path)
        except ImportError as e:
            raise ImportError(f"加载{model_arch}需要安装特定库: {str(e)}")
        except Exception as e:
            raise Exception(f"加载{model_arch}模型失败: {str(e)}")
            
    def _load_ssd_model(self, model_path: str, model_arch: str) -> None:
        """加载SSD模型"""
        try:
            if model_arch in ["SSD", "SSD300"]:
                from torchvision.models.detection import ssd300_vgg16
                self.model = ssd300_vgg16(
                    pretrained=False, 
                    num_classes=len(self.class_names) + 1
                )
            elif model_arch == "SSD512":
                raise NotImplementedError("SSD512需要特定的实现，目前尚未支持")
            else:
                raise ValueError(f"不支持的SSD架构: {model_arch}")
            
            # 加载模型权重
            self._load_model_weights(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("成功加载SSD模型: %s", model_path)
            
        except ImportError:
            raise ImportError("加载SSD模型需要安装PyTorch和torchvision")
            
    def _load_rcnn_model(self, model_path: str, model_arch: str) -> None:
        """加载R-CNN系列模型"""
        try:
            if model_arch == "Faster R-CNN":
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                self.model = fasterrcnn_resnet50_fpn(
                    pretrained=False, 
                    num_classes=len(self.class_names) + 1
                )
            elif model_arch == "Mask R-CNN":
                from torchvision.models.detection import maskrcnn_resnet50_fpn
                self.model = maskrcnn_resnet50_fpn(
                    pretrained=False, 
                    num_classes=len(self.class_names) + 1
                )
            else:
                raise ValueError(f"不支持的R-CNN架构: {model_arch}")
                
            # 加载模型权重
            self._load_model_weights(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("成功加载%s模型: %s", model_arch, model_path)
            
        except ImportError:
            raise ImportError(f"加载{model_arch}模型需要安装PyTorch和torchvision")
            
    def _load_retinanet_model(self, model_path: str) -> None:
        """加载RetinaNet模型"""
        try:
            from torchvision.models.detection import retinanet_resnet50_fpn
            self.model = retinanet_resnet50_fpn(
                pretrained=False, 
                num_classes=len(self.class_names) + 1
            )
            
            # 加载模型权重
            self._load_model_weights(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("成功加载RetinaNet模型: %s", model_path)
            
        except ImportError:
            raise ImportError("加载RetinaNet模型需要安装PyTorch和torchvision")
            
    def _load_detr_model(self, model_path: str) -> None:
        """加载DETR模型"""
        try:
            # 尝试使用DETR
            try:
                from transformers import DetrForObjectDetection
                self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                # 修改分类头以适应自定义类别数量
                self.model.config.num_labels = len(self.class_names)
                self.model.class_labels_classifier = nn.Linear(
                    in_features=self.model.class_labels_classifier.in_features,
                    out_features=len(self.class_names)
                )
            except ImportError:
                # 备选：使用torchvision的DETR实现
                from torchvision.models.detection import detr_resnet50
                self.model = detr_resnet50(
                    pretrained=False, 
                    num_classes=len(self.class_names) + 1
                )
            
            # 加载模型权重
            self._load_model_weights(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("成功加载DETR模型: %s", model_path)
            
        except ImportError:
            raise ImportError("加载DETR模型需要安装transformers库或最新版本的torchvision")

    def predict(self, image_path: str, top_k: int = 3) -> None:
        """预测单张图片类别并发送信号"""
        try:
            result = self.predict_image(image_path, top_k)
            if result:
                self.prediction_finished.emit(result)
            else:
                self.prediction_error.emit("预测失败")
        except Exception as e:
            error_msg = f'预测过程中出错: {str(e)}\n{traceback.format_exc()}'
            logger.error(error_msg)
            self.prediction_error.emit(error_msg)

    def predict_image(self, image_path: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """预测单张图片类别并返回结果（不发送信号）"""
        start_time = time.time()
        
        try:
            if self.model is None:
                logger.error("模型未加载，请先加载模型")
                return None
                
            # 加载和预处理图片
            image_tensor = self._preprocess_image(image_path)
            if image_tensor is None:
                return None
                
            # 执行预测
            predictions = self._execute_prediction(image_tensor, top_k)
            if not predictions:
                logger.error("无法获取有效的预测结果")
                return None
                
            total_time = time.time() - start_time
            logger.debug("单张图片预测总时间: %.4f秒", total_time)
            logger.debug("图片: %s, 最高置信度: %s (%.2f%%)", 
                        os.path.basename(image_path), 
                        predictions[0]['class_name'], 
                        predictions[0]['probability'])
                
            return {
                'predictions': predictions,
                'image_path': image_path
            }

        except Exception as e:
            logger.error('预测图片 %s 时出错: %s\n%s', 
                        image_path, str(e), traceback.format_exc())
            return None
            
    def _preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """预处理图片"""
        try:
            load_start = time.time()
            image = Image.open(image_path).convert('RGB')
            load_time = time.time() - load_start
            logger.debug("图片加载时间: %.4f秒", load_time)
            
            preprocess_start = time.time()
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            preprocess_time = time.time() - preprocess_start
            logger.debug("图片预处理时间: %.4f秒", preprocess_time)
            
            return image_tensor
        except Exception as e:
            logger.error("无法加载图像: %s", str(e))
            return None
            
    def _execute_prediction(self, image_tensor: torch.Tensor, top_k: int) -> List[Dict[str, Any]]:
        """执行预测"""
        try:
            predict_start = time.time()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                # 检查输出格式
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_k = min(top_k, len(self.class_names), probabilities.size(1))
                top_prob, top_class = torch.topk(probabilities, top_k)
                
            predict_time = time.time() - predict_start
            logger.debug("模型推理时间: %.4f秒", predict_time)
            logger.debug("模型输出shape: %s", outputs.shape)
            logger.debug("概率分布前3个值: %s", probabilities[0][:3].tolist())
            
            # 获取预测结果
            predictions = []
            for i in range(top_k):
                try:
                    class_idx = top_class[0][i].item()
                    if 0 <= class_idx < len(self.class_names):
                        prob = top_prob[0][i].item()
                        predictions.append({
                            'class_name': self.class_names[class_idx],
                            'probability': prob * 100  # 转换为百分比
                        })
                        logger.debug("预测结果 %d: %s - %.2f%%", 
                                   i+1, self.class_names[class_idx], prob*100)
                except Exception as e:
                    logger.error("处理预测结果 %d 时出错: %s", i, str(e))
                    continue
                    
            return predictions
            
        except Exception as e:
            logger.error("预测过程出错: %s", str(e))
            return []

    def batch_predict(self, params: Dict[str, Any]) -> None:
        """启动批量预测线程"""
        with self._lock:
            # 如果已有线程在运行，先停止它
            if self.batch_prediction_thread and self.batch_prediction_thread.isRunning():
                logger.warning("已有批量预测线程在运行，先停止它")
                self.stop_batch_processing()
                self.batch_prediction_thread.wait()
            
            # 创建新的批量预测线程
            self.batch_prediction_thread = BatchPredictionThread(self, params)
                
            # 连接线程信号到Predictor的信号
            self.batch_prediction_thread.progress_updated.connect(self.batch_prediction_progress.emit)
            self.batch_prediction_thread.status_updated.connect(self.batch_prediction_status.emit)
            self.batch_prediction_thread.prediction_finished.connect(self.batch_prediction_finished.emit)
            self.batch_prediction_thread.prediction_error.connect(self.prediction_error.emit)
                
            # 启动线程
            logger.info("启动批量预测独立线程")
            self.batch_prediction_thread.start()

    def stop_batch_processing(self) -> None:
        """停止批量处理"""
        self._stop_batch_processing = True
        
        # 如果有运行中的批量预测线程，停止它
        if self.batch_prediction_thread and self.batch_prediction_thread.isRunning():
            logger.info("正在停止批量预测线程...")
            self.batch_prediction_thread.stop_processing()
    
    def is_batch_prediction_running(self) -> bool:
        """检查批量预测线程是否正在运行"""
        return (self.batch_prediction_thread is not None and 
                self.batch_prediction_thread.isRunning())
    
    def wait_for_batch_prediction_to_finish(self, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> bool:
        """等待批量预测线程完成"""
        if self.batch_prediction_thread and self.batch_prediction_thread.isRunning():
            return self.batch_prediction_thread.wait(timeout_ms)
        return True
    
    def cleanup_batch_prediction_thread(self) -> None:
        """清理批量预测线程"""
        with self._lock:
            if self.batch_prediction_thread:
                if self.batch_prediction_thread.isRunning():
                    self.batch_prediction_thread.stop_processing()
                    self.batch_prediction_thread.wait(THREAD_WAIT_TIMEOUT_MS)
                self.batch_prediction_thread.deleteLater()
                self.batch_prediction_thread = None

    def _create_model(self, model_name: str, num_classes: int) -> nn.Module:
        """创建模型（已弃用，保留用于向后兼容）"""
        logger.warning("_create_model方法已弃用，请使用_create_classification_model")
        if model_name == 'ResNet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        else:
            raise ValueError(f'不支持的模型: {model_name}') 