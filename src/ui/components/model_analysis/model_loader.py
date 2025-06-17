import json
import torch
import logging
from PyQt5.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


def load_model_from_main_window(widget, model_info):
    """从主窗口加载模型"""
    try:
        # 寻找主窗口引用
        main_window = None
        parent = widget.parent()
        
        # 逐级向上查找主窗口
        while parent:
            if hasattr(parent, 'worker') and hasattr(parent.worker, 'predictor'):
                main_window = parent
                break
            elif hasattr(parent, 'main_window'):
                main_window = parent.main_window
                break
            parent = parent.parent()
            
        if main_window and hasattr(main_window, 'worker') and hasattr(main_window.worker, 'predictor'):
            # 使用找到的主窗口加载模型
            main_window.worker.predictor.load_model_with_info(model_info)
            
            # 获取加载后的模型
            model = main_window.worker.predictor.model
            return model
        else:
            raise ValueError("无法找到主窗口预测器，请重启应用程序")
            
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise


def load_class_names(class_info_file):
    """加载类别名称"""
    try:
        with open(class_info_file, 'r', encoding='utf-8') as f:
            class_info = json.load(f)
            class_names = class_info['class_names']
        return class_names
    except Exception as e:
        logger.error(f"加载类别信息失败: {str(e)}")
        raise


def preprocess_image(image):
    """预处理图像为模型输入格式"""
    import numpy as np
    
    # 转换为tensor
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float() / 255.0
    
    # 标准化 (假设使用ImageNet预训练模型)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor 