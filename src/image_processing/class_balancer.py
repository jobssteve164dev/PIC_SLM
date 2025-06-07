"""
类别平衡处理器
专门处理多类别图像数据的平衡预处理
"""

import os
import shutil
from sklearn.model_selection import train_test_split
from typing import List, Dict, Callable
from .image_transformer import ImageTransformer
from .augmentation_manager import AugmentationManager


class ClassBalancer:
    """类别平衡处理器，处理多类别数据的平衡预处理"""

    def __init__(self, image_transformer: ImageTransformer, 
                 augmentation_manager: AugmentationManager):
        self.image_transformer = image_transformer
        self.augmentation_manager = augmentation_manager

    def preprocess_with_class_balance(self, params: Dict,
                                    get_image_files_func: Callable[[str], List[str]],
                                    progress_callback: Callable[[int], None] = None,
                                    status_callback: Callable[[str], None] = None,
                                    stop_check: Callable[[], bool] = None) -> None:
        """使用类别平衡的方式预处理图片"""
        
        source_folder = params['source_folder']
        output_folder = params['target_folder']
        dataset_folder = params['dataset_folder']
        train_ratio = params['train_ratio']
        
        # 调试输出参数信息
        if status_callback:
            status_callback(f"预处理参数: 增强模式={params.get('augmentation_mode', '未指定')}")
            status_callback(f"启用的增强方法: 水平翻转={params.get('flip_horizontal', False)}, "
                          f"亮度={params.get('brightness', False)}, 对比度={params.get('contrast', False)}等")
        
        # 创建必要的文件夹
        os.makedirs(output_folder, exist_ok=True)
        train_folder = os.path.join(dataset_folder, 'train')
        val_folder = os.path.join(dataset_folder, 'val')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        
        # 获取类别名称
        class_names = params['class_names']
        
        # 遍历所有类别
        for class_name in class_names:
            # 检查是否需要停止处理
            if stop_check and stop_check():
                if status_callback:
                    status_callback('预处理已停止')
                return
            
            # 构建类别源文件夹路径
            class_source_folder = os.path.join(source_folder, class_name)
            
            # 跳过不存在的类别文件夹
            if not os.path.exists(class_source_folder) or not os.path.isdir(class_source_folder):
                if status_callback:
                    status_callback(f"警告：类别文件夹不存在: {class_source_folder}，跳过该类别")
                continue
            
            if status_callback:
                status_callback(f"处理类别: {class_name}")
            
            # 创建输出文件夹中的类别文件夹
            class_output_folder = os.path.join(output_folder, class_name)
            os.makedirs(class_output_folder, exist_ok=True)
            
            # 创建类别的训练集和验证集文件夹
            class_train_folder = os.path.join(train_folder, class_name)
            class_val_folder = os.path.join(val_folder, class_name)
            os.makedirs(class_train_folder, exist_ok=True)
            os.makedirs(class_val_folder, exist_ok=True)
            
            # 获取该类别的所有图片
            image_files = get_image_files_func(class_source_folder)
            
            if not image_files:
                if status_callback:
                    status_callback(f"警告：类别 {class_name} 中没有图片文件，跳过该类别")
                continue
            
            # 先划分训练集和验证集
            train_images, val_images = train_test_split(
                image_files, train_size=train_ratio, random_state=42)
            
            if status_callback:
                status_callback(f"类别 {class_name} 划分为训练集{len(train_images)}张和验证集{len(val_images)}张")
                
            # 处理训练集图片
            self._process_class_training_images(
                class_source_folder, class_output_folder, class_train_folder,
                train_images, params, status_callback, stop_check)
            
            # 处理验证集图片 - 只进行基本预处理，不增强
            self._process_class_validation_images(
                class_source_folder, class_output_folder, class_val_folder,
                val_images, params, status_callback, stop_check)

    def _process_class_training_images(self, class_source_folder: str, 
                                     class_output_folder: str, class_train_folder: str,
                                     train_images: List[str], params: Dict,
                                     status_callback: Callable[[str], None] = None,
                                     stop_check: Callable[[], bool] = None):
        """处理类别的训练集图片"""
        
        for img_file in train_images:
            if stop_check and stop_check():
                return
                
            # 预处理原始图片并保存到输出文件夹
            source_path = os.path.join(class_source_folder, img_file)
            file_name = os.path.splitext(img_file)[0]
            img_format = params['format']
            output_path = os.path.join(class_output_folder, f"{file_name}.{img_format}")
            
            # 基本图像处理（调整大小、亮度、对比度）
            self.image_transformer.process_single_image(
                source_path, 
                output_path, 
                params['width'], 
                params['height'], 
                params['brightness_value'], 
                params['contrast_value']
            )
            
            # 直接复制预处理后的图片到训练集文件夹
            train_dest_path = os.path.join(class_train_folder, f"{file_name}.{img_format}")
            shutil.copy2(output_path, train_dest_path)
            
            # 对训练集图片应用增强处理
            self.augmentation_manager.apply_augmentations_to_image(
                output_path, 
                class_train_folder, 
                file_name, 
                img_format, 
                params,
                status_callback
            )

    def _process_class_validation_images(self, class_source_folder: str, 
                                       class_output_folder: str, class_val_folder: str,
                                       val_images: List[str], params: Dict,
                                       status_callback: Callable[[str], None] = None,
                                       stop_check: Callable[[], bool] = None):
        """处理类别的验证集图片"""
        
        for img_file in val_images:
            if stop_check and stop_check():
                return
                
            # 预处理原始图片并保存到输出文件夹
            source_path = os.path.join(class_source_folder, img_file)
            file_name = os.path.splitext(img_file)[0]
            img_format = params['format']
            output_path = os.path.join(class_output_folder, f"{file_name}.{img_format}")
            
            # 基本图像处理（调整大小、亮度、对比度）
            self.image_transformer.process_single_image(
                source_path, 
                output_path, 
                params['width'], 
                params['height'], 
                params['brightness_value'], 
                params['contrast_value']
            )
            
            # 只复制预处理后的原始图片到验证集文件夹，不进行增强
            val_dest_path = os.path.join(class_val_folder, f"{file_name}.{img_format}")
            shutil.copy2(output_path, val_dest_path) 