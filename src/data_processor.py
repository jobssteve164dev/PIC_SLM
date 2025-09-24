import os
import shutil
from typing import Tuple, List
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split
from PyQt5.QtCore import QObject, pyqtSignal

class DataProcessor(QObject):
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    processing_finished = pyqtSignal()
    processing_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.augmentation_configs = {
            '基础': self._get_basic_augmentation(),
            '中等': self._get_medium_augmentation(),
            '强化': self._get_strong_augmentation()
        }

    def process_data(self, 
                    input_folder: str,
                    output_folder: str,
                    train_ratio: float = 0.8,
                    augmentation_level: str = '基础') -> None:
        """
        处理数据集，包括划分训练集和验证集，以及数据增强
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_folder, exist_ok=True)
            train_folder = os.path.join(output_folder, 'train')
            val_folder = os.path.join(output_folder, 'val')
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(val_folder, exist_ok=True)

            # 获取所有类别
            classes = [d for d in os.listdir(input_folder) 
                      if os.path.isdir(os.path.join(input_folder, d))]
            
            total_files = sum([len(os.listdir(os.path.join(input_folder, cls))) 
                             for cls in classes])
            processed_files = 0

            # 处理每个类别
            for class_name in classes:
                self.status_updated.emit(f'正在处理类别: {class_name}')
                
                # 创建目标目录
                train_class_dir = os.path.join(train_folder, class_name)
                val_class_dir = os.path.join(val_folder, class_name)
                os.makedirs(train_class_dir, exist_ok=True)
                os.makedirs(val_class_dir, exist_ok=True)

                # 获取该类别的所有图片
                class_dir = os.path.join(input_folder, class_name)
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                # 划分训练集和验证集
                train_images, val_images = train_test_split(
                    images, train_size=train_ratio, random_state=42)

                # 处理训练集
                self._process_image_set(
                    class_dir, train_class_dir, train_images,
                    augmentation_level, True)
                
                # 处理验证集
                self._process_image_set(
                    class_dir, val_class_dir, val_images,
                    augmentation_level, False)

                processed_files += len(images)
                progress = int((processed_files / total_files) * 100)
                self.progress_updated.emit(progress)

            self.status_updated.emit('数据处理完成')
            self.processing_finished.emit()

        except Exception as e:
            self.processing_error.emit(f'处理数据时出错: {str(e)}')

    def _process_image_set(self,
                          source_dir: str,
                          target_dir: str,
                          images: List[str],
                          augmentation_level: str,
                          apply_augmentation: bool) -> None:
        """
        处理图片集合，包括复制和数据增强
        """
        for image_name in images:
            # 复制原始图片
            src_path = os.path.join(source_dir, image_name)
            dst_path = os.path.join(target_dir, image_name)
            shutil.copy2(src_path, dst_path)

            # 对训练集进行数据增强
            if apply_augmentation:
                image = np.array(Image.open(src_path))
                aug = self.augmentation_configs[augmentation_level]
                
                # 生成增强后的图片
                for i in range(3):  # 每张图片生成3个增强版本
                    augmented = aug(image=image)['image']
                    aug_name = f'{os.path.splitext(image_name)[0]}_aug_{i}{os.path.splitext(image_name)[1]}'
                    aug_path = os.path.join(target_dir, aug_name)
                    Image.fromarray(augmented).save(aug_path)

    @staticmethod
    def _get_basic_augmentation() -> A.Compose:
        """获取基础数据增强配置"""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

    @staticmethod
    def _get_medium_augmentation() -> A.Compose:
        """获取中等数据增强配置"""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(p=0.3),
            A.RandomScale(scale_limit=0.2, p=0.3),
        ])

    @staticmethod
    def _get_strong_augmentation() -> A.Compose:
        """获取强化数据增强配置"""
        return A.Compose([
            A.RandomRotate90(p=0.7),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.7),
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.RandomScale(scale_limit=0.3, p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-45, 45), p=0.5),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
        ]) 