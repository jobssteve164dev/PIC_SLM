import os
import cv2
import numpy as np
import shutil
from PIL import Image, ImageEnhance
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

class ImagePreprocessor(QObject):
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    preprocessing_finished = pyqtSignal()
    preprocessing_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.augmentation_configs = {
            '基础': self._get_basic_augmentation(),
            '中等': self._get_medium_augmentation(),
            '强化': self._get_strong_augmentation()
        }
        self._stop_preprocessing = False

    def preprocess_images(self, params: Dict) -> None:
        """
        预处理图片并创建数据集
        
        参数:
            params: 包含预处理参数的字典
                - source_folder: 源文件夹路径
                - target_folder: 预处理目标文件夹路径
                - width: 目标宽度
                - height: 目标高度
                - format: 目标格式 (jpg, png, bmp)
                - brightness: 亮度调整值 (-50 到 50)
                - contrast: 对比度调整值 (-50 到 50)
                - train_ratio: 训练集比例 (0.1 到 0.9)
                - augmentation_level: 数据增强级别 ('基础', '中等', '强化')
                - dataset_folder: 数据集输出文件夹
        """
        try:
            # 重置停止标志
            self._stop_preprocessing = False
            
            # 第一步：预处理图片
            self.status_updated.emit('第1步：开始预处理图片...')
            self._preprocess_raw_images(params)
            
            # 如果处理被停止，则退出
            if self._stop_preprocessing:
                self.status_updated.emit('预处理已停止')
                return
            
            # 第二步：创建数据集
            self.status_updated.emit('第2步：开始创建数据集...')
            self._create_dataset(params)
            
            # 如果处理被停止，则退出
            if self._stop_preprocessing:
                self.status_updated.emit('预处理已停止')
                return
                
            self.status_updated.emit('预处理完成')
            self.preprocessing_finished.emit()
            
        except Exception as e:
            self.preprocessing_error.emit(f'预处理过程中出错: {str(e)}')
            
    def stop(self):
        """停止预处理过程"""
        self._stop_preprocessing = True
        self.status_updated.emit('正在停止预处理...')

    def _preprocess_raw_images(self, params: Dict) -> None:
        """预处理原始图片"""
        source_folder = params['source_folder']
        target_folder = params['target_folder']
        width = params['width']
        height = params['height']
        img_format = params['format']
        brightness = params['brightness']
        contrast = params['contrast']
        
        # 确保目标文件夹存在
        os.makedirs(target_folder, exist_ok=True)
        
        # 获取所有图片文件
        image_files = self._get_image_files(source_folder)
        total_files = len(image_files)
        
        if total_files == 0:
            self.preprocessing_error.emit('未找到图片文件')
            return
            
        self.status_updated.emit(f'找到 {total_files} 个图片文件')
        
        # 处理每个图片
        for i, img_file in enumerate(image_files):
            # 检查是否需要停止处理
            if self._stop_preprocessing:
                self.status_updated.emit('图片预处理已停止')
                return
                
            try:
                # 构建源文件和目标文件路径
                source_path = os.path.join(source_folder, img_file)
                file_name = os.path.splitext(img_file)[0]
                target_path = os.path.join(target_folder, f"{file_name}.{img_format}")
                
                # 处理图片
                self._process_single_image(
                    source_path, 
                    target_path, 
                    width, 
                    height, 
                    brightness, 
                    contrast
                )
                
                # 更新进度
                progress = int(((i + 1) / total_files) * 50)  # 预处理占总进度的50%
                self.progress_updated.emit(progress)
                
            except Exception as e:
                self.status_updated.emit(f'处理图片 {img_file} 时出错: {str(e)}')
        
        self.status_updated.emit('图片预处理完成')

    def _create_dataset(self, params: Dict) -> None:
        """创建训练和验证数据集"""
        preprocessed_folder = params['target_folder']
        dataset_folder = params['dataset_folder']
        train_ratio = params['train_ratio']
        augmentation_level = params['augmentation_level']
        
        # 检查是否需要停止处理
        if self._stop_preprocessing:
            return
            
        # 确保数据集文件夹存在
        train_folder = os.path.join(dataset_folder, 'train')
        val_folder = os.path.join(dataset_folder, 'val')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        
        # 获取所有预处理后的图片
        image_files = self._get_image_files(preprocessed_folder)
        total_files = len(image_files)
        
        if total_files == 0:
            self.preprocessing_error.emit('未找到预处理后的图片')
            return
            
        self.status_updated.emit(f'找到 {total_files} 个预处理后的图片')
        
        # 划分训练集和验证集
        train_images, val_images = train_test_split(
            image_files, train_size=train_ratio, random_state=42)
            
        self.status_updated.emit(f'划分为 {len(train_images)} 个训练图片和 {len(val_images)} 个验证图片')
        
        # 检查是否需要停止处理
        if self._stop_preprocessing:
            self.status_updated.emit('数据集创建已停止')
            return
            
        # 处理训练集
        self._process_dataset_images(
            preprocessed_folder, train_folder, train_images,
            augmentation_level, True, 50, 75)  # 训练集处理占总进度的25%
        
        # 处理验证集
        self._process_dataset_images(
            preprocessed_folder, val_folder, val_images,
            augmentation_level, False, 75, 100)  # 验证集处理占总进度的25%
            
        self.status_updated.emit('数据集创建完成')

    def _process_dataset_images(self,
                               source_dir: str,
                               target_dir: str,
                               images: List[str],
                               augmentation_level: str,
                               apply_augmentation: bool,
                               progress_start: int,
                               progress_end: int) -> None:
        """处理数据集图片，包括复制和数据增强"""
        total_images = len(images)
        
        for i, image_name in enumerate(images):
            # 检查是否需要停止处理
            if self._stop_preprocessing:
                self.status_updated.emit('数据集处理已停止')
                return
                
            try:
                # 复制原始图片
                src_path = os.path.join(source_dir, image_name)
                dst_path = os.path.join(target_dir, image_name)
                shutil.copy2(src_path, dst_path)
                
                # 对训练集进行数据增强
                if apply_augmentation:
                    image = np.array(Image.open(src_path).convert('RGB'))
                    aug = self.augmentation_configs[augmentation_level]
                    
                    # 生成增强后的图片
                    for j in range(3):  # 每张图片生成3个增强版本
                        # 检查是否需要停止处理
                        if self._stop_preprocessing:
                            return
                            
                        augmented = aug(image=image)['image']
                        aug_name = f'{os.path.splitext(image_name)[0]}_aug_{j}{os.path.splitext(image_name)[1]}'
                        aug_path = os.path.join(target_dir, aug_name)
                        Image.fromarray(augmented).save(aug_path)
                
                # 更新进度
                progress = int(progress_start + ((i + 1) / total_images) * (progress_end - progress_start))
                self.progress_updated.emit(progress)
                
            except Exception as e:
                self.status_updated.emit(f'处理数据集图片 {image_name} 时出错: {str(e)}')

    def _get_image_files(self, folder_path: str) -> List[str]:
        """获取文件夹中的所有图片文件"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        return [f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) and 
                os.path.splitext(f.lower())[1] in valid_extensions]

    def _process_single_image(self, 
                             source_path: str, 
                             target_path: str, 
                             width: int, 
                             height: int, 
                             brightness: int, 
                             contrast: int) -> None:
        """处理单个图片"""
        # 使用PIL打开图片
        img = Image.open(source_path).convert('RGB')
        
        # 调整大小
        img = img.resize((width, height), Image.LANCZOS)
        
        # 调整亮度
        if brightness != 0:
            factor = 1.0 + (brightness / 50.0)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        # 调整对比度
        if contrast != 0:
            factor = 1.0 + (contrast / 50.0)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        # 保存图片
        img.save(target_path)
        
    def create_class_folders(self, base_folder: str, class_names: List[str]) -> None:
        """为每个类别创建文件夹"""
        try:
            for class_name in class_names:
                class_folder = os.path.join(base_folder, class_name)
                os.makedirs(class_folder, exist_ok=True)
                
            self.status_updated.emit(f'已创建 {len(class_names)} 个类别文件夹')
        except Exception as e:
            self.preprocessing_error.emit(f'创建类别文件夹时出错: {str(e)}')
            
    def _get_basic_augmentation(self):
        """获取基础数据增强配置"""
        import albumentations as A
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

    def _get_medium_augmentation(self):
        """获取中等数据增强配置"""
        import albumentations as A
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(p=0.3),
            A.RandomScale(scale_limit=0.2, p=0.3),
        ])

    def _get_strong_augmentation(self):
        """获取强化数据增强配置"""
        import albumentations as A
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