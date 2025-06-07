"""
数据集创建器
专门处理训练集和验证集的划分和创建
"""

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Callable
from PIL import Image
from .augmentation_manager import AugmentationManager


class DatasetCreator:
    """数据集创建器，处理训练集和验证集的划分"""

    def __init__(self, augmentation_manager: AugmentationManager):
        self.augmentation_manager = augmentation_manager

    def create_dataset(self, preprocessed_folder: str, dataset_folder: str, 
                      train_ratio: float, augmentation_level: str, 
                      image_files: List[str], params: Dict,
                      progress_callback: Callable[[int], None] = None,
                      status_callback: Callable[[str], None] = None,
                      stop_check: Callable[[], bool] = None) -> None:
        """创建训练和验证数据集"""
        
        # 确保数据集文件夹存在
        train_folder = os.path.join(dataset_folder, 'train')
        val_folder = os.path.join(dataset_folder, 'val')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        
        if not image_files:
            if status_callback:
                status_callback('未找到预处理后的图片')
            return
            
        if status_callback:
            status_callback(f'找到 {len(image_files)} 个预处理后的图片')
        
        # 划分训练集和验证集
        train_images, val_images = train_test_split(
            image_files, train_size=train_ratio, random_state=42)
            
        if status_callback:
            status_callback(f'划分为 {len(train_images)} 个训练图片和 {len(val_images)} 个验证图片')
        
        # 检查是否需要停止处理
        if stop_check and stop_check():
            if status_callback:
                status_callback('数据集创建已停止')
            return
            
        # 处理训练集 - 应用增强
        self._process_dataset_images(
            preprocessed_folder, train_folder, train_images,
            augmentation_level, True, 50, 75, params,
            progress_callback, status_callback, stop_check)
        
        # 处理验证集 - 不应用增强，只复制
        self._process_dataset_images(
            preprocessed_folder, val_folder, val_images,
            augmentation_level, False, 75, 100, params,
            progress_callback, status_callback, stop_check)
            
        if status_callback:
            status_callback('数据集创建完成')

    def _process_dataset_images(self, source_dir: str, target_dir: str,
                               images: List[str], augmentation_level: str,
                               apply_augmentation: bool, progress_start: int,
                               progress_end: int, params: Dict,
                               progress_callback: Callable[[int], None] = None,
                               status_callback: Callable[[str], None] = None,
                               stop_check: Callable[[], bool] = None) -> None:
        """处理数据集图片，包括复制和数据增强"""
        
        total_images = len(images)
        
        # 获取增强强度参数
        aug_intensity = params.get('augmentation_intensity', 0.5)
        if status_callback:
            status_callback(f"增强强度: {aug_intensity}")
        
        # 调试输出所有参数和选项
        aug_mode = params.get('augmentation_mode', 'combined')
        if status_callback:
            status_callback(f"处理数据集图片，增强模式：{aug_mode}, 是否应用增强：{apply_augmentation}")
            if aug_mode == 'separate':
                self._log_augmentation_params(params, aug_intensity, status_callback)
        
        # 记录处理的图片数量
        processed_count = 0
        
        for i, image_name in enumerate(images):
            # 检查是否需要停止处理
            if stop_check and stop_check():
                if status_callback:
                    status_callback('数据集处理已停止')
                return
                
            try:
                # 复制原始图片
                src_path = os.path.join(source_dir, image_name)
                dst_path = os.path.join(target_dir, image_name)
                
                # 只有当目标文件不存在时才复制
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    processed_count += 1
                
                # 对训练集进行数据增强
                if apply_augmentation:
                    self._apply_training_augmentation(
                        src_path, target_dir, image_name, augmentation_level, 
                        params, aug_intensity, aug_mode, status_callback)
                
                # 更新进度
                if progress_callback:
                    progress = int(progress_start + ((i + 1) / total_images) * (progress_end - progress_start))
                    progress_callback(progress)
                
            except Exception as e:
                import traceback
                if status_callback:
                    status_callback(f'处理数据集图片 {image_name} 时出错: {str(e)}\n{traceback.format_exc()}')
        
        if status_callback:
            status_callback(f'完成数据集处理，共处理 {processed_count} 张图片')

    def _apply_training_augmentation(self, src_path: str, target_dir: str, 
                                   image_name: str, augmentation_level: str,
                                   params: Dict, aug_intensity: float, 
                                   aug_mode: str, status_callback: Callable[[str], None] = None):
        """对训练集图片应用增强"""
        
        # 如果是独立模式，我们已经在预处理阶段应用了所有增强
        if aug_mode == 'separate':
            return
        
        # 对于组合模式，我们应用额外的增强
        try:
            image = np.array(Image.open(src_path).convert('RGB'))
        except Exception as e:
            if status_callback:
                status_callback(f"打开图片 {image_name} 失败: {str(e)}")
            return
        
        # 使用组合增强处理
        aug = self.augmentation_manager.get_augmentation_with_intensity(augmentation_level, aug_intensity)
        
        # 为每张图片生成2个增强版本
        for j in range(2):
            try:
                augmented = aug(image=image)['image']
                aug_name = f'{os.path.splitext(image_name)[0]}_aug_{j}{os.path.splitext(image_name)[1]}'
                aug_path = os.path.join(target_dir, aug_name)
                
                # 只有当增强图片不存在时才保存
                if not os.path.exists(aug_path):
                    Image.fromarray(augmented).save(aug_path)
                    
                if status_callback and j == 0:  # 只为第一个增强版本记录日志
                    status_callback(f"对 {image_name} 应用了组合增强")
                    
            except Exception as e:
                if status_callback:
                    status_callback(f"组合增强 #{j+1} 失败: {str(e)}")

    def _log_augmentation_params(self, params: Dict, aug_intensity: float, 
                                status_callback: Callable[[str], None]):
        """记录增强参数信息"""
        scale_limit = aug_intensity * 0.4
        noise_limit_min = 5.0 + aug_intensity * 15.0
        noise_limit_max = 20.0 + aug_intensity * 80.0
        blur_limit = int(2 + aug_intensity * 5)
        hue_shift_limit = int(10 + aug_intensity * 30)
        sat_val_limit = int(15 + aug_intensity * 35)
        
        status_callback(f"启用的增强方法：")
        status_callback(f"- 水平翻转: {params.get('flip_horizontal', False)}")
        status_callback(f"- 垂直翻转: {params.get('flip_vertical', False)}")
        status_callback(f"- 旋转: {params.get('rotate', False)}")
        status_callback(f"- 裁剪: {params.get('random_crop', False)}")
        status_callback(f"- 缩放: {params.get('random_scale', False)} (强度: {scale_limit:.2f})")
        status_callback(f"- 亮度: {params.get('brightness', False)}")
        status_callback(f"- 对比度: {params.get('contrast', False)}")
        status_callback(f"- 噪声: {params.get('noise', False)} (强度: {noise_limit_min:.1f}-{noise_limit_max:.1f})")
        status_callback(f"- 模糊: {params.get('blur', False)} (强度: {blur_limit})")
        status_callback(f"- 色相: {params.get('hue', False)} (强度: 色相{hue_shift_limit}, 饱和度/明度{sat_val_limit})")

    def create_class_folders(self, base_folder: str, class_names: List[str],
                           status_callback: Callable[[str], None] = None) -> None:
        """为每个类别创建文件夹"""
        try:
            for class_name in class_names:
                class_folder = os.path.join(base_folder, class_name)
                os.makedirs(class_folder, exist_ok=True)
                
            if status_callback:
                status_callback(f'已创建 {len(class_names)} 个类别文件夹')
        except Exception as e:
            if status_callback:
                status_callback(f'创建类别文件夹时出错: {str(e)}') 