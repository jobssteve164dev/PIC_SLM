"""
数据增强管理器
专门处理各种数据增强方法，包括独立增强和组合增强
"""

import numpy as np
import albumentations as A
from PIL import Image, ImageEnhance
from typing import Dict, List, Tuple
import os
import shutil


class AugmentationManager:
    """数据增强管理器，处理各种数据增强方法"""

    def __init__(self):
        self.augmentation_configs = {
            '基础': self._get_basic_augmentation(),
            '中等': self._get_medium_augmentation(),
            '强化': self._get_strong_augmentation()
        }

    def get_augmentation_with_intensity(self, level: str, intensity: float) -> A.Compose:
        """根据增强级别和强度获取相应的增强配置"""
        # 根据增强强度调整各增强参数
        scale_limit = intensity * 0.4
        rotate_prob = min(0.9, 0.4 + intensity * 0.5)
        flip_prob = min(0.9, 0.4 + intensity * 0.5)
        vflip_prob = min(0.8, 0.1 + intensity * 0.6)
        brightness_contrast_prob = min(0.9, 0.2 + intensity * 0.7)
        noise_prob = min(0.8, 0.1 + intensity * 0.6)
        blur_prob = min(0.8, 0.1 + intensity * 0.6)
        hue_prob = min(0.8, 0.1 + intensity * 0.4)
        
        if level == '基础':
            return A.Compose([
                A.RandomRotate90(p=rotate_prob),
                A.HorizontalFlip(p=flip_prob),
                A.RandomBrightnessContrast(
                    brightness_limit=intensity * 0.3,
                    contrast_limit=intensity * 0.3,
                    p=brightness_contrast_prob
                ),
            ])
        elif level == '中等':
            return A.Compose([
                A.RandomRotate90(p=rotate_prob),
                A.HorizontalFlip(p=flip_prob),
                A.VerticalFlip(p=vflip_prob),
                A.RandomBrightnessContrast(
                    brightness_limit=intensity * 0.4,
                    contrast_limit=intensity * 0.4,
                    p=brightness_contrast_prob
                ),
                A.GaussNoise(var_limit=(5.0 + intensity * 15.0, 20.0 + intensity * 40.0), p=noise_prob),
                A.GaussianBlur(blur_limit=int(2 + intensity * 4), p=blur_prob),
                A.RandomScale(scale_limit=scale_limit, p=0.3),
            ])
        else:  # 强化
            return A.Compose([
                A.RandomRotate90(p=rotate_prob),
                A.HorizontalFlip(p=flip_prob),
                A.VerticalFlip(p=vflip_prob),
                A.RandomBrightnessContrast(
                    brightness_limit=intensity * 0.5,
                    contrast_limit=intensity * 0.5,
                    p=brightness_contrast_prob
                ),
                A.GaussNoise(var_limit=(5.0 + intensity * 15.0, 20.0 + intensity * 60.0), p=noise_prob),
                A.GaussianBlur(blur_limit=int(2 + intensity * 5), p=blur_prob),
                A.RandomScale(scale_limit=scale_limit, p=0.5),
                A.Affine(
                    scale=(0.9, 1.0 + intensity * 0.2),
                    translate_percent=(intensity * 0.1, intensity * 0.1),
                    rotate=(-45 * intensity, 45 * intensity),
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=int(10 + intensity * 30),
                    sat_shift_limit=int(15 + intensity * 35),
                    val_shift_limit=int(15 + intensity * 35),
                    p=hue_prob
                ),
                A.RandomGamma(gamma_limit=(80, 120 + int(intensity * 20)), p=0.3),
            ])

    def apply_augmentations_to_image(self, image_path: str, target_dir: str, 
                                   file_name: str, img_format: str, 
                                   params: Dict, status_callback=None) -> None:
        """对单张图片应用所有选择的增强方法"""
        try:
            # 读取图像
            image = np.array(Image.open(image_path).convert('RGB'))
            success_count = 0
            generated_methods = []
            
            # 获取增强模式
            augmentation_mode = params.get('augmentation_mode', 'combined')
            
            # 获取增强强度参数
            aug_intensity = params.get('augmentation_intensity', 0.5)
            
            # 根据增强强度调整各增强参数
            scale_limit = aug_intensity * 0.4
            brightness_limit = aug_intensity * 0.6
            contrast_limit = aug_intensity * 0.6
            noise_limit_min = 5.0 + aug_intensity * 15.0
            noise_limit_max = 20.0 + aug_intensity * 80.0
            blur_limit = int(2 + aug_intensity * 5)
            hue_shift_limit = int(10 + aug_intensity * 30)
            sat_val_limit = int(15 + aug_intensity * 35)
            
            if augmentation_mode == 'separate':
                # 应用所有选择的独立增强方法
                success_count += self._apply_flip_augmentations(image, target_dir, file_name, img_format, params)
                success_count += self._apply_geometric_augmentations(image, target_dir, file_name, img_format, params, scale_limit)
                success_count += self._apply_color_augmentations(image, target_dir, file_name, img_format, params, brightness_limit, contrast_limit)
                success_count += self._apply_noise_blur_augmentations(image, target_dir, file_name, img_format, params, noise_limit_min, noise_limit_max, blur_limit)
                success_count += self._apply_hue_augmentations(image, target_dir, file_name, img_format, params, hue_shift_limit, sat_val_limit)
                
                if success_count > 0 and status_callback:
                    status_callback(f"对图片 {file_name} 成功应用 {success_count} 种增强方法")
            
            else:
                # 组合模式，使用预定义的增强配置
                augmentation_level = params.get('augmentation_level', '基础')
                aug = self.get_augmentation_with_intensity(augmentation_level, aug_intensity)
                # 为每张图片生成2个增强版本
                generated_count = 0
                for j in range(2):
                    try:
                        augmented = aug(image=image)['image']
                        aug_name = f'{file_name}_aug_{j}.{img_format}'
                        aug_path = os.path.join(target_dir, aug_name)
                        Image.fromarray(augmented).save(aug_path)
                        generated_count += 1
                    except Exception as e:
                        if status_callback:
                            status_callback(f"组合增强 #{j+1} 失败: {str(e)}")
                
                if generated_count > 0 and status_callback:
                    status_callback(f"对图片 {file_name} 应用了 {generated_count} 个组合增强版本")
                    
        except Exception as e:
            import traceback
            if status_callback:
                status_callback(f"增强图片 {file_name} 时出错: {str(e)}\n{traceback.format_exc()}")

    def _apply_flip_augmentations(self, image: np.ndarray, target_dir: str, 
                                file_name: str, img_format: str, params: Dict) -> int:
        """应用翻转增强"""
        success_count = 0
        
        if params.get('flip_horizontal'):
            try:
                aug = A.Compose([A.HorizontalFlip(p=1.0)])
                augmented = aug(image=image)['image']
                aug_name = f'{file_name}_flip_h.{img_format}'
                aug_path = os.path.join(target_dir, aug_name)
                Image.fromarray(augmented).save(aug_path)
                success_count += 1
            except Exception:
                pass
        
        if params.get('flip_vertical'):
            try:
                aug = A.Compose([A.VerticalFlip(p=1.0)])
                augmented = aug(image=image)['image']
                aug_name = f'{file_name}_flip_v.{img_format}'
                aug_path = os.path.join(target_dir, aug_name)
                Image.fromarray(augmented).save(aug_path)
                success_count += 1
            except Exception:
                pass
                
        return success_count

    def _apply_geometric_augmentations(self, image: np.ndarray, target_dir: str, 
                                     file_name: str, img_format: str, params: Dict, scale_limit: float) -> int:
        """应用几何变换增强"""
        success_count = 0
        
        if params.get('rotate'):
            try:
                aug = A.Compose([A.RandomRotate90(p=1.0)])
                augmented = aug(image=image)['image']
                aug_name = f'{file_name}_rotate.{img_format}'
                aug_path = os.path.join(target_dir, aug_name)
                Image.fromarray(augmented).save(aug_path)
                success_count += 1
            except Exception:
                pass
        
        if params.get('random_crop'):
            try:
                height = int(params['height'])
                width = int(params['width'])
                crop_height = min(height, image.shape[0])
                crop_width = min(width, image.shape[1])
                aug = A.Compose([A.RandomCrop(height=crop_height, width=crop_width, p=1.0)])
                augmented = aug(image=image)['image']
                aug_name = f'{file_name}_crop.{img_format}'
                aug_path = os.path.join(target_dir, aug_name)
                Image.fromarray(augmented).save(aug_path)
                success_count += 1
            except Exception:
                pass
        
        if params.get('random_scale'):
            try:
                aug = A.Compose([A.RandomScale(scale_limit=scale_limit, p=1.0)])
                augmented = aug(image=image)['image']
                aug_name = f'{file_name}_scale.{img_format}'
                aug_path = os.path.join(target_dir, aug_name)
                Image.fromarray(augmented).save(aug_path)
                success_count += 1
            except Exception:
                pass
                
        return success_count

    def _apply_color_augmentations(self, image: np.ndarray, target_dir: str, 
                                 file_name: str, img_format: str, params: Dict, 
                                 brightness_limit: float, contrast_limit: float) -> int:
        """应用颜色增强"""
        success_count = 0
        
        if params.get('brightness'):
            try:
                pil_img = Image.fromarray(image)
                enhancer = ImageEnhance.Brightness(pil_img)
                brightness_factor = 1.0 + brightness_limit
                brightened = enhancer.enhance(brightness_factor)
                aug_name = f'{file_name}_bright.{img_format}'
                aug_path = os.path.join(target_dir, aug_name)
                brightened.save(aug_path)
                success_count += 1
            except Exception:
                pass
        
        if params.get('contrast'):
            try:
                pil_img = Image.fromarray(image)
                enhancer = ImageEnhance.Contrast(pil_img)
                contrast_factor = 1.0 + contrast_limit
                contrasted = enhancer.enhance(contrast_factor)
                aug_name = f'{file_name}_contrast.{img_format}'
                aug_path = os.path.join(target_dir, aug_name)
                contrasted.save(aug_path)
                success_count += 1
            except Exception:
                pass
                
        return success_count

    def _apply_noise_blur_augmentations(self, image: np.ndarray, target_dir: str, 
                                      file_name: str, img_format: str, params: Dict,
                                      noise_limit_min: float, noise_limit_max: float, blur_limit: int) -> int:
        """应用噪声和模糊增强"""
        success_count = 0
        
        if params.get('noise'):
            try:
                aug = A.Compose([A.GaussNoise(var_limit=(noise_limit_min, noise_limit_max), p=1.0)])
                augmented = aug(image=image)['image']
                aug_name = f'{file_name}_noise.{img_format}'
                aug_path = os.path.join(target_dir, aug_name)
                Image.fromarray(augmented).save(aug_path)
                success_count += 1
            except Exception:
                pass
        
        if params.get('blur'):
            try:
                aug = A.Compose([A.GaussianBlur(blur_limit=blur_limit, p=1.0)])
                augmented = aug(image=image)['image']
                aug_name = f'{file_name}_blur.{img_format}'
                aug_path = os.path.join(target_dir, aug_name)
                Image.fromarray(augmented).save(aug_path)
                success_count += 1
            except Exception:
                pass
                
        return success_count

    def _apply_hue_augmentations(self, image: np.ndarray, target_dir: str, 
                               file_name: str, img_format: str, params: Dict,
                               hue_shift_limit: int, sat_val_limit: int) -> int:
        """应用色相增强"""
        success_count = 0
        
        if params.get('hue'):
            try:
                aug = A.Compose([A.HueSaturationValue(
                    hue_shift_limit=hue_shift_limit, 
                    sat_shift_limit=sat_val_limit, 
                    val_shift_limit=sat_val_limit, 
                    p=1.0
                )])
                augmented = aug(image=image)['image']
                aug_name = f'{file_name}_hue.{img_format}'
                aug_path = os.path.join(target_dir, aug_name)
                Image.fromarray(augmented).save(aug_path)
                success_count += 1
            except Exception:
                pass
                
        return success_count

    def _get_basic_augmentation(self):
        """获取基础数据增强配置"""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

    def _get_medium_augmentation(self):
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

    def _get_strong_augmentation(self):
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
        
    def apply_single_augmentation_to_image(self, image_path: str, target_dir: str, 
                                         file_name: str, img_format: str, params: Dict) -> str:
        """
        对单张图片应用随机增强，用于过采样
        
        Args:
            image_path: 输入图片路径
            target_dir: 输出目录
            file_name: 输出文件名（不含扩展名）
            img_format: 输出格式
            params: 增强参数
            
        Returns:
            str: 生成的增强图片路径
        """
        try:
            # 读取图片
            image = np.array(Image.open(image_path))
            
            # 选择随机增强
            augmentation = self._get_random_single_augmentation(params)
            
            # 应用增强
            augmented = augmentation(image=image)['image']
            
            # 保存增强后的图片
            output_path = os.path.join(target_dir, f"{file_name}.{img_format}")
            Image.fromarray(augmented).save(output_path)
            
            return output_path
            
        except Exception as e:
            # 如果增强失败，直接复制原图
            output_path = os.path.join(target_dir, f"{file_name}.{img_format}")
            shutil.copy2(image_path, output_path)
            return output_path
            
    def _get_random_single_augmentation(self, params: Dict) -> A.Compose:
        """获取随机单个增强操作"""
        augmentations = []
        
        # 根据参数添加可能的增强
        if params.get('flip_horizontal', False):
            augmentations.append(A.HorizontalFlip(p=1.0))
        if params.get('flip_vertical', False):
            augmentations.append(A.VerticalFlip(p=1.0))
        if params.get('rotate', False):
            augmentations.append(A.RandomRotate90(p=1.0))
        if params.get('brightness', False):
            augmentations.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0))
        if params.get('contrast', False):
            augmentations.append(A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1.0))
        if params.get('noise', False):
            augmentations.append(A.GaussNoise(var_limit=(10, 50), p=1.0))
        if params.get('blur', False):
            augmentations.append(A.GaussianBlur(blur_limit=3, p=1.0))
        if params.get('hue', False):
            augmentations.append(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0))
            
        # 如果没有启用任何增强，使用默认增强
        if not augmentations:
            augmentations = [A.HorizontalFlip(p=1.0)]
            
        # 随机选择一个增强
        import random
        selected_aug = random.choice(augmentations)
        
        return A.Compose([selected_aug]) 