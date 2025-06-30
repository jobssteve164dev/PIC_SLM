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
        """获取随机增强操作组合 - 改进版，添加更多连续随机参数"""
        import random
        import numpy as np
        
        # 生成随机强度参数
        intensity = random.uniform(0.3, 1.0)
        
        # 构建可用的增强方法列表（使用连续随机参数）
        available_augmentations = []
        
        # 1. 几何变换 - 连续随机参数
        if params.get('flip_horizontal', False):
            available_augmentations.append(A.HorizontalFlip(p=1.0))
        if params.get('flip_vertical', False):
            available_augmentations.append(A.VerticalFlip(p=1.0))
        if params.get('rotate', False):
            # 改进：使用连续角度而不是90度倍数
            available_augmentations.append(A.Rotate(limit=(-45 * intensity, 45 * intensity), p=1.0))
            available_augmentations.append(A.RandomRotate90(p=1.0))  # 保留原有的90度旋转
        
        # 2. 颜色变换 - 连续随机参数
        if params.get('brightness', False):
            brightness_limit = 0.1 + intensity * 0.3  # 0.1-0.4 范围
            available_augmentations.append(A.RandomBrightnessContrast(
                brightness_limit=brightness_limit, contrast_limit=0, p=1.0))
        if params.get('contrast', False):
            contrast_limit = 0.1 + intensity * 0.3  # 0.1-0.4 范围
            available_augmentations.append(A.RandomBrightnessContrast(
                brightness_limit=0, contrast_limit=contrast_limit, p=1.0))
        
        # 3. 噪声和模糊 - 连续随机参数
        if params.get('noise', False):
            noise_var = (5 + intensity * 10, 15 + intensity * 35)  # 动态范围
            available_augmentations.append(A.GaussNoise(var_limit=noise_var, p=1.0))
        if params.get('blur', False):
            blur_limit = int(1 + intensity * 4)  # 1-5 像素模糊
            available_augmentations.append(A.GaussianBlur(blur_limit=blur_limit, p=1.0))
        
        # 4. 色彩变换 - 连续随机参数
        if params.get('hue', False):
            hue_limit = int(10 + intensity * 30)  # 10-40 度色相偏移
            sat_limit = int(15 + intensity * 35)  # 15-50 饱和度偏移
            val_limit = int(10 + intensity * 30)  # 10-40 明度偏移
            available_augmentations.append(A.HueSaturationValue(
                hue_shift_limit=hue_limit, sat_shift_limit=sat_limit, val_shift_limit=val_limit, p=1.0))
        
        # 5. 新增高级连续随机变换
        # 随机仿射变换
        available_augmentations.append(A.Affine(
            scale=(0.9 - intensity * 0.1, 1.0 + intensity * 0.2),  # 0.8-1.2 缩放
            translate_percent=(-intensity * 0.1, intensity * 0.1),  # ±10% 平移
            rotate=(-20 * intensity, 20 * intensity),  # ±20度旋转
            shear=(-10 * intensity, 10 * intensity),  # ±10度剪切
            p=1.0
        ))
        
        # 随机透视变换
        available_augmentations.append(A.Perspective(
            scale=(0.02 + intensity * 0.08, 0.05 + intensity * 0.15),  # 0.02-0.2 透视强度
            p=1.0
        ))
        
        # 随机弹性变形
        available_augmentations.append(A.ElasticTransform(
            alpha=1 + intensity * 50,  # 1-51 变形强度
            sigma=5 + intensity * 15,  # 5-20 平滑度
            alpha_affine=5 + intensity * 15,  # 5-20 仿射强度
            p=1.0
        ))
        
        # 随机网格扭曲
        available_augmentations.append(A.GridDistortion(
            num_steps=3 + int(intensity * 2),  # 3-5 网格步数
            distort_limit=(-0.1 - intensity * 0.2, 0.1 + intensity * 0.2),  # 扭曲强度
            p=1.0
        ))
        
        # 随机光学扭曲
        available_augmentations.append(A.OpticalDistortion(
            distort_limit=(-0.1 - intensity * 0.4, 0.1 + intensity * 0.4),  # -0.5 到 0.5
            shift_limit=(-0.05 - intensity * 0.1, 0.05 + intensity * 0.1),  # 位移限制
            p=1.0
        ))
        
        # 随机伽马校正
        available_augmentations.append(A.RandomGamma(
            gamma_limit=(80 - int(intensity * 20), 120 + int(intensity * 30)),  # 60-150 伽马范围
            p=1.0
        ))
        
        # 随机阴影
        available_augmentations.append(A.RandomShadow(
            shadow_roi=(0, 0.3 + intensity * 0.2, 1, 0.7 + intensity * 0.3),  # 阴影区域
            num_shadows_lower=1,
            num_shadows_upper=1 + int(intensity * 2),  # 1-3 个阴影
            shadow_dimension=3 + int(intensity * 2),  # 阴影维度
            p=1.0
        ))
        
        # 随机太阳耀斑
        available_augmentations.append(A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.3 + intensity * 0.4),  # 耀斑区域
            angle_lower=0,
            angle_upper=1,
            num_flare_circles_lower=1,
            num_flare_circles_upper=1 + int(intensity * 2),  # 1-3 个耀斑圆
            src_radius=50 + int(intensity * 100),  # 50-150 源半径
            p=1.0
        ))
        
        # 随机雨滴效果
        available_augmentations.append(A.RandomRain(
            slant_lower=-5 - int(intensity * 5),  # -10 到 -5
            slant_upper=5 + int(intensity * 5),   # 5 到 10
            drop_length=1 + int(intensity * 4),   # 1-5 雨滴长度
            drop_width=1 + int(intensity * 2),    # 1-3 雨滴宽度
            drop_color=(200, 200, 200),
            blur_value=1 + int(intensity * 2),    # 1-3 模糊值
            brightness_coefficient=0.6 + intensity * 0.3,  # 0.6-0.9 亮度系数
            rain_type="drizzle" if intensity < 0.5 else "heavy",
            p=1.0
        ))
        
        # 随机雾效果
        available_augmentations.append(A.RandomFog(
            fog_coef_lower=0.1 + intensity * 0.2,  # 0.1-0.3
            fog_coef_upper=0.2 + intensity * 0.4,  # 0.2-0.6
            alpha_coef=0.08 + intensity * 0.1,     # 0.08-0.18
            p=1.0
        ))
        
        # 随机雪效果
        available_augmentations.append(A.RandomSnow(
            snow_point_lower=0.1 + intensity * 0.1,  # 0.1-0.2
            snow_point_upper=0.2 + intensity * 0.2,  # 0.2-0.4
            brightness_coeff=1.5 + intensity * 0.5,  # 1.5-2.0
            p=1.0
        ))
        
        # CLAHE (对比度限制自适应直方图均衡)
        available_augmentations.append(A.CLAHE(
            clip_limit=(1 + intensity * 3, 2 + intensity * 6),  # 1-4, 2-8
            tile_grid_size=(4 + int(intensity * 4), 4 + int(intensity * 4)),  # 4-8, 4-8
            p=1.0
        ))
        
        # 随机色调分离
        available_augmentations.append(A.Solarize(
            threshold=(32 + int(intensity * 96), 64 + int(intensity * 128)),  # 32-128, 64-192
            p=1.0
        ))
        
        # 随机后验化
        available_augmentations.append(A.Posterize(
            num_bits=(3 + int(intensity * 2), 4 + int(intensity * 2)),  # 3-5, 4-6
            p=1.0
        ))
        
        # 随机均衡化
        available_augmentations.append(A.Equalize(p=1.0))
        
        # 随机自动对比度
        available_augmentations.append(A.AutoContrast(p=1.0))
        
        # 如果没有启用任何增强，添加默认的连续随机增强
        if not available_augmentations:
            available_augmentations = [
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.GaussNoise(var_limit=(10, 50), p=1.0)
            ]
        
        # 随机选择1-3个增强方法进行组合
        num_augmentations = random.randint(1, min(3, len(available_augmentations)))
        selected_augs = random.sample(available_augmentations, num_augmentations)
        
        # 为每个选中的增强添加随机概率
        final_augs = []
        for aug in selected_augs:
            # 为每个增强设置随机概率 (0.7-1.0)
            aug_prob = 0.7 + random.uniform(0, 0.3)
            # 创建带概率的增强
            if hasattr(aug, 'p'):
                aug.p = aug_prob
            final_augs.append(aug)
        
        return A.Compose(final_augs)
        
    def get_enhanced_random_augmentation(self, params: Dict) -> A.Compose:
        """获取增强版随机增强 - 专门用于高质量过采样"""
        import random
        import numpy as np
        
        # 生成更强的随机强度
        intensity = random.uniform(0.5, 1.0)
        
        # 构建高级增强序列
        augmentation_sequence = []
        
        # 第一阶段：几何变换 (随机选择1-2个)
        geometric_augs = []
        if random.random() < 0.7:  # 70% 概率应用旋转
            angle_range = random.uniform(10, 45) * intensity
            geometric_augs.append(A.Rotate(limit=(-angle_range, angle_range), p=1.0))
        
        if random.random() < 0.5:  # 50% 概率应用仿射变换
            geometric_augs.append(A.Affine(
                scale=(0.85 + random.uniform(0, 0.1), 1.15 + random.uniform(0, 0.1)),
                translate_percent=(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)),
                rotate=(random.uniform(-15, 15), random.uniform(-15, 15)),
                shear=(random.uniform(-10, 10), random.uniform(-10, 10)),
                p=1.0
            ))
        
        if random.random() < 0.3:  # 30% 概率应用透视变换
            perspective_scale = random.uniform(0.02, 0.08) * intensity
            geometric_augs.append(A.Perspective(scale=(perspective_scale, perspective_scale * 2), p=1.0))
        
        if geometric_augs:
            augmentation_sequence.extend(random.sample(geometric_augs, min(2, len(geometric_augs))))
        
        # 第二阶段：颜色变换 (随机选择2-3个)
        color_augs = []
        
        # 亮度对比度 (高概率)
        if random.random() < 0.8:
            brightness_range = random.uniform(0.1, 0.3) * intensity
            contrast_range = random.uniform(0.1, 0.3) * intensity
            color_augs.append(A.RandomBrightnessContrast(
                brightness_limit=brightness_range, contrast_limit=contrast_range, p=1.0))
        
        # 色相饱和度 (高概率)
        if random.random() < 0.8:
            hue_range = int(random.uniform(10, 30) * intensity)
            sat_range = int(random.uniform(15, 40) * intensity)
            val_range = int(random.uniform(10, 25) * intensity)
            color_augs.append(A.HueSaturationValue(
                hue_shift_limit=hue_range, sat_shift_limit=sat_range, val_shift_limit=val_range, p=1.0))
        
        # 伽马校正
        if random.random() < 0.6:
            gamma_lower = 80 - int(random.uniform(0, 20) * intensity)
            gamma_upper = 120 + int(random.uniform(0, 30) * intensity)
            color_augs.append(A.RandomGamma(gamma_limit=(gamma_lower, gamma_upper), p=1.0))
        
        # CLAHE
        if random.random() < 0.4:
            clip_limit = random.uniform(1, 4) * intensity
            tile_size = 4 + int(random.uniform(0, 4) * intensity)
            color_augs.append(A.CLAHE(clip_limit=(clip_limit, clip_limit * 2), 
                                    tile_grid_size=(tile_size, tile_size), p=1.0))
        
        if color_augs:
            augmentation_sequence.extend(random.sample(color_augs, min(3, len(color_augs))))
        
        # 第三阶段：噪声和质量变换 (随机选择1-2个)
        quality_augs = []
        
        # 高斯噪声
        if random.random() < 0.6:
            noise_min = random.uniform(5, 15) * intensity
            noise_max = random.uniform(20, 50) * intensity
            quality_augs.append(A.GaussNoise(var_limit=(noise_min, noise_max), p=1.0))
        
        # 模糊
        if random.random() < 0.4:
            blur_limit = int(random.uniform(1, 3) * intensity)
            quality_augs.append(A.GaussianBlur(blur_limit=blur_limit, p=1.0))
        
        # 锐化
        if random.random() < 0.3:
            quality_augs.append(A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0))
        
        if quality_augs:
            augmentation_sequence.extend(random.sample(quality_augs, min(2, len(quality_augs))))
        
        # 第四阶段：特殊效果 (低概率，高多样性)
        special_augs = []
        
        if random.random() < 0.2:  # 雾效果
            fog_coef = random.uniform(0.1, 0.4) * intensity
            special_augs.append(A.RandomFog(fog_coef_lower=fog_coef, fog_coef_upper=fog_coef * 2, p=1.0))
        
        if random.random() < 0.15:  # 阴影效果
            shadow_roi = (0, random.uniform(0.2, 0.4), 1, random.uniform(0.6, 0.8))
            special_augs.append(A.RandomShadow(shadow_roi=shadow_roi, p=1.0))
        
        if random.random() < 0.1:  # 太阳耀斑
            flare_roi = (0, 0, 1, random.uniform(0.3, 0.5))
            special_augs.append(A.RandomSunFlare(flare_roi=flare_roi, p=1.0))
        
        if special_augs:
            augmentation_sequence.extend(random.sample(special_augs, min(1, len(special_augs))))
        
        # 确保至少有一个增强
        if not augmentation_sequence:
            augmentation_sequence = [A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)]
        
        # 随机打乱增强顺序
        random.shuffle(augmentation_sequence)
        
        return A.Compose(augmentation_sequence) 