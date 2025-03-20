import os
import cv2
import numpy as np
import shutil
from PIL import Image, ImageEnhance
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import albumentations as A

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
                - brightness_value: 亮度调整值 (-50 到 50)
                - contrast_value: 对比度调整值 (-50 到 50)
                - train_ratio: 训练集比例 (0.1 到 0.9)
                - augmentation_level: 数据增强级别 ('基础', '中等', '强化')
                - dataset_folder: 数据集输出文件夹
                - brightness: 是否启用亮度增强 (布尔值)
                - contrast: 是否启用对比度增强 (布尔值)
                - noise: 是否启用噪声增强 (布尔值)
                - blur: 是否启用模糊增强 (布尔值)
                - hue: 是否启用色相增强 (布尔值)
                - balance_classes: 是否保持类别平衡 (布尔值)
                - class_names: 类别名称列表
        """
        try:
            # 重置停止标志
            self._stop_preprocessing = False
            
            # 用于调试的打印
            self.status_updated.emit(f"增强模式: {params.get('augmentation_mode')}")
            self.status_updated.emit(f"亮度调整: {params.get('brightness')}")
            self.status_updated.emit(f"对比度调整: {params.get('contrast')}")
            self.status_updated.emit(f"噪声: {params.get('noise')}")
            self.status_updated.emit(f"模糊: {params.get('blur')}")
            self.status_updated.emit(f"色相: {params.get('hue')}")
            self.status_updated.emit(f"类别平衡: {params.get('balance_classes', False)}")
            
            # 重命名亮度和对比度参数，避免与增强方法名称冲突
            if isinstance(params.get('brightness'), bool):
                # 如果是布尔值，说明这是增强方法的开关
                # 保留，但修改亮度调整值的参数名
                brightness_value = params.get('brightness_value', 0)
                params['brightness_value'] = brightness_value
            elif isinstance(params.get('brightness'), (int, float)):
                # 如果是数值，保存为亮度调整值
                brightness_value = params.get('brightness', 0)
                params['brightness_value'] = brightness_value
            
            if isinstance(params.get('contrast'), bool):
                # 如果是布尔值，说明这是增强方法的开关
                # 保留，但修改对比度调整值的参数名
                contrast_value = params.get('contrast_value', 0)
                params['contrast_value'] = contrast_value
            elif isinstance(params.get('contrast'), (int, float)):
                # 如果是数值，保存为对比度调整值
                contrast_value = params.get('contrast', 0)
                params['contrast_value'] = contrast_value

            # 检查是否要使用类别平衡处理
            use_class_balance = params.get('balance_classes', False)
            class_names = params.get('class_names', [])
            
            # 如果需要类别平衡且有类别名称
            if use_class_balance and class_names:
                self.status_updated.emit('使用类别平衡预处理模式')
                self._preprocess_with_class_balance(params)
            else:
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
            
            # 如果没有被停止，发出完成信号，更新状态消息以区分处理模式
            if not self._stop_preprocessing:
                if use_class_balance and class_names:
                    self.status_updated.emit('类别平衡预处理全部完成')
                else:
                    self.status_updated.emit('标准预处理全部完成')
                
            # 无论如何，都发出预处理完成信号 - 确保UI始终得到更新
            self.preprocessing_finished.emit()
            print("ImagePreprocessor: 已发出 preprocessing_finished 信号")
            
        except Exception as e:
            import traceback
            self.preprocessing_error.emit(f'预处理过程中出错: {str(e)}\n{traceback.format_exc()}')
            # 即使出错也发出完成信号，以确保UI恢复正常状态
            self.preprocessing_finished.emit()
            print("ImagePreprocessor: 虽然处理出错，但仍发出 preprocessing_finished 信号")
            
    def stop(self):
        """停止预处理过程"""
        self._stop_preprocessing = True
        self.status_updated.emit('正在停止预处理...')
        # 在停止预处理时也发出完成信号，确保UI恢复正常状态
        self.preprocessing_finished.emit()
        print("ImagePreprocessor.stop: 发出 preprocessing_finished 信号")

    def _preprocess_with_class_balance(self, params: Dict) -> None:
        """使用类别平衡的方式预处理图片"""
        source_folder = params['source_folder']
        output_folder = params['target_folder']
        dataset_folder = params['dataset_folder']
        train_ratio = params['train_ratio']
        
        # 调试输出参数信息
        self.status_updated.emit(f"预处理参数: 增强模式={params.get('augmentation_mode', '未指定')}")
        self.status_updated.emit(f"启用的增强方法: 水平翻转={params.get('flip_horizontal', False)}, 亮度={params.get('brightness', False)}, 对比度={params.get('contrast', False)}等")
        
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
            if self._stop_preprocessing:
                self.status_updated.emit('预处理已停止')
                return
            
            # 构建类别源文件夹路径
            class_source_folder = os.path.join(source_folder, class_name)
            
            # 跳过不存在的类别文件夹
            if not os.path.exists(class_source_folder) or not os.path.isdir(class_source_folder):
                self.status_updated.emit(f"警告：类别文件夹不存在: {class_source_folder}，跳过该类别")
                continue
            
            self.status_updated.emit(f"处理类别: {class_name}")
            
            # 创建输出文件夹中的类别文件夹
            class_output_folder = os.path.join(output_folder, class_name)
            os.makedirs(class_output_folder, exist_ok=True)
            
            # 创建类别的训练集和验证集文件夹
            class_train_folder = os.path.join(train_folder, class_name)
            class_val_folder = os.path.join(val_folder, class_name)
            os.makedirs(class_train_folder, exist_ok=True)
            os.makedirs(class_val_folder, exist_ok=True)
            
            # 获取该类别的所有图片
            image_files = self._get_image_files(class_source_folder)
            
            if not image_files:
                self.status_updated.emit(f"警告：类别 {class_name} 中没有图片文件，跳过该类别")
                continue
            
            # 先划分训练集和验证集
            train_images, val_images = train_test_split(
                image_files, train_size=train_ratio, random_state=42)
            
            self.status_updated.emit(f"类别 {class_name} 划分为训练集{len(train_images)}张和验证集{len(val_images)}张")
                
            # 第1步：对原始图片进行基本预处理
            # 先处理训练集图片
            for img_file in train_images:
                if self._stop_preprocessing:
                    return
                    
                # 预处理原始图片并保存到输出文件夹
                source_path = os.path.join(class_source_folder, img_file)
                file_name = os.path.splitext(img_file)[0]
                img_format = params['format']
                output_path = os.path.join(class_output_folder, f"{file_name}.{img_format}")
                
                # 基本图像处理（调整大小、亮度、对比度）
                self._process_single_image(
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
                self._apply_augmentations_to_image(
                    output_path, 
                    class_train_folder, 
                    file_name, 
                    img_format, 
                    params
                )
            
            # 再处理验证集图片 - 只进行基本预处理，不增强
            for img_file in val_images:
                if self._stop_preprocessing:
                    return
                    
                # 预处理原始图片并保存到输出文件夹
                source_path = os.path.join(class_source_folder, img_file)
                file_name = os.path.splitext(img_file)[0]
                img_format = params['format']
                output_path = os.path.join(class_output_folder, f"{file_name}.{img_format}")
                
                # 基本图像处理（调整大小、亮度、对比度）
                self._process_single_image(
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
        
    def _apply_augmentations_to_image(self, image_path: str, target_dir: str, file_name: str, img_format: str, params: Dict) -> None:
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
                if params.get('flip_horizontal'):
                    try:
                        aug = A.Compose([A.HorizontalFlip(p=1.0)])
                        augmented = aug(image=image)['image']
                        aug_name = f'{file_name}_flip_h.{img_format}'
                        aug_path = os.path.join(target_dir, aug_name)
                        Image.fromarray(augmented).save(aug_path)
                        success_count += 1
                        generated_methods.append("水平翻转")
                    except Exception as e:
                        self.status_updated.emit(f"水平翻转增强失败: {str(e)}")
                
                if params.get('flip_vertical'):
                    try:
                        aug = A.Compose([A.VerticalFlip(p=1.0)])
                        augmented = aug(image=image)['image']
                        aug_name = f'{file_name}_flip_v.{img_format}'
                        aug_path = os.path.join(target_dir, aug_name)
                        Image.fromarray(augmented).save(aug_path)
                        success_count += 1
                        generated_methods.append("垂直翻转")
                    except Exception as e:
                        self.status_updated.emit(f"垂直翻转增强失败: {str(e)}")
                
                if params.get('rotate'):
                    try:
                        aug = A.Compose([A.RandomRotate90(p=1.0)])
                        augmented = aug(image=image)['image']
                        aug_name = f'{file_name}_rotate.{img_format}'
                        aug_path = os.path.join(target_dir, aug_name)
                        Image.fromarray(augmented).save(aug_path)
                        success_count += 1
                        generated_methods.append("旋转")
                    except Exception as e:
                        self.status_updated.emit(f"旋转增强失败: {str(e)}")
                
                if params.get('random_crop'):
                    try:
                        # 确保裁剪尺寸不超过图片尺寸
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
                        generated_methods.append("裁剪")
                    except Exception as e:
                        self.status_updated.emit(f"裁剪增强失败: {str(e)}")
                
                if params.get('random_scale'):
                    try:
                        aug = A.Compose([A.RandomScale(scale_limit=scale_limit, p=1.0)])
                        augmented = aug(image=image)['image']
                        aug_name = f'{file_name}_scale.{img_format}'
                        aug_path = os.path.join(target_dir, aug_name)
                        Image.fromarray(augmented).save(aug_path)
                        success_count += 1
                        generated_methods.append("缩放")
                    except Exception as e:
                        self.status_updated.emit(f"缩放增强失败: {str(e)}")
                
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
                        generated_methods.append("亮度")
                    except Exception as e:
                        self.status_updated.emit(f"亮度增强失败: {str(e)}")
                
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
                        generated_methods.append("对比度")
                    except Exception as e:
                        self.status_updated.emit(f"对比度增强失败: {str(e)}")
                
                if params.get('noise'):
                    try:
                        aug = A.Compose([A.GaussNoise(var_limit=(noise_limit_min, noise_limit_max), p=1.0)])
                        augmented = aug(image=image)['image']
                        aug_name = f'{file_name}_noise.{img_format}'
                        aug_path = os.path.join(target_dir, aug_name)
                        Image.fromarray(augmented).save(aug_path)
                        success_count += 1
                        generated_methods.append("噪声")
                    except Exception as e:
                        self.status_updated.emit(f"噪声增强失败: {str(e)}")
                
                if params.get('blur'):
                    try:
                        aug = A.Compose([A.GaussianBlur(blur_limit=blur_limit, p=1.0)])
                        augmented = aug(image=image)['image']
                        aug_name = f'{file_name}_blur.{img_format}'
                        aug_path = os.path.join(target_dir, aug_name)
                        Image.fromarray(augmented).save(aug_path)
                        success_count += 1
                        generated_methods.append("模糊")
                    except Exception as e:
                        self.status_updated.emit(f"模糊增强失败: {str(e)}")
                
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
                        generated_methods.append("色相")
                    except Exception as e:
                        self.status_updated.emit(f"色相增强失败: {str(e)}")
                
                if success_count > 0:
                    self.status_updated.emit(f"对图片 {file_name} 成功应用 {success_count} 种增强方法: {', '.join(generated_methods)}")
            
            else:
                # 组合模式，使用预定义的增强配置
                augmentation_level = params.get('augmentation_level', '基础')
                aug = self._get_augmentation_with_intensity(augmentation_level, aug_intensity)
                # 为每张图片生成2个增强版本
                generated_count = 0
                for j in range(2):
                    if self._stop_preprocessing:
                        return
                    try:
                        augmented = aug(image=image)['image']
                        aug_name = f'{file_name}_aug_{j}.{img_format}'
                        aug_path = os.path.join(target_dir, aug_name)
                        Image.fromarray(augmented).save(aug_path)
                        generated_count += 1
                    except Exception as e:
                        self.status_updated.emit(f"组合增强 #{j+1} 失败: {str(e)}")
                
                if generated_count > 0:
                    self.status_updated.emit(f"对图片 {file_name} 应用了 {generated_count} 个组合增强版本")
                    
        except Exception as e:
            import traceback
            self.status_updated.emit(f"增强图片 {file_name} 时出错: {str(e)}\n{traceback.format_exc()}")

    def _preprocess_class_images(self, source_folder: str, target_folder: str, 
                                image_files: List[str], params: Dict) -> None:
        """预处理特定类别的原始图片"""
        width = params['width']
        height = params['height']
        img_format = params['format']
        brightness = params['brightness_value']
        contrast = params['contrast_value']
        
        # 获取增强模式
        augmentation_mode = params.get('augmentation_mode', 'combined')
        self.status_updated.emit(f"类别预处理使用增强模式: {augmentation_mode}")
        
        # 获取增强强度参数
        aug_intensity = params.get('augmentation_intensity', 0.5)
        
        # 根据增强强度调整各增强参数
        # 注意：有些效果需要较小强度(如亮度)，有些需要较大强度(如噪声)
        scale_limit = aug_intensity * 0.4  # 缩放范围: 0.04 to 0.4
        brightness_limit = aug_intensity * 0.6  # 亮度范围: 0.06 to 0.6
        contrast_limit = aug_intensity * 0.6  # 对比度范围: 0.06 to 0.6
        noise_limit_min = 5.0 + aug_intensity * 15.0  # 噪声最小值范围: 5.0 to 20.0
        noise_limit_max = 20.0 + aug_intensity * 80.0  # 噪声最大值范围: 20.0 to 100.0
        blur_limit = int(2 + aug_intensity * 5)  # 模糊范围: 2 to 7
        hue_shift_limit = int(10 + aug_intensity * 30)  # 色相范围: 10 to 40
        sat_val_limit = int(15 + aug_intensity * 35)  # 饱和度/明度范围: 15 to 50
        
        # 处理每个图片
        total_files = len(image_files)
        processed_count = 0
        
        for i, img_file in enumerate(image_files):
            # 检查是否需要停止处理
            if self._stop_preprocessing:
                return
                
            try:
                # 构建源文件和目标文件路径
                source_path = os.path.join(source_folder, img_file)
                file_name = os.path.splitext(img_file)[0]
                target_path = os.path.join(target_folder, f"{file_name}.{img_format}")
                
                # 步骤1：基本预处理（调整大小、亮度、对比度等）
                self._process_single_image(
                    source_path, 
                    target_path, 
                    width, 
                    height, 
                    brightness, 
                    contrast
                )
                processed_count += 1
                
                # 步骤2：如果是独立模式且启用了增强，则为每种方法单独生成增强图像
                if augmentation_mode == 'separate':
                    try:
                        # 读取预处理后的图像作为增强的输入
                        image = np.array(Image.open(target_path).convert('RGB'))
                        success_count = 0
                        generated_methods = []
                        
                        # 应用所有选择的增强方法
                        if params.get('flip_horizontal'):
                            try:
                                aug = A.Compose([A.HorizontalFlip(p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{file_name}_flip_h.{img_format}'
                                aug_path = os.path.join(target_folder, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("水平翻转")
                                processed_count += 1
                            except Exception as e:
                                self.status_updated.emit(f"水平翻转增强失败: {str(e)}")
                            
                        if params.get('flip_vertical'):
                            try:
                                aug = A.Compose([A.VerticalFlip(p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{file_name}_flip_v.{img_format}'
                                aug_path = os.path.join(target_folder, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("垂直翻转")
                                processed_count += 1
                            except Exception as e:
                                self.status_updated.emit(f"垂直翻转增强失败: {str(e)}")
                        
                        if params.get('rotate'):
                            try:
                                aug = A.Compose([A.RandomRotate90(p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{file_name}_rotate.{img_format}'
                                aug_path = os.path.join(target_folder, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("旋转")
                                processed_count += 1
                            except Exception as e:
                                self.status_updated.emit(f"旋转增强失败: {str(e)}")
                        
                        if params.get('random_crop'):
                            try:
                                # 确保裁剪尺寸不超过图片尺寸
                                crop_height = min(height, image.shape[0])
                                crop_width = min(width, image.shape[1])
                                aug = A.Compose([A.RandomCrop(height=crop_height, width=crop_width, p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{file_name}_crop.{img_format}'
                                aug_path = os.path.join(target_folder, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("裁剪")
                                processed_count += 1
                            except Exception as e:
                                self.status_updated.emit(f"裁剪增强失败: {str(e)}")
                        
                        if params.get('random_scale'):
                            try:
                                # 使用增强强度调整缩放范围
                                aug = A.Compose([A.RandomScale(scale_limit=scale_limit, p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{file_name}_scale.{img_format}'
                                aug_path = os.path.join(target_folder, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("缩放")
                            except Exception as e:
                                self.status_updated.emit(f"缩放增强失败: {str(e)}")
                        
                        if params.get('brightness'):
                            try:
                                pil_img = Image.fromarray(image)
                                enhancer = ImageEnhance.Brightness(pil_img)
                                # 使用增强强度调整亮度
                                brightness_factor = 1.0 + brightness_limit
                                brightened = enhancer.enhance(brightness_factor)
                                aug_name = f'{file_name}_bright.{img_format}'
                                aug_path = os.path.join(target_folder, aug_name)
                                brightened.save(aug_path)
                                success_count += 1
                                generated_methods.append("亮度")
                            except Exception as e:
                                self.status_updated.emit(f"亮度增强失败: {str(e)}")
                        
                        if params.get('contrast'):
                            try:
                                pil_img = Image.fromarray(image)
                                enhancer = ImageEnhance.Contrast(pil_img)
                                # 使用增强强度调整对比度
                                contrast_factor = 1.0 + contrast_limit
                                contrasted = enhancer.enhance(contrast_factor)
                                aug_name = f'{file_name}_contrast.{img_format}'
                                aug_path = os.path.join(target_folder, aug_name)
                                contrasted.save(aug_path)
                                success_count += 1
                                generated_methods.append("对比度")
                            except Exception as e:
                                self.status_updated.emit(f"对比度增强失败: {str(e)}")
                                
                        if params.get('noise'):
                            try:
                                # 使用增强强度调整噪声增强效果
                                aug = A.Compose([A.GaussNoise(var_limit=(noise_limit_min, noise_limit_max), p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{file_name}_noise.{img_format}'
                                aug_path = os.path.join(target_folder, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("噪声")
                            except Exception as e:
                                self.status_updated.emit(f"噪声增强失败: {str(e)}")
                            
                        if params.get('blur'):
                            try:
                                # 使用增强强度调整模糊增强效果
                                aug = A.Compose([A.GaussianBlur(blur_limit=blur_limit, p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{file_name}_blur.{img_format}'
                                aug_path = os.path.join(target_folder, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("模糊")
                            except Exception as e:
                                self.status_updated.emit(f"模糊增强失败: {str(e)}")
                            
                        if params.get('hue'):
                            try:
                                # 使用增强强度调整色相增强效果
                                aug = A.Compose([A.HueSaturationValue(
                                    hue_shift_limit=hue_shift_limit, 
                                    sat_shift_limit=sat_val_limit, 
                                    val_shift_limit=sat_val_limit, 
                                    p=1.0
                                )])
                                augmented = aug(image=image)['image']
                                aug_name = f'{file_name}_hue.{img_format}'
                                aug_path = os.path.join(target_folder, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("色相")
                            except Exception as e:
                                self.status_updated.emit(f"色相增强失败: {str(e)}")
                                
                        if i == 0:  # 仅对第一张图片详细记录应用的方法
                            self.status_updated.emit(f"对图片 {img_file} 成功应用 {success_count} 种增强方法: {', '.join(generated_methods)}")
                        
                    except Exception as e:
                        self.status_updated.emit(f"应用增强时出错: {str(e)}")
                
                # 更新进度
                progress = int(((i + 1) / total_files) * 50)  # 预处理占总进度的50%
                self.progress_updated.emit(progress)
                
            except Exception as e:
                self.status_updated.emit(f'处理图片 {img_file} 时出错: {str(e)}')
        
        self.status_updated.emit(f'完成类别图片预处理，共处理 {processed_count} 张图片')

    def _process_dataset_class_images(self,
                                     source_dir: str,
                                     target_dir: str,
                                     images: List[str],
                                     augmentation_level: str,
                                     apply_augmentation: bool,
                                     params: Dict) -> None:
        """处理特定类别的数据集图片，包括复制和数据增强"""
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
                    try:
                        image = np.array(Image.open(src_path).convert('RGB'))
                    except Exception as e:
                        self.status_updated.emit(f"打开图片 {image_name} 失败: {str(e)}")
                        continue
                    
                    # 根据增强模式选择不同的处理方式
                    if params.get('augmentation_mode') == 'separate':
                        success_count = 0
                        # 记录强制生成的增强方法
                        generated_methods = []
                        
                        # 独立模式：为每个选中的增强方法生成单独的图片
                        if params.get('flip_horizontal'):
                            try:
                                aug = A.Compose([A.HorizontalFlip(p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{os.path.splitext(image_name)[0]}_flip_h{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("水平翻转")
                            except Exception as e:
                                self.status_updated.emit(f"水平翻转增强失败: {str(e)}")
                            
                        if params.get('flip_vertical'):
                            try:
                                aug = A.Compose([A.VerticalFlip(p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{os.path.splitext(image_name)[0]}_flip_v{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("垂直翻转")
                            except Exception as e:
                                self.status_updated.emit(f"垂直翻转增强失败: {str(e)}")
                            
                        if params.get('rotate'):
                            try:
                                aug = A.Compose([A.RandomRotate90(p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{os.path.splitext(image_name)[0]}_rotate{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("旋转")
                            except Exception as e:
                                self.status_updated.emit(f"旋转增强失败: {str(e)}")
                            
                        if params.get('random_crop'):
                            try:
                                height = int(params['height'])
                                width = int(params['width'])
                                # 确保裁剪尺寸不超过图片尺寸
                                crop_height = min(height, image.shape[0])
                                crop_width = min(width, image.shape[1])
                                aug = A.Compose([A.RandomCrop(height=crop_height, width=crop_width, p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{os.path.splitext(image_name)[0]}_crop{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("裁剪")
                            except Exception as e:
                                self.status_updated.emit(f"裁剪增强失败: {str(e)}")
                            
                        if params.get('random_scale'):
                            try:
                                # 使用增强强度调整缩放范围
                                aug = A.Compose([A.RandomScale(scale_limit=scale_limit, p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{os.path.splitext(image_name)[0]}_scale{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("缩放")
                            except Exception as e:
                                self.status_updated.emit(f"缩放增强失败: {str(e)}")
                        
                        # 亮度增强 - 只在用户勾选时应用
                        if params.get('brightness'):
                            try:
                                pil_img = Image.fromarray(image)
                                enhancer = ImageEnhance.Brightness(pil_img)
                                # 使用增强强度调整亮度
                                brightness_factor = 1.0 + brightness_limit
                                brightened = enhancer.enhance(brightness_factor)
                                aug_name = f'{os.path.splitext(image_name)[0]}_bright{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                brightened.save(aug_path)
                                success_count += 1
                                generated_methods.append("亮度")
                            except Exception as e:
                                self.status_updated.emit(f"亮度增强失败: {str(e)}")
                        
                        # 对比度增强 - 只在用户勾选时应用
                        if params.get('contrast'):
                            try:
                                pil_img = Image.fromarray(image)
                                enhancer = ImageEnhance.Contrast(pil_img)
                                # 使用增强强度调整对比度
                                contrast_factor = 1.0 + contrast_limit
                                contrasted = enhancer.enhance(contrast_factor)
                                aug_name = f'{os.path.splitext(image_name)[0]}_contrast{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                contrasted.save(aug_path)
                                success_count += 1
                                generated_methods.append("对比度")
                            except Exception as e:
                                self.status_updated.emit(f"对比度增强失败: {str(e)}")
                                
                        if params.get('noise'):
                            try:
                                # 使用增强强度调整噪声增强效果
                                aug = A.Compose([A.GaussNoise(var_limit=(noise_limit_min, noise_limit_max), p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{os.path.splitext(image_name)[0]}_noise{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("噪声")
                            except Exception as e:
                                self.status_updated.emit(f"噪声增强失败: {str(e)}")
                            
                        if params.get('blur'):
                            try:
                                # 使用增强强度调整模糊增强效果
                                aug = A.Compose([A.GaussianBlur(blur_limit=blur_limit, p=1.0)])
                                augmented = aug(image=image)['image']
                                aug_name = f'{os.path.splitext(image_name)[0]}_blur{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("模糊")
                            except Exception as e:
                                self.status_updated.emit(f"模糊增强失败: {str(e)}")
                            
                        if params.get('hue'):
                            try:
                                # 使用增强强度调整色相增强效果
                                aug = A.Compose([A.HueSaturationValue(
                                    hue_shift_limit=hue_shift_limit, 
                                    sat_shift_limit=sat_val_limit, 
                                    val_shift_limit=sat_val_limit, 
                                    p=1.0
                                )])
                                augmented = aug(image=image)['image']
                                aug_name = f'{os.path.splitext(image_name)[0]}_hue{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                success_count += 1
                                generated_methods.append("色相")
                            except Exception as e:
                                self.status_updated.emit(f"色相增强失败: {str(e)}")
                                
                        if i == 0:  # 仅对第一张图片记录
                            self.status_updated.emit(f"成功应用 {success_count} 种增强方法: {', '.join(generated_methods)}")
                    else:
                        # 组合模式：创建一个组合多种增强方法的增强器
                        aug_transforms = []
                        
                        if params.get('flip_horizontal'):
                            aug_transforms.append(A.HorizontalFlip(p=0.5))
                            
                        if params.get('flip_vertical'):
                            aug_transforms.append(A.VerticalFlip(p=0.5))
                            
                        if params.get('rotate'):
                            aug_transforms.append(A.RandomRotate90(p=0.5))
                            
                        if aug_transforms:
                            try:
                                aug = A.Compose(aug_transforms)
                                augmented = aug(image=image)['image']
                                aug_name = f'{os.path.splitext(image_name)[0]}_aug{os.path.splitext(image_name)[1]}'
                                aug_path = os.path.join(target_dir, aug_name)
                                Image.fromarray(augmented).save(aug_path)
                                self.status_updated.emit(f"对 {image_name} 应用了组合增强")
                            except Exception as e:
                                self.status_updated.emit(f"组合增强失败: {str(e)}")
                        
            except Exception as e:
                self.status_updated.emit(f"处理数据集图片 {image_name} 时出错: {str(e)}")

    def _preprocess_raw_images(self, params: Dict) -> None:
        """预处理原始图片"""
        source_folder = params['source_folder']
        target_folder = params['target_folder']
        width = params['width']
        height = params['height']
        img_format = params['format']
        brightness = params['brightness_value']
        contrast = params['contrast_value']
        
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
            
        # 处理训练集 - 应用增强
        self._process_dataset_images(
            preprocessed_folder, train_folder, train_images,
            augmentation_level, True, 50, 75, params)  # 训练集处理占总进度的25%
        
        # 处理验证集 - 不应用增强，只复制
        self._process_dataset_images(
            preprocessed_folder, val_folder, val_images,
            augmentation_level, False, 75, 100, params)  # 验证集处理占总进度的25%
            
        self.status_updated.emit('数据集创建完成')

    def _process_dataset_images(self,
                               source_dir: str,
                               target_dir: str,
                               images: List[str],
                               augmentation_level: str,
                               apply_augmentation: bool,
                               progress_start: int,
                               progress_end: int,
                               params: Dict) -> None:
        """处理数据集图片，包括复制和数据增强
        
        参数:
            source_dir: 源文件夹路径
            target_dir: 目标文件夹路径
            images: 要处理的图片文件名列表
            augmentation_level: 增强级别 ('基础', '中等', '强化')
            apply_augmentation: 是否应用增强 (True: 训练集, False: 验证集)
            progress_start: 进度条起始百分比
            progress_end: 进度条结束百分比
            params: 包含预处理参数的字典
        """
        total_images = len(images)
        
        # 获取增强强度参数
        aug_intensity = params.get('augmentation_intensity', 0.5)
        self.status_updated.emit(f"增强强度: {aug_intensity}")
        
        # 根据增强强度调整各增强参数
        # 注意：有些效果需要较小强度(如亮度)，有些需要较大强度(如噪声)
        scale_limit = aug_intensity * 0.4  # 缩放范围: 0.04 to 0.4
        brightness_limit = aug_intensity * 0.6  # 亮度范围: 0.06 to 0.6
        contrast_limit = aug_intensity * 0.6  # 对比度范围: 0.06 to 0.6
        noise_limit_min = 5.0 + aug_intensity * 15.0  # 噪声最小值范围: 5.0 to 20.0
        noise_limit_max = 20.0 + aug_intensity * 80.0  # 噪声最大值范围: 20.0 to 100.0
        blur_limit = int(2 + aug_intensity * 5)  # 模糊范围: 2 to 7
        hue_shift_limit = int(10 + aug_intensity * 30)  # 色相范围: 10 to 40
        sat_val_limit = int(15 + aug_intensity * 35)  # 饱和度/明度范围: 15 to 50
        
        # 调试输出所有参数和选项
        self.status_updated.emit(f"调试：总增强参数 {len(params)} 个")
        self.status_updated.emit(f"调试：亮度增强开关={params.get('brightness')}, 类型={type(params.get('brightness'))}")
        self.status_updated.emit(f"调试：对比度增强开关={params.get('contrast')}, 类型={type(params.get('contrast'))}")
        
        # 验证参数类型
        brightness_arg = params.get('brightness')
        contrast_arg = params.get('contrast')
        
        # 确保参数为布尔值
        if isinstance(brightness_arg, str):
            if brightness_arg.lower() in ('true', 'yes', '1'):
                params['brightness'] = True
            elif brightness_arg.lower() in ('false', 'no', '0'):
                params['brightness'] = False
        
        if isinstance(contrast_arg, str):
            if contrast_arg.lower() in ('true', 'yes', '1'):
                params['contrast'] = True
            elif contrast_arg.lower() in ('false', 'no', '0'):
                params['contrast'] = False
        
        # 为调试添加状态输出
        aug_mode = params.get('augmentation_mode', 'combined')
        self.status_updated.emit(f"处理数据集图片，增强模式：{aug_mode}, 是否应用增强：{apply_augmentation}")
        if aug_mode == 'separate':
            self.status_updated.emit(f"启用的增强方法：")
            self.status_updated.emit(f"- 水平翻转: {params.get('flip_horizontal', False)}")
            self.status_updated.emit(f"- 垂直翻转: {params.get('flip_vertical', False)}")
            self.status_updated.emit(f"- 旋转: {params.get('rotate', False)}")
            self.status_updated.emit(f"- 裁剪: {params.get('random_crop', False)}")
            self.status_updated.emit(f"- 缩放: {params.get('random_scale', False)} (强度: {scale_limit:.2f})")
            self.status_updated.emit(f"- 亮度: {params.get('brightness', False)} (强度: {brightness_limit:.2f})")
            self.status_updated.emit(f"- 对比度: {params.get('contrast', False)} (强度: {contrast_limit:.2f})")
            self.status_updated.emit(f"- 噪声: {params.get('noise', False)} (强度: {noise_limit_min:.1f}-{noise_limit_max:.1f})")
            self.status_updated.emit(f"- 模糊: {params.get('blur', False)} (强度: {blur_limit})")
            self.status_updated.emit(f"- 色相: {params.get('hue', False)} (强度: 色相{hue_shift_limit}, 饱和度/明度{sat_val_limit})")
        
        # 记录处理的图片数量
        processed_count = 0
        
        for i, image_name in enumerate(images):
            # 检查是否需要停止处理
            if self._stop_preprocessing:
                self.status_updated.emit('数据集处理已停止')
                return
                
            try:
                # 复制原始图片（如果不是已经存在于目标文件夹中）
                src_path = os.path.join(source_dir, image_name)
                dst_path = os.path.join(target_dir, image_name)
                
                # 只有当目标文件不存在时才复制
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    processed_count += 1
                
                # 对训练集进行数据增强（只有apply_augmentation为True时）
                # 注意：验证集 apply_augmentation 为 False，不应该应用增强
                if apply_augmentation:
                    # 如果是独立模式，我们已经在_preprocess_class_images中应用了所有增强
                    # 所以这里不需要再做增强
                    if aug_mode == 'separate':
                        continue
                    
                    # 对于组合模式，我们应用额外的增强
                    try:
                        image = np.array(Image.open(src_path).convert('RGB'))
                    except Exception as e:
                        self.status_updated.emit(f"打开图片 {image_name} 失败: {str(e)}")
                        continue
                    
                    # 使用组合增强处理
                    aug = self._get_augmentation_with_intensity(augmentation_level, aug_intensity)
                    # 为每张图片生成2个增强版本
                    for j in range(2):
                        if self._stop_preprocessing:
                            return
                        try:
                            augmented = aug(image=image)['image']
                            aug_name = f'{os.path.splitext(image_name)[0]}_aug_{j}{os.path.splitext(image_name)[1]}'
                            aug_path = os.path.join(target_dir, aug_name)
                            # 只有当增强图片不存在时才保存
                            if not os.path.exists(aug_path):
                                Image.fromarray(augmented).save(aug_path)
                                processed_count += 1
                            self.status_updated.emit(f"对 {image_name} 应用了组合增强 #{j+1}")
                        except Exception as e:
                            self.status_updated.emit(f"组合增强 #{j+1} 失败: {str(e)}")
                
                # 更新进度
                progress = int(progress_start + ((i + 1) / total_images) * (progress_end - progress_start))
                self.progress_updated.emit(progress)
                
            except Exception as e:
                import traceback
                self.status_updated.emit(f'处理数据集图片 {image_name} 时出错: {str(e)}\n{traceback.format_exc()}')
        
        self.status_updated.emit(f'完成数据集处理，共处理 {processed_count} 张图片')

    def _get_augmentation_with_intensity(self, level: str, intensity: float) -> A.Compose:
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