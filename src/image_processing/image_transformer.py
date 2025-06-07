"""
图像变换器
专门处理图像的基本变换，如大小调整、亮度、对比度调整等
"""

from PIL import Image, ImageEnhance
from typing import Optional
import os


class ImageTransformer:
    """图像变换器，处理基本的图像变换操作"""

    @staticmethod
    def process_single_image(source_path: str, 
                           target_path: str, 
                           width: int, 
                           height: int, 
                           brightness: int = 0, 
                           contrast: int = 0) -> None:
        """
        处理单个图片
        
        参数:
            source_path: 源图片路径
            target_path: 目标图片路径
            width: 目标宽度
            height: 目标高度
            brightness: 亮度调整值 (-50 到 50)
            contrast: 对比度调整值 (-50 到 50)
        """
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
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # 保存图片
        img.save(target_path)

    @staticmethod
    def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
        """调整图片大小"""
        return image.resize((width, height), Image.LANCZOS)

    @staticmethod
    def adjust_brightness(image: Image.Image, brightness_value: int) -> Image.Image:
        """调整图片亮度"""
        if brightness_value == 0:
            return image
        factor = 1.0 + (brightness_value / 50.0)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_contrast(image: Image.Image, contrast_value: int) -> Image.Image:
        """调整图片对比度"""
        if contrast_value == 0:
            return image
        factor = 1.0 + (contrast_value / 50.0)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def convert_to_rgb(image_path: str) -> Image.Image:
        """将图片转换为RGB格式"""
        return Image.open(image_path).convert('RGB') 