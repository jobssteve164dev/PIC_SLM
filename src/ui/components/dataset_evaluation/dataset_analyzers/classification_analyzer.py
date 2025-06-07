"""
分类数据集分析器

专门处理分类数据集的分析任务，包括类别分布、图像质量、标注质量和特征分布分析
"""

import os
import numpy as np
from collections import Counter
from .base_analyzer import BaseDatasetAnalyzer


class ClassificationAnalyzer(BaseDatasetAnalyzer):
    """分类数据集分析器"""
    
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        
    def analyze_data_distribution(self):
        """分析分类数据集的类别分布"""
        self.validate_dataset_structure()
        
        # 统计训练集类别分布
        train_classes = {}
        for class_name in os.listdir(self.train_path):
            class_path = os.path.join(self.train_path, class_name)
            if os.path.isdir(class_path):
                train_classes[class_name] = len([f for f in os.listdir(class_path) 
                                               if self.is_image_file(f)])
                
        # 统计验证集类别分布
        val_classes = {}
        for class_name in os.listdir(self.val_path):
            class_path = os.path.join(self.val_path, class_name)
            if os.path.isdir(class_path):
                val_classes[class_name] = len([f for f in os.listdir(class_path) 
                                             if self.is_image_file(f)])
        
        # 计算类别不平衡度
        imbalance_ratio = self.calculate_imbalance_ratio(train_classes)
        
        return {
            'train_classes': train_classes,
            'val_classes': val_classes,
            'total_classes': len(train_classes),
            'train_total': sum(train_classes.values()),
            'val_total': sum(val_classes.values()),
            'imbalance_ratio': imbalance_ratio
        }
        
    def analyze_image_quality(self):
        """分析分类数据集的图像质量"""
        self.validate_dataset_structure()
        
        image_sizes = []
        image_qualities = []
        brightness_values = []
        contrast_values = []
        
        # 统计总图像数量用于进度更新
        total_images = 0
        for class_name in os.listdir(self.train_path):
            class_path = os.path.join(self.train_path, class_name)
            if os.path.isdir(class_path):
                total_images += len([f for f in os.listdir(class_path) 
                                   if self.is_image_file(f)])
        
        processed = 0
        for class_name in os.listdir(self.train_path):
            class_path = os.path.join(self.train_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if self.is_image_file(img_name):
                        img_path = os.path.join(class_path, img_name)
                        info = self.get_image_basic_info(img_path)
                        
                        if info:
                            image_sizes.append(info['size'])
                            image_qualities.append(info['quality'])
                            brightness_values.append(info['brightness'])
                            contrast_values.append(info['contrast'])
                            
                        processed += 1
                        if total_images > 0:
                            progress = int(processed / total_images * 100)
                            self.progress_updated.emit(progress)
                            
        return {
            'image_sizes': image_sizes,
            'image_qualities': image_qualities,
            'brightness_values': brightness_values,
            'contrast_values': contrast_values,
            'avg_size': np.mean(image_sizes) if image_sizes else 0,
            'size_std': np.std(image_sizes) if image_sizes else 0,
            'avg_quality': np.mean(image_qualities) if image_qualities else 0,
            'quality_std': np.std(image_qualities) if image_qualities else 0,
            'avg_brightness': np.mean(brightness_values) if brightness_values else 0,
            'brightness_cv': self.calculate_variation_coefficient(brightness_values),
            'avg_contrast': np.mean(contrast_values) if contrast_values else 0,
            'contrast_cv': self.calculate_variation_coefficient(contrast_values)
        }
        
    def analyze_annotation_quality(self):
        """分析分类数据集的标注质量"""
        self.validate_dataset_structure()
        
        annotation_counts = []
        annotation_sizes = []
        class_names = []
        
        # 获取所有类别文件夹
        class_dirs = [d for d in os.listdir(self.train_path) 
                     if os.path.isdir(os.path.join(self.train_path, d))]
        
        for i, class_name in enumerate(class_dirs):
            class_path = os.path.join(self.train_path, class_name)
            
            # 统计该类别的图像数量
            image_files = [f for f in os.listdir(class_path) if self.is_image_file(f)]
            annotation_count = len(image_files)
            annotation_counts.append(annotation_count)
            class_names.append(class_name)
            
            # 计算该类别图像文件的平均大小
            total_size = 0
            for img_name in image_files:
                img_path = os.path.join(class_path, img_name)
                try:
                    size = os.path.getsize(img_path)
                    total_size += size
                    annotation_sizes.append(size)
                except Exception as e:
                    print(f"获取文件大小失败 {img_path}: {str(e)}")
                    
            # 更新进度
            progress = int((i + 1) / len(class_dirs) * 100)
            self.progress_updated.emit(progress)
            
        # 计算标注一致性指标（使用变异系数）
        cv = self.calculate_variation_coefficient(annotation_counts)
        
        return {
            'class_names': class_names,
            'annotation_counts': annotation_counts,
            'annotation_sizes': annotation_sizes,
            'avg_annotations': np.mean(annotation_counts) if annotation_counts else 0,
            'max_annotations': max(annotation_counts) if annotation_counts else 0,
            'min_annotations': min(annotation_counts) if annotation_counts else 0,
            'avg_file_size': np.mean(annotation_sizes) if annotation_sizes else 0,
            'consistency_cv': cv
        }
        
    def analyze_feature_distribution(self):
        """分析分类数据集的特征分布"""
        self.validate_dataset_structure()
        
        brightness_values = []
        contrast_values = []
        
        # 统计总图像数量
        total_images = 0
        for class_name in os.listdir(self.train_path):
            class_path = os.path.join(self.train_path, class_name)
            if os.path.isdir(class_path):
                total_images += len([f for f in os.listdir(class_path) 
                                   if self.is_image_file(f)])
        
        processed = 0
        for class_name in os.listdir(self.train_path):
            class_path = os.path.join(self.train_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if self.is_image_file(img_name):
                        img_path = os.path.join(class_path, img_name)
                        info = self.get_image_basic_info(img_path)
                        
                        if info:
                            brightness_values.append(info['brightness'])
                            contrast_values.append(info['contrast'])
                            
                        processed += 1
                        if total_images > 0:
                            progress = int(processed / total_images * 100)
                            self.progress_updated.emit(progress)
                            
        # 计算变异系数
        brightness_cv = self.calculate_variation_coefficient(brightness_values)
        contrast_cv = self.calculate_variation_coefficient(contrast_values)
        
        return {
            'brightness_values': brightness_values,
            'contrast_values': contrast_values,
            'avg_brightness': np.mean(brightness_values) if brightness_values else 0,
            'brightness_cv': brightness_cv,
            'avg_contrast': np.mean(contrast_values) if contrast_values else 0,
            'contrast_cv': contrast_cv
        }
        
    def get_class_structure_info(self):
        """获取分类数据集的结构信息"""
        self.validate_dataset_structure()
        
        train_structure = {}
        val_structure = {}
        
        # 分析训练集结构
        for class_name in os.listdir(self.train_path):
            class_path = os.path.join(self.train_path, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) if self.is_image_file(f)]
                train_structure[class_name] = {
                    'count': len(image_files),
                    'files': image_files[:5]  # 只保存前5个文件名作为示例
                }
                
        # 分析验证集结构  
        for class_name in os.listdir(self.val_path):
            class_path = os.path.join(self.val_path, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) if self.is_image_file(f)]
                val_structure[class_name] = {
                    'count': len(image_files),
                    'files': image_files[:5]  # 只保存前5个文件名作为示例
                }
                
        return {
            'train_structure': train_structure,
            'val_structure': val_structure,
            'common_classes': set(train_structure.keys()) & set(val_structure.keys()),
            'train_only': set(train_structure.keys()) - set(val_structure.keys()),
            'val_only': set(val_structure.keys()) - set(train_structure.keys())
        } 