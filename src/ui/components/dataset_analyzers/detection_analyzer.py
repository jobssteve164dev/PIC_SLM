"""
目标检测数据集分析器

专门处理目标检测数据集的分析任务，包括标注分布、图像质量、边界框质量和特征分布分析
"""

import os
import json
import numpy as np
from scipy import spatial
from .base_analyzer import BaseDatasetAnalyzer


class DetectionAnalyzer(BaseDatasetAnalyzer):
    """目标检测数据集分析器"""
    
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.class_names = self._load_class_info()
        
    def _load_class_info(self):
        """加载类别信息"""
        class_info_path = os.path.join(self.dataset_path, "class_info.json")
        if os.path.exists(class_info_path):
            with open(class_info_path, 'r', encoding='utf-8') as f:
                class_info = json.load(f)
                return class_info.get('classes', [])
        return []
        
    def analyze_data_distribution(self):
        """分析目标检测数据集的标注分布"""
        self.validate_dataset_structure()
        
        # 统计训练集和验证集的标注数量
        train_annotations = self._count_annotations(self.train_path)
        val_annotations = self._count_annotations(self.val_path)
        
        # 计算类别不平衡度
        train_values = np.array(list(train_annotations.values()))
        imbalance_ratio = np.max(train_values) / np.min(train_values) if np.min(train_values) > 0 else float('inf')
        
        # 计算训练集与验证集分布相似度（使用余弦相似度）
        all_classes = set(train_annotations.keys()) | set(val_annotations.keys())
        train_vector = [train_annotations.get(cls, 0) for cls in all_classes]
        val_vector = [val_annotations.get(cls, 0) for cls in all_classes]
        
        if sum(train_vector) > 0 and sum(val_vector) > 0:
            similarity = 1 - spatial.distance.cosine(train_vector, val_vector)
        else:
            similarity = 0
            
        return {
            'train_annotations': train_annotations,
            'val_annotations': val_annotations,
            'total_classes': len(all_classes),
            'train_total': sum(train_annotations.values()),
            'val_total': sum(val_annotations.values()),
            'imbalance_ratio': imbalance_ratio,
            'similarity': similarity
        }
        
    def analyze_image_quality(self):
        """分析目标检测数据集的图像质量"""
        self.validate_dataset_structure()
        
        images_path = os.path.join(self.train_path, "images")
        if not os.path.exists(images_path):
            raise ValueError("未找到图像文件夹")
            
        image_sizes = []
        image_qualities = []
        brightness_values = []
        contrast_values = []
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(images_path) if self.is_image_file(f)]
        total_images = len(image_files)
        
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(images_path, img_name)
            info = self.get_image_basic_info(img_path)
            
            if info:
                image_sizes.append(info['size'])
                image_qualities.append(info['quality'])
                brightness_values.append(info['brightness'])
                contrast_values.append(info['contrast'])
                
            # 更新进度
            if total_images > 0:
                progress = int((i + 1) / total_images * 100)
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
        """分析目标检测数据集的标注质量"""
        self.validate_dataset_structure()
        
        labels_path = os.path.join(self.train_path, "labels")
        if not os.path.exists(labels_path):
            raise ValueError("未找到标注文件夹")
            
        box_sizes = []
        box_counts = []
        
        # 获取所有标注文件
        label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
        total_labels = len(label_files)
        
        for i, label_file in enumerate(label_files):
            box_count = 0
            with open(os.path.join(labels_path, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # 确保有足够的坐标信息
                        # 计算边界框大小
                        try:
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            box_size = width * height
                            box_sizes.append(box_size)
                            box_count += 1
                        except ValueError:
                            continue
                            
            box_counts.append(box_count)
            
            # 更新进度
            if total_labels > 0:
                progress = int((i + 1) / total_labels * 100)
                self.progress_updated.emit(progress)
                
        # 计算重要统计量
        empty_images = box_counts.count(0)
        empty_ratio = empty_images / len(box_counts) if box_counts else 0
        small_boxes = sum(1 for size in box_sizes if size < 0.01)  # 小目标定义为相对面积<1%
        small_box_ratio = small_boxes / len(box_sizes) if box_sizes else 0
        
        return {
            'box_sizes': box_sizes,
            'box_counts': box_counts,
            'avg_boxes_per_image': np.mean(box_counts) if box_counts else 0,
            'empty_ratio': empty_ratio,
            'small_box_ratio': small_box_ratio,
            'avg_box_size': np.mean(box_sizes) if box_sizes else 0,
            'box_size_variance': np.var(box_sizes) if box_sizes else 0
        }
        
    def analyze_feature_distribution(self):
        """分析目标检测数据集的特征分布"""
        self.validate_dataset_structure()
        
        images_path = os.path.join(self.train_path, "images")
        if not os.path.exists(images_path):
            raise ValueError("未找到图像文件夹")
            
        brightness_values = []
        contrast_values = []
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(images_path) if self.is_image_file(f)]
        total_images = len(image_files)
        
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(images_path, img_name)
            info = self.get_image_basic_info(img_path)
            
            if info:
                brightness_values.append(info['brightness'])
                contrast_values.append(info['contrast'])
                
            # 更新进度
            if total_images > 0:
                progress = int((i + 1) / total_images * 100)
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
        
    def _count_annotations(self, dataset_path):
        """统计目标检测数据集的标注数量"""
        annotations = {}
        
        # 统计标注数量
        labels_path = os.path.join(dataset_path, "labels")
        if os.path.exists(labels_path):
            for label_file in os.listdir(labels_path):
                if label_file.endswith('.txt'):
                    with open(os.path.join(labels_path, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                try:
                                    class_id = int(parts[0])
                                    if class_id < len(self.class_names):
                                        class_name = self.class_names[class_id]
                                    else:
                                        class_name = f"class_{class_id}"
                                    annotations[class_name] = annotations.get(class_name, 0) + 1
                                except ValueError:
                                    continue
                                    
        return annotations
        
    def get_dataset_structure_info(self):
        """获取目标检测数据集的结构信息"""
        self.validate_dataset_structure()
        
        structure_info = {
            'train_images': 0,
            'train_labels': 0,
            'val_images': 0,
            'val_labels': 0,
            'class_info_exists': os.path.exists(os.path.join(self.dataset_path, "class_info.json")),
            'class_count': len(self.class_names)
        }
        
        # 统计训练集
        train_images_path = os.path.join(self.train_path, "images")
        train_labels_path = os.path.join(self.train_path, "labels")
        
        if os.path.exists(train_images_path):
            structure_info['train_images'] = len([f for f in os.listdir(train_images_path) 
                                                if self.is_image_file(f)])
                                                
        if os.path.exists(train_labels_path):
            structure_info['train_labels'] = len([f for f in os.listdir(train_labels_path) 
                                                if f.endswith('.txt')])
        
        # 统计验证集
        val_images_path = os.path.join(self.val_path, "images")
        val_labels_path = os.path.join(self.val_path, "labels")
        
        if os.path.exists(val_images_path):
            structure_info['val_images'] = len([f for f in os.listdir(val_images_path) 
                                              if self.is_image_file(f)])
                                              
        if os.path.exists(val_labels_path):
            structure_info['val_labels'] = len([f for f in os.listdir(val_labels_path) 
                                              if f.endswith('.txt')])
        
        return structure_info 