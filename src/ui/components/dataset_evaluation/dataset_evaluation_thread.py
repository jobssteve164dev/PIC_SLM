"""
数据集评估独立线程

为数据集评估功能提供独立线程支持，避免UI冻结，提供更好的用户体验
"""

import os
import sys
import time
import traceback
from PyQt5.QtCore import QThread, pyqtSignal

from .dataset_analyzers import ClassificationAnalyzer, DetectionAnalyzer


class DatasetEvaluationThread(QThread):
    """数据集评估线程"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    evaluation_finished = pyqtSignal(dict)
    evaluation_error = pyqtSignal(str)
    evaluation_stopped = pyqtSignal()
    
    def __init__(self, dataset_path, eval_type, metric_type):
        super().__init__()
        self.dataset_path = dataset_path
        self.eval_type = eval_type
        self.metric_type = metric_type
        self.stop_evaluation = False
        self.current_analyzer = None
        
    def run(self):
        """执行数据集评估"""
        try:
            self.stop_evaluation = False
            
            # 创建对应的分析器
            if self.eval_type == "分类数据集":
                self.current_analyzer = ClassificationAnalyzer(self.dataset_path)
                self._evaluate_classification_dataset()
            else:
                self.current_analyzer = DetectionAnalyzer(self.dataset_path)
                self._evaluate_detection_dataset()
                
        except Exception as e:
            error_msg = f"评估过程中出错: {str(e)}\n{traceback.format_exc()}"
            print(f"数据集评估线程错误: {error_msg}")
            self.evaluation_error.emit(error_msg)
            
    def stop(self):
        """停止评估"""
        self.stop_evaluation = True
        self.status_updated.emit("正在停止评估...")
        
    def _evaluate_classification_dataset(self):
        """评估分类数据集"""
        # 连接分析器的进度信号
        self.current_analyzer.progress_updated.connect(self.progress_updated)
        self.current_analyzer.status_updated.connect(self.status_updated)
        
        try:
            result = None
            
            if self.metric_type == "数据分布分析":
                result = self._analyze_classification_distribution()
            elif self.metric_type == "图像质量分析":
                result = self._analyze_classification_image_quality()
            elif self.metric_type == "标注质量分析":
                result = self._analyze_classification_annotation_quality()
            elif self.metric_type == "特征分布分析":
                result = self._analyze_classification_feature_distribution()
            elif self.metric_type == "生成类别权重":
                result = self._generate_classification_weights()
            
            if not self.stop_evaluation and result is not None:
                # 添加评估类型和指标类型信息
                result['eval_type'] = self.eval_type
                result['metric_type'] = self.metric_type
                result['dataset_path'] = self.dataset_path
                
                self.evaluation_finished.emit(result)
            elif self.stop_evaluation:
                self.evaluation_stopped.emit()
                
        finally:
            # 断开信号连接
            if self.current_analyzer:
                self.current_analyzer.progress_updated.disconnect()
                self.current_analyzer.status_updated.disconnect()
                
    def _evaluate_detection_dataset(self):
        """评估目标检测数据集"""
        # 连接分析器的进度信号
        self.current_analyzer.progress_updated.connect(self.progress_updated)
        self.current_analyzer.status_updated.connect(self.status_updated)
        
        try:
            result = None
            
            if self.metric_type == "数据分布分析":
                result = self._analyze_detection_distribution()
            elif self.metric_type == "图像质量分析":
                result = self._analyze_detection_image_quality()
            elif self.metric_type == "标注质量分析":
                result = self._analyze_detection_annotation_quality()
            elif self.metric_type == "特征分布分析":
                result = self._analyze_detection_feature_distribution()
            
            if not self.stop_evaluation and result is not None:
                # 添加评估类型和指标类型信息
                result['eval_type'] = self.eval_type
                result['metric_type'] = self.metric_type
                result['dataset_path'] = self.dataset_path
                
                self.evaluation_finished.emit(result)
            elif self.stop_evaluation:
                self.evaluation_stopped.emit()
                
        finally:
            # 断开信号连接
            if self.current_analyzer:
                self.current_analyzer.progress_updated.disconnect()
                self.current_analyzer.status_updated.disconnect()
    
    def _analyze_classification_distribution(self):
        """分析分类数据集分布"""
        if self.stop_evaluation:
            return None
            
        self.status_updated.emit("正在分析分类数据集分布...")
        result = self.current_analyzer.analyze_data_distribution()
        
        if self.stop_evaluation:
            return None
            
        return {
            'analysis_type': 'distribution',
            'train_classes': result['train_classes'],
            'val_classes': result['val_classes'],
            'total_classes': result['total_classes'],
            'train_total': result['train_total'],
            'val_total': result['val_total'],
            'imbalance_ratio': result['imbalance_ratio']
        }
        
    def _analyze_classification_image_quality(self):
        """分析分类数据集图像质量"""
        if self.stop_evaluation:
            return None
            
        self.status_updated.emit("正在分析图像质量...")
        result = self.current_analyzer.analyze_image_quality()
        
        if self.stop_evaluation:
            return None
            
        return {
            'analysis_type': 'image_quality',
            'image_sizes': result['image_sizes'],
            'image_qualities': result['image_qualities'],
            'brightness_values': result['brightness_values'],
            'contrast_values': result['contrast_values'],
            'avg_size': result['avg_size'],
            'size_std': result['size_std'],
            'avg_quality': result['avg_quality'],
            'quality_std': result['quality_std'],
            'avg_brightness': result['avg_brightness'],
            'brightness_cv': result['brightness_cv'],
            'avg_contrast': result['avg_contrast'],
            'contrast_cv': result['contrast_cv']
        }
        
    def _analyze_classification_annotation_quality(self):
        """分析分类数据集标注质量"""
        if self.stop_evaluation:
            return None
            
        self.status_updated.emit("正在分析标注质量...")
        result = self.current_analyzer.analyze_annotation_quality()
        
        if self.stop_evaluation:
            return None
            
        return {
            'analysis_type': 'annotation_quality',
            'class_names': result['class_names'],
            'annotation_counts': result['annotation_counts'],
            'annotation_sizes': result['annotation_sizes'],
            'avg_annotations': result['avg_annotations'],
            'max_annotations': result['max_annotations'],
            'min_annotations': result['min_annotations'],
            'avg_file_size': result['avg_file_size'],
            'consistency_cv': result['consistency_cv']
        }
        
    def _analyze_classification_feature_distribution(self):
        """分析分类数据集特征分布"""
        if self.stop_evaluation:
            return None
            
        self.status_updated.emit("正在分析特征分布...")
        result = self.current_analyzer.analyze_feature_distribution()
        
        if self.stop_evaluation:
            return None
            
        return {
            'analysis_type': 'feature_distribution',
            'brightness_values': result['brightness_values'],
            'contrast_values': result['contrast_values'],
            'avg_brightness': result['avg_brightness'],
            'brightness_cv': result['brightness_cv'],
            'avg_contrast': result['avg_contrast'],
            'contrast_cv': result['contrast_cv']
        }
        
    def _generate_classification_weights(self):
        """生成分类数据集权重"""
        if self.stop_evaluation:
            return None
            
        self.status_updated.emit("正在生成类别权重...")
        
        # 首先获取类别分布
        distribution_result = self.current_analyzer.analyze_data_distribution()
        
        if self.stop_evaluation:
            return None
        
        # 导入权重生成器
        from .weight_generator import WeightGenerator
        weight_generator = WeightGenerator()
        
        # 生成权重
        train_classes = distribution_result['train_classes']
        weight_result = weight_generator.generate_weights(train_classes)
        
        if self.stop_evaluation:
            return None
        
        # 合并结果
        return {
            'analysis_type': 'weight_generation',
            'train_classes': train_classes,
            'val_classes': distribution_result['val_classes'],
            'total_classes': distribution_result['total_classes'],
            'train_total': distribution_result['train_total'],
            'val_total': distribution_result['val_total'],
            'imbalance_ratio': distribution_result['imbalance_ratio'],
            'class_weights': weight_result['class_weights'],
            'weight_strategies': weight_result['weight_strategies'],
            'recommended_strategy': weight_result['recommended_strategy'],
            'imbalance_analysis': weight_result['imbalance_analysis']
        }
        
    def _analyze_detection_distribution(self):
        """分析检测数据集分布"""
        if self.stop_evaluation:
            return None
            
        self.status_updated.emit("正在分析检测数据集分布...")
        result = self.current_analyzer.analyze_data_distribution()
        
        if self.stop_evaluation:
            return None
            
        return {
            'analysis_type': 'distribution',
            'class_distribution': result.get('class_distribution', {}),
            'bbox_count': result.get('bbox_count', 0),
            'image_count': result.get('image_count', 0),
            'avg_objects_per_image': result.get('avg_objects_per_image', 0),
            'bbox_size_distribution': result.get('bbox_size_distribution', [])
        }
        
    def _analyze_detection_image_quality(self):
        """分析检测数据集图像质量"""
        if self.stop_evaluation:
            return None
            
        self.status_updated.emit("正在分析检测数据集图像质量...")
        result = self.current_analyzer.analyze_image_quality()
        
        if self.stop_evaluation:
            return None
            
        return {
            'analysis_type': 'image_quality',
            'image_sizes': result.get('image_sizes', []),
            'image_qualities': result.get('image_qualities', []),
            'brightness_values': result.get('brightness_values', []),
            'contrast_values': result.get('contrast_values', []),
            'avg_size': result.get('avg_size', 0),
            'avg_quality': result.get('avg_quality', 0),
            'avg_brightness': result.get('avg_brightness', 0),
            'avg_contrast': result.get('avg_contrast', 0)
        }
        
    def _analyze_detection_annotation_quality(self):
        """分析检测数据集标注质量"""
        if self.stop_evaluation:
            return None
            
        self.status_updated.emit("正在分析检测数据集标注质量...")
        result = self.current_analyzer.analyze_annotation_quality()
        
        if self.stop_evaluation:
            return None
            
        return {
            'analysis_type': 'annotation_quality',
            'annotation_files': result.get('annotation_files', []),
            'valid_annotations': result.get('valid_annotations', 0),
            'invalid_annotations': result.get('invalid_annotations', 0),
            'annotation_consistency': result.get('annotation_consistency', 0),
            'bbox_quality_scores': result.get('bbox_quality_scores', [])
        }
        
    def _analyze_detection_feature_distribution(self):
        """分析检测数据集特征分布"""
        if self.stop_evaluation:
            return None
            
        self.status_updated.emit("正在分析检测数据集特征分布...")
        result = self.current_analyzer.analyze_feature_distribution()
        
        if self.stop_evaluation:
            return None
            
        return {
            'analysis_type': 'feature_distribution',
            'bbox_aspect_ratios': result.get('bbox_aspect_ratios', []),
            'bbox_areas': result.get('bbox_areas', []),
            'object_density': result.get('object_density', []),
            'spatial_distribution': result.get('spatial_distribution', {})
        } 