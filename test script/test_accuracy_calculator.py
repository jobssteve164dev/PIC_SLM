#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试准确率计算功能
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 直接导入需要的类，避免复杂的模块依赖
from PyQt5.QtCore import QThread, pyqtSignal
from collections import defaultdict
import re


class AccuracyCalculationThread:
    """简化的准确率计算类（仅用于测试）"""
    
    def __init__(self, source_folder, output_folder, class_names=None):
        self.source_folder = source_folder
        self.output_folder = output_folder
        self.class_names = class_names or []
        
    def _get_source_images(self):
        """获取源文件夹中的图片信息"""
        source_images = {}
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        for root, _, files in os.walk(self.source_folder):
            for file in files:
                if file.lower().endswith(supported_formats):
                    # 从文件名中提取类别信息
                    true_class = self._extract_class_from_filename(file)
                    if true_class:
                        file_path = os.path.join(root, file)
                        source_images[file] = {
                            'path': file_path,
                            'true_class': true_class,
                            'relative_path': os.path.relpath(file_path, self.source_folder)
                        }
        
        return source_images
    
    def _get_output_images(self):
        """获取输出文件夹中的图片信息"""
        output_images = {}
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        for root, _, files in os.walk(self.output_folder):
            for file in files:
                if file.lower().endswith(supported_formats):
                    # 从文件夹路径中提取预测类别
                    predicted_class = self._extract_class_from_path(root)
                    if predicted_class:
                        file_path = os.path.join(root, file)
                        output_images[file] = {
                            'path': file_path,
                            'predicted_class': predicted_class,
                            'relative_path': os.path.relpath(file_path, self.output_folder)
                        }
        
        return output_images
    
    def _extract_class_from_filename(self, filename):
        """从文件名中提取类别信息"""
        # 移除文件扩展名
        name_without_ext = os.path.splitext(filename)[0]
        
        # 尝试多种模式匹配
        patterns = [
            r'^([A-Za-z]+)\d*.*',  # A123.jpg -> A
            r'^([A-Za-z]+)\(\d+\).*',  # A(1).jpg -> A
            r'^([A-Za-z]+)_.*',  # A_something.jpg -> A
            r'^([A-Za-z]+)-.*',  # A-something.jpg -> A
            r'^(\w+?)[\d_\-\(\)].*',  # 更通用的模式
        ]
        
        for pattern in patterns:
            match = re.match(pattern, name_without_ext)
            if match:
                return match.group(1).upper()
        
        # 如果没有匹配到，尝试提取第一个字母序列
        match = re.match(r'^([A-Za-z]+)', name_without_ext)
        if match:
            return match.group(1).upper()
        
        return None
    
    def _extract_class_from_path(self, path):
        """从文件夹路径中提取类别信息"""
        # 获取相对于输出文件夹的路径
        rel_path = os.path.relpath(path, self.output_folder)
        
        # 分割路径，取最后一个文件夹名作为类别
        path_parts = rel_path.split(os.sep)
        if path_parts and path_parts[-1] != '.':
            return path_parts[-1].upper()
        
        return None
    
    def _calculate_accuracy(self, source_images, output_images):
        """计算准确率"""
        # 统计结果
        total_images = len(source_images)
        matched_images = 0
        class_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'predicted_as': defaultdict(int)})
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        # 遍历源图片，查找对应的输出图片
        for filename, source_info in source_images.items():
            true_class = source_info['true_class']
            class_stats[true_class]['total'] += 1
            
            if filename in output_images:
                predicted_class = output_images[filename]['predicted_class']
                class_stats[true_class]['predicted_as'][predicted_class] += 1
                confusion_matrix[true_class][predicted_class] += 1
                
                if true_class == predicted_class:
                    matched_images += 1
                    class_stats[true_class]['correct'] += 1
            else:
                # 图片未被分类（可能因为置信度太低）
                class_stats[true_class]['predicted_as']['未分类'] += 1
                confusion_matrix[true_class]['未分类'] += 1
        
        # 计算总体准确率
        overall_accuracy = (matched_images / total_images * 100) if total_images > 0 else 0
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for class_name, stats in class_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                class_accuracies[class_name] = accuracy
        
        # 构建结果
        results = {
            'total_images': total_images,
            'matched_images': matched_images,
            'overall_accuracy': overall_accuracy,
            'class_stats': dict(class_stats),
            'class_accuracies': class_accuracies,
            'confusion_matrix': dict(confusion_matrix),
            'unprocessed_images': total_images - len(output_images)
        }
        
        return results


def create_test_data():
    """创建测试数据"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    source_dir = os.path.join(temp_dir, "source")
    output_dir = os.path.join(temp_dir, "output")
    
    os.makedirs(source_dir)
    os.makedirs(output_dir)
    
    # 创建源文件夹结构和文件（测试多种格式）
    source_files = [
        # 测试数字_类别_数字格式
        "01_spur_07.jpg", "01_spur_08.jpg", "01_spur_09.jpg", "01_spur_10.jpg", "01_spur_11.jpg",
        # 测试类别_数字格式
        "crack_01.jpg", "crack_02.jpg", "crack_03.jpg", "crack_04.jpg", "crack_05.jpg",
        # 测试传统A(1)格式
        "C(1).jpg", "C(2).jpg", "C(3).jpg", "C(4).jpg", "C(5).jpg"
    ]
    
    for filename in source_files:
        file_path = os.path.join(source_dir, filename)
        with open(file_path, 'w') as f:
            f.write("fake image content")
    
    # 创建输出文件夹结构
    os.makedirs(os.path.join(output_dir, "SPUR"))
    os.makedirs(os.path.join(output_dir, "CRACK"))
    os.makedirs(os.path.join(output_dir, "C"))
    
    # 模拟预测结果 - 一些正确，一些错误
    prediction_results = {
        # SPUR类别 - 4个正确，1个错误
        "01_spur_07.jpg": "SPUR",
        "01_spur_08.jpg": "SPUR", 
        "01_spur_09.jpg": "SPUR",
        "01_spur_10.jpg": "CRACK",  # 错误预测
        "01_spur_11.jpg": "SPUR",
        
        # CRACK类别 - 3个正确，2个错误
        "crack_01.jpg": "CRACK",
        "crack_02.jpg": "C",  # 错误预测
        "crack_03.jpg": "CRACK",
        "crack_04.jpg": "SPUR",  # 错误预测
        "crack_05.jpg": "CRACK",
        
        # C类别 - 5个正确，0个错误
        "C(1).jpg": "C",
        "C(2).jpg": "C",
        "C(3).jpg": "C",
        "C(4).jpg": "C",
        "C(5).jpg": "C",
    }
    
    # 创建输出文件
    for filename, predicted_class in prediction_results.items():
        source_path = os.path.join(source_dir, filename)
        output_path = os.path.join(output_dir, predicted_class, filename)
        shutil.copy2(source_path, output_path)
    
    return temp_dir, source_dir, output_dir


def test_accuracy_calculation():
    """测试准确率计算"""
    print("=" * 60)
    print("测试准确率计算功能")
    print("=" * 60)
    
    # 创建测试数据
    temp_dir, source_dir, output_dir = create_test_data()
    
    try:
        print(f"源文件夹: {source_dir}")
        print(f"输出文件夹: {output_dir}")
        
        # 创建计算线程
        calc_thread = AccuracyCalculationThread(source_dir, output_dir)
        
        # 直接调用计算方法（不使用线程）
        source_images = calc_thread._get_source_images()
        print(f"\n源图片数量: {len(source_images)}")
        
        for filename, info in list(source_images.items())[:3]:
            print(f"  {filename} -> 真实类别: {info['true_class']}")
        
        output_images = calc_thread._get_output_images()
        print(f"\n输出图片数量: {len(output_images)}")
        
        for filename, info in list(output_images.items())[:3]:
            print(f"  {filename} -> 预测类别: {info['predicted_class']}")
        
        # 计算准确率
        results = calc_thread._calculate_accuracy(source_images, output_images)
        
        print(f"\n" + "=" * 40)
        print("计算结果:")
        print("=" * 40)
        print(f"总图片数: {results['total_images']}")
        print(f"正确预测数: {results['matched_images']}")
        print(f"总体准确率: {results['overall_accuracy']:.2f}%")
        
        print(f"\n各类别准确率:")
        for class_name, accuracy in sorted(results['class_accuracies'].items()):
            print(f"  {class_name}: {accuracy:.2f}%")
        
        print(f"\n类别详细统计:")
        for class_name, stats in sorted(results['class_stats'].items()):
            print(f"  {class_name}: 总数={stats['total']}, 正确={stats['correct']}")
            print(f"    预测分布: {dict(stats['predicted_as'])}")
        
        print(f"\n混淆矩阵:")
        confusion_matrix = results['confusion_matrix']
        all_classes = sorted(set(confusion_matrix.keys()) | 
                           set(pred for true_preds in confusion_matrix.values() 
                               for pred in true_preds.keys()))
        
        # 打印表头
        print("真实\\预测", end="")
        for pred_class in all_classes:
            print(f"\t{pred_class}", end="")
        print()
        
        # 打印矩阵
        for true_class in all_classes:
            print(f"{true_class}", end="")
            for pred_class in all_classes:
                count = confusion_matrix.get(true_class, {}).get(pred_class, 0)
                print(f"\t{count}", end="")
            print()
        
        # 验证期望结果
        print(f"\n" + "=" * 40)
        print("验证结果:")
        print("=" * 40)
        
        expected_accuracy = (4 + 3 + 5) / 15 * 100  # 12/15 = 80%
        print(f"期望总体准确率: {expected_accuracy:.2f}%")
        print(f"实际总体准确率: {results['overall_accuracy']:.2f}%")
        
        if abs(results['overall_accuracy'] - expected_accuracy) < 0.01:
            print("✅ 总体准确率计算正确")
        else:
            print("❌ 总体准确率计算错误")
        
        # 验证各类别准确率
        expected_class_accuracies = {'SPUR': 80.0, 'CRACK': 60.0, 'C': 100.0}
        for class_name, expected in expected_class_accuracies.items():
            actual = results['class_accuracies'].get(class_name, 0)
            if abs(actual - expected) < 0.01:
                print(f"✅ {class_name}类别准确率计算正确: {actual:.2f}%")
            else:
                print(f"❌ {class_name}类别准确率计算错误: 期望{expected:.2f}%, 实际{actual:.2f}%")
        
        print("\n🎉 测试完成！")
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)
        print(f"已清理临时文件: {temp_dir}")


if __name__ == "__main__":
    test_accuracy_calculation() 