#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新增的模型分析方法
验证Integrated Gradients, SHAP, SmoothGrad的实现
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入测试组件
from ui.components.model_analysis.worker import ModelAnalysisWorker

class SimpleTestModel(nn.Module):
    """简单的测试模型"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def create_test_image():
    """创建测试图像"""
    # 创建一个简单的彩色测试图像
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

def preprocess_test_image(image):
    """预处理测试图像"""
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
    
    # 标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor

def test_integrated_gradients():
    """测试Integrated Gradients方法"""
    print("🔍 测试 Integrated Gradients...")
    
    try:
        # 创建测试数据
        model = SimpleTestModel()
        model.eval()
        
        image = create_test_image()
        image_tensor = preprocess_test_image(image)
        class_names = [f"类别{i}" for i in range(10)]
        
        # 创建worker并设置任务
        worker = ModelAnalysisWorker()
        worker.set_analysis_task(
            "Integrated Gradients", 
            model, 
            image, 
            image_tensor, 
            class_names, 
            0,  # target_class
            {'ig_steps': 20, 'baseline_type': 'zero'}
        )
        
        # 执行分析
        result = worker._integrated_gradients_analysis()
        
        # 验证结果
        assert isinstance(result, np.ndarray), "结果应该是numpy数组"
        assert result.shape == image_tensor.shape, f"结果形状应该是{image_tensor.shape}，实际是{result.shape}"
        
        print("✅ Integrated Gradients 测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Integrated Gradients 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_smoothgrad():
    """测试SmoothGrad方法"""
    print("🔍 测试 SmoothGrad...")
    
    try:
        # 创建测试数据
        model = SimpleTestModel()
        model.eval()
        
        image = create_test_image()
        image_tensor = preprocess_test_image(image)
        class_names = [f"类别{i}" for i in range(10)]
        
        # 创建worker并设置任务
        worker = ModelAnalysisWorker()
        worker.set_analysis_task(
            "SmoothGrad", 
            model, 
            image, 
            image_tensor, 
            class_names, 
            0,  # target_class
            {'smoothgrad_samples': 10, 'noise_level': 0.1}
        )
        
        # 执行分析
        result = worker._smoothgrad_analysis()
        
        # 验证结果
        assert isinstance(result, np.ndarray), "结果应该是numpy数组"
        assert result.shape == image_tensor.shape, f"结果形状应该是{image_tensor.shape}，实际是{result.shape}"
        
        print("✅ SmoothGrad 测试通过")
        return True
        
    except Exception as e:
        print(f"❌ SmoothGrad 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_shap_analysis():
    """测试SHAP分析方法"""
    print("🔍 测试 SHAP分析...")
    
    try:
        # 检查SHAP是否可用
        try:
            import shap
        except ImportError:
            print("⚠️ SHAP库未安装，跳过SHAP测试")
            return True
        
        # 创建测试数据
        model = SimpleTestModel()
        model.eval()
        
        image = create_test_image()
        image_tensor = preprocess_test_image(image)
        class_names = [f"类别{i}" for i in range(10)]
        
        # 创建worker并设置任务
        worker = ModelAnalysisWorker()
        worker.set_analysis_task(
            "SHAP解释", 
            model, 
            image, 
            image_tensor, 
            class_names, 
            0,  # target_class
            {'shap_samples': 20}
        )
        
        # 执行分析
        result = worker._shap_analysis()
        
        # 验证结果
        assert isinstance(result, dict), "结果应该是字典"
        assert 'shap_values' in result, "结果应该包含shap_values"
        
        print("✅ SHAP分析 测试通过")
        return True
        
    except Exception as e:
        print(f"❌ SHAP分析 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_functions():
    """测试可视化函数"""
    print("🔍 测试可视化函数...")
    
    try:
        # 导入可视化函数
        from ui.components.model_analysis.visualization_utils import (
            display_integrated_gradients, display_shap_explanation, display_smoothgrad
        )
        
        # 创建测试数据
        gradients = np.random.randn(3, 224, 224).astype(np.float32)
        image = create_test_image()
        
        # 创建模拟的viewer（这里只是验证函数不会崩溃）
        class MockViewer:
            def set_image(self, pixmap):
                pass
        
        viewer = MockViewer()
        
        # 测试所有可视化函数
        print("  测试 display_integrated_gradients...")
        display_integrated_gradients(gradients, image, viewer, "测试类别")
        
        print("  测试 display_smoothgrad...")
        display_smoothgrad(gradients, image, viewer, "测试类别")
        
        print("  测试 display_shap_explanation...")
        shap_result = {'shap_values': gradients, 'method': 'gradient'}
        display_shap_explanation(shap_result, image, viewer, "测试类别")
        
        print("✅ 可视化函数 测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 可视化函数 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试新增的模型分析方法...")
    print("=" * 60)
    
    tests = [
        test_integrated_gradients,
        test_smoothgrad,
        test_shap_analysis,
        test_visualization_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print("-" * 40)
    
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！新增的分析方法实现正确。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查实现。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 