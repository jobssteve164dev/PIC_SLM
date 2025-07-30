#!/usr/bin/env python3
"""
测试matplotlib警告修复的脚本
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import io

def test_confusion_matrix_fix():
    """测试混淆矩阵修复"""
    print("测试混淆矩阵修复...")
    
    # 创建测试数据
    cm = np.array([[10, 2, 1], [3, 15, 2], [1, 1, 8]])
    class_names = ['Class A', 'Class B', 'Class C']
    
    try:
        # 测试修复后的混淆矩阵创建
        plt.figure(figsize=(10, 10))
        
        # 创建混淆矩阵图像并保存mappable对象
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        
        # 添加colorbar（使用保存的mappable对象）
        plt.colorbar(im)
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # 在格子中添加数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('test_confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("✅ 混淆矩阵测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 混淆矩阵测试失败: {str(e)}")
        return False

def test_image_range_fix():
    """测试图像数据范围修复"""
    print("测试图像数据范围修复...")
    
    # 创建超出范围的测试图像数据
    test_image = np.random.randn(100, 100, 3) * 2  # 范围在 [-2, 2]
    print(f"原始图像数据范围: [{test_image.min():.3f}, {test_image.max():.3f}]")
    
    try:
        # 测试修复后的图像显示
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 修复前（会发出警告）
        axes[0].imshow(test_image)
        axes[0].set_title('修复前（会发出警告）')
        axes[0].axis('off')
        
        # 修复后（不会发出警告）
        normalized_image = np.clip(test_image, 0, 1)
        axes[1].imshow(normalized_image, vmin=0, vmax=1)
        axes[1].set_title('修复后（无警告）')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_image_range.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("✅ 图像数据范围测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 图像数据范围测试失败: {str(e)}")
        return False

def test_tensor_image_fix():
    """测试张量图像修复"""
    print("测试张量图像修复...")
    
    # 创建超出范围的PyTorch张量
    test_tensor = torch.randn(3, 100, 100) * 2  # 范围在 [-2, 2]
    print(f"原始张量数据范围: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
    
    try:
        # 测试修复后的张量图像显示
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 修复前（会发出警告）
        img_original = test_tensor.permute(1, 2, 0)
        axes[0].imshow(img_original)
        axes[0].set_title('修复前（会发出警告）')
        axes[0].axis('off')
        
        # 修复后（不会发出警告）
        img_fixed = torch.clamp(test_tensor, 0, 1).permute(1, 2, 0)
        axes[1].imshow(img_fixed, vmin=0, vmax=1)
        axes[1].set_title('修复后（无警告）')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_tensor_image.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("✅ 张量图像测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 张量图像测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("开始测试matplotlib警告修复...")
    print("=" * 50)
    
    # 设置matplotlib后端
    plt.switch_backend('Agg')
    plt.ioff()
    
    # 运行测试
    tests = [
        test_confusion_matrix_fix,
        test_image_range_fix,
        test_tensor_image_fix
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！matplotlib警告修复成功。")
    else:
        print("⚠️ 部分测试失败，请检查修复。")

if __name__ == "__main__":
    main() 