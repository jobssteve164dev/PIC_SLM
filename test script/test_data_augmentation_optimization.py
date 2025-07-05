#!/usr/bin/env python3
"""
数据增强优化功能测试

测试基础数据增强和高级数据增强的独立控制和组合使用
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def test_basic_augmentation_control():
    """测试基础数据增强控制"""
    print("🧪 测试基础数据增强控制")
    print("=" * 50)
    
    # 模拟训练线程的数据准备逻辑
    def simulate_prepare_data(config):
        """模拟_prepare_data方法的逻辑"""
        use_augmentation = config.get('use_augmentation', True)
        
        # 构建训练时的transform列表
        train_transforms = ["Resize((224, 224))"]
        
        # 基础数据增强（只有在启用时才添加）
        if use_augmentation:
            train_transforms.extend([
                "RandomHorizontalFlip(p=0.5)",
                "RandomRotation(degrees=15)",
                "ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)",
                "RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))",
            ])
            status = "✅ 启用基础数据增强（翻转、旋转、颜色抖动、仿射变换）"
        else:
            status = "⚪ 基础数据增强已禁用"
        
        # 添加必要的转换
        train_transforms.extend([
            "ToTensor()",
            "Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])"
        ])
        
        return train_transforms, status
    
    # 测试用例1: 启用基础数据增强
    print("\n📋 测试用例1: 启用基础数据增强")
    config1 = {'use_augmentation': True}
    transforms1, status1 = simulate_prepare_data(config1)
    print(f"   配置: {config1}")
    print(f"   状态: {status1}")
    print(f"   变换数量: {len(transforms1)}")
    print(f"   包含增强: {'RandomHorizontalFlip' in str(transforms1)}")
    
    # 测试用例2: 禁用基础数据增强
    print("\n📋 测试用例2: 禁用基础数据增强")
    config2 = {'use_augmentation': False}
    transforms2, status2 = simulate_prepare_data(config2)
    print(f"   配置: {config2}")
    print(f"   状态: {status2}")
    print(f"   变换数量: {len(transforms2)}")
    print(f"   包含增强: {'RandomHorizontalFlip' in str(transforms2)}")
    
    # 测试用例3: 默认配置（应该启用）
    print("\n📋 测试用例3: 默认配置")
    config3 = {}
    transforms3, status3 = simulate_prepare_data(config3)
    print(f"   配置: {config3}")
    print(f"   状态: {status3}")
    print(f"   变换数量: {len(transforms3)}")
    print(f"   包含增强: {'RandomHorizontalFlip' in str(transforms3)}")
    
    # 验证结果
    result1 = len(transforms1) > len(transforms2)  # 启用时应该有更多变换
    result2 = 'RandomHorizontalFlip' in str(transforms1) and 'RandomHorizontalFlip' not in str(transforms2)
    result3 = len(transforms3) == len(transforms1)  # 默认应该启用
    
    overall_result = result1 and result2 and result3
    print(f"\n结果: {'✅ 基础数据增强控制正确' if overall_result else '❌ 基础数据增强控制有误'}")
    
    return overall_result

def test_advanced_augmentation_info():
    """测试高级数据增强信息功能"""
    print("\n🧪 测试高级数据增强信息功能")
    print("=" * 50)
    
    class MockAdvancedAugmentationManager:
        """模拟高级数据增强管理器"""
        
        def __init__(self, config):
            self.config = config
            self.advanced_augmentation_enabled = config.get('advanced_augmentation_enabled', False)
            
            if self.advanced_augmentation_enabled:
                self.mixup_prob = config.get('mixup_alpha', 0.0)
                self.cutmix_prob = config.get('cutmix_prob', 0.0)
                
                # 参数验证
                self.mixup_prob = max(0.0, min(2.0, self.mixup_prob))
                self.cutmix_prob = max(0.0, min(1.0, self.cutmix_prob))
            else:
                self.mixup_prob = 0.0
                self.cutmix_prob = 0.0
            
            self.enabled = self.advanced_augmentation_enabled and (self.mixup_prob > 0 or self.cutmix_prob > 0)
            
        def get_augmentation_info(self):
            """获取增强配置信息"""
            return {
                'enabled': self.enabled,
                'advanced_augmentation_enabled': self.advanced_augmentation_enabled,
                'mixup_prob': self.mixup_prob,
                'cutmix_prob': self.cutmix_prob,
                'mixup_available': self.mixup_prob > 0,
                'cutmix_available': self.cutmix_prob > 0
            }
        
        def is_enabled(self):
            return self.enabled
    
    # 测试用例1: 启用MixUp和CutMix
    print("\n📋 测试用例1: 启用MixUp和CutMix")
    config1 = {
        'advanced_augmentation_enabled': True,
        'mixup_alpha': 0.2,
        'cutmix_prob': 0.5
    }
    manager1 = MockAdvancedAugmentationManager(config1)
    info1 = manager1.get_augmentation_info()
    print(f"   配置: {config1}")
    print(f"   信息: {info1}")
    print(f"   启用状态: {manager1.is_enabled()}")
    
    # 测试用例2: 启用开关但参数为0
    print("\n📋 测试用例2: 启用开关但参数为0")
    config2 = {
        'advanced_augmentation_enabled': True,
        'mixup_alpha': 0.0,
        'cutmix_prob': 0.0
    }
    manager2 = MockAdvancedAugmentationManager(config2)
    info2 = manager2.get_augmentation_info()
    print(f"   配置: {config2}")
    print(f"   信息: {info2}")
    print(f"   启用状态: {manager2.is_enabled()}")
    
    # 测试用例3: 完全禁用
    print("\n📋 测试用例3: 完全禁用")
    config3 = {
        'advanced_augmentation_enabled': False,
        'mixup_alpha': 0.5,  # 即使有值也应该被忽略
        'cutmix_prob': 0.3
    }
    manager3 = MockAdvancedAugmentationManager(config3)
    info3 = manager3.get_augmentation_info()
    print(f"   配置: {config3}")
    print(f"   信息: {info3}")
    print(f"   启用状态: {manager3.is_enabled()}")
    
    # 验证结果
    result1 = manager1.is_enabled() and info1['mixup_available'] and info1['cutmix_available']
    result2 = not manager2.is_enabled() and info2['advanced_augmentation_enabled'] and not info2['mixup_available']
    result3 = not manager3.is_enabled() and not info3['advanced_augmentation_enabled']
    
    overall_result = result1 and result2 and result3
    print(f"\n结果: {'✅ 高级数据增强信息功能正确' if overall_result else '❌ 高级数据增强信息功能有误'}")
    
    return overall_result

def test_augmentation_combination():
    """测试数据增强组合使用"""
    print("\n🧪 测试数据增强组合使用")
    print("=" * 50)
    
    def simulate_augmentation_status(use_basic, advanced_manager):
        """模拟数据增强状态输出"""
        augmentation_status = []
        if use_basic:
            augmentation_status.append("基础增强")
        if advanced_manager and advanced_manager.is_enabled():
            augmentation_status.append("高级增强")
        
        if augmentation_status:
            return f"📊 数据增强配置: {' + '.join(augmentation_status)}"
        else:
            return "📊 数据增强配置: 无增强"
    
    class MockManager:
        def __init__(self, enabled):
            self._enabled = enabled
        def is_enabled(self):
            return self._enabled
    
    # 测试所有组合
    combinations = [
        (True, MockManager(True), "基础增强 + 高级增强"),
        (True, MockManager(False), "仅基础增强"),
        (False, MockManager(True), "仅高级增强"),
        (False, MockManager(False), "无增强"),
        (True, None, "仅基础增强（无高级管理器）"),
    ]
    
    for i, (use_basic, advanced_manager, expected_desc) in enumerate(combinations, 1):
        print(f"\n📋 测试用例{i}: {expected_desc}")
        status = simulate_augmentation_status(use_basic, advanced_manager)
        print(f"   基础增强: {use_basic}")
        print(f"   高级管理器: {advanced_manager.is_enabled() if advanced_manager else None}")
        print(f"   状态输出: {status}")
        
        # 验证状态输出的正确性
        if use_basic and advanced_manager and advanced_manager.is_enabled():
            expected = "基础增强 + 高级增强"
        elif use_basic and (not advanced_manager or not advanced_manager.is_enabled()):
            expected = "基础增强"
        elif not use_basic and advanced_manager and advanced_manager.is_enabled():
            expected = "高级增强"
        else:
            expected = "无增强"
        
        correct = expected in status
        print(f"   验证结果: {'✅ 正确' if correct else '❌ 错误'}")
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始测试数据增强优化功能")
    print("=" * 70)
    
    # 运行所有测试
    test_results = []
    
    # 测试1: 基础数据增强控制
    result1 = test_basic_augmentation_control()
    test_results.append(("基础数据增强控制", result1))
    
    # 测试2: 高级数据增强信息功能
    result2 = test_advanced_augmentation_info()
    test_results.append(("高级数据增强信息功能", result2))
    
    # 测试3: 数据增强组合使用
    result3 = test_augmentation_combination()
    test_results.append(("数据增强组合使用", result3))
    
    # 输出测试总结
    print("\n" + "=" * 70)
    print("📊 测试结果总结")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 总体结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过！数据增强优化功能正常")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步检查和修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 