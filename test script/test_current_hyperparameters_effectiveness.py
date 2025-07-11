#!/usr/bin/env python3
"""
测试当前版本超参数生效性验证脚本

验证智能推断逻辑是否正确识别超参数启用状态，
以及超参数是否真正应用到训练过程中。
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 直接导入需要的模块，避免UI组件的复杂依赖
try:
    from training_components.optimizer_factory import OptimizerFactory
    from training_components.training_validator import TrainingValidator
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录下运行此脚本")
    sys.exit(1)


def enhance_config_with_enable_states(config):
    """
    智能推断并增强配置中的启用状态字段
    
    直接实现ConfigApplier的逻辑，避免UI依赖
    """
    enhanced_config = config.copy()
    
    # 学习率预热启用状态推断
    if 'warmup_enabled' not in enhanced_config:
        warmup_steps = enhanced_config.get('warmup_steps', 0)
        warmup_ratio = enhanced_config.get('warmup_ratio', 0.0)
        enhanced_config['warmup_enabled'] = warmup_steps > 0 or warmup_ratio > 0.0
    
    # 最小学习率启用状态推断
    if 'min_lr_enabled' not in enhanced_config:
        min_lr = enhanced_config.get('min_lr', 0.0)
        enhanced_config['min_lr_enabled'] = min_lr > 0.0
    
    # 标签平滑启用状态推断
    if 'label_smoothing_enabled' not in enhanced_config:
        label_smoothing = enhanced_config.get('label_smoothing', 0.0)
        enhanced_config['label_smoothing_enabled'] = label_smoothing > 0.0
    
    # 梯度累积启用状态推断
    if 'gradient_accumulation_enabled' not in enhanced_config:
        gradient_accumulation_steps = enhanced_config.get('gradient_accumulation_steps', 1)
        enhanced_config['gradient_accumulation_enabled'] = gradient_accumulation_steps > 1
    
    # 高级数据增强启用状态推断
    if 'advanced_augmentation_enabled' not in enhanced_config:
        cutmix_prob = enhanced_config.get('cutmix_prob', 0.0)
        mixup_alpha = enhanced_config.get('mixup_alpha', 0.0)
        enhanced_config['advanced_augmentation_enabled'] = cutmix_prob > 0.0 or mixup_alpha > 0.0
    
    # 损失缩放启用状态推断
    if 'loss_scaling_enabled' not in enhanced_config:
        loss_scale = enhanced_config.get('loss_scale', 'none')
        enhanced_config['loss_scaling_enabled'] = loss_scale != 'none'
    
    return enhanced_config


def test_config_enhancement():
    """测试配置增强功能"""
    print("🔍 测试1: 配置增强功能")
    
    # 模拟旧版本配置文件（没有启用状态字段）
    old_config = {
        'warmup_steps': 100,
        'warmup_ratio': 0.0,
        'min_lr': 1e-6,
        'label_smoothing': 0.1,
        'gradient_accumulation_steps': 4,
        'cutmix_prob': 0.5,
        'mixup_alpha': 0.2,
        'loss_scale': 'dynamic'
    }
    
    # 应用智能推断
    enhanced_config = enhance_config_with_enable_states(old_config)
    
    # 验证推断结果
    expected_states = {
        'warmup_enabled': True,  # warmup_steps > 0
        'min_lr_enabled': True,  # min_lr > 0
        'label_smoothing_enabled': True,  # label_smoothing > 0
        'gradient_accumulation_enabled': True,  # gradient_accumulation_steps > 1
        'advanced_augmentation_enabled': True,  # cutmix_prob > 0 or mixup_alpha > 0
        'loss_scaling_enabled': True  # loss_scale != 'none'
    }
    
    success = True
    for key, expected in expected_states.items():
        actual = enhanced_config.get(key, False)
        if actual != expected:
            print(f"❌ {key}: 期望 {expected}, 实际 {actual}")
            success = False
        else:
            print(f"✅ {key}: {actual}")
    
    return success


def test_scheduler_creation():
    """测试学习率调度器创建"""
    print("\n🔍 测试2: 学习率调度器创建")
    
    # 创建简单模型用于测试
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 测试场景1: 启用最小学习率
    config1 = {
        'lr_scheduler': 'CosineAnnealingLR',
        'T_max': 50,
        'min_lr_enabled': True,
        'min_lr': 1e-6
    }
    
    scheduler1 = OptimizerFactory.create_scheduler(optimizer, config1)
    if scheduler1 and hasattr(scheduler1, 'eta_min'):
        print(f"✅ 启用最小学习率: eta_min = {scheduler1.eta_min}")
    else:
        print("❌ 启用最小学习率: 调度器创建失败")
        return False
    
    # 测试场景2: 禁用最小学习率
    config2 = {
        'lr_scheduler': 'CosineAnnealingLR',
        'T_max': 50,
        'min_lr_enabled': False,
        'min_lr': 1e-6
    }
    
    scheduler2 = OptimizerFactory.create_scheduler(optimizer, config2)
    if scheduler2 and hasattr(scheduler2, 'eta_min'):
        print(f"✅ 禁用最小学习率: eta_min = {scheduler2.eta_min}")
        if scheduler2.eta_min != 0:
            print("❌ 禁用最小学习率时，eta_min应该为0")
            return False
    else:
        print("❌ 禁用最小学习率: 调度器创建失败")
        return False
    
    # 测试场景3: 旧配置文件（智能推断）
    old_config = {
        'lr_scheduler': 'CosineAnnealingLR',
        'T_max': 50,
        'min_lr': 1e-6  # 没有min_lr_enabled字段
    }
    
    enhanced_config = enhance_config_with_enable_states(old_config)
    scheduler3 = OptimizerFactory.create_scheduler(optimizer, enhanced_config)
    
    if scheduler3 and hasattr(scheduler3, 'eta_min'):
        min_lr_enabled = enhanced_config.get('min_lr_enabled', False)
        expected_eta_min = 1e-6 if min_lr_enabled else 0
        print(f"✅ 智能推断: min_lr_enabled = {min_lr_enabled}, eta_min = {scheduler3.eta_min}")
        
        if scheduler3.eta_min != expected_eta_min:
            print(f"❌ 智能推断结果不正确: 期望 eta_min = {expected_eta_min}, 实际 = {scheduler3.eta_min}")
            return False
    else:
        print("❌ 智能推断: 调度器创建失败")
        return False
    
    return True


def test_warmup_functionality():
    """测试预热功能"""
    print("\n🔍 测试3: 预热功能")
    
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 测试场景1: 启用预热
    config1 = {
        'lr_scheduler': 'CosineAnnealingLR',
        'T_max': 50,
        'warmup_enabled': True,
        'warmup_steps': 10,
        'warmup_method': 'linear'
    }
    
    scheduler1 = OptimizerFactory.create_scheduler(optimizer, config1, total_steps=100)
    if scheduler1:
        # 检查是否是预热调度器
        if hasattr(scheduler1, 'warmup_steps'):
            print(f"✅ 启用预热: warmup_steps = {scheduler1.warmup_steps}")
        else:
            print("❌ 启用预热: 未创建预热调度器")
            return False
    else:
        print("❌ 启用预热: 调度器创建失败")
        return False
    
    # 测试场景2: 禁用预热
    config2 = {
        'lr_scheduler': 'CosineAnnealingLR',
        'T_max': 50,
        'warmup_enabled': False,
        'warmup_steps': 10  # 即使设置了步数，也不应该启用
    }
    
    scheduler2 = OptimizerFactory.create_scheduler(optimizer, config2, total_steps=100)
    if scheduler2:
        # 检查是否是基础调度器（不是预热调度器）
        if not hasattr(scheduler2, 'warmup_steps'):
            print("✅ 禁用预热: 使用基础调度器")
        else:
            print("❌ 禁用预热: 错误地创建了预热调度器")
            return False
    else:
        print("❌ 禁用预热: 调度器创建失败")
        return False
    
    return True


def test_loss_function_creation():
    """测试损失函数创建"""
    print("\n🔍 测试4: 损失函数创建")
    
    # 测试场景1: 启用标签平滑
    config1 = {
        'label_smoothing_enabled': True,
        'label_smoothing': 0.1
    }
    
    criterion1 = OptimizerFactory.create_criterion(config1)
    if hasattr(criterion1, 'smoothing'):
        print(f"✅ 启用标签平滑: smoothing = {criterion1.smoothing}")
    else:
        print("❌ 启用标签平滑: 未创建标签平滑损失函数")
        return False
    
    # 测试场景2: 禁用标签平滑
    config2 = {
        'label_smoothing_enabled': False,
        'label_smoothing': 0.1  # 即使设置了值，也不应该启用
    }
    
    criterion2 = OptimizerFactory.create_criterion(config2)
    if isinstance(criterion2, nn.CrossEntropyLoss):
        print("✅ 禁用标签平滑: 使用标准交叉熵损失")
    else:
        print("❌ 禁用标签平滑: 错误地创建了标签平滑损失函数")
        return False
    
    return True


def test_critical_min_lr_issue():
    """测试关键的最小学习率问题"""
    print("\n🔍 测试5: 关键的最小学习率问题")
    
    # 这是导致学习率曲线差异的关键问题
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟7月4日之前的配置（没有启用状态字段）
    old_config = {
        'lr_scheduler': 'CosineAnnealingLR',
        'T_max': 50,
        'min_lr': 1e-6  # 旧版本会直接使用这个值
    }
    
    # 应用智能推断
    enhanced_config = enhance_config_with_enable_states(old_config)
    
    # 检查推断结果
    min_lr_enabled = enhanced_config.get('min_lr_enabled', False)
    print(f"智能推断结果: min_lr_enabled = {min_lr_enabled}")
    
    # 创建调度器
    scheduler = OptimizerFactory.create_scheduler(optimizer, enhanced_config)
    
    if scheduler and hasattr(scheduler, 'eta_min'):
        print(f"实际 eta_min = {scheduler.eta_min}")
        
        # 关键问题：如果min_lr_enabled被推断为False，eta_min会是0而不是1e-6
        if min_lr_enabled:
            expected_eta_min = 1e-6
            print("✅ 最小学习率启用状态正确推断")
        else:
            expected_eta_min = 0
            print("❌ 最小学习率启用状态推断错误！")
            print("   这会导致学习率可以降到接近0，与7月4日版本行为不同")
            return False
        
        if abs(scheduler.eta_min - expected_eta_min) < 1e-10:
            print("✅ eta_min值正确")
        else:
            print(f"❌ eta_min值错误: 期望 {expected_eta_min}, 实际 {scheduler.eta_min}")
            return False
    else:
        print("❌ 调度器创建失败")
        return False
    
    return True


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n🔍 测试6: 向后兼容性")
    
    # 模拟各种旧配置文件场景
    test_cases = [
        {
            'name': '默认min_lr配置',
            'config': {'lr_scheduler': 'CosineAnnealingLR', 'T_max': 50, 'min_lr': 1e-6},
            'expected_min_lr_enabled': True,  # 应该推断为启用
            'expected_eta_min': 1e-6
        },
        {
            'name': '零min_lr配置',
            'config': {'lr_scheduler': 'CosineAnnealingLR', 'T_max': 50, 'min_lr': 0.0},
            'expected_min_lr_enabled': False,  # 应该推断为禁用
            'expected_eta_min': 0
        },
        {
            'name': '无min_lr配置',
            'config': {'lr_scheduler': 'CosineAnnealingLR', 'T_max': 50},
            'expected_min_lr_enabled': False,  # 应该推断为禁用
            'expected_eta_min': 0
        },
        {
            'name': '自定义min_lr配置',
            'config': {'lr_scheduler': 'CosineAnnealingLR', 'T_max': 50, 'min_lr': 1e-5},
            'expected_min_lr_enabled': True,  # 应该推断为启用
            'expected_eta_min': 1e-5
        }
    ]
    
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    all_passed = True
    
    for case in test_cases:
        print(f"\n  测试场景: {case['name']}")
        
        # 应用智能推断
        enhanced_config = enhance_config_with_enable_states(case['config'])
        
        # 检查推断结果
        min_lr_enabled = enhanced_config.get('min_lr_enabled', False)
        if min_lr_enabled != case['expected_min_lr_enabled']:
            print(f"    ❌ 推断错误: 期望 {case['expected_min_lr_enabled']}, 实际 {min_lr_enabled}")
            all_passed = False
            continue
        
        # 创建调度器
        scheduler = OptimizerFactory.create_scheduler(optimizer, enhanced_config)
        
        if scheduler and hasattr(scheduler, 'eta_min'):
            if abs(scheduler.eta_min - case['expected_eta_min']) < 1e-10:
                print(f"    ✅ 正确: eta_min = {scheduler.eta_min}")
            else:
                print(f"    ❌ eta_min错误: 期望 {case['expected_eta_min']}, 实际 {scheduler.eta_min}")
                all_passed = False
        else:
            print("    ❌ 调度器创建失败")
            all_passed = False
    
    return all_passed


def main():
    """主测试函数"""
    print("🚀 开始测试当前版本超参数生效性...\n")
    
    test_results = []
    
    # 执行所有测试
    test_results.append(("配置增强功能", test_config_enhancement()))
    test_results.append(("学习率调度器创建", test_scheduler_creation()))
    test_results.append(("预热功能", test_warmup_functionality()))
    test_results.append(("损失函数创建", test_loss_function_creation()))
    test_results.append(("关键最小学习率问题", test_critical_min_lr_issue()))
    test_results.append(("向后兼容性", test_backward_compatibility()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("📊 测试结果汇总:")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过！当前版本超参数功能正常")
        return True
    else:
        print("⚠️  部分测试失败，存在超参数生效性问题")
        
        # 提供修复建议
        print("\n🔧 修复建议:")
        print("1. 检查智能推断逻辑是否正确处理默认值")
        print("2. 确认向后兼容性是否符合预期")
        print("3. 验证关键的最小学习率推断逻辑")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 