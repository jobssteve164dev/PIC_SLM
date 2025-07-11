#!/usr/bin/env python3
"""
测试当前版本超参数生效性验证脚本

验证智能推断逻辑是否正确识别超参数启用状态，
以及超参数是否真正应用到训练过程中。
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

# 直接导入需要的模块
try:
    from training_components.optimizer_factory import OptimizerFactory
    import torch
    import torch.nn as nn
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在项目根目录下运行此脚本")
    sys.exit(1)


def enhance_config_with_enable_states(config):
    """
    智能推断并增强配置中的启用状态字段
    
    直接实现ConfigApplier的逻辑，避免UI依赖
    """
    enhanced_config = config.copy()
    
    # 最小学习率启用状态推断
    if 'min_lr_enabled' not in enhanced_config:
        min_lr = enhanced_config.get('min_lr', 0.0)
        enhanced_config['min_lr_enabled'] = min_lr > 0.0
    
    # 学习率预热启用状态推断
    if 'warmup_enabled' not in enhanced_config:
        warmup_steps = enhanced_config.get('warmup_steps', 0)
        warmup_ratio = enhanced_config.get('warmup_ratio', 0.0)
        enhanced_config['warmup_enabled'] = warmup_steps > 0 or warmup_ratio > 0.0
    
    # 标签平滑启用状态推断
    if 'label_smoothing_enabled' not in enhanced_config:
        label_smoothing = enhanced_config.get('label_smoothing', 0.0)
        enhanced_config['label_smoothing_enabled'] = label_smoothing > 0.0
    
    return enhanced_config


def test_critical_min_lr_issue():
    """测试关键的最小学习率问题"""
    print("\n🔍 测试: 关键的最小学习率问题")
    
    # 这是导致学习率曲线差异的关键问题
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟7月4日之前的配置（没有启用状态字段）
    old_config = {
        'lr_scheduler': 'CosineAnnealingLR',
        'T_max': 50,
        'min_lr': 1e-6  # 旧版本会直接使用这个值
    }
    
    print(f"原始配置: {old_config}")
    
    # 应用智能推断
    enhanced_config = enhance_config_with_enable_states(old_config)
    
    # 检查推断结果
    min_lr_enabled = enhanced_config.get('min_lr_enabled', False)
    print(f"智能推断结果: min_lr_enabled = {min_lr_enabled}")
    print(f"增强后配置: {enhanced_config}")
    
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


def test_scheduler_comparison():
    """测试不同配置下的调度器行为"""
    print("\n🔍 测试: 调度器行为对比")
    
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 测试场景
    test_cases = [
        {
            'name': '7月4日版本模拟（直接使用min_lr）',
            'config': {'lr_scheduler': 'CosineAnnealingLR', 'T_max': 50, 'min_lr': 1e-6},
            'use_enhancement': False,  # 不使用智能推断
            'expected_eta_min': 1e-6
        },
        {
            'name': '当前版本（智能推断）',
            'config': {'lr_scheduler': 'CosineAnnealingLR', 'T_max': 50, 'min_lr': 1e-6},
            'use_enhancement': True,  # 使用智能推断
            'expected_eta_min': 1e-6  # 期望推断为启用
        },
        {
            'name': '显式启用最小学习率',
            'config': {'lr_scheduler': 'CosineAnnealingLR', 'T_max': 50, 'min_lr_enabled': True, 'min_lr': 1e-6},
            'use_enhancement': False,
            'expected_eta_min': 1e-6
        },
        {
            'name': '显式禁用最小学习率',
            'config': {'lr_scheduler': 'CosineAnnealingLR', 'T_max': 50, 'min_lr_enabled': False, 'min_lr': 1e-6},
            'use_enhancement': False,
            'expected_eta_min': 0
        }
    ]
    
    all_passed = True
    
    for case in test_cases:
        print(f"\n  场景: {case['name']}")
        
        config = case['config'].copy()
        if case['use_enhancement']:
            config = enhance_config_with_enable_states(config)
        
        # 模拟7月4日版本的行为（直接使用min_lr）
        if case['name'] == '7月4日版本模拟（直接使用min_lr）':
            # 直接创建调度器，不经过当前版本的逻辑
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['T_max'], eta_min=config['min_lr']
            )
        else:
            scheduler = OptimizerFactory.create_scheduler(optimizer, config)
        
        if scheduler and hasattr(scheduler, 'eta_min'):
            print(f"    配置: {config}")
            print(f"    eta_min: {scheduler.eta_min}")
            
            if abs(scheduler.eta_min - case['expected_eta_min']) < 1e-10:
                print("    ✅ 结果正确")
            else:
                print(f"    ❌ 结果错误: 期望 {case['expected_eta_min']}, 实际 {scheduler.eta_min}")
                all_passed = False
        else:
            print("    ❌ 调度器创建失败")
            all_passed = False
    
    return all_passed


def main():
    """主测试函数"""
    print("🚀 开始测试当前版本超参数生效性...\n")
    
    # 执行关键测试
    test1_result = test_critical_min_lr_issue()
    test2_result = test_scheduler_comparison()
    
    print("\n" + "="*60)
    print("📊 测试结果汇总:")
    print("="*60)
    
    print(f"关键最小学习率问题: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"调度器行为对比: {'✅ 通过' if test2_result else '❌ 失败'}")
    
    if test1_result and test2_result:
        print("\n🎉 所有测试都通过！")
        print("当前版本的智能推断逻辑工作正常，超参数应该能正确生效。")
    else:
        print("\n⚠️  测试失败，发现超参数生效性问题！")
        print("\n🔧 问题分析:")
        if not test1_result:
            print("- 智能推断逻辑可能存在问题")
            print("- 旧配置文件的min_lr可能不会被正确识别")
        if not test2_result:
            print("- 不同配置场景下的行为不一致")
        
        print("\n💡 建议:")
        print("1. 检查ConfigApplier中的智能推断逻辑")
        print("2. 确保向后兼容性处理正确")
        print("3. 在训练配置中显式设置启用状态字段")
    
    return test1_result and test2_result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 