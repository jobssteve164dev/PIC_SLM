#!/usr/bin/env python3
"""
高级超参数启用状态测试脚本

测试高级超参数的启用状态是否能正确保存和应用：
1. 测试配置保存时是否包含启用状态
2. 测试配置应用时是否正确设置启用状态
3. 测试智能推断功能是否正常工作
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from PyQt5.QtWidgets import QApplication
from ui.components.training.advanced_hyperparameters_widget import AdvancedHyperparametersWidget
from ui.components.training.config_applier import ConfigApplier

def test_advanced_hyperparameters_enable_states():
    """测试高级超参数启用状态的保存和应用"""
    
    print("=" * 60)
    print("🧪 高级超参数启用状态测试")
    print("=" * 60)
    
    # 创建QApplication实例
    app = QApplication(sys.argv)
    
    try:
        # 创建高级超参数组件
        widget = AdvancedHyperparametersWidget()
        
        # 测试1: 设置一些启用状态并获取配置
        print("\n📋 测试1: 配置保存测试")
        print("-" * 40)
        
        # 启用学习率预热
        widget.warmup_enabled_checkbox.setChecked(True)
        widget.warmup_steps_spin.setValue(100)
        widget.warmup_ratio_spin.setValue(0.1)
        
        # 启用最小学习率限制
        widget.min_lr_enabled_checkbox.setChecked(True)
        widget.min_lr_spin.setValue(1e-6)
        
        # 启用标签平滑
        widget.label_smoothing_enabled_checkbox.setChecked(True)
        widget.label_smoothing_spin.setValue(0.1)
        
        # 启用梯度累积
        widget.gradient_accumulation_enabled_checkbox.setChecked(True)
        widget.gradient_accumulation_steps_spin.setValue(4)
        
        # 启用高级数据增强
        widget.advanced_augmentation_enabled_checkbox.setChecked(True)
        widget.cutmix_prob_spin.setValue(0.5)
        widget.mixup_alpha_spin.setValue(0.2)
        
        # 启用损失缩放
        widget.loss_scaling_enabled_checkbox.setChecked(True)
        widget.loss_scale_combo.setCurrentText('dynamic')
        
        # 获取配置
        config = widget.get_config()
        
        # 检查启用状态是否正确保存
        enable_states = {
            'warmup_enabled': config.get('warmup_enabled'),
            'min_lr_enabled': config.get('min_lr_enabled'),
            'label_smoothing_enabled': config.get('label_smoothing_enabled'),
            'gradient_accumulation_enabled': config.get('gradient_accumulation_enabled'),
            'advanced_augmentation_enabled': config.get('advanced_augmentation_enabled'),
            'loss_scaling_enabled': config.get('loss_scaling_enabled'),
        }
        
        print("启用状态保存结果:")
        all_enabled = True
        for key, value in enable_states.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
            if not value:
                all_enabled = False
        
        if all_enabled:
            print("\n✅ 配置保存测试通过：所有启用状态都正确保存")
        else:
            print("\n❌ 配置保存测试失败：部分启用状态未正确保存")
            return False
        
        # 测试2: 配置应用测试
        print("\n📋 测试2: 配置应用测试")
        print("-" * 40)
        
        # 创建新的组件实例
        widget2 = AdvancedHyperparametersWidget()
        
        # 应用配置
        widget2.set_config(config)
        
        # 检查启用状态是否正确应用
        applied_states = {
            'warmup_enabled': widget2.warmup_enabled_checkbox.isChecked(),
            'min_lr_enabled': widget2.min_lr_enabled_checkbox.isChecked(),
            'label_smoothing_enabled': widget2.label_smoothing_enabled_checkbox.isChecked(),
            'gradient_accumulation_enabled': widget2.gradient_accumulation_enabled_checkbox.isChecked(),
            'advanced_augmentation_enabled': widget2.advanced_augmentation_enabled_checkbox.isChecked(),
            'loss_scaling_enabled': widget2.loss_scaling_enabled_checkbox.isChecked(),
        }
        
        print("启用状态应用结果:")
        all_applied = True
        for key, value in applied_states.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
            if not value:
                all_applied = False
        
        if all_applied:
            print("\n✅ 配置应用测试通过：所有启用状态都正确应用")
        else:
            print("\n❌ 配置应用测试失败：部分启用状态未正确应用")
            return False
        
        # 测试3: 智能推断测试
        print("\n📋 测试3: 智能推断测试")
        print("-" * 40)
        
        # 创建一个缺少启用状态的配置（模拟旧配置文件）
        old_config = {
            'warmup_steps': 100,
            'warmup_ratio': 0.1,
            'min_lr': 1e-6,
            'label_smoothing': 0.1,
            'gradient_accumulation_steps': 4,
            'cutmix_prob': 0.5,
            'mixup_alpha': 0.2,
            'loss_scale': 'dynamic',
        }
        
        # 使用智能推断功能
        enhanced_config = ConfigApplier._enhance_config_with_enable_states(old_config)
        
        # 检查推断结果
        inferred_states = {
            'warmup_enabled': enhanced_config.get('warmup_enabled'),
            'min_lr_enabled': enhanced_config.get('min_lr_enabled'),
            'label_smoothing_enabled': enhanced_config.get('label_smoothing_enabled'),
            'gradient_accumulation_enabled': enhanced_config.get('gradient_accumulation_enabled'),
            'advanced_augmentation_enabled': enhanced_config.get('advanced_augmentation_enabled'),
            'loss_scaling_enabled': enhanced_config.get('loss_scaling_enabled'),
        }
        
        print("智能推断结果:")
        all_inferred = True
        for key, value in inferred_states.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
            if not value:
                all_inferred = False
        
        if all_inferred:
            print("\n✅ 智能推断测试通过：所有启用状态都正确推断")
        else:
            print("\n❌ 智能推断测试失败：部分启用状态未正确推断")
            return False
        
        # 测试4: 边界情况测试
        print("\n📋 测试4: 边界情况测试")
        print("-" * 40)
        
        # 测试值为0的情况
        zero_config = {
            'warmup_steps': 0,
            'warmup_ratio': 0.0,
            'min_lr': 0.0,
            'label_smoothing': 0.0,
            'gradient_accumulation_steps': 1,
            'cutmix_prob': 0.0,
            'mixup_alpha': 0.0,
            'loss_scale': 'none',
        }
        
        enhanced_zero_config = ConfigApplier._enhance_config_with_enable_states(zero_config)
        
        # 检查边界情况推断结果
        boundary_states = {
            'warmup_enabled': enhanced_zero_config.get('warmup_enabled'),
            'min_lr_enabled': enhanced_zero_config.get('min_lr_enabled'),
            'label_smoothing_enabled': enhanced_zero_config.get('label_smoothing_enabled'),
            'gradient_accumulation_enabled': enhanced_zero_config.get('gradient_accumulation_enabled'),
            'advanced_augmentation_enabled': enhanced_zero_config.get('advanced_augmentation_enabled'),
            'loss_scaling_enabled': enhanced_zero_config.get('loss_scaling_enabled'),
        }
        
        print("边界情况推断结果:")
        all_disabled = True
        for key, value in boundary_states.items():
            status = "✅" if not value else "❌"
            print(f"  {status} {key}: {value} (应为False)")
            if value:
                all_disabled = False
        
        if all_disabled:
            print("\n✅ 边界情况测试通过：所有启用状态都正确推断为False")
        else:
            print("\n❌ 边界情况测试失败：部分启用状态推断错误")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！高级超参数启用状态功能正常")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        app.quit()

def test_config_file_compatibility():
    """测试配置文件兼容性"""
    
    print("\n" + "=" * 60)
    print("🔄 配置文件兼容性测试")
    print("=" * 60)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建一个旧格式的配置文件
        old_config_path = os.path.join(temp_dir, "old_config.json")
        old_config = {
            "model_name": "MobileNetV2",
            "num_epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.001,
            "warmup_steps": 100,
            "warmup_ratio": 0.05,
            "min_lr": 1e-6,
            "label_smoothing": 0.1,
            "gradient_accumulation_steps": 4,
            "cutmix_prob": 0.5,
            "mixup_alpha": 0.2,
            "loss_scale": "dynamic"
        }
        
        with open(old_config_path, 'w', encoding='utf-8') as f:
            json.dump(old_config, f, indent=2)
        
        # 读取并应用智能推断
        with open(old_config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        enhanced_config = ConfigApplier._enhance_config_with_enable_states(loaded_config)
        
        # 检查是否正确添加了启用状态
        expected_states = {
            'warmup_enabled': True,
            'min_lr_enabled': True,
            'label_smoothing_enabled': True,
            'gradient_accumulation_enabled': True,
            'advanced_augmentation_enabled': True,
            'loss_scaling_enabled': True,
        }
        
        print("配置文件兼容性测试结果:")
        all_compatible = True
        for key, expected in expected_states.items():
            actual = enhanced_config.get(key)
            status = "✅" if actual == expected else "❌"
            print(f"  {status} {key}: {actual} (期望: {expected})")
            if actual != expected:
                all_compatible = False
        
        if all_compatible:
            print("\n✅ 配置文件兼容性测试通过：旧配置文件可以正确处理")
        else:
            print("\n❌ 配置文件兼容性测试失败：旧配置文件处理有问题")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ 配置文件兼容性测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    success = test_advanced_hyperparameters_enable_states()
    if success:
        success = test_config_file_compatibility()
    
    if success:
        print("\n🎉 所有测试通过！")
        exit(0)
    else:
        print("\n❌ 测试失败！")
        exit(1) 