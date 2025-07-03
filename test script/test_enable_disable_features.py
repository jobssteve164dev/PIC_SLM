#!/usr/bin/env python3
"""
测试高级超参数启用/禁用功能

验证新增的复选框控制功能是否正常工作
"""

import sys
import os

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

try:
    # 尝试直接导入
    import training_components.training_validator as tv
    TrainingValidator = tv.TrainingValidator
    print("✅ 成功导入TrainingValidator")
except ImportError as e:
    print(f"❌ 导入TrainingValidator失败: {e}")
    # 尝试添加更多路径
    import importlib.util
    validator_path = os.path.join(src_dir, 'training_components', 'training_validator.py')
    if os.path.exists(validator_path):
        spec = importlib.util.spec_from_file_location("training_validator", validator_path)
        tv_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tv_module)
        TrainingValidator = tv_module.TrainingValidator
        print("✅ 通过文件路径成功导入TrainingValidator")
    else:
        print(f"❌ 找不到验证器文件: {validator_path}")
        sys.exit(1)


def test_warmup_enable_disable():
    """测试预热功能启用/禁用"""
    print("\n" + "=" * 60)
    print("测试预热功能启用/禁用")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    # 测试1：启用预热但未设置参数
    config_warmup_no_params = {
        'warmup_enabled': True,
        'warmup_steps': 0,
        'warmup_ratio': 0.0,
        'optimizer': 'Adam',
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_warmup_no_params)
    
    print(f"测试1 - 启用预热但未设置参数:")
    print(f"  发现冲突: {len(conflicts)} 个")
    for conflict in conflicts:
        print(f"    - {conflict['type']}: {conflict['description']}")
    
    # 测试2：禁用预热，不应该检测到预热相关冲突
    config_warmup_disabled = {
        'warmup_enabled': False,
        'warmup_steps': 1000,  # 虽然设置了参数
        'warmup_ratio': 0.1,   # 但预热被禁用
        'optimizer': 'Adam',
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts2, suggestions2 = validator._detect_hyperparameter_conflicts(config_warmup_disabled)
    
    print(f"\n测试2 - 禁用预热:")
    print(f"  发现冲突: {len(conflicts2)} 个")
    for conflict in conflicts2:
        print(f"    - {conflict['type']}: {conflict['description']}")
    
    return len(conflicts) > 0 and len(conflicts2) == 0


def test_augmentation_enable_disable():
    """测试高级数据增强启用/禁用"""
    print("\n" + "=" * 60)
    print("测试高级数据增强启用/禁用")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    # 测试1：启用数据增强但任务类型为检测
    config_detection_augmentation = {
        'advanced_augmentation_enabled': True,
        'cutmix_prob': 0.5,
        'mixup_alpha': 0.2,
        'task_type': 'detection',
        'optimizer': 'Adam',
        'data_dir': 'test_data',
        'model_name': 'YOLOv5',
        'num_epochs': 10,
        'batch_size': 16,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_detection_augmentation)
    
    print(f"测试1 - 检测任务启用高级数据增强:")
    print(f"  发现冲突: {len(conflicts)} 个")
    for conflict in conflicts:
        print(f"    - {conflict['type']}: {conflict['description']}")
    
    # 测试2：禁用数据增强，即使任务类型为检测也不应该冲突
    config_detection_no_augmentation = {
        'advanced_augmentation_enabled': False,
        'cutmix_prob': 0.5,  # 虽然设置了参数
        'mixup_alpha': 0.2,  # 但数据增强被禁用
        'task_type': 'detection',
        'optimizer': 'Adam',
        'data_dir': 'test_data',
        'model_name': 'YOLOv5',
        'num_epochs': 10,
        'batch_size': 16,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models'
    }
    
    conflicts2, suggestions2 = validator._detect_hyperparameter_conflicts(config_detection_no_augmentation)
    
    print(f"\n测试2 - 检测任务禁用高级数据增强:")
    print(f"  发现冲突: {len(conflicts2)} 个")
    for conflict in conflicts2:
        print(f"    - {conflict['type']}: {conflict['description']}")
    
    return len(conflicts) > 0 and len(conflicts2) == 0


def test_label_smoothing_enable_disable():
    """测试标签平滑启用/禁用"""
    print("\n" + "=" * 60)
    print("测试标签平滑启用/禁用")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    # 测试1：启用标签平滑但任务类型为检测
    config_detection_label_smoothing = {
        'label_smoothing_enabled': True,
        'label_smoothing': 0.1,
        'task_type': 'detection',
        'optimizer': 'Adam',
        'data_dir': 'test_data',
        'model_name': 'YOLOv5',
        'num_epochs': 10,
        'batch_size': 16,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_detection_label_smoothing)
    
    print(f"测试1 - 检测任务启用标签平滑:")
    print(f"  发现冲突: {len(conflicts)} 个")
    for conflict in conflicts:
        print(f"    - {conflict['type']}: {conflict['description']}")
    
    # 测试2：禁用标签平滑，即使任务类型为检测也不应该冲突
    config_detection_no_label_smoothing = {
        'label_smoothing_enabled': False,
        'label_smoothing': 0.1,  # 虽然设置了参数
        'task_type': 'detection',
        'optimizer': 'Adam',
        'data_dir': 'test_data',
        'model_name': 'YOLOv5',
        'num_epochs': 10,
        'batch_size': 16,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models'
    }
    
    conflicts2, suggestions2 = validator._detect_hyperparameter_conflicts(config_detection_no_label_smoothing)
    
    print(f"\n测试2 - 检测任务禁用标签平滑:")
    print(f"  发现冲突: {len(conflicts2)} 个")
    for conflict in conflicts2:
        print(f"    - {conflict['type']}: {conflict['description']}")
    
    return len(conflicts) > 0 and len(conflicts2) == 0


def test_gradient_accumulation_enable_disable():
    """测试梯度累积启用/禁用"""
    print("\n" + "=" * 60)
    print("测试梯度累积启用/禁用")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    # 测试1：启用梯度累积但批次过大
    config_large_batch = {
        'gradient_accumulation_enabled': True,
        'batch_size': 128,
        'gradient_accumulation_steps': 8,  # 有效批次 = 128 * 8 = 1024
        'optimizer': 'Adam',
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_large_batch)
    
    print(f"测试1 - 启用梯度累积但批次过大:")
    print(f"  发现冲突: {len(conflicts)} 个")
    for conflict in conflicts:
        print(f"    - {conflict['type']}: {conflict['description']}")
    
    # 测试2：禁用梯度累积，不应该检测到批次相关冲突
    config_no_accumulation = {
        'gradient_accumulation_enabled': False,
        'batch_size': 128,
        'gradient_accumulation_steps': 8,  # 虽然设置了参数
        'optimizer': 'Adam',
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts2, suggestions2 = validator._detect_hyperparameter_conflicts(config_no_accumulation)
    
    print(f"\n测试2 - 禁用梯度累积:")
    print(f"  发现冲突: {len(conflicts2)} 个")
    for conflict in conflicts2:
        print(f"    - {conflict['type']}: {conflict['description']}")
    
    return len(conflicts) > 0 and len(conflicts2) == 0


def test_loss_scaling_enable_disable():
    """测试损失缩放启用/禁用"""
    print("\n" + "=" * 60)
    print("测试损失缩放启用/禁用")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    # 测试1：启用损失缩放但未启用混合精度
    config_loss_scaling_no_mixed_precision = {
        'loss_scaling_enabled': True,
        'loss_scale': 'static',
        'mixed_precision': False,
        'optimizer': 'Adam',
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_loss_scaling_no_mixed_precision)
    
    print(f"测试1 - 启用损失缩放但未启用混合精度:")
    print(f"  发现冲突: {len(conflicts)} 个")
    for conflict in conflicts:
        print(f"    - {conflict['type']}: {conflict['description']}")
    
    # 测试2：禁用损失缩放，不应该检测到混合精度相关冲突
    config_no_loss_scaling = {
        'loss_scaling_enabled': False,
        'loss_scale': 'static',  # 虽然设置了参数
        'mixed_precision': False,
        'optimizer': 'Adam',
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts2, suggestions2 = validator._detect_hyperparameter_conflicts(config_no_loss_scaling)
    
    print(f"\n测试2 - 禁用损失缩放:")
    print(f"  发现冲突: {len(conflicts2)} 个")
    for conflict in conflicts2:
        print(f"    - {conflict['type']}: {conflict['description']}")
    
    return len(conflicts) > 0 and len(conflicts2) == 0


def main():
    """主测试函数"""
    print("🔧 高级超参数启用/禁用功能测试")
    print("=" * 80)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("预热功能启用/禁用", test_warmup_enable_disable()))
    test_results.append(("高级数据增强启用/禁用", test_augmentation_enable_disable()))
    test_results.append(("标签平滑启用/禁用", test_label_smoothing_enable_disable()))
    test_results.append(("梯度累积启用/禁用", test_gradient_accumulation_enable_disable()))
    test_results.append(("损失缩放启用/禁用", test_loss_scaling_enable_disable()))
    
    # 汇总测试结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 个测试通过, {failed} 个测试失败")
    
    if failed == 0:
        print("🎉 所有测试都通过了！启用/禁用功能工作正常。")
    else:
        print("⚠️  部分测试失败，需要检查启用/禁用功能的实现。")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 