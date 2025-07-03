#!/usr/bin/env python3
"""
超参数冲突检测测试脚本

测试新增的超参数冲突检测功能，验证各种冲突场景
"""

import sys
import os

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

# 设置环境变量以避免相对导入问题
os.environ['PYTHONPATH'] = src_dir

# 现在可以直接导入
try:
    from training_components.training_validator import TrainingValidator
    print("成功导入TrainingValidator")
except ImportError as e:
    print(f"导入TrainingValidator失败: {e}")
    # 尝试直接导入验证器文件
    sys.path.insert(0, os.path.join(src_dir, 'training_components'))
    from training_validator import TrainingValidator
    print("通过直接路径导入TrainingValidator成功")


def test_optimizer_conflicts():
    """测试优化器参数冲突"""
    print("=" * 60)
    print("测试优化器参数冲突")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    # 测试SGD优化器配置了Adam参数
    config_sgd_with_adam = {
        'optimizer': 'SGD',
        'beta1': 0.95,  # 非默认值
        'beta2': 0.99,  # 非默认值
        'momentum': 0.9,
        'nesterov': False,
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_sgd_with_adam)
    
    print(f"SGD配置Adam参数冲突检测:")
    print(f"发现冲突: {len(conflicts)} 个")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict['type']}: {conflict['description']}")
    
    print(f"修复建议: {len(suggestions)} 个")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion['parameter']}: {suggestion['action']}")
    
    # 测试Adam优化器配置了SGD参数
    config_adam_with_sgd = {
        'optimizer': 'Adam',
        'beta1': 0.9,
        'beta2': 0.999,
        'momentum': 0.95,  # 非默认值
        'nesterov': True,  # 非默认值
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_adam_with_sgd)
    
    print(f"\nAdam配置SGD参数冲突检测:")
    print(f"发现冲突: {len(conflicts)} 个")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict['type']}: {conflict['description']}")


def test_warmup_conflicts():
    """测试学习率预热冲突"""
    print("\n" + "=" * 60)
    print("测试学习率预热冲突")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    config_double_warmup = {
        'optimizer': 'Adam',
        'warmup_steps': 1000,
        'warmup_ratio': 0.1,  # 同时设置了两个预热参数
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_double_warmup)
    
    print(f"双重预热参数冲突检测:")
    print(f"发现冲突: {len(conflicts)} 个")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict['type']}: {conflict['description']}")


def test_data_augmentation_conflicts():
    """测试数据增强冲突"""
    print("\n" + "=" * 60)
    print("测试数据增强冲突")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    # 测试检测任务使用分类数据增强
    config_detection_with_mixup = {
        'optimizer': 'Adam',
        'task_type': 'detection',
        'cutmix_prob': 0.5,
        'mixup_alpha': 0.2,
        'data_dir': 'test_data',
        'model_name': 'YOLOv5',
        'num_epochs': 10,
        'batch_size': 16,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_detection_with_mixup)
    
    print(f"检测任务使用分类数据增强冲突检测:")
    print(f"发现冲突: {len(conflicts)} 个")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict['type']}: {conflict['description']}")
    
    # 测试同时使用CutMix和MixUp
    config_double_augmentation = {
        'optimizer': 'Adam',
        'task_type': 'classification',
        'cutmix_prob': 0.5,
        'mixup_alpha': 0.2,
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_double_augmentation)
    
    print(f"\n同时使用CutMix和MixUp冲突检测:")
    print(f"发现冲突: {len(conflicts)} 个")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict['type']}: {conflict['description']}")


def test_batch_size_conflicts():
    """测试批次大小冲突"""
    print("\n" + "=" * 60)
    print("测试批次大小冲突")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    config_large_batch = {
        'optimizer': 'Adam',
        'batch_size': 128,
        'gradient_accumulation_steps': 8,  # 有效批次 = 128 * 8 = 1024
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_large_batch)
    
    print(f"大批次冲突检测:")
    print(f"发现冲突: {len(conflicts)} 个")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict['type']}: {conflict['description']}")


def test_ema_conflicts():
    """测试EMA冲突"""
    print("\n" + "=" * 60)
    print("测试EMA冲突")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    config_ema_high_beta = {
        'optimizer': 'Adam',
        'beta1': 0.98,  # 高beta1值
        'model_ema': True,
        'model_ema_decay': 0.9999,
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_ema_high_beta)
    
    print(f"EMA与高beta1冲突检测:")
    print(f"发现冲突: {len(conflicts)} 个")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict['type']}: {conflict['description']}")


def test_scheduler_conflicts():
    """测试学习率调度器冲突"""
    print("\n" + "=" * 60)
    print("测试学习率调度器冲突")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    config_onecycle_with_warmup = {
        'optimizer': 'Adam',
        'lr_scheduler': 'OneCycleLR',
        'warmup_steps': 1000,
        'warmup_ratio': 0.1,
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_onecycle_with_warmup)
    
    print(f"OneCycleLR与预热冲突检测:")
    print(f"发现冲突: {len(conflicts)} 个")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict['type']}: {conflict['description']}")


def test_mixed_precision_conflicts():
    """测试混合精度冲突"""
    print("\n" + "=" * 60)
    print("测试混合精度冲突")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    config_no_mixed_precision = {
        'optimizer': 'Adam',
        'mixed_precision': False,
        'loss_scale': 'static',
        'static_loss_scale': 128.0,
        'data_dir': 'test_data',
        'model_name': 'ResNet50',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models',
        'task_type': 'classification'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_no_mixed_precision)
    
    print(f"混合精度关闭但设置损失缩放冲突检测:")
    print(f"发现冲突: {len(conflicts)} 个")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict['type']}: {conflict['description']}")


def test_conflict_fixes():
    """测试冲突修复功能"""
    print("\n" + "=" * 60)
    print("测试冲突修复功能")
    print("=" * 60)
    
    validator = TrainingValidator()
    
    # 创建一个包含多种冲突的配置
    config_with_conflicts = {
        'optimizer': 'SGD',
        'beta1': 0.95,  # SGD不需要
        'beta2': 0.99,  # SGD不需要
        'momentum': 0.9,
        'nesterov': False,
        'warmup_steps': 1000,
        'warmup_ratio': 0.1,  # 冲突
        'task_type': 'detection',
        'cutmix_prob': 0.5,  # 检测任务不应使用
        'mixup_alpha': 0.2,  # 检测任务不应使用
        'batch_size': 64,
        'gradient_accumulation_steps': 10,  # 有效批次 = 640
        'data_dir': 'test_data',
        'model_name': 'YOLOv5',
        'num_epochs': 10,
        'learning_rate': 0.001,
        'model_save_dir': 'test_models'
    }
    
    conflicts, suggestions = validator._detect_hyperparameter_conflicts(config_with_conflicts)
    
    print(f"修复前配置冲突:")
    print(f"发现冲突: {len(conflicts)} 个")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict['type']}: {conflict['description']}")
    
    # 应用修复
    fixed_config = validator._apply_conflict_fixes(config_with_conflicts, suggestions)
    
    print(f"\n修复后配置:")
    print(f"optimizer: {fixed_config['optimizer']}")
    print(f"beta1: {fixed_config['beta1']}")
    print(f"beta2: {fixed_config['beta2']}")
    print(f"warmup_steps: {fixed_config['warmup_steps']}")
    print(f"warmup_ratio: {fixed_config['warmup_ratio']}")
    print(f"cutmix_prob: {fixed_config['cutmix_prob']}")
    print(f"mixup_alpha: {fixed_config['mixup_alpha']}")
    print(f"gradient_accumulation_steps: {fixed_config['gradient_accumulation_steps']}")
    
    # 验证修复后是否还有冲突
    new_conflicts, _ = validator._detect_hyperparameter_conflicts(fixed_config)
    print(f"\n修复后剩余冲突: {len(new_conflicts)} 个")


def main():
    """主函数"""
    print("超参数冲突检测功能测试")
    print("=" * 60)
    
    # 运行各种冲突测试
    test_optimizer_conflicts()
    test_warmup_conflicts()
    test_data_augmentation_conflicts()
    test_batch_size_conflicts()
    test_ema_conflicts()
    test_scheduler_conflicts()
    test_mixed_precision_conflicts()
    test_conflict_fixes()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main() 