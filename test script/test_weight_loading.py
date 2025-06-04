#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试类别权重加载功能
验证训练算法是否能正确识别和使用类别权重信息
"""

import json
import os
import sys

# 添加src目录到路径
sys.path.append('src')

from model_trainer import TrainingThread

def test_weight_loading():
    """测试不同格式的权重配置加载"""
    print("=" * 60)
    print("测试类别权重加载功能")
    print("=" * 60)
    
    # 测试类别名称
    test_class_names = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
    
    # 测试配置1：直接在配置中包含class_weights
    print("\n1. 测试直接配置格式 (class_weights)")
    config1 = {
        'class_weights': {
            "Missing_hole": 1.5,
            "Mouse_bite": 2.0,
            "Open_circuit": 1.2,
            "Short": 1.8,
            "Spur": 1.0,
            "Spurious_copper": 2.5
        },
        'weight_strategy': 'custom',
        'use_class_weights': True
    }
    
    trainer1 = TrainingThread(config1)
    trainer1.class_distribution = {name: 100 for name in test_class_names}  # 模拟类别分布
    
    # 模拟权重计算
    try:
        weights1 = trainer1._calculate_class_weights(None, test_class_names, 'custom')
        print(f"✓ 成功加载权重: {weights1.cpu().numpy()}")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
    
    # 测试配置2：从权重配置文件读取
    print("\n2. 测试从权重配置文件读取")
    
    # 创建测试权重配置文件
    test_weight_file = "test_weights.json"
    weight_config_data = {
        "weight_config": {
            "classes": test_class_names,
            "class_weights": {
                "Missing_hole": 2.0,
                "Mouse_bite": 1.5,
                "Open_circuit": 3.0,
                "Short": 1.0,
                "Spur": 2.5,
                "Spurious_copper": 1.8
            },
            "weight_strategy": "custom"
        }
    }
    
    with open(test_weight_file, 'w', encoding='utf-8') as f:
        json.dump(weight_config_data, f, ensure_ascii=False, indent=2)
    
    config2 = {
        'weight_config_file': test_weight_file,
        'weight_strategy': 'custom',
        'use_class_weights': True
    }
    
    trainer2 = TrainingThread(config2)
    trainer2.class_distribution = {name: 100 for name in test_class_names}
    
    try:
        weights2 = trainer2._calculate_class_weights(None, test_class_names, 'custom')
        print(f"✓ 成功从文件加载权重: {weights2.cpu().numpy()}")
    except Exception as e:
        print(f"✗ 从文件加载失败: {e}")
    
    # 测试配置3：包含all_strategies的格式
    print("\n3. 测试all_strategies格式")
    config3 = {
        'all_strategies': {
            'balanced': {name: 1.0 for name in test_class_names},
            'custom': {
                "Missing_hole": 3.0,
                "Mouse_bite": 1.0,
                "Open_circuit": 2.2,
                "Short": 1.5,
                "Spur": 2.8,
                "Spurious_copper": 1.3
            }
        },
        'weight_strategy': 'custom',
        'use_class_weights': True
    }
    
    trainer3 = TrainingThread(config3)
    trainer3.class_distribution = {name: 100 for name in test_class_names}
    
    try:
        weights3 = trainer3._calculate_class_weights(None, test_class_names, 'custom')
        print(f"✓ 成功从all_strategies加载权重: {weights3.cpu().numpy()}")
    except Exception as e:
        print(f"✗ 从all_strategies加载失败: {e}")
    
    # 测试配置4：旧版custom_class_weights格式
    print("\n4. 测试旧版custom_class_weights格式")
    config4 = {
        'custom_class_weights': {
            "Missing_hole": 1.2,
            "Mouse_bite": 2.5,
            "Open_circuit": 1.0,
            "Short": 3.0,
            "Spur": 1.8,
            "Spurious_copper": 2.2
        },
        'weight_strategy': 'custom',
        'use_class_weights': True
    }
    
    trainer4 = TrainingThread(config4)
    trainer4.class_distribution = {name: 100 for name in test_class_names}
    
    try:
        weights4 = trainer4._calculate_class_weights(None, test_class_names, 'custom')
        print(f"✓ 成功从custom_class_weights加载权重: {weights4.cpu().numpy()}")
    except Exception as e:
        print(f"✗ 从custom_class_weights加载失败: {e}")
    
    # 测试配置5：无权重信息的情况
    print("\n5. 测试无权重信息的情况")
    config5 = {
        'weight_strategy': 'custom',
        'use_class_weights': True
    }
    
    trainer5 = TrainingThread(config5)
    trainer5.class_distribution = {name: 100 for name in test_class_names}
    
    try:
        weights5 = trainer5._calculate_class_weights(None, test_class_names, 'custom')
        print(f"✓ 无权重信息时使用默认权重: {weights5.cpu().numpy()}")
    except Exception as e:
        print(f"✗ 无权重信息处理失败: {e}")
    
    # 清理测试文件
    if os.path.exists(test_weight_file):
        os.remove(test_weight_file)
        print(f"\n清理测试文件: {test_weight_file}")
    
    print("\n" + "=" * 60)
    print("权重加载测试完成")
    print("=" * 60)

def test_real_config_loading():
    """测试真实的配置文件加载"""
    print("\n" + "=" * 60)
    print("测试真实配置文件加载")
    print("=" * 60)
    
    # 测试加载现有的权重配置文件
    test_files = [
        "class_weights_balanced.json",
        "example_dataset_weights_config.json"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n测试文件: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 创建配置来测试加载
                config = {}
                
                # 根据文件格式设置配置
                if 'weight_config' in data:
                    # 数据集评估导出格式
                    weight_config = data['weight_config']
                    config['class_weights'] = weight_config.get('class_weights', {})
                    config['weight_strategy'] = weight_config.get('weight_strategy', 'custom')
                    class_names = weight_config.get('classes', [])
                elif 'class_weights' in data:
                    # 直接权重格式
                    config['class_weights'] = data.get('class_weights', {})
                    config['weight_strategy'] = data.get('weight_strategy', 'custom')
                    class_names = list(config['class_weights'].keys())
                else:
                    print(f"  不支持的文件格式")
                    continue
                
                if class_names:
                    trainer = TrainingThread(config)
                    trainer.class_distribution = {name: 100 for name in class_names}
                    
                    weights = trainer._calculate_class_weights(None, class_names, 'custom')
                    print(f"  ✓ 成功加载，类别数: {len(class_names)}")
                    print(f"  权重范围: {weights.min().item():.3f} - {weights.max().item():.3f}")
                    
                    # 显示前3个类别的权重
                    for i, (name, weight) in enumerate(zip(class_names[:3], weights[:3])):
                        print(f"    {name}: {weight.item():.3f}")
                    if len(class_names) > 3:
                        print(f"    ... 还有 {len(class_names) - 3} 个类别")
                
            except Exception as e:
                print(f"  ✗ 加载失败: {e}")
        else:
            print(f"\n文件不存在: {file_path}")

if __name__ == "__main__":
    # 运行测试
    test_weight_loading()
    test_real_config_loading()
    
    print(f"\n测试脚本位置: {os.path.abspath(__file__)}")
    print("你可以运行此脚本来验证权重加载功能是否正常工作") 