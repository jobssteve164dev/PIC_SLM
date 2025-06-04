#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试配置文件加载功能
"""

import json
import os

def test_config_loading():
    """测试配置文件加载"""
    
    # 测试示例配置文件
    config_file = "example_dataset_weights_config.json"
    
    if not os.path.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        return False
        
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"文件加载成功")
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"主要键: {list(data.keys())}")
            
            # 检查数据集评估导出格式
            if 'weight_config' in data:
                print("✓ 检测到数据集评估导出格式")
                weight_config = data.get('weight_config', {})
                classes = weight_config.get('classes', [])
                class_weights = weight_config.get('class_weights', {})
                weight_strategy = weight_config.get('weight_strategy', 'balanced')
                
                print(f"  类别数量: {len(classes)}")
                print(f"  类别列表: {classes}")
                print(f"  权重策略: {weight_strategy}")
                print(f"  权重信息: {class_weights}")
                
                # 检查数据集信息
                if 'dataset_info' in data:
                    dataset_info = data['dataset_info']
                    print(f"  数据集路径: {dataset_info.get('dataset_path', '未知')}")
                    print(f"  不平衡度: {dataset_info.get('imbalance_ratio', '未知')}")
                    print(f"  分析日期: {dataset_info.get('analysis_date', '未知')}")
                
                return True
            else:
                print("❌ 未检测到'weight_config'键")
                return False
        else:
            print(f"❌ 数据类型错误: {type(data)}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

if __name__ == "__main__":
    print("开始测试配置文件加载...")
    success = test_config_loading()
    print(f"\n测试结果: {'成功' if success else '失败'}") 