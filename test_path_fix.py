#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试路径修复的脚本
"""

import os
import json

def test_path_normalization():
    """测试路径标准化"""
    print("=== 测试路径标准化 ===")
    
    # 测试原始问题路径
    original_path = "F:\\Qsync\\00.AI_PROJECT\\图片分类模型训练\\C1\\src\\..\\models\\saved_models"
    print(f"原始路径: {original_path}")
    
    # 标准化路径
    normalized_path = os.path.normpath(original_path)
    print(f"标准化路径: {normalized_path}")
    
    # 转换为绝对路径
    absolute_path = os.path.abspath(normalized_path)
    print(f"绝对路径: {absolute_path}")
    
    # 检查路径是否存在
    print(f"路径是否存在: {os.path.exists(absolute_path)}")
    
    # 测试写入权限
    try:
        test_file = os.path.join(absolute_path, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ 路径可写")
    except Exception as e:
        print(f"❌ 路径不可写: {str(e)}")

def test_config_loading():
    """测试配置文件加载"""
    print("\n=== 测试配置文件加载 ===")
    
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_save_dir = config.get('default_model_save_dir', '')
        print(f"配置文件中的模型保存路径: {model_save_dir}")
        
        # 标准化路径
        normalized_path = os.path.normpath(model_save_dir)
        print(f"标准化后的路径: {normalized_path}")
        
        # 检查路径是否存在
        print(f"路径是否存在: {os.path.exists(normalized_path)}")
        
        # 测试写入权限
        try:
            test_file = os.path.join(normalized_path, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print("✅ 配置文件中的路径可写")
        except Exception as e:
            print(f"❌ 配置文件中的路径不可写: {str(e)}")
            
    except Exception as e:
        print(f"❌ 加载配置文件失败: {str(e)}")

if __name__ == "__main__":
    test_path_normalization()
    test_config_loading()