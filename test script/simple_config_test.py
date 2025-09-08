#!/usr/bin/env python3
"""
简化的智能训练配置测试
"""

import os
import json

def test_config_files():
    """测试配置文件"""
    print("Testing config files...")
    
    # 检查主配置文件
    if os.path.exists("config.json"):
        with open("config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
            intelligent_config = config.get('intelligent_training', {})
            print(f"Main config found: {bool(intelligent_config)}")
            if intelligent_config:
                print(f"Max iterations: {intelligent_config.get('max_iterations', 'Not set')}")
    else:
        print("Main config not found")
    
    # 检查智能训练配置文件
    if os.path.exists("setting/intelligent_training_config.json"):
        print("Intelligent training config found")
    else:
        print("Intelligent training config not found")

if __name__ == "__main__":
    test_config_files()
