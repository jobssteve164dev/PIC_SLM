#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试UNKNOW类别识别逻辑
"""

def debug_noise_filter(filename, class_names):
    """调试噪音过滤逻辑"""
    filename_upper = filename.upper()
    
    print(f"调试文件名: {filename}")
    print(f"大写文件名: {filename_upper}")
    
    # 如果提取的类别太短（单字符），直接拒绝
    if len(filename.strip()) <= 1:
        print("❌ 被单字符规则拒绝")
        return None
    
    # 噪音关键词检查
    noise_keywords = ['UNKNOWN', 'TEST', 'SAMPLE', 'TEMP', 'DEBUG', 'CLASS']
    print(f"噪音关键词列表: {noise_keywords}")
    
    # 当前的逻辑：any(noise in filename_upper for noise in noise_keywords)
    noise_matches = []
    for noise in noise_keywords:
        if noise in filename_upper:
            noise_matches.append(noise)
    
    print(f"文件名中包含的噪音关键词: {noise_matches}")
    
    if noise_matches:
        print(f"❌ 被噪音关键词规则拒绝: {noise_matches}")
        return None
    else:
        print("✅ 通过噪音关键词检查")
        return "PASS"


def test_unknow_debug():
    """调试UNKNOW相关的具体案例"""
    print("=" * 80)
    print("UNKNOW类别识别调试")
    print("=" * 80)
    
    test_cases = [
        ("UNKNOW (1)", ["UNKNOW", "GOOD", "BAD"]),
        ("UNKNOW_001", ["UNKNOW", "GOOD", "BAD"]),
        ("UNKNOW_test", ["UNKNOW", "GOOD", "BAD"]),
        ("unknown_class", ["UNKNOW", "GOOD", "BAD"]),
        ("UNKNOWN_noise", ["UNKNOW", "GOOD", "BAD"]),
        ("TEST (1)", ["TEST", "GOOD", "BAD"]),
        ("test_file", ["TEST", "GOOD", "BAD"]),
    ]
    
    for filename, class_names in test_cases:
        print(f"\n" + "─" * 60)
        result = debug_noise_filter(filename, class_names)
        print(f"最终结果: {result}")


if __name__ == "__main__":
    test_unknow_debug() 