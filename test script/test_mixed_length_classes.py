#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试长短不一的类名识别场景
"""

import sys
import os
import re
from difflib import SequenceMatcher

def smart_class_matching(filename, class_names):
    """当前的智能类别匹配算法"""
    filename_upper = filename.upper()
    
    # 如果提取的类别太短（单字符），直接拒绝
    if len(filename.strip()) <= 1:
        return None
    
    # 如果包含明显的"噪音"关键词，直接拒绝
    noise_keywords = ['UNKNOWN', 'TEST', 'SAMPLE', 'TEMP', 'DEBUG', 'CLASS']
    if any(noise in filename_upper for noise in noise_keywords):
        return None
    
    # 对每个类别进行匹配分析
    match_details = []
    
    for class_name in class_names:
        class_upper = class_name.upper()
        max_similarity = 0.0
        best_match_type = ""
        
        # 1. 精确匹配（最高优先级）
        if filename_upper == class_upper:
            max_similarity = 1.0
            best_match_type = "精确匹配"
        
        # 2. 包含关系检查
        elif class_upper in filename_upper:
            precision = len(class_upper) / len(filename_upper)
            max_similarity = 0.95 * precision
            best_match_type = f"包含匹配(精确度:{precision:.3f})"
        
        elif filename_upper in class_upper:
            precision = len(filename_upper) / len(class_upper)
            max_similarity = 0.85 * precision
            best_match_type = f"反向包含匹配(精确度:{precision:.3f})"
        
        else:
            # 3. 字符串相似度
            full_similarity = SequenceMatcher(None, filename_upper, class_upper).ratio()
            max_similarity = full_similarity
            best_match_type = f"相似度匹配({full_similarity:.3f})"
        
        match_details.append({
            'class_name': class_name,
            'similarity': max_similarity,
            'match_type': best_match_type,
            'length_diff': abs(len(class_upper) - len(filename_upper))
        })
    
    # 排序：相似度优先，然后是长度差异最小的
    match_details.sort(key=lambda x: (-x['similarity'], x['length_diff']))
    
    if match_details:
        best_match = match_details[0]
        if best_match['similarity'] >= 0.4:  # 合理的阈值
            return best_match['class_name'], match_details  # 返回详细信息用于分析
        else:
            return None, match_details  # 没有达到阈值
    
    return None, []  # 没有任何匹配


def test_mixed_length_scenario():
    """测试长短不一的类名场景"""
    print("=" * 80)
    print("长短不一类名识别测试")
    print("=" * 80)
    
    # 你的场景：A_B 和 D_E_F 长短不一
    class_names = ["A_B", "D_E_F", "Short", "Very_Long_Class_Name"]
    
    test_cases = [
        # 短类名测试
        ("A_B (1)", "A_B"),
        ("A_B_001", "A_B"),
        ("A_B_image", "A_B"),
        
        # 中等长度类名测试
        ("D_E_F (1)", "D_E_F"),
        ("D_E_F_002", "D_E_F"),
        ("D_E_F_sample", "D_E_F"),
        
        # 其他长度测试
        ("Short_003", "Short"),
        ("Very_Long_Class_Name_004", "Very_Long_Class_Name"),
    ]
    
    print(f"类别列表: {class_names}")
    print(f"测试用例: {len(test_cases)} 个")
    print()
    
    correct = 0
    total = len(test_cases)
    
    for filename, expected in test_cases:
        result, details = smart_class_matching(filename, class_names)
        
        is_correct = result == expected
        status = "✅" if is_correct else "❌"
        if is_correct:
            correct += 1
        
        expected_str = expected if expected else "None"
        result_str = result if result else "None"
        
        print(f"{status} {filename:<25} -> 期望: {expected_str:<20} | 实际: {result_str}")
        
        # 如果匹配错误，显示详细的匹配分析
        if not is_correct and details:
            print(f"    详细分析:")
            for i, detail in enumerate(details[:3]):  # 显示前3个候选
                print(f"      {i+1}. {detail['class_name']:<20} 相似度: {detail['similarity']:.3f} ({detail['match_type']}) 长度差: {detail['length_diff']}")
            print()
    
    print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")
    
    return correct/total


def test_problematic_cases():
    """测试当前有问题的案例"""
    print("=" * 80)
    print("问题案例分析")
    print("=" * 80)
    
    # 重现当前的问题
    class_names = ["A_B", "A_B_C", "A_B_C_D"]
    
    test_cases = [
        ("A_B (1)", "A_B"),          # 当前错误匹配到 A_B_C
        ("A_B_C (1)", "A_B_C"),      # 当前错误匹配到 A_B_C_D
    ]
    
    print(f"类别列表: {class_names}")
    print("问题分析:")
    print()
    
    for filename, expected in test_cases:
        result, details = smart_class_matching(filename, class_names)
        
        is_correct = result == expected
        status = "✅" if is_correct else "❌"
        
        expected_str = expected if expected else "None"
        result_str = result if result else "None"
        
        print(f"{status} {filename:<15} -> 期望: {expected_str:<10} | 实际: {result_str}")
        print(f"    所有候选匹配:")
        
        for i, detail in enumerate(details):
            marker = "👈 选中" if detail['class_name'] == result else ""
            print(f"      {i+1}. {detail['class_name']:<10} 相似度: {detail['similarity']:.3f} 长度差: {detail['length_diff']} {detail['match_type']} {marker}")
        print()


if __name__ == "__main__":
    # 先测试问题案例
    test_problematic_cases()
    
    # 再测试混合长度场景
    accuracy = test_mixed_length_scenario()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    if accuracy >= 0.9:
        print("🎉 算法在混合长度场景下表现优秀")
    elif accuracy >= 0.8:
        print("✅ 算法在混合长度场景下表现良好")
    else:
        print("⚠️ 算法在混合长度场景下需要改进")
        print("主要问题：偏向选择更长的类名，而不是最精确的匹配") 