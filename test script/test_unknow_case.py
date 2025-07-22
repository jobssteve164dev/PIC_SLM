#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试UNKNOW类别识别问题
"""

import sys
import os
import re
from difflib import SequenceMatcher

def current_smart_class_matching(filename, class_names):
    """当前的智能类别匹配算法（有问题的版本）"""
    filename_upper = filename.upper()
    
    # 如果提取的类别太短（单字符），直接拒绝
    if len(filename.strip()) <= 1:
        return None
    
    # 如果包含明显的"噪音"关键词，直接拒绝
    noise_keywords = ['UNKNOWN', 'TEST', 'SAMPLE', 'TEMP', 'DEBUG', 'CLASS']
    if any(noise in filename_upper for noise in noise_keywords):
        return None  # ❌ 这里会误判 UNKNOW
    
    # 先去掉文件名中的数字和括号，获得核心类名
    filename_core = re.sub(r'[\d\(\)\s_]+$', '', filename_upper).rstrip('_')
    
    # 对每个类别进行匹配分析
    match_details = []
    
    for class_name in class_names:
        class_upper = class_name.upper()
        max_similarity = 0.0
        best_match_type = ""
        
        # 1. 核心类名精确匹配（最高优先级）
        if filename_core == class_upper:
            max_similarity = 1.0
            best_match_type = "核心精确匹配"
        
        # 2. 完整精确匹配
        elif filename_upper == class_upper:
            max_similarity = 0.99
            best_match_type = "完整精确匹配"
        
        # 3. 核心类名包含关系检查（高优先级）
        elif class_upper == filename_core:
            max_similarity = 0.95
            best_match_type = "核心类名匹配"
        
        # 4. 传统包含关系检查
        elif class_upper in filename_upper:
            precision = len(class_upper) / len(filename_upper)
            max_similarity = 0.9 + precision * 0.05  # 0.9-0.95 之间
            best_match_type = f"包含匹配(精确度:{precision:.3f})"
        
        elif filename_upper in class_upper:
            precision = len(filename_upper) / len(class_upper)
            max_similarity = 0.85 + precision * 0.05  # 0.85-0.9 之间
            best_match_type = f"反向包含匹配(精确度:{precision:.3f})"
        
        else:
            # 5. 字符串相似度（最低优先级）
            full_similarity = SequenceMatcher(None, filename_upper, class_upper).ratio()
            max_similarity = full_similarity * 0.8  # 最高只能到0.8
            best_match_type = f"相似度匹配({full_similarity:.3f})"
        
        match_details.append({
            'class_name': class_name,
            'similarity': max_similarity,
            'match_type': best_match_type,
            'class_length': len(class_upper),
            'length_diff': abs(len(class_upper) - len(filename_upper))
        })
    
    # 排序：优先相似度，然后优先较短的类名（避免过度匹配）
    match_details.sort(key=lambda x: (-x['similarity'], x['class_length'], x['length_diff']))
    
    if match_details:
        best_match = match_details[0]
        if best_match['similarity'] >= 0.4:
            return best_match['class_name'], match_details
        else:
            return None, match_details
    
    return None, []


def fixed_smart_class_matching(filename, class_names):
    """修复后的智能类别匹配算法"""
    filename_upper = filename.upper()
    
    # 如果提取的类别太短（单字符），直接拒绝
    if len(filename.strip()) <= 1:
        return None
    
    # 修复的关键：更智能的噪音关键词过滤
    # 只有当文件名不在已知类别列表中时，才进行噪音过滤
    class_names_upper = [name.upper() for name in class_names]
    filename_core = re.sub(r'[\d\(\)\s_]+$', '', filename_upper).rstrip('_')
    
    # 如果核心类名或完整文件名在类别列表中，跳过噪音过滤
    if filename_core not in class_names_upper and filename_upper not in class_names_upper:
        # 只对明确的噪音关键词进行过滤，并且要更精确
        noise_keywords = ['UNKNOWN', 'TEST', 'SAMPLE', 'TEMP', 'DEBUG', 'CLASS']
        # 使用精确匹配而不是包含匹配，避免误判
        if filename_core in noise_keywords or filename_upper in noise_keywords:
            return None
        
        # 额外检查：如果文件名完全等于噪音关键词，才拒绝
        for noise in noise_keywords:
            if filename_upper == noise or filename_core == noise:
                return None
    
    # 对每个类别进行匹配分析
    match_details = []
    
    for class_name in class_names:
        class_upper = class_name.upper()
        max_similarity = 0.0
        best_match_type = ""
        
        # 1. 核心类名精确匹配（最高优先级）
        if filename_core == class_upper:
            max_similarity = 1.0
            best_match_type = "核心精确匹配"
        
        # 2. 完整精确匹配
        elif filename_upper == class_upper:
            max_similarity = 0.99
            best_match_type = "完整精确匹配"
        
        # 3. 核心类名包含关系检查（高优先级）
        elif class_upper == filename_core:
            max_similarity = 0.95
            best_match_type = "核心类名匹配"
        
        # 4. 传统包含关系检查
        elif class_upper in filename_upper:
            precision = len(class_upper) / len(filename_upper)
            max_similarity = 0.9 + precision * 0.05  # 0.9-0.95 之间
            best_match_type = f"包含匹配(精确度:{precision:.3f})"
        
        elif filename_upper in class_upper:
            precision = len(filename_upper) / len(class_upper)
            max_similarity = 0.85 + precision * 0.05  # 0.85-0.9 之间
            best_match_type = f"反向包含匹配(精确度:{precision:.3f})"
        
        else:
            # 5. 字符串相似度（最低优先级）
            full_similarity = SequenceMatcher(None, filename_upper, class_upper).ratio()
            max_similarity = full_similarity * 0.8  # 最高只能到0.8
            best_match_type = f"相似度匹配({full_similarity:.3f})"
        
        match_details.append({
            'class_name': class_name,
            'similarity': max_similarity,
            'match_type': best_match_type,
            'class_length': len(class_upper),
            'length_diff': abs(len(class_upper) - len(filename_upper))
        })
    
    # 排序：优先相似度，然后优先较短的类名（避免过度匹配）
    match_details.sort(key=lambda x: (-x['similarity'], x['class_length'], x['length_diff']))
    
    if match_details:
        best_match = match_details[0]
        if best_match['similarity'] >= 0.4:
            return best_match['class_name'], match_details
        else:
            return None, match_details
    
    return None, []


def test_unknow_scenarios():
    """测试UNKNOW相关场景"""
    print("=" * 80)
    print("UNKNOW类别识别测试")
    print("=" * 80)
    
    # 测试场景
    test_scenarios = [
        {
            "name": "包含UNKNOW类别的场景",
            "class_names": ["UNKNOW", "GOOD", "BAD", "DEFECT"],
            "test_cases": [
                ("UNKNOW (1)", "UNKNOW"),          # 核心问题：应该识别为UNKNOW类别
                ("UNKNOW_001", "UNKNOW"),          # 应该识别为UNKNOW类别
                ("UNKNOW_test", "UNKNOW"),         # 应该识别为UNKNOW类别
                ("GOOD_001", "GOOD"),              # 正常情况
                ("unknown_class", None),           # 真正的噪音，应该拒绝
                ("UNKNOWN_noise", None),           # 真正的噪音，应该拒绝
            ]
        },
        {
            "name": "类似噪音但实际是类别名的场景",
            "class_names": ["TEST", "SAMPLE", "CLASS_A", "DEBUG_MODE"],
            "test_cases": [
                ("TEST (1)", "TEST"),              # TEST是真实类别名
                ("SAMPLE_001", "SAMPLE"),          # SAMPLE是真实类别名
                ("CLASS_A_002", "CLASS_A"),        # CLASS_A是真实类别名
                ("DEBUG_MODE_003", "DEBUG_MODE"),  # DEBUG_MODE是真实类别名
                ("temp_file", None),               # 真正的噪音
                ("debug_log", None),               # 真正的噪音
            ]
        }
    ]
    
    print("对比测试：当前算法 vs 修复后算法")
    print("=" * 80)
    
    total_current_correct = 0
    total_fixed_correct = 0
    total_cases = 0
    
    for scenario in test_scenarios:
        print(f"\n【{scenario['name']}】")
        print(f"类别列表: {scenario['class_names']}")
        print("-" * 80)
        print(f"{'文件名':<20} {'期望':<15} {'当前算法':<15} {'修复算法':<15} {'状态'}")
        print("-" * 80)
        
        for filename, expected in scenario['test_cases']:
            # 当前算法测试
            current_result, _ = current_smart_class_matching(filename, scenario['class_names'])
            
            # 修复算法测试
            fixed_result, _ = fixed_smart_class_matching(filename, scenario['class_names'])
            
            # 评估结果
            current_correct = (current_result == expected)
            fixed_correct = (fixed_result == expected)
            
            if current_correct:
                total_current_correct += 1
            if fixed_correct:
                total_fixed_correct += 1
            total_cases += 1
            
            # 显示结果
            expected_str = expected if expected else "None"
            current_str = current_result if current_result else "None"
            fixed_str = fixed_result if fixed_result else "None"
            
            # 状态标识
            if current_correct and fixed_correct:
                status = "✅✅"  # 两个都对
            elif not current_correct and fixed_correct:
                status = "❌✅"  # 修复成功
            elif current_correct and not fixed_correct:
                status = "✅❌"  # 修复引入问题
            else:
                status = "❌❌"  # 两个都错
            
            print(f"{filename:<20} {expected_str:<15} {current_str:<15} {fixed_str:<15} {status}")
    
    print(f"\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"当前算法准确率: {total_current_correct}/{total_cases} = {total_current_correct/total_cases*100:.1f}%")
    print(f"修复算法准确率: {total_fixed_correct}/{total_cases} = {total_fixed_correct/total_cases*100:.1f}%")
    print(f"改进幅度: {total_fixed_correct - total_current_correct} 个案例")
    
    if total_fixed_correct > total_current_correct:
        print("🎉 修复成功！UNKNOW类别识别问题已解决")
    elif total_fixed_correct == total_current_correct:
        print("⚖️ 修复无害，保持原有准确率")
    else:
        print("⚠️ 修复可能引入新问题，需要进一步调整")


if __name__ == "__main__":
    test_unknow_scenarios() 