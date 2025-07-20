#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终匹配算法测试 - 专注于关键问题
"""

import sys
import os
import re
from difflib import SequenceMatcher

def smart_class_matching(filename, class_names):
    """优化后的智能类别匹配算法"""
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
    
    # 特殊处理：对于高分匹配，选择最精确的
    if match_details and match_details[0]['similarity'] >= 0.8:
        high_score_matches = [m for m in match_details if m['similarity'] >= 0.8]
        
        if len(high_score_matches) > 1:
            # 去掉数字和括号，比较核心部分
            filename_clean = re.sub(r'[\d\(\)\s]+', '', filename_upper)
            
            best_candidate = None
            best_exactness = 0
            
            for match in high_score_matches:
                class_clean = re.sub(r'[\d\(\)\s]+', '', match['class_name'].upper())
                
                if filename_clean == class_clean:
                    exactness = 1.0  # 完全匹配
                elif class_clean in filename_clean:
                    exactness = 0.9 - (len(filename_clean) - len(class_clean)) * 0.05
                elif filename_clean in class_clean:
                    exactness = 0.8 - (len(class_clean) - len(filename_clean)) * 0.05
                else:
                    exactness = match['similarity'] * 0.7
                
                if exactness > best_exactness:
                    best_exactness = exactness
                    best_candidate = match
            
            if best_candidate:
                match_details[0] = best_candidate
    
    if match_details:
        best_match = match_details[0]
        if best_match['similarity'] >= 0.4:  # 合理的阈值
            return best_match['class_name']
    
    return None


def test_key_scenarios():
    """测试关键场景"""
    print("=" * 80)
    print("关键场景测试")
    print("=" * 80)
    
    class_names = [
        "Missing_hole",
        "Mouse_bite", 
        "Open_circuit",
        "Short",
        "Spur",
        "Spurious_copper",
        "A_B",
        "A_B_C",
        "X_Y_Z", 
        "A_B_C_D",
    ]
    
    # 你最关心的测试用例
    test_cases = [
        # 核心问题：复合类名匹配
        ("A_B_C (1)", "A_B_C"),        # 应该匹配A_B_C，不是A_B_C_D
        ("A_B_C_001", "A_B_C"),        # 应该匹配A_B_C
        ("A_B (1)", "A_B"),            # 应该匹配A_B
        ("A_B_C_D (1)", "A_B_C_D"),    # 应该匹配A_B_C_D
        
        # 应该被拒绝的
        ("A", None),                   # 单字符
        ("B", None),                   # 单字符
        ("unknown_class", None),       # 噪音关键词
        
        # 正常匹配
        ("Missing_hole_001", "Missing_hole"),
        ("X_Y_Z_002", "X_Y_Z"),
        ("Short_004", "Short"),
    ]
    
    print(f"类别列表: {class_names}")
    print(f"测试用例: {len(test_cases)} 个")
    print()
    
    correct = 0
    total = len(test_cases)
    
    for filename, expected in test_cases:
        result = smart_class_matching(filename, class_names)
        
        is_correct = result == expected
        status = "✅" if is_correct else "❌"
        if is_correct:
            correct += 1
        
        expected_str = expected if expected else "None"
        result_str = result if result else "None"
        
        print(f"{status} {filename:<20} -> 期望: {expected_str:<15} | 实际: {result_str}")
    
    print(f"\n准确率: {correct}/{total} = {correct/total*100:.1f}%")
    
    if correct/total >= 0.9:
        print("🎉 优秀！算法表现良好")
    elif correct/total >= 0.8:
        print("✅ 良好，基本满足需求")
    else:
        print("⚠️ 需要进一步调整")


if __name__ == "__main__":
    test_key_scenarios() 