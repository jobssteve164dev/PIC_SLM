#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的匹配算法
"""

import sys
import os
import re
from difflib import SequenceMatcher

def fixed_smart_class_matching(filename, class_names):
    """修复后的智能类别匹配算法"""
    filename_upper = filename.upper()
    
    # 如果提取的类别太短（单字符），直接拒绝
    if len(filename.strip()) <= 1:
        return None
    
    # 如果包含明显的"噪音"关键词，直接拒绝
    noise_keywords = ['UNKNOWN', 'TEST', 'SAMPLE', 'TEMP', 'DEBUG', 'CLASS']
    if any(noise in filename_upper for noise in noise_keywords):
        return None
    
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


def test_critical_cases():
    """测试关键修复案例"""
    print("=" * 80)
    print("关键修复案例测试")
    print("=" * 80)
    
    # 最关键的测试用例
    test_scenarios = [
        {
            "name": "复合类名层次测试",
            "class_names": ["A_B", "A_B_C", "A_B_C_D"],
            "test_cases": [
                ("A_B (1)", "A_B"),          # 核心问题1
                ("A_B_C (1)", "A_B_C"),      # 核心问题2
                ("A_B_C_D (1)", "A_B_C_D"),  # 应该正确
            ]
        },
        {
            "name": "长短不一类名测试",
            "class_names": ["A_B", "D_E_F", "Short", "Very_Long_Class_Name"],
            "test_cases": [
                ("A_B (1)", "A_B"),
                ("A_B_001", "A_B"),
                ("D_E_F (1)", "D_E_F"),
                ("D_E_F_002", "D_E_F"),
                ("Short_003", "Short"),
                ("Very_Long_Class_Name_004", "Very_Long_Class_Name"),
            ]
        },
        {
            "name": "边界情况测试",
            "class_names": ["Missing_hole", "Mouse_bite", "Open_circuit"],
            "test_cases": [
                ("A", None),                    # 单字符拒绝
                ("unknown_class", None),        # 噪音关键词拒绝
                ("Missing_hole_001", "Missing_hole"),
                ("Mouse_bite_test", "Mouse_bite"),
            ]
        }
    ]
    
    total_correct = 0
    total_cases = 0
    
    for scenario in test_scenarios:
        print(f"\n【{scenario['name']}】")
        print(f"类别列表: {scenario['class_names']}")
        print("-" * 60)
        
        scenario_correct = 0
        
        for filename, expected in scenario['test_cases']:
            result, details = fixed_smart_class_matching(filename, scenario['class_names'])
            
            is_correct = result == expected
            status = "✅" if is_correct else "❌"
            if is_correct:
                scenario_correct += 1
                total_correct += 1
            
            expected_str = expected if expected else "None"
            result_str = result if result else "None"
            
            print(f"{status} {filename:<20} -> 期望: {expected_str:<15} | 实际: {result_str}")
            
            # 如果匹配错误，显示前3个候选的详细分析
            if not is_correct and details:
                print(f"    匹配分析:")
                for i, detail in enumerate(details[:3]):
                    marker = "👈 选中" if detail['class_name'] == result else ""
                    print(f"      {i+1}. {detail['class_name']:<15} 相似度: {detail['similarity']:.3f} ({detail['match_type']}) {marker}")
        
        scenario_total = len(scenario['test_cases'])
        total_cases += scenario_total
        print(f"场景准确率: {scenario_correct}/{scenario_total} = {scenario_correct/scenario_total*100:.1f}%")
    
    print(f"\n" + "=" * 80)
    print(f"总体准确率: {total_correct}/{total_cases} = {total_correct/total_cases*100:.1f}%")
    
    if total_correct/total_cases >= 0.95:
        print("🎉 优秀！修复成功，算法表现出色")
    elif total_correct/total_cases >= 0.9:
        print("✅ 良好！修复基本成功")
    elif total_correct/total_cases >= 0.8:
        print("⚠️ 一般，还需要进一步调整")
    else:
        print("❌ 修复失败，需要重新设计")


if __name__ == "__main__":
    test_critical_cases() 