#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别对比匹配问题测试脚本

专门测试源文件夹类别与配置文件类别之间的匹配问题
"""

import sys
import os
import re
from difflib import SequenceMatcher
import tempfile
import json

def test_class_comparison_matching():
    """测试类别对比匹配问题"""
    print("=" * 80)
    print("类别对比匹配问题测试")
    print("=" * 80)
    
    # 模拟真实场景：配置文件中的类别名称
    config_class_names = [
        "Missing_hole",      # 配置文件中的标准格式
        "Mouse_bite", 
        "Open_circuit",
        "Short",
        "Spur",
        "Spurious_copper",
        "A_B",               # 复合类名
        "A_B_C",             # 三段复合类名
    ]
    
    # 模拟从源文件夹中提取的类别名称（可能格式不完全一致）
    extracted_from_filenames = [
        "MISSING_HOLE",      # 大小写不同
        "Mouse_Bite",        # 部分大小写不同
        "OPEN_CIRCUIT",      # 全大写
        "short",             # 全小写
        "Spur",              # 正确格式
        "spurious_copper",   # 小写+下划线
        "A_B",               # 正确格式
        "a_b_c",             # 小写
        "Missing",           # 部分匹配
        "Mouse",             # 部分匹配
        "Circuit",           # 部分匹配
        "A",                 # 错误提取（原本应该是A_B）
        "B",                 # 错误提取
        "Unknown_class",     # 不存在的类别
    ]
    
    print("配置文件类别列表:")
    for i, class_name in enumerate(config_class_names, 1):
        print(f"  {i}. {class_name}")
    
    print(f"\n从文件名提取的类别列表:")
    for i, class_name in enumerate(extracted_from_filenames, 1):
        print(f"  {i}. {class_name}")
    
    print("\n" + "=" * 80)
    print("匹配测试结果:")
    print("=" * 80)
    
    # 测试不同的匹配策略
    strategies = {
        "精确匹配": exact_match_strategy,
        "忽略大小写匹配": case_insensitive_match_strategy,
        "智能相似度匹配": smart_similarity_match_strategy,
        "改进的智能匹配": improved_smart_match_strategy,
    }
    
    for strategy_name, strategy_func in strategies.items():
        print(f"\n--- {strategy_name} ---")
        correct_matches = 0
        total_tests = len(extracted_from_filenames)
        
        # 预期的正确匹配（人工标注）
        expected_matches = {
            "MISSING_HOLE": "Missing_hole",
            "Mouse_Bite": "Mouse_bite",
            "OPEN_CIRCUIT": "Open_circuit",
            "short": "Short",
            "Spur": "Spur",
            "spurious_copper": "Spurious_copper",
            "A_B": "A_B",
            "a_b_c": "A_B_C",
            "Missing": "Missing_hole",  # 部分匹配应该成功
            "Mouse": "Mouse_bite",      # 部分匹配应该成功
            "Circuit": "Open_circuit",  # 部分匹配应该成功
            "A": None,                  # 应该匹配失败（太模糊）
            "B": None,                  # 应该匹配失败（太模糊）
            "Unknown_class": None,      # 应该匹配失败
        }
        
        for extracted_class in extracted_from_filenames:
            matched_class = strategy_func(extracted_class, config_class_names)
            expected_class = expected_matches.get(extracted_class)
            
            is_correct = matched_class == expected_class
            if is_correct:
                correct_matches += 1
                status = "✅"
            else:
                status = "❌"
            
            expected_str = str(expected_class) if expected_class else "None"
            matched_str = str(matched_class) if matched_class else "None"
            print(f"{status} {extracted_class:<15} -> 期望: {expected_str:<15} | 实际: {matched_str}")
        
        accuracy = (correct_matches / total_tests) * 100
        print(f"策略准确率: {accuracy:.1f}% ({correct_matches}/{total_tests})")


def exact_match_strategy(extracted_class, config_classes):
    """精确匹配策略"""
    return extracted_class if extracted_class in config_classes else None


def case_insensitive_match_strategy(extracted_class, config_classes):
    """忽略大小写匹配策略"""
    extracted_upper = extracted_class.upper()
    for config_class in config_classes:
        if config_class.upper() == extracted_upper:
            return config_class
    return None


def smart_similarity_match_strategy(extracted_class, config_classes):
    """智能相似度匹配策略（当前算法）"""
    extracted_upper = extracted_class.upper()
    best_match = None
    best_score = 0.0
    
    for config_class in config_classes:
        config_upper = config_class.upper()
        
        # 1. 完整字符串相似度
        similarity = SequenceMatcher(None, extracted_upper, config_upper).ratio()
        
        # 2. 包含关系检查
        if config_upper in extracted_upper or extracted_upper in config_upper:
            similarity = max(similarity, 0.8)
        
        # 3. 部分匹配
        extracted_parts = re.split(r'[_\-\s]+', extracted_upper)
        config_parts = re.split(r'[_\-\s]+', config_upper)
        
        for ext_part in extracted_parts:
            if ext_part and len(ext_part) >= 2:
                for conf_part in config_parts:
                    if conf_part and len(conf_part) >= 2:
                        part_similarity = SequenceMatcher(None, ext_part, conf_part).ratio()
                        similarity = max(similarity, part_similarity)
        
        if similarity > best_score:
            best_score = similarity
            best_match = config_class
    
    # 阈值设置为0.4
    return best_match if best_score >= 0.4 else None


def improved_smart_match_strategy(extracted_class, config_classes):
    """改进的智能匹配策略"""
    extracted_upper = extracted_class.upper()
    best_match = None
    best_score = 0.0
    
    # 如果提取的类别太短（单字符），直接拒绝
    if len(extracted_class.strip()) <= 1:
        return None
    
    # 如果包含明显的"噪音"关键词，直接拒绝
    noise_keywords = ['UNKNOWN', 'TEST', 'SAMPLE', 'TEMP', 'DEBUG']
    if any(noise in extracted_upper for noise in noise_keywords):
        return None
    
    for config_class in config_classes:
        config_upper = config_class.upper()
        max_similarity = 0.0
        
        # 1. 精确匹配（最高优先级）
        if extracted_upper == config_upper:
            return config_class
        
        # 2. 完整字符串相似度
        full_similarity = SequenceMatcher(None, extracted_upper, config_upper).ratio()
        max_similarity = max(max_similarity, full_similarity)
        
        # 3. 包含关系检查（高优先级）
        if config_upper in extracted_upper:
            max_similarity = max(max_similarity, 0.9)
        elif extracted_upper in config_upper:
            max_similarity = max(max_similarity, 0.85)
        
        # 4. 智能部分匹配
        extracted_parts = [p for p in re.split(r'[_\-\s\d]+', extracted_upper) if p and len(p) >= 2]
        config_parts = [p for p in re.split(r'[_\-\s\d]+', config_upper) if p and len(p) >= 2]
        
        if extracted_parts and config_parts:
            # 计算部分匹配的最高相似度
            part_similarities = []
            for ext_part in extracted_parts:
                for conf_part in config_parts:
                    part_sim = SequenceMatcher(None, ext_part, conf_part).ratio()
                    part_similarities.append(part_sim)
            
            if part_similarities:
                # 使用最高的部分相似度，但给予权重
                best_part_similarity = max(part_similarities)
                max_similarity = max(max_similarity, best_part_similarity * 0.8)  # 降权重
        
        # 5. 字符重叠度检查
        common_chars = set(extracted_upper) & set(config_upper)
        if common_chars:
            char_overlap = len(common_chars) / max(len(set(extracted_upper)), len(set(config_upper)))
            max_similarity = max(max_similarity, char_overlap * 0.6)  # 更低权重
        
        if max_similarity > best_score:
            best_score = max_similarity
            best_match = config_class
    
    # 使用更高的阈值，减少误匹配
    threshold = 0.6 if len(extracted_class) <= 3 else 0.5  # 短字符串用更高阈值
    return best_match if best_score >= threshold else None


def create_real_world_test_data():
    """创建真实世界的测试数据"""
    # 创建临时配置文件
    temp_dir = tempfile.mkdtemp()
    config_file = os.path.join(temp_dir, "config.json")
    
    config_data = {
        "default_classes": [
            "Missing_hole",
            "Mouse_bite", 
            "Open_circuit",
            "Short",
            "Spur",
            "Spurious_copper",
            "A_B",
            "A_B_C"
        ]
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n创建了测试配置文件: {config_file}")
    
    return temp_dir, config_file


if __name__ == "__main__":
    test_class_comparison_matching()
    
    print("\n" + "=" * 80)
    print("问题分析:")
    print("=" * 80)
    print("1. 精确匹配: 只有完全相同才匹配，对大小写敏感")
    print("2. 忽略大小写匹配: 解决了大小写问题，但不处理部分匹配")
    print("3. 智能相似度匹配: 当前算法，可能阈值过低导致误匹配")
    print("4. 改进的智能匹配: 提高阈值，增加噪音过滤，减少误匹配")
    print()
    print("建议:")
    print("- 对于单字符提取结果（如'A'、'B'），应该直接拒绝")
    print("- 提高相似度阈值，减少误匹配")
    print("- 增加噪音关键词过滤")
    print("- 优先使用精确匹配和包含关系匹配")
    print("=" * 80) 