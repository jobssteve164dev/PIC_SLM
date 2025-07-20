#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实使用场景测试脚本

基于用户实际的文件命名模式和使用场景进行测试
"""

import sys
import os
import re
from difflib import SequenceMatcher
import tempfile
import shutil

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 模拟真实的智能匹配算法
class RealScenarioMatcher:
    """真实场景匹配器"""
    
    def __init__(self, class_names):
        self.class_names = class_names or []
    
    def extract_class_from_filename(self, filename):
        """从文件名中提取类别信息 - 真实场景版本"""
        # 移除文件扩展名
        name_without_ext = os.path.splitext(filename)[0]
        
        # 如果有已知的类别列表，使用智能匹配
        if self.class_names:
            return self._smart_class_matching(name_without_ext, self.class_names)
        
        # 如果没有类别列表，使用传统的模式匹配
        return self._extract_class_traditional(name_without_ext)
    
    def _smart_class_matching(self, filename, class_names):
        """智能类别匹配算法"""
        filename_upper = filename.upper()
        best_match = None
        best_score = 0.0
        
        print(f"\n分析文件名: {filename}")
        
        # 对每个类别进行全面的相似度分析
        for class_name in class_names:
            class_upper = class_name.upper()
            max_similarity = 0.0
            best_match_type = ""
            
            # 1. 完整字符串相似度
            full_similarity = SequenceMatcher(None, filename_upper, class_upper).ratio()
            if full_similarity > max_similarity:
                max_similarity = full_similarity
                best_match_type = f"完整匹配(相似度:{full_similarity:.3f})"
            
            # 2. 类别名的各个部分与文件名的相似度
            class_parts = re.split(r'[_\-\s]+', class_upper)
            for part in class_parts:
                if part and len(part) >= 2:  # 忽略过短的部分
                    part_similarity = SequenceMatcher(None, filename_upper, part).ratio()
                    if part_similarity > max_similarity:
                        max_similarity = part_similarity
                        best_match_type = f"部分匹配({part},相似度:{part_similarity:.3f})"
            
            # 3. 文件名的各个部分与类别名的相似度
            filename_parts = re.split(r'[_\-\s\d\(\)]+', filename_upper)
            for file_part in filename_parts:
                if file_part and len(file_part) >= 1:  # 对于A_B这种情况，允许单字符
                    file_part_similarity = SequenceMatcher(None, file_part, class_upper).ratio()
                    if file_part_similarity > max_similarity:
                        max_similarity = file_part_similarity
                        best_match_type = f"文件名部分匹配({file_part},相似度:{file_part_similarity:.3f})"
                    
                    # 文件名部分与类别名部分的交叉匹配
                    for class_part in class_parts:
                        if class_part and len(class_part) >= 1:
                            cross_similarity = SequenceMatcher(None, file_part, class_part).ratio()
                            if cross_similarity > max_similarity:
                                max_similarity = cross_similarity
                                best_match_type = f"交叉匹配({file_part}↔{class_part},相似度:{cross_similarity:.3f})"
            
            # 4. 包含关系检查（作为高分奖励）
            if class_upper in filename_upper:
                max_similarity = max(max_similarity, 0.9)  # 给包含关系高分
                best_match_type = f"包含匹配({class_upper} in {filename_upper})"
            elif filename_upper in class_upper:
                max_similarity = max(max_similarity, 0.85)  # 反向包含也给高分
                best_match_type = f"反向包含匹配({filename_upper} in {class_upper})"
            
            print(f"  {class_name}: {max_similarity:.3f} ({best_match_type})")
            
            # 更新最佳匹配
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = class_name
        
        # 设置阈值
        if best_score >= 0.3:  # 降低阈值以适应真实场景
            print(f"最终匹配: {filename} -> {best_match} (相似度: {best_score:.3f})")
            return best_match
        
        print(f"无法匹配: {filename} (最高相似度: {best_score:.3f})")
        return None
    
    def _extract_class_traditional(self, filename):
        """传统的类别提取算法"""
        # 处理带空格和括号的情况
        patterns = [
            # 处理 A_B (1) 格式
            r'^([A-Za-z]+)_([A-Za-z]+)\s*\(\d+\).*',  # A_B (1) -> A_B
            # 处理 A_B_001 格式  
            r'^([A-Za-z]+)_([A-Za-z]+)_\d+.*',        # A_B_001 -> A_B
            # 处理 A_B 格式
            r'^([A-Za-z]+)_([A-Za-z]+).*',            # A_B -> A_B
            # 传统格式
            r'^([A-Za-z]+)\(\d+\).*',                 # A(1) -> A
            r'^([A-Za-z]+)\d+.*',                     # A123 -> A
            r'^([A-Za-z]+)',                          # A -> A
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, filename)
            if match:
                if len(match.groups()) == 2:  # 复合类名
                    class_name = f"{match.group(1)}_{match.group(2)}".upper()
                else:  # 单一类名
                    class_name = match.group(1).upper()
                print(f"传统匹配: {filename} 使用模式 {i+1} -> {class_name}")
                return class_name
        
        print(f"传统匹配失败: {filename}")
        return None


def test_real_scenario():
    """测试真实使用场景"""
    print("=" * 80)
    print("真实使用场景测试")
    print("=" * 80)
    
    # 真实的类别列表（基于用户实际使用）
    real_class_names = [
        "Missing_hole",
        "Mouse_bite", 
        "Open_circuit",
        "Short",
        "Spur",
        "Spurious_copper",
        "A_B",  # 添加复合类名
        "C_D",  # 添加另一个复合类名
    ]
    
    # 真实的测试用例（基于用户实际文件命名）
    real_test_cases = [
        # 用户实际遇到的复合类名问题
        ("A_B (1).jpg", "A_B"),
        ("A_B (2).jpg", "A_B"), 
        ("A_B_001.jpg", "A_B"),
        ("C_D (1).jpg", "C_D"),
        ("C_D_002.jpg", "C_D"),
        
        # 传统格式
        ("Missing_hole_001.jpg", "Missing_hole"),
        ("Mouse_bite_002.jpg", "Mouse_bite"),
        ("Short_004.jpg", "Short"),
        ("Spur_005.jpg", "Spur"),
        
        # 边界情况
        ("unknown_class.jpg", None),
        ("123.jpg", None),
        
        # 拼写错误
        ("A_B(1).jpg", "A_B"),  # 没有空格
        ("AB_001.jpg", None),   # 连写可能无法匹配
    ]
    
    print(f"真实类别列表: {real_class_names}")
    print(f"测试用例数量: {len(real_test_cases)}")
    
    # 创建匹配器
    matcher = RealScenarioMatcher(real_class_names)
    
    print("\n" + "=" * 80)
    print("测试结果:")
    print("=" * 80)
    
    # 测试每个文件名
    correct_matches = 0
    total_tests = len(real_test_cases)
    
    for filename, expected_class in real_test_cases:
        # 使用智能匹配算法
        matched_class = matcher.extract_class_from_filename(filename)
        
        # 判断匹配结果
        is_correct = matched_class == expected_class
        if is_correct:
            correct_matches += 1
            status = "✅ 正确"
        else:
            status = "❌ 错误"
        
        expected_str = str(expected_class) if expected_class is not None else "None"
        matched_str = str(matched_class) if matched_class is not None else "None"
        print(f"{status} | 文件: {filename:<20} | 期望: {expected_str:<15} | 实际: {matched_str}")
    
    print("\n" + "=" * 80)
    print("测试统计:")
    print("=" * 80)
    print(f"总测试数: {total_tests}")
    print(f"正确匹配: {correct_matches}")
    print(f"错误匹配: {total_tests - correct_matches}")
    print(f"准确率: {correct_matches / total_tests * 100:.2f}%")
    
    # 分析失败原因
    print("\n" + "=" * 80)
    print("失败案例分析:")
    print("=" * 80)
    
    for filename, expected_class in real_test_cases:
        matched_class = matcher.extract_class_from_filename(filename)
        if matched_class != expected_class:
            print(f"失败: {filename}")
            print(f"  期望: {expected_class}")
            print(f"  实际: {matched_class}")
            print(f"  分析: 需要改进匹配算法")


if __name__ == "__main__":
    test_real_scenario()
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print("如果测试结果显示准确率不高，说明我们的算法确实")
    print("没有很好地适应真实使用场景，需要重新设计。")
    print("=" * 80) 