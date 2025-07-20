#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复杂复合类名测试脚本

测试A_B_C (1)、A_B_C_D (1)等更复杂的复合类名格式
"""

import sys
import os
import re
from difflib import SequenceMatcher

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class ComplexCompoundMatcher:
    """复杂复合类名匹配器"""
    
    def __init__(self, class_names):
        self.class_names = class_names or []
    
    def extract_class_from_filename(self, filename):
        """从文件名中提取类别信息"""
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
                if part and len(part) >= 1:  # 允许单字符部分
                    part_similarity = SequenceMatcher(None, filename_upper, part).ratio()
                    if part_similarity > max_similarity:
                        max_similarity = part_similarity
                        best_match_type = f"部分匹配({part},相似度:{part_similarity:.3f})"
            
            # 3. 文件名的各个部分与类别名的相似度
            filename_parts = re.split(r'[_\-\s\d\(\)]+', filename_upper)
            for file_part in filename_parts:
                if file_part and len(file_part) >= 1:
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
                max_similarity = max(max_similarity, 0.95)  # 给包含关系高分
                best_match_type = f"包含匹配({class_upper} in {filename_upper})"
            elif filename_upper in class_upper:
                max_similarity = max(max_similarity, 0.9)  # 反向包含也给高分
                best_match_type = f"反向包含匹配({filename_upper} in {class_upper})"
            
            # 5. 特殊处理：复合类名的精确匹配
            # 提取文件名中的字母部分，忽略数字和符号
            filename_letters = re.sub(r'[\d\(\)\s]+', '', filename_upper)
            if filename_letters == class_upper:
                max_similarity = 1.0
                best_match_type = f"复合类名精确匹配({filename_letters}={class_upper})"
            
            print(f"  {class_name}: {max_similarity:.3f} ({best_match_type})")
            
            # 更新最佳匹配
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = class_name
        
        # 设置阈值
        if best_score >= 0.4:  # 使用较高的阈值
            print(f"最终匹配: {filename} -> {best_match} (相似度: {best_score:.3f})")
            return best_match
        
        print(f"无法匹配: {filename} (最高相似度: {best_score:.3f})")
        return None
    
    def _extract_class_traditional(self, filename):
        """传统的类别提取算法 - 支持多段复合类名"""
        patterns = [
            # 处理 A_B_C_D (1) 格式 - 四段
            r'^([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\s*\(\d+\).*',
            # 处理 A_B_C (1) 格式 - 三段
            r'^([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\s*\(\d+\).*',
            # 处理 A_B (1) 格式 - 两段
            r'^([A-Za-z]+)_([A-Za-z]+)\s*\(\d+\).*',
            # 处理 A_B_C_D_001 格式 - 四段数字后缀
            r'^([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_\d+.*',
            # 处理 A_B_C_001 格式 - 三段数字后缀
            r'^([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_\d+.*',
            # 处理 A_B_001 格式 - 两段数字后缀
            r'^([A-Za-z]+)_([A-Za-z]+)_\d+.*',
            # 处理 A_B_C_D 格式 - 四段基本
            r'^([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+).*',
            # 处理 A_B_C 格式 - 三段基本
            r'^([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+).*',
            # 处理 A_B 格式 - 两段基本
            r'^([A-Za-z]+)_([A-Za-z]+).*',
            # 传统格式
            r'^([A-Za-z]+)\(\d+\).*',
            r'^([A-Za-z]+)\d+.*',
            r'^([A-Za-z]+)',
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, filename)
            if match:
                groups = match.groups()
                if len(groups) >= 2:  # 复合类名
                    class_name = "_".join(groups).upper()
                else:  # 单一类名
                    class_name = groups[0].upper()
                print(f"传统匹配: {filename} 使用模式 {i+1} -> {class_name}")
                return class_name
        
        print(f"传统匹配失败: {filename}")
        return None


def test_complex_compound_names():
    """测试复杂复合类名"""
    print("=" * 80)
    print("复杂复合类名测试")
    print("=" * 80)
    
    # 复杂的类别列表
    complex_class_names = [
        # 两段复合类名
        "Missing_hole",
        "Mouse_bite", 
        "Open_circuit",
        "A_B",
        "C_D",
        
        # 三段复合类名
        "A_B_C",
        "X_Y_Z", 
        "Short_open_circuit",
        "Missing_via_hole",
        
        # 四段复合类名
        "A_B_C_D",
        "Very_long_class_name",
        "Complex_defect_type_one",
        
        # 传统单段类名
        "Short",
        "Spur",
        "Spurious_copper",
    ]
    
    # 复杂的测试用例
    complex_test_cases = [
        # 三段复合类名测试
        ("A_B_C (1).jpg", "A_B_C"),
        ("A_B_C (2).jpg", "A_B_C"),
        ("A_B_C_001.jpg", "A_B_C"),
        ("X_Y_Z (1).jpg", "X_Y_Z"),
        ("X_Y_Z_002.jpg", "X_Y_Z"),
        
        # 四段复合类名测试
        ("A_B_C_D (1).jpg", "A_B_C_D"),
        ("A_B_C_D_001.jpg", "A_B_C_D"),
        ("Very_long_class_name (1).jpg", "Very_long_class_name"),
        ("Complex_defect_type_one_001.jpg", "Complex_defect_type_one"),
        
        # 混合测试
        ("Short_open_circuit (1).jpg", "Short_open_circuit"),
        ("Missing_via_hole_001.jpg", "Missing_via_hole"),
        
        # 两段复合类名（之前已测试过的）
        ("A_B (1).jpg", "A_B"),
        ("C_D_002.jpg", "C_D"),
        
        # 传统单段类名
        ("Short_004.jpg", "Short"),
        ("Spur_005.jpg", "Spur"),
        
        # 边界情况
        ("unknown_class.jpg", None),
        ("A_B_C_D_E_001.jpg", None),  # 五段，超出预期
    ]
    
    print(f"复杂类别列表: {complex_class_names}")
    print(f"测试用例数量: {len(complex_test_cases)}")
    
    # 创建匹配器
    matcher = ComplexCompoundMatcher(complex_class_names)
    
    print("\n" + "=" * 80)
    print("测试结果:")
    print("=" * 80)
    
    # 测试每个文件名
    correct_matches = 0
    total_tests = len(complex_test_cases)
    
    for filename, expected_class in complex_test_cases:
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
        print(f"{status} | 文件: {filename:<35} | 期望: {expected_str:<25} | 实际: {matched_str}")
    
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
    
    failed_cases = []
    for filename, expected_class in complex_test_cases:
        matched_class = matcher.extract_class_from_filename(filename)
        if matched_class != expected_class:
            failed_cases.append((filename, expected_class, matched_class))
            print(f"失败: {filename}")
            print(f"  期望: {expected_class}")
            print(f"  实际: {matched_class}")
    
    if not failed_cases:
        print("🎉 所有测试用例都通过了！")
    
    return correct_matches / total_tests * 100


if __name__ == "__main__":
    accuracy = test_complex_compound_names()
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    if accuracy >= 90:
        print(f"🎉 优秀！准确率 {accuracy:.2f}% 表明算法能很好地处理复杂复合类名")
    elif accuracy >= 80:
        print(f"✅ 良好！准确率 {accuracy:.2f}% 表明算法基本能处理复杂复合类名")
    elif accuracy >= 70:
        print(f"⚠️  一般！准确率 {accuracy:.2f}% 表明算法需要改进")
    else:
        print(f"❌ 较差！准确率 {accuracy:.2f}% 表明算法无法有效处理复杂复合类名")
    print("=" * 80) 