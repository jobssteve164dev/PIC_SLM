#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能类别匹配算法测试脚本

测试新的类别匹配算法是否能正确处理各种文件名格式，
特别是A_B这种复合类名的情况。
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

# 直接在这里实现测试需要的类别匹配算法，避免复杂的导入问题
class TestClassMatcher:
    """测试用的类别匹配器"""
    
    def __init__(self, class_names):
        self.class_names = class_names or []
    
    def _smart_class_matching(self, filename, class_names):
        """智能类别匹配算法 - 基于字符串相似度的真正智能匹配"""
        filename_upper = filename.upper()
        best_match = None
        best_score = 0.0
        match_details = []
        
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
            filename_parts = re.split(r'[_\-\s\d]+', filename_upper)
            for file_part in filename_parts:
                if file_part and len(file_part) >= 2:  # 忽略过短的部分
                    file_part_similarity = SequenceMatcher(None, file_part, class_upper).ratio()
                    if file_part_similarity > max_similarity:
                        max_similarity = file_part_similarity
                        best_match_type = f"文件名部分匹配({file_part},相似度:{file_part_similarity:.3f})"
                    
                    # 文件名部分与类别名部分的交叉匹配
                    for class_part in class_parts:
                        if class_part and len(class_part) >= 2:
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
            
            # 记录匹配详情
            match_details.append({
                'class_name': class_name,
                'similarity': max_similarity,
                'match_type': best_match_type
            })
            
            # 更新最佳匹配
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = class_name
        
        # 按相似度排序，用于调试
        match_details.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 记录详细的匹配分析
        print(f"文件名: {filename} 的匹配分析:")
        for detail in match_details[:3]:  # 只显示前3个最佳匹配
            print(f"  {detail['class_name']}: {detail['similarity']:.3f} ({detail['match_type']})")
        
        # 设置一个较低的阈值，因为我们现在有更精确的相似度计算
        if best_score >= 0.4:  # 40%相似度阈值
            print(f"最终匹配: {filename} -> {best_match} (相似度: {best_score:.3f})")
            return best_match
        
        # 如果所有方法都失败，返回None
        print(f"无法匹配类别: {filename} (最高相似度: {best_score:.3f}, 可用类别: {class_names})")
        return None
    
    def _extract_identifiers_from_filename(self, filename):
        """从文件名中提取可能的类别标识符"""
        identifiers = []
        
        # 提取模式
        patterns = [
            r'([A-Z][A-Z_]*[A-Z])',  # 大写字母组合，如 MOUSE_BITE
            r'([A-Z]+)',             # 连续大写字母，如 SPUR
            r'([A-Za-z]+)(?=\d)',    # 字母后跟数字，如 Missing123
            r'([A-Za-z]+)(?=_)',     # 字母后跟下划线，如 Open_
            r'([A-Za-z]+)(?=-)',     # 字母后跟连字符，如 Short-
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, filename)
            for match in matches:
                if len(match) >= 2:  # 至少2个字符
                    identifiers.append(match)
        
        # 去重并按长度排序（长的优先）
        identifiers = list(set(identifiers))
        identifiers.sort(key=len, reverse=True)
        
        return identifiers
    
    def _extract_class_from_filename_traditional(self, filename):
        """传统的类别提取算法（作为备选方案）"""
        # 尝试多种模式匹配，按优先级排序
        patterns = [
            # 模式1: 数字_类别_数字格式，如 01_spur_07 -> spur
            r'^\d+_([A-Za-z_]+?)_\d+.*',
            # 模式2: 类别_数字格式，如 spur_07 -> spur
            r'^([A-Za-z_]+?)_\d+.*',
            # 模式3: 数字_类别格式，如 01_spur -> spur
            r'^\d+_([A-Za-z_]+?).*',
            # 模式4: 字母+数字组合，如 A123, B456
            r'^([A-Za-z_]+?)\d+.*',
            # 模式5: 字母+括号数字，如 A(1), B(2)
            r'^([A-Za-z_]+?)\(\d+\).*',
            # 模式6: 字母+下划线，如 A_001, B_test
            r'^([A-Za-z_]+?)_.*',
            # 模式7: 字母+连字符，如 A-001, B-test
            r'^([A-Za-z_]+?)-.*',
            # 模式8: 字母+点，如 A.001, B.test
            r'^([A-Za-z_]+?)\..*',
            # 模式9: 纯字母开头，如 Apple123, Banana456
            r'^([A-Za-z_]+?)[\d_\-\(\)\.].*',
            # 模式10: 任何字母序列开头
            r'^([A-Za-z_]+)',
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, filename)
            if match:
                class_name = match.group(1).upper()
                # 清理类别名称（移除尾部的下划线）
                class_name = class_name.rstrip('_')
                print(f"传统匹配: {filename} 使用模式 {i+1}: {pattern} -> {class_name}")
                return class_name
        
        # 如果所有模式都不匹配，返回None
        print(f"传统匹配失败: {filename}")
        return None

    def extract_class_from_filename_smart(self, filename):
        """智能类别提取算法 - 支持字符串相似度匹配"""
        # 移除文件扩展名
        name_without_ext = os.path.splitext(filename)[0]
        
        # 如果有已知的类别列表，使用智能匹配
        if self.class_names:
            return self._smart_class_matching(name_without_ext, self.class_names)
        
        # 如果没有类别列表，使用传统的模式匹配
        return self._extract_class_from_filename_traditional(name_without_ext)


def create_test_files():
    """创建测试文件"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    source_dir = os.path.join(temp_dir, "source")
    os.makedirs(source_dir)
    
    # 测试文件名和期望的类别匹配
    test_cases = [
        # 文件名, 期望匹配的类别
        ("Missing_hole_001.jpg", "Missing_hole"),
        ("Mouse_bite_002.jpg", "Mouse_bite"),
        ("Open_circuit_003.jpg", "Open_circuit"),
        ("Short_004.jpg", "Short"),
        ("Spur_005.jpg", "Spur"),
        ("Spurious_copper_006.jpg", "Spurious_copper"),
        
        # 复合类名测试
        ("A_B_001.jpg", None),  # 这种情况应该无法匹配，需要更智能的处理
        ("Missing_123.jpg", "Missing_hole"),  # 部分匹配
        ("Mouse_test.jpg", "Mouse_bite"),  # 部分匹配
        ("OPEN_CIRCUIT_456.jpg", "Open_circuit"),  # 大小写不敏感
        ("spur_data.jpg", "Spur"),  # 小写匹配
        ("spurious_789.jpg", "Spurious_copper"),  # 部分匹配
        
        # 边界情况测试
        ("unknown_class.jpg", None),  # 无法匹配
        ("123.jpg", None),  # 纯数字
        ("test.jpg", None),  # 通用名称
        
        # 相似度匹配测试
        ("Missin_hole.jpg", "Missing_hole"),  # 拼写错误
        ("Mouse_bit.jpg", "Mouse_bite"),  # 部分相似
        ("Open_circui.jpg", "Open_circuit"),  # 部分相似
    ]
    
    # 创建测试文件
    for filename, _ in test_cases:
        filepath = os.path.join(source_dir, filename)
        with open(filepath, 'w') as f:
            f.write("test image content")
    
    return temp_dir, source_dir, test_cases


def test_smart_class_matching():
    """测试智能类别匹配算法"""
    print("=" * 80)
    print("智能类别匹配算法测试")
    print("=" * 80)
    
    # 定义测试类别
    class_names = [
        "Missing_hole",
        "Mouse_bite", 
        "Open_circuit",
        "Short",
        "Spur",
        "Spurious_copper"
    ]
    
    # 创建测试文件
    temp_dir, source_dir, test_cases = create_test_files()
    
    try:
        # 创建测试匹配器
        matcher = TestClassMatcher(class_names)
        
        print(f"测试类别列表: {class_names}")
        print(f"测试文件目录: {source_dir}")
        print("\n" + "=" * 80)
        print("测试结果:")
        print("=" * 80)
        
        # 测试每个文件名
        correct_matches = 0
        total_tests = len(test_cases)
        
        for filename, expected_class in test_cases:
            # 使用智能匹配算法
            matched_class = matcher.extract_class_from_filename_smart(filename)
            
            # 判断匹配结果
            is_correct = matched_class == expected_class
            if is_correct:
                correct_matches += 1
                status = "✅ 正确"
            else:
                status = "❌ 错误"
            
            expected_str = str(expected_class) if expected_class is not None else "None"
            matched_str = str(matched_class) if matched_class is not None else "None"
            print(f"{status} | 文件: {filename:<25} | 期望: {expected_str:<15} | 实际: {matched_str}")
        
        print("\n" + "=" * 80)
        print("测试统计:")
        print("=" * 80)
        print(f"总测试数: {total_tests}")
        print(f"正确匹配: {correct_matches}")
        print(f"错误匹配: {total_tests - correct_matches}")
        print(f"准确率: {correct_matches / total_tests * 100:.2f}%")
        
        # 详细的匹配分析
        print("\n" + "=" * 80)
        print("详细分析:")
        print("=" * 80)
        
        # 测试传统算法对比
        print("\n传统算法对比:")
        traditional_correct = 0
        
        for filename, expected_class in test_cases:
            # 使用传统匹配算法
            name_without_ext = os.path.splitext(filename)[0]
            traditional_match = matcher._extract_class_from_filename_traditional(name_without_ext)
            
            # 判断匹配结果
            is_correct = traditional_match == expected_class
            if is_correct:
                traditional_correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} | 文件: {filename:<25} | 传统算法: {traditional_match}")
        
        print(f"\n传统算法准确率: {traditional_correct / total_tests * 100:.2f}%")
        print(f"智能算法准确率: {correct_matches / total_tests * 100:.2f}%")
        print(f"提升: {(correct_matches - traditional_correct) / total_tests * 100:.2f}%")
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)
        print(f"\n清理临时文件: {temp_dir}")


def test_similarity_matching():
    """测试相似度匹配功能"""
    print("\n" + "=" * 80)
    print("相似度匹配测试")
    print("=" * 80)
    
    class_names = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
    test_filenames = [
        "Missin_hole",
        "Mouse_bit", 
        "Open_circui",
        "Sho",
        "Spu",
        "Spurious_copp"
    ]
    
    for filename in test_filenames:
        print(f"\n测试文件名: {filename}")
        similarities = []
        
        for class_name in class_names:
            similarity = SequenceMatcher(None, filename.upper(), class_name.upper()).ratio()
            similarities.append((class_name, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("相似度排序:")
        for class_name, similarity in similarities:
            print(f"  {class_name}: {similarity:.3f}")
        
        # 判断是否达到阈值
        best_match, best_score = similarities[0]
        if best_score >= 0.6:
            print(f"匹配结果: {best_match} (相似度: {best_score:.3f})")
        else:
            print(f"无匹配 (最高相似度: {best_score:.3f})")


def test_identifier_extraction():
    """测试标识符提取功能"""
    print("\n" + "=" * 80)
    print("标识符提取测试")
    print("=" * 80)
    
    # 创建测试匹配器
    matcher = TestClassMatcher([])
    
    test_filenames = [
        "A_B_001",
        "MOUSE_BITE_TEST",
        "Missing123",
        "Open_circuit_data",
        "Short-test",
        "SPUR_ANALYSIS",
        "Spurious_copper_sample"
    ]
    
    for filename in test_filenames:
        identifiers = matcher._extract_identifiers_from_filename(filename)
        print(f"文件名: {filename:<25} | 提取的标识符: {identifiers}")


if __name__ == "__main__":
    test_smart_class_matching()
    test_similarity_matching()
    test_identifier_extraction()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80) 