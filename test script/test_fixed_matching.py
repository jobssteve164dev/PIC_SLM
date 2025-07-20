#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的匹配算法
"""

import sys
import os
import re
from difflib import SequenceMatcher

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class FixedMatcher:
    """修复后的匹配器"""
    
    def __init__(self, class_names):
        self.class_names = class_names or []
    
    def _smart_class_matching(self, filename, class_names):
        """修复后的智能类别匹配算法 - 最终优化版本"""
        filename_upper = filename.upper()
        best_match = None
        best_score = 0.0
        match_details = []
        
        # 如果提取的类别太短（单字符），直接拒绝
        if len(filename.strip()) <= 1:
            print(f"单字符类别提取被拒绝: {filename}")
            return None
        
        # 如果包含明显的"噪音"关键词，直接拒绝
        noise_keywords = ['UNKNOWN', 'TEST', 'SAMPLE', 'TEMP', 'DEBUG', 'CLASS']
        if any(noise in filename_upper for noise in noise_keywords):
            print(f"噪音关键词类别被拒绝: {filename}")
            return None
        
        # 对每个类别进行全面的相似度分析
        for class_name in class_names:
            class_upper = class_name.upper()
            max_similarity = 0.0
            best_match_type = ""
            
            # 1. 精确匹配（最高优先级）
            if filename_upper == class_upper:
                max_similarity = 1.0
                best_match_type = f"精确匹配"
            
            # 2. 包含关系检查 - 优化精确度计算
            elif class_upper in filename_upper:
                # 计算精确匹配度，避免短类别匹配长文件名
                precision = len(class_upper) / len(filename_upper)
                # 只有当类别占文件名的比例足够大时才给高分
                if precision >= 0.5:  # 类别至少要占文件名的50%
                    max_similarity = 0.95 * precision
                    best_match_type = f"包含匹配(精确度:{precision:.3f})"
                else:
                    # 如果比例太小，降低分数
                    max_similarity = 0.7 * precision
                    best_match_type = f"部分包含匹配(精确度:{precision:.3f})"
            
            elif filename_upper in class_upper:
                # 反向包含：文件名在类别名中
                precision = len(filename_upper) / len(class_upper)
                # 只有当文件名占类别的比例足够大时才给高分
                if precision >= 0.7:  # 文件名至少要占类别的70%
                    max_similarity = 0.90 * precision
                    best_match_type = f"反向包含匹配(精确度:{precision:.3f})"
                else:
                    max_similarity = 0.6 * precision
                    best_match_type = f"部分反向包含匹配(精确度:{precision:.3f})"
            
            else:
                # 3. 完整字符串相似度
                full_similarity = SequenceMatcher(None, filename_upper, class_upper).ratio()
                if full_similarity > max_similarity:
                    max_similarity = full_similarity
                    best_match_type = f"完整匹配(相似度:{full_similarity:.3f})"
                
                # 4. 智能部分匹配
                filename_parts = [p for p in re.split(r'[_\-\s\d\(\)]+', filename_upper) if p and len(p) >= 2]
                class_parts = [p for p in re.split(r'[_\-\s\d]+', class_upper) if p and len(p) >= 2]
                
                if filename_parts and class_parts:
                    # 计算部分匹配的最高相似度
                    part_similarities = []
                    for file_part in filename_parts:
                        for class_part in class_parts:
                            part_sim = SequenceMatcher(None, file_part, class_part).ratio()
                            part_similarities.append(part_sim)
                    
                    if part_similarities:
                        # 使用最高的部分相似度，但给予权重
                        best_part_similarity = max(part_similarities)
                        weighted_similarity = best_part_similarity * 0.7  # 降权重
                        if weighted_similarity > max_similarity:
                            max_similarity = weighted_similarity
                            best_match_type = f"部分匹配(相似度:{best_part_similarity:.3f})"
            
            # 记录匹配详情
            match_details.append({
                'class_name': class_name,
                'similarity': max_similarity,
                'match_type': best_match_type,
                'length_diff': abs(len(class_upper) - len(filename_upper))  # 长度差异
            })
        
        # 优化排序策略：相似度优先，然后是长度差异最小的
        match_details.sort(key=lambda x: (-x['similarity'], x['length_diff']))
        
        if match_details:
            best_match_detail = match_details[0]
            best_score = best_match_detail['similarity']
            best_match = best_match_detail['class_name']
            
            # 使用更高的阈值，减少误匹配
            threshold = 0.65 if len(filename) <= 3 else 0.55  # 提高阈值
            
            if best_score >= threshold:
                return best_match
        
        return None
    
    def extract_class_from_filename(self, filename):
        """从文件名中提取类别信息"""
        name_without_ext = os.path.splitext(filename)[0]
        
        if self.class_names:
            return self._smart_class_matching(name_without_ext, self.class_names)
        
        return None


def test_fixed_matching():
    """测试修复后的匹配算法"""
    print("=" * 80)
    print("修复后的匹配算法测试")
    print("=" * 80)
    
    # 复杂的类别列表
    complex_class_names = [
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
        "Very_long_class_name",
    ]
    
    # 关键测试用例（之前失败的）
    critical_test_cases = [
        # 复合类名问题
        ("A_B_C (1).jpg", "A_B_C"),       # 之前错误匹配到A_B
        ("A_B_C (2).jpg", "A_B_C"),       # 之前错误匹配到A_B
        ("A_B_C_001.jpg", "A_B_C"),       # 之前错误匹配到A_B
        ("A_B_C_D (1).jpg", "A_B_C_D"),   # 之前错误匹配到A_B
        ("A_B_C_D_001.jpg", "A_B_C_D"),   # 之前错误匹配到A_B
        
        # 单字符问题
        ("A.jpg", None),                  # 应该被拒绝
        ("B.jpg", None),                  # 应该被拒绝
        
        # 噪音关键词
        ("unknown_class.jpg", None),      # 应该被拒绝
        ("test_sample.jpg", None),        # 应该被拒绝
        
        # 正常情况
        ("A_B (1).jpg", "A_B"),          # 应该正确
        ("Missing_hole_001.jpg", "Missing_hole"),  # 应该正确
        ("X_Y_Z_002.jpg", "X_Y_Z"),      # 应该正确
    ]
    
    print(f"类别列表: {complex_class_names}")
    print(f"关键测试用例数量: {len(critical_test_cases)}")
    
    # 创建修复后的匹配器
    matcher = FixedMatcher(complex_class_names)
    
    print("\n" + "=" * 80)
    print("关键测试结果:")
    print("=" * 80)
    
    # 测试每个关键用例
    correct_matches = 0
    total_tests = len(critical_test_cases)
    
    for filename, expected_class in critical_test_cases:
        # 使用修复后的匹配算法
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
        print(f"{status} | 文件: {filename:<25} | 期望: {expected_str:<15} | 实际: {matched_str}")
    
    print("\n" + "=" * 80)
    print("测试统计:")
    print("=" * 80)
    print(f"总测试数: {total_tests}")
    print(f"正确匹配: {correct_matches}")
    print(f"错误匹配: {total_tests - correct_matches}")
    print(f"准确率: {correct_matches / total_tests * 100:.2f}%")
    
    # 分析改进效果
    print("\n" + "=" * 80)
    print("改进效果分析:")
    print("=" * 80)
    
    if correct_matches / total_tests >= 0.9:
        print("🎉 优秀！修复后的算法显著改善了匹配准确率")
    elif correct_matches / total_tests >= 0.8:
        print("✅ 良好！修复有效，但仍有改进空间")
    else:
        print("⚠️  需要进一步优化")
    
    return correct_matches / total_tests


if __name__ == "__main__":
    accuracy = test_fixed_matching()
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print("关键修复:")
    print("1. 精确度权重：包含匹配现在考虑精确度，A_B_C不会错误匹配到A_B")
    print("2. 单字符过滤：A、B等单字符提取结果被直接拒绝")
    print("3. 噪音过滤：unknown、test等关键词被过滤")
    print("4. 排序优化：相似度相同时，优先选择更长的类别名")
    print("5. 阈值提升：减少误匹配")
    print("=" * 80) 