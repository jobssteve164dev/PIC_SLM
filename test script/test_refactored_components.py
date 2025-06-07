#!/usr/bin/env python3
"""
重构组件验证脚本

用于快速验证重构后的数据集评估组件是否能正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试所有组件是否能正常导入"""
    try:
        from components.dataset_analyzers import ClassificationAnalyzer, DetectionAnalyzer, BaseDatasetAnalyzer
        from components.weight_generator import WeightGenerator
        from components.chart_manager import ChartManager
        from components.result_display_manager import ResultDisplayManager
        
        print("✓ 所有组件导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_weight_generator():
    """测试权重生成器"""
    try:
        generator = WeightGenerator()
        
        # 测试权重生成
        test_class_counts = {
            'class_a': 100,
            'class_b': 50,
            'class_c': 200
        }
        
        weights = generator.generate_weights(test_class_counts)
        assert 'balanced' in weights
        assert 'inverse' in weights
        assert 'log_inverse' in weights
        assert 'normalized' in weights
        
        # 测试推荐策略
        imbalance_ratio = max(test_class_counts.values()) / min(test_class_counts.values())
        recommended = generator.get_recommended_strategy(imbalance_ratio)
        assert recommended is not None
        
        # 测试权重验证
        valid, msg = generator.validate_weights(weights['balanced'])
        assert valid
        
        print("✓ 权重生成器测试通过")
        return True
    except Exception as e:
        print(f"✗ 权重生成器测试失败: {e}")
        return False

def test_analyzers():
    """测试分析器基础功能"""
    try:
        # 创建测试数据集路径（不需要真实存在）
        test_path = "/test/dataset"
        
        # 测试基础分析器方法
        from components.dataset_analyzers.base_analyzer import BaseDatasetAnalyzer
        
        # 创建具体分析器（不会立即验证路径）
        from components.dataset_analyzers.classification_analyzer import ClassificationAnalyzer
        from components.dataset_analyzers.detection_analyzer import DetectionAnalyzer
        
        # 测试类创建
        cls_analyzer = ClassificationAnalyzer(test_path)
        det_analyzer = DetectionAnalyzer(test_path)
        
        # 测试工具方法
        assert cls_analyzer.is_image_file("test.jpg")
        assert cls_analyzer.is_image_file("test.png")
        assert not cls_analyzer.is_image_file("test.txt")
        
        # 测试不平衡度计算
        test_counts = {'a': 100, 'b': 50}
        ratio = cls_analyzer.calculate_imbalance_ratio(test_counts)
        assert ratio == 2.0
        
        print("✓ 分析器基础功能测试通过")
        return True
    except Exception as e:
        print(f"✗ 分析器测试失败: {e}")
        return False

def test_chart_manager():
    """测试图表管理器"""
    try:
        from components.chart_manager import ChartManager
        
        # 创建图表管理器
        chart_manager = ChartManager()
        
        # 测试基础方法
        summary = chart_manager.get_chart_summary()
        assert isinstance(summary, dict)
        
        print("✓ 图表管理器测试通过")
        return True
    except Exception as e:
        print(f"✗ 图表管理器测试失败: {e}")
        return False

def test_result_display_manager():
    """测试结果显示管理器"""
    try:
        from components.result_display_manager import ResultDisplayManager
        
        # 创建结果显示管理器
        display_manager = ResultDisplayManager()
        
        # 测试基础功能（不需要真实的UI组件）
        display_manager.clear_results()
        
        print("✓ 结果显示管理器测试通过")
        return True
    except Exception as e:
        print(f"✗ 结果显示管理器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试重构后的数据集评估组件...")
    print("=" * 50)
    
    tests = [
        ("组件导入", test_imports),
        ("权重生成器", test_weight_generator),
        ("分析器", test_analyzers),
        ("图表管理器", test_chart_manager),
        ("结果显示管理器", test_result_display_manager)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n测试 {test_name}:")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过！重构成功。")
        return True
    else:
        print(f"✗ {total - passed} 个测试失败，需要修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 