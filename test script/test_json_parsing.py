#!/usr/bin/env python3
"""
测试JSON解析功能
"""

import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_json_parsing():
    """测试JSON解析功能"""
    print("🔍 测试JSON解析功能...")
    
    # 模拟LLM返回的文本
    llm_response = """
### 1. 配置评估

当前配置整体合理，适合图像分类任务，但存在一些潜在问题：
- **学习率偏高**：MobileNetV2 使用预训练权重时，初始学习率 0.001 可能偏大，容易导致训练初期不稳定
- **类别权重配置**：所有类别权重均为 1.0，与 `use_class_weights: true` 冲突，实际未启用类别平衡

### 2. 训练状态分析

**当前状态**：训练初期表现异常，存在严重欠拟合
- **验证损失极高**（1.80），**准确率极低**（16.7%），远低于随机猜测水平
- 可能原因：学习率过高导致梯度更新过大，模型无法有效学习特征

### 3. 优化建议

```json
{
    "suggestions": [
        {
            "parameter": "learning_rate",
            "current_value": 0.001,
            "suggested_value": 0.0001,
            "reason": "验证损失极高表明学习率过大，建议降低10倍以确保训练稳定性",
            "priority": "high"
        },
        {
            "parameter": "use_class_weights",
            "current_value": true,
            "suggested_value": false,
            "reason": "当前类别权重均为1.0，实际未实现类别平衡，建议关闭或重新计算真实权重",
            "priority": "medium"
        }
    ]
}
```

### 4. 关键注意事项

1. **优先调整学习率**：这是当前最紧急的问题，直接影响训练收敛
2. **检查数据分布**：确认数据集标签是否正确，极低准确率可能暗示数据问题
"""
    
    try:
        from src.training_components.intelligent_config_generator import IntelligentConfigGenerator
        generator = IntelligentConfigGenerator()
        
        # 测试JSON解析
        suggestions = generator._parse_suggestions_from_text(llm_response)
        
        print(f"✅ 成功解析到 {len(suggestions)} 个建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. 参数: {suggestion.get('parameter', 'N/A')}")
            print(f"     当前值: {suggestion.get('current_value', 'N/A')}")
            print(f"     建议值: {suggestion.get('suggested_value', 'N/A')}")
            print(f"     优先级: {suggestion.get('priority', 'N/A')}")
            print(f"     原因: {suggestion.get('reason', 'N/A')}")
            print()
        
        if len(suggestions) == 2:
            print("✅ JSON解析测试通过")
        else:
            print(f"❌ JSON解析测试失败，期望2个建议，实际得到{len(suggestions)}个")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_json_parsing()
