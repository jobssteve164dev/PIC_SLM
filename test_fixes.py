#!/usr/bin/env python3
"""
测试修复后的功能
"""

import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fixes():
    """测试修复后的功能"""
    print("🔍 测试修复后的功能...")
    
    # 测试JSON解析功能
    print("\n1. 测试JSON解析功能...")
    try:
        from src.training_components.intelligent_config_generator import IntelligentConfigGenerator
        generator = IntelligentConfigGenerator()
        
        # 模拟LLM返回的文本
        llm_response = """
### 3. 优化建议

```json
{
    "suggestions": [
        {
            "parameter": "learning_rate",
            "current_value": 0.001,
            "suggested_value": 0.0001,
            "reason": "验证损失极高表明学习率过大",
            "priority": "high"
        },
        {
            "parameter": "use_class_weights",
            "current_value": true,
            "suggested_value": false,
            "reason": "当前类别权重均为1.0，实际未实现类别平衡",
            "priority": "medium"
        }
    ]
}
```
"""
        
        suggestions = generator._parse_suggestions_from_text(llm_response)
        print(f"✅ 成功解析到 {len(suggestions)} 个建议")
        
        if len(suggestions) == 2:
            print("✅ JSON解析测试通过")
        else:
            print(f"❌ JSON解析测试失败，期望2个建议，实际得到{len(suggestions)}个")
            
    except Exception as e:
        print(f"❌ JSON解析测试失败: {str(e)}")
    
    # 测试配置应用功能
    print("\n2. 测试配置应用功能...")
    try:
        current_config = {
            'learning_rate': 0.001,
            'use_class_weights': True,
            'batch_size': 32
        }
        
        suggestions = [
            {
                'parameter': 'learning_rate',
                'suggested_value': 0.0001
            },
            {
                'parameter': 'use_class_weights',
                'suggested_value': False
            }
        ]
        
        optimized_config = generator._apply_optimization_suggestions(current_config, suggestions)
        
        print(f"✅ 原始配置: {current_config}")
        print(f"✅ 优化配置: {optimized_config}")
        
        if (optimized_config['learning_rate'] == 0.0001 and 
            optimized_config['use_class_weights'] == False and
            optimized_config['batch_size'] == 32):
            print("✅ 配置应用测试通过")
        else:
            print("❌ 配置应用测试失败")
            
    except Exception as e:
        print(f"❌ 配置应用测试失败: {str(e)}")
    
    print("\n🎯 修复测试完成")

if __name__ == "__main__":
    test_fixes()
