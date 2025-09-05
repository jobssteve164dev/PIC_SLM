#!/usr/bin/env python3
"""
LLM配置测试脚本

用于测试智能训练系统的LLM配置是否正确。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.llm_config_checker import check_llm_config
from src.utils.production_config_validator import validate_production_config


def main():
    """主函数"""
    print("🔍 开始检查LLM配置...")
    print()
    
    # 检查LLM配置
    print("1. 检查LLM配置:")
    print("-" * 40)
    llm_success, llm_report = check_llm_config()
    print(llm_report)
    print()
    
    # 检查生产环境配置
    print("2. 检查生产环境配置:")
    print("-" * 40)
    prod_success, prod_report = validate_production_config()
    print(prod_report)
    print()
    
    # 总结
    print("📋 检查总结:")
    print("-" * 40)
    if llm_success and prod_success:
        print("✅ 所有检查都通过了！")
        print("🎉 智能训练系统已准备好使用真实的LLM服务")
        return 0
    else:
        print("❌ 检查未通过，请修复上述问题")
        print("💡 建议:")
        print("   1. 在智能训练设置中选择真实的LLM适配器（如OpenAI、DeepSeek等）")
        print("   2. 在AI设置中配置正确的API密钥")
        print("   3. 避免在生产环境中使用mock适配器")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
