"""
生产环境配置验证器

用于验证智能训练系统在生产环境中的配置是否正确，
特别是确保不会使用模拟LLM适配器。
"""

import os
import json
from typing import Dict, Any, List, Tuple


class ProductionConfigValidator:
    """生产环境配置验证器"""
    
    def __init__(self):
        self.warnings = []
        self.errors = []
    
    def validate_intelligent_training_config(self) -> Tuple[bool, List[str], List[str]]:
        """验证智能训练配置"""
        self.warnings = []
        self.errors = []
        
        # 检查智能训练配置文件
        config_file = "setting/intelligent_training_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 检查LLM配置
                llm_config = config.get('llm_config', {})
                adapter_type = llm_config.get('adapter_type', 'mock')
                
                if adapter_type == 'mock':
                    self.errors.append("❌ 智能训练配置中使用了mock LLM适配器，不适合生产环境")
                elif adapter_type == 'openai':
                    # 检查是否有API密钥
                    ai_config_file = "setting/ai_config.json"
                    if os.path.exists(ai_config_file):
                        with open(ai_config_file, 'r', encoding='utf-8') as f:
                            ai_config = json.load(f)
                        if not ai_config.get('api_key'):
                            self.warnings.append("⚠️ OpenAI适配器未配置API密钥")
                    else:
                        self.warnings.append("⚠️ 未找到AI配置文件，无法验证API密钥")
                elif adapter_type in ['deepseek', 'ollama', 'custom']:
                    self.warnings.append(f"ℹ️ 使用{adapter_type}适配器，请确保服务配置正确")
                
                # 检查其他配置
                if not config.get('enabled', False):
                    self.warnings.append("ℹ️ 智能训练功能未启用")
                
                if config.get('max_iterations', 0) > 10:
                    self.warnings.append("⚠️ 最大迭代次数设置较高，可能影响训练效率")
                
            except Exception as e:
                self.errors.append(f"❌ 读取智能训练配置文件失败: {str(e)}")
        else:
            self.warnings.append("ℹ️ 未找到智能训练配置文件，将使用默认配置")
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def validate_ai_config(self) -> Tuple[bool, List[str], List[str]]:
        """验证AI配置"""
        self.warnings = []
        self.errors = []
        
        config_file = "setting/ai_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                adapter_type = config.get('adapter_type', 'mock')
                
                if adapter_type == 'mock':
                    self.errors.append("❌ AI配置中使用了mock适配器，不适合生产环境")
                elif adapter_type == 'openai':
                    if not config.get('api_key'):
                        self.errors.append("❌ OpenAI适配器缺少API密钥")
                    if not config.get('model'):
                        self.warnings.append("⚠️ 未指定OpenAI模型")
                elif adapter_type == 'deepseek':
                    if not config.get('api_key'):
                        self.errors.append("❌ DeepSeek适配器缺少API密钥")
                elif adapter_type == 'custom':
                    if not config.get('api_key') or not config.get('base_url'):
                        self.errors.append("❌ 自定义适配器缺少API密钥或基础URL")
                
            except Exception as e:
                self.errors.append(f"❌ 读取AI配置文件失败: {str(e)}")
        else:
            self.warnings.append("ℹ️ 未找到AI配置文件")
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def validate_all_configs(self) -> Tuple[bool, List[str], List[str]]:
        """验证所有配置"""
        all_errors = []
        all_warnings = []
        
        # 验证智能训练配置
        success1, errors1, warnings1 = self.validate_intelligent_training_config()
        all_errors.extend(errors1)
        all_warnings.extend(warnings1)
        
        # 验证AI配置
        success2, errors2, warnings2 = self.validate_ai_config()
        all_errors.extend(errors2)
        all_warnings.extend(warnings2)
        
        return len(all_errors) == 0, all_errors, all_warnings
    
    def generate_report(self) -> str:
        """生成验证报告"""
        success, errors, warnings = self.validate_all_configs()
        
        report = []
        report.append("=" * 60)
        report.append("生产环境配置验证报告")
        report.append("=" * 60)
        
        if success:
            report.append("✅ 配置验证通过")
        else:
            report.append("❌ 配置验证失败")
        
        if errors:
            report.append("\n🚨 错误:")
            for error in errors:
                report.append(f"  {error}")
        
        if warnings:
            report.append("\n⚠️ 警告:")
            for warning in warnings:
                report.append(f"  {warning}")
        
        if not errors and not warnings:
            report.append("\n🎉 所有配置都符合生产环境要求")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def validate_production_config() -> Tuple[bool, str]:
    """验证生产环境配置的便捷函数"""
    validator = ProductionConfigValidator()
    success, errors, warnings = validator.validate_all_configs()
    report = validator.generate_report()
    return success, report


if __name__ == "__main__":
    # 运行验证
    success, report = validate_production_config()
    print(report)
    
    if not success:
        exit(1)
