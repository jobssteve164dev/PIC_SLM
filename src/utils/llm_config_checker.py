"""
LLM配置检查工具

用于检查智能训练系统的LLM配置是否正确，
并提供详细的诊断信息。
"""

import os
import json
from typing import Dict, Any, List, Tuple


class LLMConfigChecker:
    """LLM配置检查器"""
    
    def __init__(self):
        self.config_files = [
            "setting/intelligent_training_config.json",
            "setting/ai_config.json"
        ]
    
    def check_all_configs(self) -> Dict[str, Any]:
        """检查所有LLM配置"""
        result = {
            'overall_status': 'unknown',
            'configs': {},
            'recommendations': [],
            'errors': [],
            'warnings': []
        }
        
        # 检查智能训练配置
        intelligent_config = self._check_intelligent_training_config()
        result['configs']['intelligent_training'] = intelligent_config
        
        # 检查AI配置
        ai_config = self._check_ai_config()
        result['configs']['ai'] = ai_config
        
        # 综合分析
        self._analyze_overall_status(result)
        
        return result
    
    def _check_intelligent_training_config(self) -> Dict[str, Any]:
        """检查智能训练配置"""
        config_file = "setting/intelligent_training_config.json"
        result = {
            'file_exists': False,
            'llm_config_exists': False,
            'adapter_type': None,
            'status': 'unknown',
            'issues': []
        }
        
        if not os.path.exists(config_file):
            result['issues'].append("智能训练配置文件不存在")
            return result
        
        result['file_exists'] = True
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            llm_config = config.get('llm_config', {})
            if llm_config:
                result['llm_config_exists'] = True
                adapter_type = llm_config.get('adapter_type', 'mock')
                result['adapter_type'] = adapter_type
                
                if adapter_type == 'mock':
                    result['status'] = 'error'
                    result['issues'].append("使用mock适配器，不适合生产环境")
                elif adapter_type in ['openai', 'deepseek', 'custom']:
                    result['status'] = 'warning'
                    result['issues'].append(f"使用{adapter_type}适配器，需要验证API配置")
                else:
                    result['status'] = 'unknown'
                    result['issues'].append(f"未知的适配器类型: {adapter_type}")
            else:
                result['status'] = 'warning'
                result['issues'].append("未找到LLM配置")
                
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"读取配置文件失败: {str(e)}")
        
        return result
    
    def _check_ai_config(self) -> Dict[str, Any]:
        """检查AI配置"""
        config_file = "setting/ai_config.json"
        result = {
            'file_exists': False,
            'adapter_type': None,
            'api_key_configured': False,
            'status': 'unknown',
            'issues': []
        }
        
        if not os.path.exists(config_file):
            result['issues'].append("AI配置文件不存在")
            return result
        
        result['file_exists'] = True
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            adapter_type = config.get('adapter_type', 'mock')
            result['adapter_type'] = adapter_type
            
            api_key = config.get('api_key', '')
            result['api_key_configured'] = bool(api_key and api_key.strip())
            
            if adapter_type == 'mock':
                result['status'] = 'error'
                result['issues'].append("使用mock适配器，不适合生产环境")
            elif adapter_type == 'openai':
                if not result['api_key_configured']:
                    result['status'] = 'error'
                    result['issues'].append("OpenAI适配器缺少API密钥")
                else:
                    result['status'] = 'ok'
            elif adapter_type == 'deepseek':
                if not result['api_key_configured']:
                    result['status'] = 'error'
                    result['issues'].append("DeepSeek适配器缺少API密钥")
                else:
                    result['status'] = 'ok'
            elif adapter_type == 'custom':
                base_url = config.get('base_url', '')
                if not result['api_key_configured'] or not base_url:
                    result['status'] = 'error'
                    result['issues'].append("自定义适配器缺少API密钥或基础URL")
                else:
                    result['status'] = 'ok'
            else:
                result['status'] = 'unknown'
                result['issues'].append(f"未知的适配器类型: {adapter_type}")
                
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"读取配置文件失败: {str(e)}")
        
        return result
    
    def _analyze_overall_status(self, result: Dict[str, Any]):
        """分析整体状态"""
        errors = []
        warnings = []
        recommendations = []
        
        # 收集所有错误和警告
        for config_name, config_data in result['configs'].items():
            if config_data['status'] == 'error':
                errors.extend(config_data['issues'])
            elif config_data['status'] == 'warning':
                warnings.extend(config_data['issues'])
        
        # 生成建议
        if errors:
            result['overall_status'] = 'error'
            recommendations.append("请修复上述错误配置")
        elif warnings:
            result['overall_status'] = 'warning'
            recommendations.append("请检查上述警告配置")
        else:
            result['overall_status'] = 'ok'
            recommendations.append("配置看起来正常")
        
        # 添加具体建议
        if any('mock' in str(issue).lower() for issue in errors + warnings):
            recommendations.append("建议在智能训练设置中选择真实的LLM服务（如OpenAI、DeepSeek等）")
        
        if any('api_key' in str(issue).lower() for issue in errors):
            recommendations.append("请在AI设置中配置正确的API密钥")
        
        result['errors'] = errors
        result['warnings'] = warnings
        result['recommendations'] = recommendations
    
    def generate_report(self) -> str:
        """生成检查报告"""
        result = self.check_all_configs()
        
        report = []
        report.append("=" * 60)
        report.append("LLM配置检查报告")
        report.append("=" * 60)
        
        # 整体状态
        status_emoji = {
            'ok': '✅',
            'warning': '⚠️',
            'error': '❌',
            'unknown': '❓'
        }
        emoji = status_emoji.get(result['overall_status'], '❓')
        report.append(f"{emoji} 整体状态: {result['overall_status'].upper()}")
        
        # 详细配置信息
        for config_name, config_data in result['configs'].items():
            report.append(f"\n📁 {config_name.upper()} 配置:")
            report.append(f"  文件存在: {'是' if config_data['file_exists'] else '否'}")
            if config_data.get('adapter_type'):
                report.append(f"  适配器类型: {config_data['adapter_type']}")
            if config_data.get('api_key_configured') is not None:
                report.append(f"  API密钥配置: {'是' if config_data['api_key_configured'] else '否'}")
            
            if config_data['issues']:
                for issue in config_data['issues']:
                    report.append(f"  ⚠️ {issue}")
        
        # 错误和警告
        if result['errors']:
            report.append(f"\n🚨 错误 ({len(result['errors'])}):")
            for error in result['errors']:
                report.append(f"  ❌ {error}")
        
        if result['warnings']:
            report.append(f"\n⚠️ 警告 ({len(result['warnings'])}):")
            for warning in result['warnings']:
                report.append(f"  ⚠️ {warning}")
        
        # 建议
        if result['recommendations']:
            report.append(f"\n💡 建议:")
            for rec in result['recommendations']:
                report.append(f"  💡 {rec}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def check_llm_config() -> Tuple[bool, str]:
    """检查LLM配置的便捷函数"""
    checker = LLMConfigChecker()
    result = checker.check_all_configs()
    report = checker.generate_report()
    return result['overall_status'] in ['ok', 'warning'], report


if __name__ == "__main__":
    # 运行检查
    success, report = check_llm_config()
    print(report)
    
    if not success:
        exit(1)
