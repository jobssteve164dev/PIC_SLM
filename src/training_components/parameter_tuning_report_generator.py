"""
参数微调报告生成器

用于生成智能训练参数微调的Markdown格式报告
主要功能：
- 生成参数调整的详细报告
- 包含LLM分析结果
- 记录配置变更历史
- 提供训练指标对比
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ParameterTuningReport:
    """参数微调报告数据结构"""
    report_id: str
    timestamp: float
    session_id: str
    adjustment_id: str
    original_config: Dict[str, Any]
    adjusted_config: Dict[str, Any]
    changes: Dict[str, Any]
    llm_analysis: Dict[str, Any]
    training_metrics: Dict[str, Any]
    reason: str
    status: str


class ParameterTuningReportGenerator:
    """参数微调报告生成器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化报告生成器
        
        Args:
            config: 报告配置，包含保存路径、格式等设置
        """
        self.config = config or {}
        self.report_config = self.config.get('parameter_tuning_reports', {})
        
        # 默认配置
        self.default_config = {
            'enabled': True,
            'save_path': 'reports/parameter_tuning',
            'format': 'markdown',
            'include_llm_analysis': True,
            'include_metrics_comparison': True,
            'include_config_changes': True
        }
        
        # 合并配置
        for key, value in self.default_config.items():
            if key not in self.report_config:
                self.report_config[key] = value
    
    def generate_report(self, 
                       original_config: Dict[str, Any],
                       adjusted_config: Dict[str, Any],
                       changes: Dict[str, Any],
                       llm_analysis: Dict[str, Any],
                       training_metrics: Dict[str, Any],
                       reason: str = "智能参数优化",
                       session_id: str = "",
                       adjustment_id: str = "") -> str:
        """
        生成参数微调报告
        
        Args:
            original_config: 原始配置
            adjusted_config: 调整后配置
            changes: 具体变更内容
            llm_analysis: LLM分析结果
            training_metrics: 训练指标
            reason: 调整原因
            session_id: 会话ID
            adjustment_id: 调整ID
            
        Returns:
            生成的报告文件路径
        """
        try:
            if not self.report_config.get('enabled', True):
                return ""
            
            # 创建报告数据结构
            report = ParameterTuningReport(
                report_id=f"report_{int(time.time())}",
                timestamp=time.time(),
                session_id=session_id,
                adjustment_id=adjustment_id,
                original_config=original_config,
                adjusted_config=adjusted_config,
                changes=changes,
                llm_analysis=llm_analysis,
                training_metrics=training_metrics,
                reason=reason,
                status='generated'
            )
            
            # 生成报告内容
            report_content = self._generate_markdown_content(report)
            
            # 保存报告
            file_path = self._save_report(report_content, report)
            
            return file_path
            
        except Exception as e:
            print(f"[ERROR] 生成参数微调报告失败: {str(e)}")
            return ""
    
    def _generate_markdown_content(self, report: ParameterTuningReport) -> str:
        """生成Markdown格式的报告内容"""
        try:
            # 获取时间信息
            report_time = datetime.fromtimestamp(report.timestamp)
            time_str = report_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 开始构建报告内容
            content = []
            
            # 报告标题
            content.append("# 智能训练参数微调报告")
            content.append("")
            content.append(f"**生成时间**: {time_str}")
            content.append(f"**报告ID**: {report.report_id}")
            content.append(f"**会话ID**: {report.session_id}")
            content.append(f"**调整ID**: {report.adjustment_id}")
            content.append("")
            
            # 调整原因
            content.append("## 📋 调整原因")
            content.append(f"{report.reason}")
            content.append("")
            
            # 配置变更详情
            if self.report_config.get('include_config_changes', True) and report.changes:
                content.append("## 🔧 配置变更详情")
                content.append("")
                
                for param_name, change_info in report.changes.items():
                    if isinstance(change_info, dict) and 'from' in change_info and 'to' in change_info:
                        old_value = change_info['from']
                        new_value = change_info['to']
                        content.append(f"### {param_name}")
                        content.append(f"- **原始值**: `{old_value}`")
                        content.append(f"- **新值**: `{new_value}`")
                        content.append(f"- **变更类型**: {self._get_change_type(old_value, new_value)}")
                        content.append("")
                    else:
                        content.append(f"### {param_name}")
                        content.append(f"- **变更**: {change_info}")
                        content.append("")
            
            # LLM分析结果
            if self.report_config.get('include_llm_analysis', True) and report.llm_analysis:
                content.append("## 🤖 LLM分析结果")
                content.append("")
                
                if isinstance(report.llm_analysis, dict):
                    # 如果是字典格式，提取关键信息
                    if 'reason' in report.llm_analysis:
                        content.append(f"**分析原因**: {report.llm_analysis['reason']}")
                        content.append("")
                    
                    if 'analysis' in report.llm_analysis:
                        content.append("**详细分析**:")
                        content.append("")
                        content.append(report.llm_analysis['analysis'])
                        content.append("")
                    
                    if 'suggestions' in report.llm_analysis:
                        content.append("**优化建议**:")
                        content.append("")
                        for i, suggestion in enumerate(report.llm_analysis['suggestions'], 1):
                            if isinstance(suggestion, dict):
                                param = suggestion.get('parameter', '未知参数')
                                reason = suggestion.get('reason', '无说明')
                                priority = suggestion.get('priority', 'medium')
                                content.append(f"{i}. **{param}**: {reason} (优先级: {priority})")
                            else:
                                content.append(f"{i}. {suggestion}")
                        content.append("")
                else:
                    # 如果是字符串格式，直接显示
                    content.append("**分析内容**:")
                    content.append("")
                    content.append(str(report.llm_analysis))
                    content.append("")
            
            # 训练指标对比
            if self.report_config.get('include_metrics_comparison', True) and report.training_metrics:
                content.append("## 📊 训练指标")
                content.append("")
                
                # 显示当前训练指标
                content.append("### 当前训练状态")
                content.append("")
                
                metrics_table = []
                metrics_table.append("| 指标 | 数值 |")
                metrics_table.append("|------|------|")
                
                for metric_name, metric_value in report.training_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        metrics_table.append(f"| {metric_name} | {metric_value:.6f} |")
                    else:
                        metrics_table.append(f"| {metric_name} | {metric_value} |")
                
                content.extend(metrics_table)
                content.append("")
            
            # 配置对比
            content.append("## ⚙️ 配置对比")
            content.append("")
            
            # 原始配置
            content.append("### 原始配置")
            content.append("```json")
            content.append(json.dumps(report.original_config, indent=2, ensure_ascii=False))
            content.append("```")
            content.append("")
            
            # 调整后配置
            content.append("### 调整后配置")
            content.append("```json")
            content.append(json.dumps(report.adjusted_config, indent=2, ensure_ascii=False))
            content.append("```")
            content.append("")
            
            # 报告总结
            content.append("## 📝 报告总结")
            content.append("")
            content.append(f"- 本次调整共修改了 **{len(report.changes)}** 个参数")
            content.append(f"- 调整基于LLM智能分析结果")
            content.append(f"- 报告生成时间: {time_str}")
            content.append("")
            content.append("---")
            content.append("*此报告由智能训练系统自动生成*")
            
            return "\n".join(content)
            
        except Exception as e:
            print(f"[ERROR] 生成Markdown内容失败: {str(e)}")
            return f"# 参数微调报告\n\n生成报告时发生错误: {str(e)}"
    
    def _get_change_type(self, old_value: Any, new_value: Any) -> str:
        """获取变更类型"""
        try:
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if new_value > old_value:
                    return "增加"
                elif new_value < old_value:
                    return "减少"
                else:
                    return "无变化"
            else:
                return "修改"
        except:
            return "修改"
    
    def _save_report(self, content: str, report: ParameterTuningReport) -> str:
        """保存报告到文件"""
        try:
            # 确保保存目录存在
            save_path = self.report_config.get('save_path', 'reports/parameter_tuning')
            os.makedirs(save_path, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.fromtimestamp(report.timestamp)
            filename = f"parameter_tuning_report_{timestamp.strftime('%Y%m%d_%H%M%S')}_{report.adjustment_id}.md"
            file_path = os.path.join(save_path, filename)
            
            # 保存文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"[INFO] 参数微调报告已保存: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"[ERROR] 保存报告失败: {str(e)}")
            return ""
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新报告生成器配置"""
        try:
            self.config.update(new_config)
            self.report_config = self.config.get('parameter_tuning_reports', {})
            
            # 合并默认配置
            for key, value in self.default_config.items():
                if key not in self.report_config:
                    self.report_config[key] = value
            
            print(f"[INFO] 参数微调报告生成器配置已更新")
            
        except Exception as e:
            print(f"[ERROR] 更新报告生成器配置失败: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            'parameter_tuning_reports': self.report_config
        }
