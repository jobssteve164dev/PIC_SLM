"""
Markdown渲染器组件

功能特性：
- 将Markdown文本转换为HTML格式
- 支持基本的Markdown语法（标题、列表、代码块、链接等）
- 自定义样式和主题
- 安全的HTML输出
"""

import re
import html
from typing import Optional


class MarkdownRenderer:
    """Markdown渲染器，将Markdown文本转换为HTML"""
    
    def __init__(self):
        # 定义Markdown语法规则
        self.rules = [
            # 代码块 (```code```)
            (r'```([\s\S]*?)```', self._render_code_block),
            # 行内代码 (`code`)
            (r'`([^`]+)`', self._render_inline_code),
            # 标题 (# ## ###)
            (r'^(#{1,6})\s+(.+)$', self._render_heading, re.MULTILINE),
            # 粗体 (**text** 或 __text__)
            (r'\*\*(.+?)\*\*', self._render_bold),
            (r'__(.+?)__', self._render_bold),
            # 斜体 (*text* 或 _text_)
            (r'\*(.+?)\*', self._render_italic),
            (r'_(.+?)_', self._render_italic),
            # 链接 [text](url)
            (r'\[([^\]]+)\]\(([^)]+)\)', self._render_link),
            # 无序列表 (- item)
            (r'^[-*+]\s+(.+)$', self._render_unordered_list, re.MULTILINE),
            # 有序列表 (1. item)
            (r'^\d+\.\s+(.+)$', self._render_ordered_list, re.MULTILINE),
            # 水平分割线 (--- 或 ***)
            (r'^[-*]{3,}$', self._render_hr, re.MULTILINE),
            # 换行符
            (r'\n', self._render_line_break),
        ]
        
        # 编译正则表达式以提高性能
        self.compiled_rules = []
        for rule in self.rules:
            if len(rule) == 3:
                pattern, handler, flags = rule
                self.compiled_rules.append((re.compile(pattern, flags), handler))
            else:
                pattern, handler = rule
                self.compiled_rules.append((re.compile(pattern), handler))
    
    def render(self, markdown_text: str) -> str:
        """将Markdown文本渲染为HTML"""
        if not markdown_text:
            return ""
        
        # 转义HTML特殊字符
        html_text = html.escape(markdown_text)
        
        # 应用Markdown规则
        for pattern, handler in self.compiled_rules:
            html_text = pattern.sub(handler, html_text)
        
        # 处理段落
        html_text = self._render_paragraphs(html_text)
        
        return html_text
    
    def _render_code_block(self, match) -> str:
        """渲染代码块"""
        code = match.group(1).strip()
        escaped_code = html.escape(code)
        return f'<pre style="background-color: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin: 10px 0; overflow-x: auto; font-family: \'Consolas\', \'Monaco\', \'Courier New\', monospace; font-size: 13px; line-height: 1.45; color: #24292e;"><code>{escaped_code}</code></pre>'
    
    def _render_inline_code(self, match) -> str:
        """渲染行内代码"""
        code = match.group(1)
        escaped_code = html.escape(code)
        return f'<code style="background-color: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 3px; padding: 2px 4px; font-family: \'Consolas\', \'Monaco\', \'Courier New\', monospace; font-size: 85%; color: #e36209;">{escaped_code}</code>'
    
    def _render_heading(self, match) -> str:
        """渲染标题"""
        level = len(match.group(1))
        text = match.group(2)
        return f'<h{level} style="margin: 16px 0 8px 0; font-weight: 600; line-height: 1.25; color: #24292e;">{text}</h{level}>'
    
    def _render_bold(self, match) -> str:
        """渲染粗体"""
        text = match.group(1)
        return f'<strong style="font-weight: 600;">{text}</strong>'
    
    def _render_italic(self, match) -> str:
        """渲染斜体"""
        text = match.group(1)
        return f'<em style="font-style: italic;">{text}</em>'
    
    def _render_link(self, match) -> str:
        """渲染链接"""
        text = match.group(1)
        url = match.group(2)
        return f'<a href="{url}" style="color: #0366d6; text-decoration: none;">{text}</a>'
    
    def _render_unordered_list(self, match) -> str:
        """渲染无序列表"""
        text = match.group(1)
        return f'<li style="margin: 4px 0;">{text}</li>'
    
    def _render_ordered_list(self, match) -> str:
        """渲染有序列表"""
        text = match.group(1)
        return f'<li style="margin: 4px 0;">{text}</li>'
    
    def _render_hr(self, match) -> str:
        """渲染水平分割线"""
        return '<hr style="border: none; border-top: 1px solid #e1e4e8; margin: 16px 0;">'
    
    def _render_line_break(self, match) -> str:
        """渲染换行符"""
        return '<br>'
    
    def _render_paragraphs(self, html_text: str) -> str:
        """处理段落"""
        # 将连续的<li>标签包装在<ul>中
        html_text = re.sub(
            r'(<li[^>]*>.*?</li>)+',
            lambda m: f'<ul style="margin: 8px 0; padding-left: 20px;">{m.group(0)}</ul>',
            html_text,
            flags=re.DOTALL
        )
        
        # 将连续的文本行包装在<p>标签中
        lines = html_text.split('<br>')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append('<p style="margin: 8px 0; line-height: 1.6;">' + 
                                    '<br>'.join(current_paragraph) + '</p>')
                    current_paragraph = []
            elif line.startswith('<h') or line.startswith('<pre') or line.startswith('<ul') or line.startswith('<hr'):
                if current_paragraph:
                    paragraphs.append('<p style="margin: 8px 0; line-height: 1.6;">' + 
                                    '<br>'.join(current_paragraph) + '</p>')
                    current_paragraph = []
                paragraphs.append(line)
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append('<p style="margin: 8px 0; line-height: 1.6;">' + 
                            '<br>'.join(current_paragraph) + '</p>')
        
        return '\n'.join(paragraphs)


# 全局渲染器实例
_markdown_renderer = None


def get_markdown_renderer() -> MarkdownRenderer:
    """获取全局Markdown渲染器实例"""
    global _markdown_renderer
    if _markdown_renderer is None:
        _markdown_renderer = MarkdownRenderer()
    return _markdown_renderer


def render_markdown(markdown_text: str) -> str:
    """便捷函数：渲染Markdown文本为HTML"""
    renderer = get_markdown_renderer()
    return renderer.render(markdown_text) 