"""
LLM Model Adapters

This module provides adapters for different LLM models, enabling
a unified interface for various language model providers.
"""

import json
import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class LLMAdapter(ABC):
    """LLM适配器基类"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.request_count = 0
        self.total_tokens = 0
        
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """生成响应"""
        pass
    
    @abstractmethod
    def analyze_metrics(self, metrics_data: Dict) -> str:
        """分析训练指标"""
        pass
    
    def get_stats(self) -> Dict:
        """获取使用统计"""
        return {
            'model_name': self.model_name,
            'request_count': self.request_count,
            'total_tokens': self.total_tokens
        }


class OpenAIAdapter(LLMAdapter):
    """OpenAI模型适配器"""
    
    def __init__(self, api_key: str, model: str = 'gpt-4', base_url: Optional[str] = None):
        super().__init__(model)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        
        # 尝试导入OpenAI库
        try:
            import openai
            if base_url:
                self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = openai.OpenAI(api_key=api_key)
            self.available = True
        except ImportError:
            print("警告: OpenAI库未安装，将使用HTTP请求方式")
            self.client = None
            self.available = False
    
    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """生成响应"""
        try:
            self.request_count += 1
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            if context:
                context_msg = f"上下文信息: {json.dumps(context, ensure_ascii=False)}"
                messages.insert(1, {"role": "assistant", "content": context_msg})
            
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                content = response.choices[0].message.content
                self.total_tokens += response.usage.total_tokens
                return content
            else:
                # 使用HTTP请求方式
                return self._http_request(messages)
                
        except Exception as e:
            return f"OpenAI API调用失败: {str(e)}"
    
    def analyze_metrics(self, metrics_data: Dict) -> str:
        """分析训练指标"""
        prompt = f"""
请分析以下CV模型训练指标并提供专业建议:

训练指标:
{json.dumps(metrics_data, ensure_ascii=False, indent=2)}

请分析:
1. 当前训练状态 (收敛情况、过拟合/欠拟合)
2. 关键指标趋势
3. 具体的优化建议
4. 潜在问题诊断

请用中文回答，并保持专业性。
"""
        return self.generate_response(prompt)
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """
你是一个专业的深度学习训练分析师。你的任务是:
1. 分析训练指标数据，识别训练状态
2. 提供具体的优化建议
3. 诊断常见的训练问题
4. 使用专业但易懂的语言解释

请始终基于提供的数据进行分析，避免过度推测。
回答要简洁明了，重点突出，用中文回答。
"""
    
    def _http_request(self, messages: List[Dict]) -> str:
        """使用HTTP请求方式调用API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                self.total_tokens += result.get('usage', {}).get('total_tokens', 0)
                return content
            else:
                return f"API请求失败: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"HTTP请求失败: {str(e)}"


class DeepSeekAdapter(LLMAdapter):
    """DeepSeek模型适配器"""
    
    def __init__(self, api_key: str, model: str = 'deepseek-chat', base_url: Optional[str] = None):
        super().__init__(model)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.deepseek.com/v1"
        
        # 尝试导入OpenAI库（DeepSeek使用兼容OpenAI的API格式）
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)
            self.available = True
        except ImportError:
            print("警告: OpenAI库未安装，将使用HTTP请求方式")
            self.client = None
            self.available = False
    
    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """生成响应"""
        try:
            self.request_count += 1
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            if context:
                context_msg = f"上下文信息: {json.dumps(context, ensure_ascii=False)}"
                messages.insert(1, {"role": "assistant", "content": context_msg})
            
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                content = response.choices[0].message.content
                self.total_tokens += response.usage.total_tokens
                return content
            else:
                # 使用HTTP请求方式
                return self._http_request(messages)
                
        except Exception as e:
            return f"DeepSeek API调用失败: {str(e)}"
    
    def analyze_metrics(self, metrics_data: Dict) -> str:
        """分析训练指标"""
        prompt = f"""
请分析以下CV模型训练指标并提供专业建议:

训练指标:
{json.dumps(metrics_data, ensure_ascii=False, indent=2)}

请分析:
1. 当前训练状态 (收敛情况、过拟合/欠拟合)
2. 关键指标趋势
3. 具体的优化建议
4. 潜在问题诊断

请用中文回答，并保持专业性。
"""
        return self.generate_response(prompt)
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """
你是一个专业的深度学习训练分析师。你的任务是:
1. 分析训练指标数据，识别训练状态
2. 提供具体的优化建议
3. 诊断常见的训练问题
4. 使用专业但易懂的语言解释

请始终基于提供的数据进行分析，避免过度推测。
回答要简洁明了，重点突出，用中文回答。
"""
    
    def _http_request(self, messages: List[Dict]) -> str:
        """使用HTTP请求方式调用API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                self.total_tokens += result.get('usage', {}).get('total_tokens', 0)
                return content
            else:
                return f"API请求失败: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"HTTP请求失败: {str(e)}"


class LocalLLMAdapter(LLMAdapter):
    """本地LLM适配器 (支持Ollama等)"""
    
    def __init__(self, model_name: str = 'llama2', base_url: str = 'http://localhost:11434', 
                 timeout: int = 120):
        super().__init__(model_name)
        self.base_url = base_url
        self.timeout = timeout  # 默认2分钟超时
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """检查本地LLM服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """生成响应"""
        if not self.available:
            return "本地LLM服务不可用，请检查Ollama是否运行在 " + self.base_url
        
        try:
            self.request_count += 1
            
            full_prompt = self._build_full_prompt(prompt, context)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.model_name,
                    'prompt': full_prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'num_predict': 1000
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '无响应')
            else:
                return f"本地LLM调用失败: HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            return f"本地LLM响应超时（{self.timeout}秒），请检查模型是否过大或服务器负载过高。建议：\n1. 尝试使用更小的模型\n2. 增加超时时间\n3. 检查服务器资源使用情况"
        except requests.exceptions.ConnectionError:
            return f"无法连接到本地LLM服务 ({self.base_url})，请确保Ollama服务正在运行"
        except Exception as e:
            return f"本地LLM调用异常: {str(e)}"
    
    def analyze_metrics(self, metrics_data: Dict) -> str:
        """分析训练指标"""
        prompt = f"""
作为深度学习专家，请分析以下CV模型训练指标:

{json.dumps(metrics_data, ensure_ascii=False, indent=2)}

请提供:
1. 训练状态分析
2. 性能评估
3. 优化建议
4. 问题诊断

用中文回答，保持专业性。
"""
        return self.generate_response(prompt)
    
    def _build_full_prompt(self, prompt: str, context: Optional[Dict] = None) -> str:
        """构建完整提示词"""
        system_prompt = """
你是一个专业的深度学习训练分析师。请基于提供的数据进行分析，
给出具体的建议和解决方案。回答要简洁明了，用中文回答。
"""
        
        full_prompt = system_prompt + "\n\n"
        
        if context:
            full_prompt += f"上下文: {json.dumps(context, ensure_ascii=False)}\n\n"
        
        full_prompt += f"用户问题: {prompt}"
        
        return full_prompt


class MockLLMAdapter(LLMAdapter):
    """模拟LLM适配器 (用于测试和演示)"""
    
    def __init__(self):
        super().__init__("mock-llm")
        self.available = True
    
    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """生成模拟响应"""
        self.request_count += 1
        time.sleep(1)  # 模拟网络延迟
        
        if "训练指标" in prompt or "analyze_metrics" in str(context):
            return self._generate_metrics_analysis(context)
        elif "优化建议" in prompt:
            return self._generate_optimization_suggestions()
        elif "问题诊断" in prompt:
            return self._generate_problem_diagnosis()
        else:
            return f"这是一个模拟回答。您的问题是: {prompt[:100]}..."
    
    def analyze_metrics(self, metrics_data: Dict) -> str:
        """分析训练指标"""
        return self._generate_metrics_analysis(metrics_data)
    
    def _generate_metrics_analysis(self, metrics_data: Any) -> str:
        """生成指标分析"""
        return """
## 训练状态分析

**当前状态**: 训练进展良好，模型正在稳定收敛。

**关键发现**:
1. 训练损失持续下降，显示良好的学习趋势
2. 验证准确率稳步提升，无明显过拟合迹象
3. 学习率调度合适，梯度更新稳定

**优化建议**:
1. 可以适当增加数据增强强度
2. 考虑使用余弦退火学习率调度
3. 监控验证损失，防止过拟合

**注意**: 这是模拟分析结果，实际使用时请配置真实的LLM服务。
"""
    
    def _generate_optimization_suggestions(self) -> str:
        """生成优化建议"""
        return """
## 超参数优化建议

1. **学习率调整**: 建议降低至当前值的0.8倍
2. **批量大小**: 可以尝试增加到64
3. **正则化**: 增加dropout率到0.3
4. **数据增强**: 启用更多的变换操作

这些是基于经验的建议，请根据实际情况调整。
"""
    
    def _generate_problem_diagnosis(self) -> str:
        """生成问题诊断"""
        return """
## 训练问题诊断

**检测结果**: 未发现严重问题

**监控建议**:
1. 持续观察验证损失趋势
2. 检查GPU内存使用情况
3. 监控训练速度变化

如有异常，请及时调整参数。
"""


def create_llm_adapter(adapter_type: str, **kwargs) -> LLMAdapter:
    """工厂方法：创建LLM适配器"""
    
    if adapter_type.lower() == 'openai':
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("OpenAI适配器需要提供api_key")
        return OpenAIAdapter(
            api_key=api_key,
            model=kwargs.get('model', 'gpt-4'),
            base_url=kwargs.get('base_url')
        )
    
    elif adapter_type.lower() == 'deepseek':
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("DeepSeek适配器需要提供api_key")
        return DeepSeekAdapter(
            api_key=api_key,
            model=kwargs.get('model', 'deepseek-chat'),
            base_url=kwargs.get('base_url')
        )
    
    elif adapter_type.lower() == 'local':
        return LocalLLMAdapter(
            model_name=kwargs.get('model_name', 'llama2'),
            base_url=kwargs.get('base_url', 'http://localhost:11434'),
            timeout=kwargs.get('timeout', 120)
        )
    
    elif adapter_type.lower() == 'mock':
        return MockLLMAdapter()
    
    else:
        raise ValueError(f"不支持的适配器类型: {adapter_type}")


# 使用示例
if __name__ == "__main__":
    # 测试模拟适配器
    mock_adapter = MockLLMAdapter()
    
    test_metrics = {
        "epoch": 10,
        "train_loss": 0.234,
        "val_loss": 0.287,
        "train_accuracy": 0.894,
        "val_accuracy": 0.856
    }
    
    print("=== 模拟LLM适配器测试 ===")
    result = mock_adapter.analyze_metrics(test_metrics)
    print(result)
    print(f"\n统计信息: {mock_adapter.get_stats()}") 