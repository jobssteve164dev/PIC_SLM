"""
LLM (Large Language Model) Integration Module

This module provides the framework for integrating large language models
into the CV training system for intelligent analysis and recommendations.

Components:
- model_adapters: LLM model adapter interfaces
- analysis_engine: Training analysis and recommendation engine
- llm_framework: Main LLM integration framework
- prompt_templates: System prompts and templates
"""

from .llm_framework import LLMFramework
from .model_adapters import LLMAdapter, OpenAIAdapter, LocalLLMAdapter
from .analysis_engine import TrainingAnalysisEngine

__all__ = [
    'LLMFramework',
    'LLMAdapter', 
    'OpenAIAdapter',
    'LocalLLMAdapter',
    'TrainingAnalysisEngine'
]

__version__ = '1.0.0' 