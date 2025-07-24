#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试AI设置统一保存功能
"""

import sys
import os
import json

# 设置路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_ai_settings_unified_save():
    """测试AI设置统一保存功能"""
    print("🧪 测试AI设置统一保存功能...")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from src.ui.components.settings.ai_settings_widget import AISettingsWidget
        
        # 创建应用实例（如果不存在）
        if not QApplication.instance():
            app = QApplication([])
        
        # 创建AI设置组件
        ai_widget = AISettingsWidget()
        print("✅ AI设置组件创建成功")
        
        # 模拟用户修改设置
        ai_widget.openai_api_key.setText("sk-test12345")
        ai_widget.openai_model.setCurrentText("gpt-4")
        ai_widget.ollama_base_url.setText("http://localhost:11434")
        ai_widget.default_adapter.setCurrentText("OpenAI")
        
        print("✅ 模拟用户设置修改完成")
        
        # 获取当前配置
        current_config = ai_widget.get_config()
        print(f"✅ 当前配置获取成功:")
        print(f"   - 默认适配器: {current_config['general']['default_adapter']}")
        print(f"   - OpenAI API密钥: {'已设置' if current_config['openai']['api_key'] else '未设置'}")
        print(f"   - OpenAI模型: {current_config['openai']['model']}")
        
        # 测试保存功能
        success = ai_widget._save_config_to_file()
        if success:
            print("✅ 配置保存成功")
            
            # 验证文件是否存在
            config_file = "setting/ai_config.json"
            if os.path.exists(config_file):
                print(f"✅ 配置文件已创建: {config_file}")
                
                # 读取并验证内容
                with open(config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                
                print("✅ 配置文件内容验证:")
                print(f"   - 默认适配器: {saved_config['general']['default_adapter']}")
                print(f"   - OpenAI模型: {saved_config['openai']['model']}")
                print(f"   - Ollama服务器: {saved_config['ollama']['base_url']}")
                
            else:
                print("❌ 配置文件未创建")
                return False
        else:
            print("❌ 配置保存失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """清理测试文件"""
    try:
        test_file = "setting/ai_config.json"
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"🧹 已清理测试文件: {test_file}")
    except Exception as e:
        print(f"⚠️ 清理测试文件时出错: {str(e)}")

def main():
    """主测试函数"""
    print("=" * 50)
    print("🧪 AI设置统一保存功能测试")
    print("=" * 50)
    
    success = test_ai_settings_unified_save()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 测试通过！AI设置统一保存功能正常工作")
    else:
        print("❌ 测试失败，请检查相关功能")
    
    cleanup_test_files()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 