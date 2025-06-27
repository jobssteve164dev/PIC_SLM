from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser
import os
import json
from src.utils.config_manager import config_manager


class WeightConfigManager:
    """权重配置管理器，负责加载和显示权重配置信息"""
    
    def __init__(self):
        self._last_config_hash = None  # 用于检测配置变化
    
    def _load_config_directly(self):
        """使用集中化配置管理器加载配置"""
        try:
            config = config_manager.get_config()
            
            # 生成配置哈希值以检测变化
            if config:
                current_hash = hash(str(sorted(config.items())))
                
                if self._last_config_hash != current_hash:
                    print(f"WeightConfigManager: 配置已更新，weight_strategy = {config.get('weight_strategy', 'balanced')}")
                    self._last_config_hash = current_hash
                else:
                    print(f"WeightConfigManager: 使用缓存的配置")
                    
                return config
            else:
                print(f"WeightConfigManager: 配置管理器返回空配置")
                return {}
                
        except Exception as e:
            print(f"WeightConfigManager: 读取配置失败: {str(e)}")
            return {}
    
    def load_weight_config(self):
        """加载权重配置信息"""
        try:
            weight_info = {
                'strategy': '未配置',
                'source': '无',
                'details': '暂无权重配置信息',
                'class_weights': {},
                'found_sources': []
            }
            
            # 直接读取配置文件，避免使用缓存的配置
            config = self._load_config_directly()
            
            if config:
                # 获取权重策略和启用状态 - 与主窗口逻辑一致
                use_class_weights = config.get('use_class_weights', True)
                
                # 处理weight_strategy字段的不同格式
                weight_strategy_raw = config.get('weight_strategy', 'balanced')
                if isinstance(weight_strategy_raw, list):
                    # 如果是数组格式，取第一个非显示名的值
                    weight_strategy = weight_strategy_raw[0] if weight_strategy_raw else 'balanced'
                elif isinstance(weight_strategy_raw, str):
                    weight_strategy = weight_strategy_raw
                else:
                    weight_strategy = 'balanced'
                
                print(f"WeightConfigManager: 原始weight_strategy = {weight_strategy_raw}")
                print(f"WeightConfigManager: 解析后weight_strategy = {weight_strategy}")
                
                # 检查各种权重配置源 - 按照模型训练器中的优先级顺序
                sources_found = []
                
                # 1. 首先检查class_weights字段（设置界面格式）
                if 'class_weights' in config and config['class_weights']:
                    sources_found.append('配置文件中的class_weights')
                    weight_info['class_weights'] = config['class_weights']
                    weight_info['strategy'] = weight_strategy
                    weight_info['source'] = '配置文件(class_weights)'
                
                # 2. 如果为空，检查custom_class_weights字段（旧版格式）
                elif 'custom_class_weights' in config and config['custom_class_weights']:
                    sources_found.append('配置文件中的custom_class_weights')
                    weight_info['class_weights'] = config['custom_class_weights']
                    weight_info['strategy'] = weight_strategy
                    weight_info['source'] = '配置文件(custom_class_weights)'
                
                # 3. 如果还为空，检查外部权重配置文件
                elif 'weight_config_file' in config and config['weight_config_file']:
                    weight_file = config['weight_config_file']
                    if weight_file and os.path.exists(weight_file):
                        sources_found.append(f'外部权重文件: {os.path.basename(weight_file)}')
                        try:
                            with open(weight_file, 'r', encoding='utf-8') as wf:
                                weight_data = json.load(wf)
                            
                            # 支持多种权重文件格式 - 与模型训练器逻辑一致
                            if 'weight_config' in weight_data:
                                # 数据集评估导出格式
                                weight_info['class_weights'] = weight_data['weight_config'].get('class_weights', {})
                                weight_info['strategy'] = weight_data['weight_config'].get('weight_strategy', weight_strategy)
                            elif 'class_weights' in weight_data:
                                # 直接包含class_weights的格式
                                weight_info['class_weights'] = weight_data.get('class_weights', {})
                                weight_info['strategy'] = weight_data.get('weight_strategy', weight_strategy)
                            
                            weight_info['source'] = f'外部文件({os.path.basename(weight_file)})'
                        except Exception as e:
                            sources_found.append(f'外部权重文件读取失败: {str(e)}')
                
                # 4. 如果仍然为空，检查all_strategies配置
                elif 'all_strategies' in config and config['all_strategies']:
                    strategies = config['all_strategies']
                    sources_found.append('配置文件中的all_strategies')
                    
                    # 优先使用当前weight_strategy指定的策略，如果不存在则使用custom
                    if weight_strategy in strategies:
                        weight_info['class_weights'] = strategies[weight_strategy]
                        weight_info['strategy'] = weight_strategy
                    elif 'custom' in strategies:
                        weight_info['class_weights'] = strategies['custom']
                        weight_info['strategy'] = 'custom'
                    else:
                        # 使用第一个可用的策略
                        first_strategy = list(strategies.keys())[0]
                        weight_info['class_weights'] = strategies[first_strategy]
                        weight_info['strategy'] = first_strategy
                    
                    weight_info['source'] = 'all_strategies配置'
                
                # 检查是否启用了类别权重 - 与主窗口逻辑一致
                if not use_class_weights:
                    weight_info['strategy'] += ' (已禁用)'
                
                weight_info['found_sources'] = sources_found
                
                # 生成详细信息
                if weight_info['class_weights']:
                    details = []
                    details.append(f"权重策略: {weight_info['strategy']}")
                    details.append(f"配置源: {weight_info['source']}")
                    details.append(f"启用状态: {'启用' if use_class_weights else '禁用'}")
                    details.append(f"类别数量: {len(weight_info['class_weights'])}")
                    
                    # 显示权重范围
                    weights = list(weight_info['class_weights'].values())
                    if weights:
                        details.append(f"权重范围: {min(weights):.3f} - {max(weights):.3f}")
                        details.append(f"权重均值: {sum(weights)/len(weights):.3f}")
                    
                    details.append("")
                    details.append("类别权重详情:")
                    
                    # 显示前5个类别的权重
                    for i, (class_name, weight) in enumerate(list(weight_info['class_weights'].items())[:5]):
                        details.append(f"  {class_name}: {weight:.3f}")
                    
                    if len(weight_info['class_weights']) > 5:
                        details.append(f"  ... 还有 {len(weight_info['class_weights']) - 5} 个类别")
                    
                    details.append("")
                    details.append("注意: 此配置与实际训练时使用的配置保持同步")
                    
                    weight_info['details'] = '\n'.join(details)
                else:
                    if sources_found:
                        weight_info['details'] = f"发现权重配置源但无有效权重数据:\n" + '\n'.join(f"- {s}" for s in sources_found)
                        weight_info['details'] += f"\n\n当前权重策略: {weight_strategy}"
                        weight_info['details'] += f"\n启用状态: {'启用' if use_class_weights else '禁用'}"
                    else:
                        weight_info['details'] = f"未发现任何权重配置\n\n当前设置:\n- 权重策略: {weight_strategy}\n- 启用状态: {'启用' if use_class_weights else '禁用'}\n\n建议:\n- 在设置页面配置类别权重\n- 使用数据集评估功能生成权重配置\n- 手动编辑配置文件添加权重信息"
            
            else:
                weight_info['details'] = "配置文件不存在或无法访问"
            
            return weight_info
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'strategy': '加载失败',
                'source': '错误',
                'details': f'加载权重配置时出错: {str(e)}',
                'class_weights': {},
                'found_sources': []
            }


class WeightConfigDisplayWidget(QWidget):
    """权重配置显示控件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.weight_manager = WeightConfigManager()
        self.init_ui()
        # 初始化时自动加载权重配置
        self.refresh_weight_config()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # 减小边距
        layout.setSpacing(3)  # 减小间距
        
        # 合并权重策略和配置源到一行
        info_layout = QHBoxLayout()
        info_layout.setSpacing(5)  # 减小水平间距
        
        # 权重策略显示
        strategy_layout = QHBoxLayout()
        strategy_layout.setSpacing(2)  # 减小组件间距
        strategy_layout.addWidget(QLabel("权重策略:"))
        self.weight_strategy_label = QLabel("未配置")
        self.weight_strategy_label.setStyleSheet("color: #666; font-weight: bold;")
        self.weight_strategy_label.setToolTip("当前使用的类别权重策略")
        strategy_layout.addWidget(self.weight_strategy_label)
        strategy_layout.addStretch(1)
        info_layout.addLayout(strategy_layout, 3)  # 分配比例权重
        
        # 权重配置源显示
        source_layout = QHBoxLayout()
        source_layout.setSpacing(2)  # 减小组件间距
        source_layout.addWidget(QLabel("配置源:"))
        self.weight_source_label = QLabel("无")
        self.weight_source_label.setStyleSheet("color: #666;")
        self.weight_source_label.setToolTip("权重配置的来源")
        source_layout.addWidget(self.weight_source_label)
        source_layout.addStretch(1)
        info_layout.addLayout(source_layout, 3)  # 分配比例权重
        
        # 刷新权重配置按钮
        refresh_weight_btn = QPushButton("刷新配置")  # 缩短按钮文本
        refresh_weight_btn.setFixedWidth(80)  # 减小按钮宽度
        refresh_weight_btn.setToolTip("重新检测和加载类别权重配置")
        refresh_weight_btn.clicked.connect(self.refresh_weight_config)
        info_layout.addWidget(refresh_weight_btn, 1)  # 分配比例权重
        
        layout.addLayout(info_layout)
        
        # 权重信息文本框
        self.weight_info_text = QTextBrowser()
        self.weight_info_text.setMaximumHeight(80)  # 减小高度
        self.weight_info_text.setStyleSheet("background-color: #F8F9FA; border: 1px solid #E0E0E0; font-family: monospace; font-size: 10px;")  # 减小字体
        self.weight_info_text.setToolTip("显示详细的类别权重配置信息")
        layout.addWidget(self.weight_info_text)
    
    def refresh_weight_config(self):
        """刷新权重配置"""
        try:
            weight_info = self.weight_manager.load_weight_config()
            self.update_display(weight_info)
        except Exception as e:
            print(f"刷新权重配置时出错: {str(e)}")
    
    def update_display(self, weight_info):
        """更新显示"""
        try:
            # 更新策略标签
            strategy_text = weight_info['strategy']
            if weight_info['class_weights']:
                self.weight_strategy_label.setStyleSheet("color: #2E7D32; font-weight: bold;")  # 绿色表示有配置
            else:
                self.weight_strategy_label.setStyleSheet("color: #D32F2F; font-weight: bold;")  # 红色表示无配置
            self.weight_strategy_label.setText(strategy_text)
            
            # 更新配置源标签
            self.weight_source_label.setText(weight_info['source'])
            
            # 更新详细信息
            self.weight_info_text.setPlainText(weight_info['details'])
            
        except Exception as e:
            print(f"更新权重显示时出错: {str(e)}") 