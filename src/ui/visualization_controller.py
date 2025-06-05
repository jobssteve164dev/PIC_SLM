import torch
import torchvision.models as models
from torchsummary import summary
import io
from contextlib import redirect_stdout
import traceback
import networkx as nx
from PyQt5.QtWidgets import QMessageBox
from .graph_builder import GraphBuilder
from .layout_algorithms import LayoutAlgorithms

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class VisualizationController:
    """可视化控制器，专门处理模型结构的可视化逻辑"""
    
    def __init__(self):
        self.graph_builder = GraphBuilder()
        self.current_graph = None
        self.max_depth = 10
        self.current_depth = 3
        
    def create_text_visualization(self, model, input_size):
        """创建文本格式的模型结构可视化"""
        try:
            # 设置设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # 检查是否为DenseNet模型
            is_densenet = self._is_densenet_model(model)
            
            if is_densenet:
                return self._create_densenet_text_summary(model, input_size)
            else:
                return self._create_standard_text_summary(model, input_size, device)
                
        except Exception as e:
            error_msg = f"创建文本可视化失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            raise Exception(error_msg)
    
    def _is_densenet_model(self, model):
        """判断是否为DenseNet模型"""
        try:
            return any(isinstance(model, getattr(models.densenet, name)) 
                      for name in dir(models.densenet) 
                      if 'DenseNet' in name and isinstance(getattr(models.densenet, name), type))
        except:
            return False
    
    def _create_densenet_text_summary(self, model, input_size):
        """创建DenseNet模型的文本摘要"""
        channels, height, width = input_size
        
        output = f"模型类型: {model.__class__.__name__}\n\n"
        output += f"特征提取层 (features):\n"
        for name, module in model.features.named_children():
            output += f"  {name}: {module}\n"
        
        output += f"\n分类器 (classifier):\n  {model.classifier}\n"
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        output += f"\n\n总参数数量: {total_params:,}\n"
        output += f"可训练参数数量: {trainable_params:,}\n"
        
        # 添加DenseNet特有信息
        additional_info = (
            f"\n\n注意: DenseNet模型结构复杂，无法使用标准方法可视化。\n"
            f"DenseNet模型的主要组成部分:\n"
            f"1. 卷积层 (Conv2d)\n"
            f"2. BatchNorm层\n"
            f"3. 密集连接块 (DenseBlock)\n"
            f"4. 转换层 (Transition)\n"
            f"5. 全局池化层\n"
            f"6. 分类器 (Linear)\n\n"
            f"输入尺寸: ({channels}, {height}, {width})"
        )
        output += additional_info
        
        return output
    
    def _create_standard_text_summary(self, model, input_size, device):
        """创建标准模型的文本摘要"""
        channels, height, width = input_size
        
        # 重定向stdout以捕获summary输出
        string_io = io.StringIO()
        with redirect_stdout(string_io):
            summary(model, input_size=(channels, height, width), device=str(device))
        
        # 获取捕获的输出
        output = string_io.getvalue()
        
        # 添加总参数数量和可训练参数数量信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        additional_info = f"\n\n总参数数量：{total_params:,}\n可训练参数数量：{trainable_params:,}"
        output += additional_info
        
        return output
    
    def create_fx_visualization(self, model):
        """创建FX可视化"""
        if not HAS_MATPLOTLIB:
            raise ImportError("需要安装matplotlib才能使用FX可视化功能")
        
        try:
            # 尝试使用FX符号跟踪
            self.current_graph, self.max_depth, dot_graph = self.graph_builder.create_fx_graph(model)
            return self.current_graph, self.max_depth, dot_graph
            
        except Exception as e:
            # 如果符号跟踪失败，使用分层结构代替
            print(f"FX符号跟踪失败，使用模型层结构替代: {str(e)}")
            self.current_graph, self.max_depth = self.graph_builder.create_hierarchical_graph(model)
            
            # 创建文本描述
            layers_text = "模型层次结构:\n"
            def format_module_info(module, prefix=""):
                nonlocal layers_text
                for name, layer in module.named_children():
                    layer_str = f"{prefix}└─ {name}: {layer.__class__.__name__}"
                    if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                        layer_str += f" ({layer.in_features} → {layer.out_features})"
                    elif hasattr(layer, "in_channels") and hasattr(layer, "out_channels"):
                        layer_str += f" ({layer.in_channels} → {layer.out_channels})"
                    layers_text += layer_str + "\n"
                    
                    if list(layer.named_children()):
                        format_module_info(layer, prefix + "  ")
            
            format_module_info(model)
            return self.current_graph, self.max_depth, layers_text
    
    def create_graph_figure(self, current_depth, layout_idx, show_types, show_params, model_name=None):
        """创建图形可视化"""
        if self.current_graph is None:
            raise ValueError("尚未创建图形，请先调用create_fx_visualization")
        
        try:
            # 创建图表
            fig = Figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
            
            # 根据当前深度获取子图
            subgraph = self.graph_builder.get_subgraph_by_depth(self.current_graph, current_depth)
            
            # 使用选定的布局算法
            pos = LayoutAlgorithms.get_layout_by_index(subgraph, layout_idx)
            
            # 获取节点属性
            node_colors, node_sizes, labels = self.graph_builder.get_node_attributes(
                subgraph, show_types, show_params)
            
            # 绘制图形
            nx.draw_networkx(
                subgraph, 
                pos=pos,
                with_labels=True,
                node_color=node_colors,
                node_size=node_sizes,
                labels=labels,
                font_size=9,
                font_weight='bold',
                arrows=True,
                arrowsize=15,
                ax=ax
            )
            
            # 添加图例
            legend_items = self.graph_builder.get_legend_items()
            if legend_items:
                ax.legend(handles=legend_items, loc='upper right', bbox_to_anchor=(1.0, 1.0))
            
            # 设置标题
            if model_name:
                ax.set_title(f"模型: {model_name} (深度: {current_depth})", fontsize=14)
            
            ax.set_axis_off()
            
            return fig
            
        except Exception as e:
            error_msg = f"创建图形可视化失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            raise Exception(error_msg)
    
    def get_max_depth(self):
        """获取最大深度"""
        return self.max_depth
    
    def has_graph(self):
        """检查是否已创建图形"""
        return self.current_graph is not None
    
    def get_graph_info(self):
        """获取图形信息"""
        if self.current_graph is None:
            return None
        
        return {
            'node_count': self.current_graph.number_of_nodes(),
            'edge_count': self.current_graph.number_of_edges(),
            'max_depth': self.max_depth
        } 