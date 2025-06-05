import torch.fx as fx
import networkx as nx
import traceback


class GraphBuilder:
    """图形构建器，专门处理模型图形的构建和FX处理"""
    
    def __init__(self):
        self.layer_types = {}
    
    def create_fx_graph(self, model):
        """使用FX创建模型图"""
        try:
            model.eval()  # 设为评估模式
            traced = fx.symbolic_trace(model)
            
            # 获取FX跟踪结果的文本表示
            dot_graph = traced.graph.print_tabular()
            
            # 创建有向图
            G = nx.DiGraph()
            
            # 添加节点和边 - 带深度信息
            depth_map = {}  # 用于跟踪节点深度
            
            # 先添加所有节点
            for i, node in enumerate(traced.graph.nodes):
                # 设置节点类型
                node_type = node.op
                if node.op == 'call_module':
                    target_type = str(type(traced.get_submodule(node.target)).__name__)
                    node_type = f"{node.op} - {target_type}"
                elif node.op == 'call_function' or node.op == 'call_method':
                    node_type = f"{node.op} - {node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)}"
                    
                # 计算参数信息
                params_info = ""
                if node.op == 'call_module':
                    module = traced.get_submodule(node.target)
                    if hasattr(module, "in_features") and hasattr(module, "out_features"):
                        params_info = f"{module.in_features}→{module.out_features}"
                    elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                        params_info = f"{module.in_channels}→{module.out_channels}"
                        if hasattr(module, "kernel_size"):
                            kernel = module.kernel_size
                            if isinstance(kernel, tuple):
                                kernel = 'x'.join(map(str, kernel))
                            params_info += f", k={kernel}"
                    
                # 添加节点
                G.add_node(node.name, type=node_type, depth=0, params=params_info)  # 初始深度为0
                
            # 添加边并计算深度
            for node in traced.graph.nodes:
                for input_node in node.all_input_nodes:
                    G.add_edge(input_node.name, node.name)
            
            # 计算深度 - 从源节点开始的最长路径
            for node in nx.topological_sort(G):
                if not list(G.predecessors(node)):  # 如果是源节点
                    depth_map[node] = 1
                else:
                    # 节点深度是所有前置节点的最大深度+1
                    max_pred_depth = max([depth_map.get(pred, 0) for pred in G.predecessors(node)])
                    depth_map[node] = max_pred_depth + 1
            
            # 更新节点深度
            for node, depth in depth_map.items():
                G.nodes[node]['depth'] = depth
                
            # 设置图的最大深度
            max_depth = max(depth_map.values())
            
            return G, max_depth, dot_graph
            
        except Exception as e:
            print(f"FX符号跟踪失败: {str(e)}")
            print(traceback.format_exc())
            raise e
    
    def create_hierarchical_graph(self, model):
        """创建带层次结构的图"""
        G = nx.DiGraph()
        
        # 重置层类型字典
        self.layer_types = {}
        
        # 添加输入节点
        G.add_node("输入", type="placeholder", depth=0)
        
        # 获取模型名称
        model_name = model.__class__.__name__
        
        # 添加模型根节点
        G.add_node(model_name, type=model_name, depth=1)
        G.add_edge("输入", model_name)
        
        # 递归添加模块
        def add_module_to_graph(module, parent_name, depth):
            for name, layer in module.named_children():
                # 创建节点名称和类型
                node_name = f"{parent_name}.{name}" if parent_name else name
                layer_type = layer.__class__.__name__
                
                # 获取参数信息
                params_info = ""
                if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                    params_info = f"{layer.in_features}→{layer.out_features}"
                elif hasattr(layer, "in_channels") and hasattr(layer, "out_channels"):
                    params_info = f"{layer.in_channels}→{layer.out_channels}"
                    if hasattr(layer, "kernel_size"):
                        kernel = layer.kernel_size
                        if isinstance(kernel, tuple):
                            kernel = 'x'.join(map(str, kernel))
                        params_info += f", k={kernel}"
                
                # 添加节点和边
                G.add_node(node_name, type=layer_type, depth=depth, params=params_info)
                G.add_edge(parent_name, node_name)
                
                # 记录层类型
                self.layer_types[node_name] = layer_type
                
                # 递归处理子模块
                if list(layer.named_children()):
                    add_module_to_graph(layer, node_name, depth+1)
                elif depth == 2:  # 如果是叶子节点且深度较浅，增加一个虚拟节点防止图太扁平
                    virtual_node = f"{node_name}.output"
                    G.add_node(virtual_node, type="output", depth=depth+1)
                    G.add_edge(node_name, virtual_node)
        
        # 从模型开始添加
        add_module_to_graph(model, model_name, 2)
        
        # 获取最大深度
        max_depth = max([attrs.get('depth', 0) for _, attrs in G.nodes(data=True)])
        
        return G, max_depth
    
    def get_subgraph_by_depth(self, graph, max_depth):
        """根据深度过滤图形节点"""
        display_nodes = []
        for node, attrs in graph.nodes(data=True):
            # 检查节点深度是否小于或等于当前设置的显示深度
            if attrs.get('depth', 0) <= max_depth:
                display_nodes.append(node)
        
        return graph.subgraph(display_nodes)
    
    def get_node_attributes(self, subgraph, show_types, show_params):
        """获取节点的颜色、大小和标签属性"""
        node_colors = []
        node_sizes = []
        labels = {}
        
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('type', '')
            node_colors.append(self._get_layer_color(node_type))
            node_sizes.append(self._get_node_size(node_type))
            
            # 设置标签
            if show_types and show_params:
                # 同时显示类型和参数
                params = subgraph.nodes[node].get('params', '')
                if params:
                    labels[node] = f"{node}\n({node_type})\n{params}"
                else:
                    labels[node] = f"{node}\n({node_type})"
            elif show_types:
                # 仅显示类型
                labels[node] = f"{node}\n({node_type})"
            elif show_params:
                # 仅显示参数
                params = subgraph.nodes[node].get('params', '')
                if params:
                    labels[node] = f"{node}\n{params}"
                else:
                    labels[node] = node
            else:
                # 仅显示名称
                labels[node] = node
                
        return node_colors, node_sizes, labels
    
    def _get_layer_color(self, layer_type):
        """根据层类型返回颜色"""
        color_map = {
            'Conv2d': '#FF9999',          # 红色
            'Linear': '#99CCFF',          # 蓝色
            'BatchNorm': '#FFCC99',       # 橙色
            'ReLU': '#99FF99',            # 绿色
            'Dropout': '#CCCCCC',         # 灰色
            'MaxPool': '#CC99FF',         # 紫色
            'AvgPool': '#FFFF99',         # 黄色
            'Flatten': '#FF99FF',         # 粉色
            'DenseBlock': '#66CCCC',      # 青色
            'Transition': '#CCFF99',      # 浅绿色
            'AdaptiveAvgPool': '#FFCC99', # 橙色
            'placeholder': '#F0F0F0',     # 占位符(输入/输出)为浅灰色
            'getattr': '#E0E0E0',         # getattr操作为灰色
            'call_method': '#D0D0D0',     # 调用方法为深灰色
            'output': '#B0E0E6'           # 输出为淡蓝色
        }
        
        # 检查是否包含已知类型的层
        for known_type, color in color_map.items():
            if known_type.lower() in layer_type.lower():
                return color
        
        # 默认颜色 - 浅灰色
        return '#F5F5F5'
    
    def _get_node_size(self, node_type):
        """根据节点类型返回大小"""
        if 'conv' in node_type.lower() or 'linear' in node_type.lower():
            return 1500  # 重要层节点大一些
        elif 'input' in node_type.lower() or 'output' in node_type.lower():
            return 1800  # 输入输出节点最大
        else:
            return 1200  # 默认大小
    
    def get_legend_items(self):
        """获取图例项目"""
        try:
            import matplotlib.pyplot as plt
            return [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9999', markersize=10, label='卷积层'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#99CCFF', markersize=10, label='全连接层'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFCC99', markersize=10, label='BatchNorm/归一化'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#99FF99', markersize=10, label='激活函数'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#CC99FF', markersize=10, label='池化层'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#66CCCC', markersize=10, label='DenseBlock'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F0F0F0', markersize=10, label='输入/占位符'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#B0E0E6', markersize=10, label='输出'),
            ]
        except ImportError:
            return [] 