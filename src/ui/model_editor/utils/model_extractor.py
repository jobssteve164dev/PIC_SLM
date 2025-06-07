"""
模型结构提取工具
"""
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QProgressDialog, QMessageBox
from PyQt5.QtCore import Qt, QPointF
from .constants import MIN_HORIZONTAL_SPACING, VERTICAL_SPACING


class ModelExtractor:
    """模型结构提取器"""
    
    def __init__(self, parent=None):
        self.parent = parent
        
    def extract_model_structure(self, model, model_name, editor):
        """提取模型结构并在编辑器中显示"""
        try:
            # 创建进度对话框
            progress = QProgressDialog("正在分析模型结构...", "取消", 0, 100, self.parent)
            progress.setWindowTitle("提取模型结构")
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(0)
            progress.show()
            
            # 清除现有内容
            editor.clear_all()
            
            # 层计数器和已处理的模块
            layer_counter = 0
            processed_modules = set()
            
            # 估计模型层数，用于进度显示
            total_layers = self._estimate_model_layers(model)
            processed_layers = 0
            
            # 跟踪每个深度的层数量和位置
            depth_layers = {}  # 用于存储每个深度级别的层数量
            depth_width_used = {}  # 用于存储每个深度已使用的水平空间
            
            # 递归函数来处理模型各层
            def process_module(module, parent_name=None, parent_layer=None, depth=0):
                nonlocal layer_counter, processed_layers
                
                # 更新进度
                processed_layers += 1
                progress_value = min(99, int(processed_layers / max(1, total_layers) * 100))
                progress.setValue(progress_value)
                
                # 检查是否取消
                if progress.wasCanceled():
                    return None
                
                # 避免处理同一个模块多次
                module_id = id(module)
                if module_id in processed_modules:
                    return
                processed_modules.add(module_id)
                
                # 为复杂模块生成有意义的名称
                if isinstance(module, nn.Sequential) and not parent_name:
                    module_name = f"Sequential_{layer_counter}"
                    layer_counter += 1
                elif isinstance(module, nn.ModuleList) and not parent_name:
                    module_name = f"ModuleList_{layer_counter}"
                    layer_counter += 1
                elif hasattr(module, '__class__'):
                    module_type = module.__class__.__name__
                    module_name = f"{module_type}_{layer_counter}"
                    layer_counter += 1
                else:
                    module_name = f"Layer_{layer_counter}"
                    layer_counter += 1
                
                # 完整名称包括父模块名称
                full_name = f"{parent_name}_{module_name}" if parent_name else module_name
                
                # 只处理叶子模块或常见的容器
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, 
                                    nn.MaxPool2d, nn.AvgPool2d, nn.BatchNorm2d,
                                    nn.Dropout, nn.ReLU, nn.LeakyReLU, nn.Sigmoid,
                                    nn.Tanh, nn.Flatten)):
                    # 基本层，添加到编辑器中
                    layer_info = self._create_layer_info(module, full_name, depth)
                    
                    # 如果有父层，创建连接
                    if parent_layer:
                        editor.connections.append({
                            'from': parent_layer,
                            'to': full_name
                        })
                    
                    # 添加层
                    editor.layers.append(layer_info)
                    
                    # 计算该层在其深度级别的位置
                    if depth not in depth_layers:
                        depth_layers[depth] = 0
                        depth_width_used[depth] = 0
                    
                    # 计算水平位置，考虑避免重叠
                    x_pos = depth_width_used[depth]
                    y_pos = depth * VERTICAL_SPACING
                    
                    # 更新该深度已使用的水平空间
                    depth_width_used[depth] += MIN_HORIZONTAL_SPACING
                    depth_layers[depth] += 1
                    
                    # 添加到场景
                    pos = QPointF(x_pos, y_pos)
                    editor.scene.add_layer(layer_info, pos)
                    
                    return full_name
                    
                else:
                    # 容器模块，递归处理
                    last_child_name = parent_layer
                    
                    # 处理子模块
                    if isinstance(module, (nn.Sequential, nn.ModuleList)):
                        for i, child in enumerate(module.children()):
                            child_name = process_module(child, full_name, last_child_name, depth + 1)
                            if child_name:
                                last_child_name = child_name
                    else:
                        # 检查是否有命名子模块
                        has_children = False
                        for name, child in module.named_children():
                            has_children = True
                            child_name = process_module(child, full_name, last_child_name, depth + 1)
                            if child_name:
                                last_child_name = child_name
                                
                        # 如果没有子模块但模块类型很重要，也添加它
                        if not has_children and type(module) not in [nn.Module]:
                            layer_info = self._create_layer_info(module, full_name, depth)
                            
                            if parent_layer:
                                editor.connections.append({
                                    'from': parent_layer,
                                    'to': full_name
                                })
                            
                            editor.layers.append(layer_info)
                            
                            # 计算该层在其深度级别的位置
                            if depth not in depth_layers:
                                depth_layers[depth] = 0
                                depth_width_used[depth] = 0
                            
                            # 计算水平位置，考虑避免重叠
                            x_pos = depth_width_used[depth]
                            y_pos = depth * VERTICAL_SPACING
                            
                            # 更新该深度已使用的水平空间
                            depth_width_used[depth] += MIN_HORIZONTAL_SPACING
                            depth_layers[depth] += 1
                            
                            # 添加到场景
                            pos = QPointF(x_pos, y_pos)
                            editor.scene.add_layer(layer_info, pos)
                            
                            return full_name
                    
                    return last_child_name
            
            # 从顶层开始处理
            process_module(model)
            
            # 处理完成进度
            progress.setValue(100)
            
            # 如果用户取消了，则不进行后续操作
            if progress.wasCanceled():
                return
            
            # 添加连接图形项
            for conn in editor.connections:
                editor.scene.add_connection(conn['from'], conn['to'])
                
            # 调整布局 - 使各深度层在水平方向居中
            self._optimize_layer_layout(editor, depth_layers, depth_width_used, MIN_HORIZONTAL_SPACING)
            
            # 更新所有连接，确保反映了新的布局
            editor.scene.update_connections()
                
            # 调整视图以适应所有内容
            editor.view.resetTransform()
            
            # 获取场景中所有项的边界矩形
            scene_items_rect = editor.scene.itemsBoundingRect()
            
            # 如果层数较多，初始显示比例较小，以便看到整体结构
            if len(editor.layers) > 100:
                initial_scale = 0.5  # 设置一个较小的初始缩放因子
                editor.view.scale(initial_scale, initial_scale)
                editor.view.zoom_factor = initial_scale
            
            # 确保视图适应所有内容
            editor.view.fitInView(scene_items_rect, Qt.KeepAspectRatio)
            
            QMessageBox.information(self.parent, "成功", f"已导入{model_name}模型结构，共{len(editor.layers)}个层")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self.parent, "错误", f"提取模型结构时出错: {str(e)}\n\n详细信息:\n{error_details}")
    
    def _estimate_model_layers(self, model):
        """估计模型中的层数，用于进度显示"""
        try:
            # 统计模型中可能的层数
            layer_count = 0
            
            # 使用非递归方法遍历模型
            stack = [model]
            while stack:
                module = stack.pop()
                # 判断是否是我们关注的层类型
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, 
                                    nn.MaxPool2d, nn.AvgPool2d, nn.BatchNorm2d,
                                    nn.Dropout, nn.ReLU, nn.LeakyReLU, nn.Sigmoid,
                                    nn.Tanh, nn.Flatten)):
                    layer_count += 1
                # 添加子模块到栈中
                for child in module.children():
                    stack.append(child)
            
            # 返回估计的层数，最少返回1
            return max(1, layer_count)
        except:
            # 出错时返回一个默认值
            return 100

    def _optimize_layer_layout(self, editor, depth_layers, depth_width_used, min_spacing):
        """优化层的布局，确保每个深度的层在水平方向居中，并避免重叠"""
        # 自适应调整间距 - 当模型层数特别多时，减小间距
        total_layer_count = sum(depth_layers.values())
        
        # 根据总层数动态调整间距
        if total_layer_count > 100:
            # 对于大型模型，采用更紧凑的布局
            adjusted_spacing = max(120, min_spacing * (1.0 - (total_layer_count - 100) / 400))
        else:
            adjusted_spacing = min_spacing
            
        # 对于每个深度级别
        for depth, count in depth_layers.items():
            if count > 0:
                # 计算该深度层的总宽度
                total_width = count * adjusted_spacing
                
                # 找出该深度的所有层
                depth_layer_items = []
                for layer_name, layer_item in editor.scene.layer_items.items():
                    y_pos = layer_item.pos().y()
                    if abs(y_pos - depth * VERTICAL_SPACING) < 1:
                        depth_layer_items.append(layer_item)
                
                # 按当前x坐标排序
                depth_layer_items.sort(key=lambda item: item.pos().x())
                
                # 计算居中所需的起始x坐标
                if len(depth_layer_items) > 0:
                    start_x = -total_width / 2
                    
                    # 重新排列该深度的所有层
                    current_x = start_x
                    for item in depth_layer_items:
                        item.setPos(current_x, item.pos().y())
                        current_x += adjusted_spacing
            
            # 更新该深度已使用的水平空间
            depth_width_used[depth] += adjusted_spacing
            
    def _create_layer_info(self, module, name, depth=0):
        """从模块创建层信息字典"""
        layer_type = module.__class__.__name__
        layer_info = {
            'name': name,
            'type': layer_type,
            'position': {'x': depth * 150, 'y': 0}
        }
        
        # 提取层特定参数
        if isinstance(module, nn.Conv2d):
            layer_info.update({
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding
            })
        elif isinstance(module, nn.ConvTranspose2d):
            layer_info.update({
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding
            })
        elif isinstance(module, nn.Linear):
            layer_info.update({
                'in_features': module.in_features,
                'out_features': module.out_features
            })
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            # 处理kernel_size可能是不同形式的情况
            if hasattr(module, 'kernel_size'):
                if isinstance(module.kernel_size, int):
                    k_size = (module.kernel_size, module.kernel_size)
                else:
                    k_size = module.kernel_size
                layer_info['kernel_size'] = k_size
        elif isinstance(module, nn.BatchNorm2d):
            layer_info['num_features'] = module.num_features
        elif isinstance(module, nn.Dropout):
            layer_info['p'] = module.p
        elif isinstance(module, nn.LeakyReLU):
            layer_info['negative_slope'] = module.negative_slope
            
        return layer_info
    
    def create_dummy_yolox(self, model_name):
        """创建YOLOX的替代模型结构"""
        import torch.nn as nn
        
        class DummyYOLOX(nn.Module):
            """YOLOX模型的替代结构"""
            def __init__(self, depth_factor=1.0):
                super().__init__()
                # 根据不同型号设置不同的深度因子
                if model_name == "YOLOX_m":
                    depth_factor = 1.5
                elif model_name == "YOLOX_l":
                    depth_factor = 2.0
                elif model_name == "YOLOX_x":
                    depth_factor = 3.0
                    
                # 特征提取主干网络
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, int(64 * depth_factor), 3, 2, 1, bias=False),
                    nn.BatchNorm2d(int(64 * depth_factor)),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(int(64 * depth_factor), int(128 * depth_factor), 3, 2, 1, bias=False),
                    nn.BatchNorm2d(int(128 * depth_factor)),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(int(128 * depth_factor), int(256 * depth_factor), 3, 2, 1, bias=False),
                    nn.BatchNorm2d(int(256 * depth_factor)),
                    nn.LeakyReLU(0.1),
                )
                
                # 检测头
                self.head = nn.Sequential(
                    nn.Conv2d(int(256 * depth_factor), int(256 * depth_factor), 3, 1, 1),
                    nn.BatchNorm2d(int(256 * depth_factor)),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(int(256 * depth_factor), 85, 1, 1, 0),  # 80类 + 4个框坐标 + 1个置信度
                )
                
            def forward(self, x):
                feat = self.backbone(x)
                out = self.head(feat)
                return out
        
        return DummyYOLOX() 