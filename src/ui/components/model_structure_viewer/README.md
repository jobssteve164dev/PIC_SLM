# 模型结构可视化组件

这是一个重构后的模型结构可视化组件，将原来的单一大型组件（1000+行）拆分为多个专门的模块，提高了代码的可维护性、可扩展性和可测试性。

## 📁 文件结构

```
src/ui/components/model_structure_viewer/
├── __init__.py                     # 模块初始化文件
├── model_structure_viewer.py       # 主组件，协调各个子模块
├── model_loader.py                 # 模型加载模块
├── graph_builder.py                # 图形构建模块
├── layout_algorithms.py            # 布局算法模块
├── visualization_controller.py     # 可视化控制模块
├── ui_components.py                # UI组件模块
└── README.md                       # 本文档
```

## 🧩 模块说明

### 1. `model_loader.py` - 模型加载模块
**职责**: 专门处理PyTorch模型的加载和创建
- 支持多种模型架构（ResNet, DenseNet, MobileNet, VGG, EfficientNet等）
- 兼容新旧PyTorch API
- 处理模型权重加载的各种兼容性问题
- 自动加载类别信息文件

### 2. `graph_builder.py` - 图形构建模块
**职责**: 专门处理模型图形的构建和FX处理
- FX符号跟踪
- 创建NetworkX图形结构
- 分层图形构建
- 节点属性计算（颜色、大小、标签）
- 图例生成

### 3. `layout_algorithms.py` - 布局算法模块
**职责**: 专门处理图形布局算法
- 自定义树形布局算法
- 多种布局方式支持
- 不依赖外部图形库

### 4. `visualization_controller.py` - 可视化控制模块
**职责**: 专门处理模型结构的可视化逻辑
- 文本格式可视化
- 图形可视化控制
- DenseNet特殊处理
- matplotlib图表生成

### 5. `ui_components.py` - UI组件模块
**职责**: 专门处理界面元素的创建和布局
- 模型选择组件
- 控制按钮
- 输出文本区域
- FX控制面板
- 图表容器

### 6. `model_structure_viewer.py` - 主组件
**职责**: 协调各个子模块，提供统一的对外接口
- 整合所有子模块
- 信号连接和事件处理
- 外部API保持兼容

## 🚀 使用方法

### 基本使用

```python
from src.ui.components.model_structure_viewer import ModelStructureViewer

# 创建组件
viewer = ModelStructureViewer()

# 设置模型（从外部）
viewer.set_model(model, class_names)

# 或者通过界面选择模型文件
# 用户可以直接使用界面进行操作
```

### 高级使用（直接使用子模块）

```python
from src.ui.components.model_structure_viewer import (
    model_loader,
    graph_builder,
    visualization_controller
)

# 使用模型加载器
loader = model_loader.ModelLoader()
loader.set_model_path("path/to/model.pth")
model = loader.load_model()

# 使用图形构建器
builder = graph_builder.GraphBuilder()
graph = builder.create_fx_graph(model)

# 使用可视化控制器
controller = visualization_controller.VisualizationController()
text_output = controller.create_text_visualization(model, (3, 224, 224))
```

## ✨ 重构优势

1. **职责分离**: 每个模块只负责一个特定的功能领域
2. **可测试性**: 每个模块可以独立测试
3. **可扩展性**: 新增功能可以在对应模块中扩展
4. **代码复用**: 子模块可以在其他项目中复用
5. **维护性**: 修改某个功能时，只需要关注对应的模块

## 🔧 功能完整性

✅ **所有原有功能都完整保留**:
- 模型文件选择和加载
- torchsummary文本可视化
- FX图形可视化
- 多种布局算法支持
- 深度控制和交互功能
- DenseNet特殊处理
- 参数信息显示
- 图例和工具栏
- 外部模型设置接口

✅ **向后兼容**:
- 外部调用接口保持不变
- 所有公开方法和属性都保留
- 不影响现有代码的使用

## 📋 版本信息

- **版本**: 2.0.0
- **重构日期**: 2024
- **维护状态**: 活跃维护
- **兼容性**: 完全向后兼容 