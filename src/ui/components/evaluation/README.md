# 评估标签页组件包

本目录包含了模型评估相关的所有组件，经过重构后按功能模块进行了清晰的组织。

## 目录结构

```
evaluation/
├── __init__.py                     # 主模块导出文件
├── README.md                       # 本说明文档
├── widgets/                        # 核心功能组件
│   ├── __init__.py
│   ├── model_evaluation_widget.py          # 模型评估和比较组件
│   ├── tensorboard_manager_widget.py       # TensorBoard管理组件
│   ├── training_curve_widget.py            # 训练曲线显示组件
│   ├── params_comparison_widget.py         # 训练参数对比组件
│   ├── training_visualization_widget.py    # 训练可视化组件
│   └── tensorboard_widget.py               # TensorBoard嵌入式组件
├── visualization/                  # 可视化相关组件
│   ├── __init__.py
│   └── visualization_container_widget.py   # 可视化容器组件
└── utils/                          # 工具类和辅助模块
    ├── __init__.py
    ├── chart_renderer.py                   # 图表渲染器
    ├── metric_explanations.py              # 指标解释工具
    └── metrics_data_manager.py             # 指标数据管理器
```

## 组件分类说明

### widgets/ - 核心功能组件
包含评估标签页的主要UI组件，每个组件负责特定的功能模块：

- **ModelEvaluationWidget**: 负责模型选择、评估和比较功能
- **TensorBoardManagerWidget**: 管理TensorBoard服务的启动、停止和配置
- **TrainingCurveWidget**: 实时显示训练过程中的各种指标曲线
- **ParamsComparisonWidget**: 对比不同模型的训练参数配置
- **TrainingVisualizationWidget**: 训练过程的综合可视化展示
- **TensorBoardWidget**: 嵌入式TensorBoard视图组件

### visualization/ - 可视化组件
专门负责各种模型可视化功能：

- **VisualizationContainerWidget**: 可视化组件的容器，统一管理特征可视化、GradCAM、敏感性分析等

### utils/ - 工具类模块
提供各种辅助功能和工具类：

- **ChartRenderer**: 负责各种图表的渲染和样式管理
- **MetricExplanations**: 提供各种评估指标的详细解释
- **MetricsDataManager**: 管理和处理评估指标数据

## 使用方法

### 直接导入使用
```python
# 从主模块导入所需组件
from ui.components.evaluation import (
    ModelEvaluationWidget,
    TensorBoardManagerWidget,
    TrainingCurveWidget,
    VisualizationContainerWidget
)

# 创建组件实例
model_eval = ModelEvaluationWidget(parent)
tensorboard = TensorBoardManagerWidget(parent)
```

### 从子模块导入
```python
# 从特定子模块导入
from ui.components.evaluation.widgets import ModelEvaluationWidget
from ui.components.evaluation.visualization import VisualizationContainerWidget
from ui.components.evaluation.utils import ChartRenderer
```

## 组织优势

### 1. 清晰的功能分离
- 核心UI组件与工具类分离
- 不同类型的组件有明确的归属

### 2. 便于维护和扩展
- 新增组件可以清楚地放在对应的子目录中
- 修改特定功能时只需关注相关目录

### 3. 更好的代码组织
- 符合Python包管理的最佳实践
- 支持选择性导入，减少不必要的依赖

### 4. 团队协作友好
- 不同开发者可以专注于不同的子模块
- 减少文件冲突的可能性

## 向后兼容性

虽然文件位置发生了变化，但通过`__init__.py`文件的统一导出，保证了向后兼容性。原有的导入方式仍然可以正常工作：

```python
# 这种导入方式仍然有效
from ui.components.evaluation import ModelEvaluationWidget
```

## 扩展指南

### 添加新的Widget组件
1. 在`widgets/`目录下创建新的组件文件
2. 在`widgets/__init__.py`中添加导入和导出
3. 在主`__init__.py`中添加相应的导入

### 添加新的可视化组件
1. 在`visualization/`目录下创建新组件
2. 更新相应的`__init__.py`文件

### 添加新的工具类
1. 在`utils/`目录下创建新的工具类文件
2. 更新相应的`__init__.py`文件

这种组织结构为评估标签页组件提供了清晰、可维护和可扩展的架构基础。 