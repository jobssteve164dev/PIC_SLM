# 数据集评估组件模块

## 概述

此目录包含了数据集评估功能的拆分组件，用于提高代码的可维护性和可测试性。

## 组件结构

### 1. 数据集分析器 (`dataset_analyzers/`)
- **`base_analyzer.py`**: 基础分析器类，定义通用接口和方法
- **`classification_analyzer.py`**: 分类数据集专用分析器
- **`detection_analyzer.py`**: 目标检测数据集专用分析器

### 2. 权重生成器 (`weight_generator.py`)
- 负责根据数据集类别分布生成多种权重策略
- 支持balanced、inverse、log_inverse、normalized等策略
- 提供权重配置的导出和验证功能

### 3. 图表管理器 (`chart_manager.py`)
- 负责所有可视化图表的绘制和管理
- 支持分类和检测数据集的各种统计图表
- 提供图表保存和样式设置功能

### 4. 结果显示管理器 (`result_display_manager.py`)
- 负责评估结果的表格显示和指标说明
- 为不同类型的分析提供专门的显示方法
- 统一管理指标说明和建议信息

## 设计原则

1. **单一职责**: 每个组件只负责特定的功能领域
2. **松耦合**: 组件间通过清晰的接口通信
3. **可扩展**: 新增分析类型或指标时只需要扩展对应组件
4. **可测试**: 各组件可以独立进行单元测试

## 使用方式

```python
from .components.dataset_analyzers import ClassificationAnalyzer, DetectionAnalyzer
from .components.weight_generator import WeightGenerator
from .components.chart_manager import ChartManager
from .components.result_display_manager import ResultDisplayManager

# 创建分析器
analyzer = ClassificationAnalyzer(dataset_path)

# 执行分析
result = analyzer.analyze_data_distribution()

# 使用图表管理器显示结果
chart_manager = ChartManager(figure, ax1, ax2)
chart_manager.plot_classification_distribution(result['train_classes'], result['val_classes'])

# 使用结果显示管理器展示表格
display_manager = ResultDisplayManager(table, info_label)
display_manager.display_classification_distribution_results(result)
```

## 扩展说明

### 添加新的分析类型
1. 在对应分析器中添加新的方法
2. 在图表管理器中添加对应的绘制方法
3. 在结果显示管理器中添加对应的显示方法
4. 在主UI组件中添加调用逻辑

### 添加新的权重策略
1. 在`weight_generator.py`的`generate_weights`方法中添加新策略
2. 更新策略名称映射和说明文档

## 优势

1. **代码可维护性**: 从1292行的大文件拆分为多个专门的小文件
2. **功能完整性**: 保持所有原有功能不变
3. **测试友好**: 各组件可以独立测试
4. **扩展性**: 新增功能时影响范围小
5. **复用性**: 组件可以在其他地方复用

## 兼容性

重构后的组件完全兼容原有的接口，主UI组件`DatasetEvaluationTab`保持相同的对外接口，确保生产环境的稳定性。 