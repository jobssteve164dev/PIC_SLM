# 训练组件模块

该目录包含所有与模型训练相关的UI组件，统一管理训练功能的用户界面。

## 组件概述

### 核心训练组件
- **`ClassificationTrainingWidget`**: 图像分类训练界面组件
  - 支持多种分类模型架构（ResNet、MobileNet、EfficientNet等）
  - 提供完整的训练参数配置
  - 集成权重配置和预训练模型支持

- **`DetectionTrainingWidget`**: 目标检测训练界面组件
  - 支持YOLO、SSD、Faster R-CNN等检测模型
  - 专门针对目标检测任务的参数优化
  - 包含检测特有的数据处理配置

- **`TrainingControlWidget`**: 训练控制组件
  - 提供开始/停止训练控制
  - 实时显示训练状态和进度
  - 集成训练帮助和模型文件夹管理

### 辅助组件
- **`LayerConfigWidget`**: 模型层配置组件
  - 支持自定义网络架构
  - 可调整卷积层、全连接层等参数
  - 分类和检测模型的专用配置

- **`TrainingHelpDialog`**: 训练帮助对话框
  - 详细的训练指导说明
  - 参数设置建议和最佳实践
  - 常见问题解答

- **`ModelDownloadDialog`**: 模型下载对话框
  - 预训练模型下载管理
  - 支持断点续传和进度显示
  - 提供多种模型源选择

- **`WeightConfigManager`**: 权重配置管理器
  - 自动分析数据集类别分布
  - 生成平衡的类别权重
  - 支持自定义权重调整

## 使用方式

```python
from src.ui.components.training import (
    ClassificationTrainingWidget,
    DetectionTrainingWidget,
    TrainingControlWidget
)

# 创建分类训练组件
classification_widget = ClassificationTrainingWidget()

# 创建检测训练组件  
detection_widget = DetectionTrainingWidget()

# 创建训练控制组件
control_widget = TrainingControlWidget()
```

## 目录结构
```
training/
├── __init__.py                      # 组件导出定义
├── classification_training_widget.py # 分类训练界面
├── detection_training_widget.py      # 检测训练界面
├── training_control_widget.py        # 训练控制界面
├── layer_config_widget.py           # 层配置组件
├── training_help_dialog.py          # 训练帮助对话框
├── model_download_dialog.py         # 模型下载对话框
├── weight_config_manager.py         # 权重配置管理
└── README.md                        # 组件说明文档
```

## 重构说明

这些组件原本分散在 `src/ui/` 目录下，为了更好的代码组织和模块化管理，现已统一整合到 `src/ui/components/training/` 目录中。

### 优势
1. **模块化**: 训练相关功能集中管理，便于维护
2. **可重用性**: 组件间依赖关系清晰，易于复用
3. **扩展性**: 新增训练相关功能时有明确的组织结构
4. **一致性**: 与其他组件模块（evaluation、annotation等）保持一致的目录结构 