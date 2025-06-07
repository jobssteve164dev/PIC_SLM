# 图像处理模块 (Image Processing)

## 概述

本模块是对原始 `image_preprocessor.py` 文件的完整重构，采用模块化设计提高可维护性和可扩展性。所有功能都被拆分到专门的组件中，每个组件负责特定的功能领域。

## 组件结构

```
src/image_processing/
├── __init__.py              # 模块初始化文件
├── README.md               # 本说明文档
├── base_processor.py       # 基础处理器（信号定义、通用功能）
├── image_transformer.py    # 图像变换器（大小调整、亮度、对比度）
├── augmentation_manager.py # 数据增强管理器（各种增强方法）
├── dataset_creator.py      # 数据集创建器（训练集/验证集分割）
├── class_balancer.py      # 类别平衡处理器（多类别平衡处理）
├── file_manager.py        # 文件管理器（文件和目录操作）
└── main_processor.py      # 主处理器（协调所有组件）
```

## 组件详解

### 1. BaseImageProcessor (base_processor.py)
- **职责**: 提供基础功能和 PyQt5 信号定义
- **功能**:
  - 定义进度更新、状态更新、完成和错误信号
  - 停止预处理功能
  - 通用工具方法（获取图片文件、创建目录、错误处理）

### 2. ImageTransformer (image_transformer.py)
- **职责**: 处理基本的图像变换操作
- **功能**:
  - 图像大小调整
  - 亮度和对比度调整
  - RGB 格式转换
  - 单张图片的完整预处理流程

### 3. AugmentationManager (augmentation_manager.py)
- **职责**: 管理所有数据增强方法
- **功能**:
  - 独立增强模式（每种方法单独应用）
  - 组合增强模式（多种方法组合）
  - 增强强度控制
  - 支持的增强方法：翻转、旋转、裁剪、缩放、亮度、对比度、噪声、模糊、色相

### 4. DatasetCreator (dataset_creator.py)
- **职责**: 创建训练和验证数据集
- **功能**:
  - 数据集划分（训练集/验证集）
  - 训练集增强处理
  - 验证集基本复制
  - 进度跟踪和状态更新

### 5. ClassBalancer (class_balancer.py)
- **职责**: 处理多类别数据的平衡预处理
- **功能**:
  - 按类别组织图片处理
  - 每个类别独立进行训练集/验证集划分
  - 训练集增强，验证集不增强
  - 类别文件夹管理

### 6. FileManager (file_manager.py)
- **职责**: 处理所有文件和目录操作
- **功能**:
  - 图片文件识别和筛选
  - 目录创建和管理
  - 文件复制操作
  - 路径处理工具
  - 文件验证功能

### 7. ImagePreprocessor (main_processor.py)
- **职责**: 主协调器，提供统一的处理接口
- **功能**:
  - 协调所有子组件
  - 提供与原始类相同的接口
  - 参数预处理和验证
  - 处理模式选择（标准模式 vs 类别平衡模式）

## 使用方式

### 基本使用（向后兼容）
```python
from src.image_processing import ImagePreprocessor

# 创建处理器实例
processor = ImagePreprocessor()

# 连接信号
processor.progress_updated.connect(your_progress_callback)
processor.status_updated.connect(your_status_callback)
processor.preprocessing_finished.connect(your_finished_callback)
processor.preprocessing_error.connect(your_error_callback)

# 设置参数并开始处理
params = {
    'source_folder': '/path/to/source',
    'target_folder': '/path/to/target',
    'dataset_folder': '/path/to/dataset',
    # ... 其他参数
}
processor.preprocess_images(params)
```

### 使用独立组件
```python
from src.image_processing.image_transformer import ImageTransformer
from src.image_processing.augmentation_manager import AugmentationManager

# 使用图像变换器
ImageTransformer.process_single_image(
    source_path, target_path, width, height, brightness, contrast
)

# 使用增强管理器
aug_manager = AugmentationManager()
aug_manager.apply_augmentations_to_image(
    image_path, target_dir, file_name, img_format, params
)
```

## 优势

### 1. 可维护性
- **单一职责**: 每个组件只负责特定功能
- **模块化**: 便于独立测试和调试
- **清晰结构**: 功能分离明确，代码更易理解

### 2. 可扩展性
- **新功能**: 可以轻松添加新的增强方法或处理模式
- **独立开发**: 不同功能可以独立开发和维护
- **插件式**: 组件可以独立使用或替换

### 3. 可重用性
- **组件重用**: 各个组件可以在其他项目中重用
- **功能组合**: 可以灵活组合不同组件实现特定需求

### 4. 可测试性
- **单元测试**: 每个组件可以独立进行单元测试
- **集成测试**: 组件间的交互可以单独测试

## 向后兼容性

为了确保不影响现有代码，原始的 `image_preprocessor.py` 文件已更新为兼容性入口：

```python
# 原始导入方式仍然有效
from src.image_preprocessor import ImagePreprocessor

# 新的导入方式
from src.image_processing import ImagePreprocessor
```

所有原有的接口和功能都得到完整保留，现有代码无需修改即可使用重构后的组件。

## 性能优化

重构后的组件在保持功能完整的同时，还带来了以下性能优化：

1. **内存效率**: 组件化设计减少了不必要的内存占用
2. **处理速度**: 专门的组件可以进行针对性优化
3. **错误处理**: 更精确的错误定位和处理
4. **资源管理**: 更好的资源分配和释放

## 注意事项

1. **依赖关系**: 确保所有必要的依赖包已安装（PIL, albumentations, sklearn 等）
2. **文件路径**: 所有文件路径参数都应该是绝对路径或相对于工作目录的路径
3. **内存使用**: 处理大量图片时注意内存使用情况
4. **停止机制**: 可以通过 `stop()` 方法安全停止处理过程 