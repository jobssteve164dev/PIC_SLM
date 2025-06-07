"""
模型编辑器常量定义
"""

# 层类型颜色映射
LAYER_COLORS = {
    'Conv2d': "#4285F4",        # 蓝色
    'ConvTranspose2d': "#34A853", # 绿色
    'Linear': "#FBBC05",        # 黄色
    'MaxPool2d': "#EA4335",     # 红色
    'AvgPool2d': "#EA4335",     # 红色
    'BatchNorm2d': "#9C27B0",   # 紫色
    'Dropout': "#FF9800",       # 橙色
    'ReLU': "#03A9F4",          # 浅蓝色
    'LeakyReLU': "#03A9F4",     # 浅蓝色
    'Sigmoid': "#03A9F4",       # 浅蓝色
    'Tanh': "#03A9F4",          # 浅蓝色
    'Flatten': "#607D8B",       # 灰蓝色
    'default': "#757575"        # 默认灰色
}

# 支持的层类型列表
LAYER_TYPES = [
    'Conv2d', 'ConvTranspose2d', 'Linear', 'MaxPool2d', 
    'AvgPool2d', 'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh',
    'BatchNorm2d', 'Dropout', 'Flatten'
]

# 分类模型列表
CLASSIFICATION_MODELS = [
    "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
    "VGG16", "VGG19", 
    "DenseNet121", "DenseNet169", "DenseNet201",
    "MobileNetV2", "MobileNetV3Small", "MobileNetV3Large",
    "EfficientNetB0", "EfficientNetB1", "EfficientNetB2",
    "RegNetX_400MF", "RegNetY_400MF",
    "ConvNeXt_Tiny", "ConvNeXt_Small",
    "ViT_B_16", "Swin_T",
    "自定义模型文件"
]

# 目标检测模型列表
DETECTION_MODELS = [
    "YOLOv5s", "YOLOv5m", "YOLOv5l", "YOLOv5x",
    "YOLOX_s", "YOLOX_m", "YOLOX_l", "YOLOX_x",
    "FasterRCNN_ResNet50_FPN", "FasterRCNN_MobileNetV3_Large_FPN",
    "RetinaNet_ResNet50_FPN", "SSD300_VGG16",
    "自定义模型文件"
]

# 视图设置
MIN_ZOOM_FACTOR = 0.01
MAX_ZOOM_FACTOR = 20.0
DEFAULT_ZOOM_FACTOR = 1.0

# 布局设置
MIN_HORIZONTAL_SPACING = 200
VERTICAL_SPACING = 150
LAYER_ITEM_WIDTH = 160
LAYER_ITEM_HEIGHT = 100 