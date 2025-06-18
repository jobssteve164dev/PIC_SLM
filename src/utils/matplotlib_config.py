"""
matplotlib全局配置模块
统一处理matplotlib的警告和显示设置
"""

import matplotlib
import matplotlib.pyplot as plt
import warnings
import numpy as np
from PIL import Image

# 过滤matplotlib的常见警告
warnings.filterwarnings('ignore', message='Clipping input data to the valid range for imshow with RGB data')
warnings.filterwarnings('ignore', message='.*GUI is implemented.*')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 配置matplotlib基本设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 100
matplotlib.rcParams['image.cmap'] = 'viridis'
matplotlib.rcParams['image.interpolation'] = 'nearest'

# 设置后端（如果需要）
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    pass  # 如果Qt5Agg不可用，使用默认后端


def normalize_image_for_matplotlib(image_data):
    """
    标准化图像数据用于matplotlib显示
    
    Args:
        image_data: PIL.Image, numpy.ndarray 或其他图像数据
        
    Returns:
        标准化后的numpy数组，数值范围在0-1之间
    """
    if isinstance(image_data, Image.Image):
        # PIL图像转换为numpy数组
        image_array = np.array(image_data)
        # 确保数据范围在0-1之间
        if image_array.max() > 1.0:
            image_array = image_array.astype(np.float32) / 255.0
        return image_array
    elif isinstance(image_data, np.ndarray):
        # numpy数组
        if image_data.dtype == np.uint8:
            # 如果是uint8类型，直接除以255
            return image_data.astype(np.float32) / 255.0
        elif image_data.max() > 1.0:
            # 如果最大值大于1，假设是0-255范围
            return image_data.astype(np.float32) / 255.0
        else:
            # 已经是0-1范围
            return image_data.astype(np.float32)
    else:
        return image_data


def normalize_feature_map(feature_map):
    """
    标准化特征图数据用于可视化
    
    Args:
        feature_map: numpy数组特征图
        
    Returns:
        标准化后的特征图，数值范围在0-1之间
    """
    if isinstance(feature_map, np.ndarray):
        if feature_map.max() > feature_map.min():
            # 标准化到0-1范围
            normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            return normalized.astype(np.float32)
        else:
            # 如果最大值等于最小值，返回全零数组
            return np.zeros_like(feature_map, dtype=np.float32)
    return feature_map


def safe_imshow(ax, image_data, **kwargs):
    """
    安全的imshow函数，自动处理数据范围
    
    Args:
        ax: matplotlib轴对象
        image_data: 图像数据
        **kwargs: 传递给imshow的其他参数
    """
    # 标准化图像数据
    normalized_data = normalize_image_for_matplotlib(image_data)
    
    # 设置默认参数
    if 'vmin' not in kwargs and 'vmax' not in kwargs:
        kwargs['vmin'] = 0
        kwargs['vmax'] = 1
    
    return ax.imshow(normalized_data, **kwargs)


def configure_matplotlib_for_chinese():
    """配置matplotlib支持中文显示"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


def suppress_matplotlib_warnings():
    """抑制matplotlib的常见警告"""
    warnings.filterwarnings('ignore', message='Clipping input data to the valid range for imshow with RGB data')
    warnings.filterwarnings('ignore', message='.*GUI is implemented.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='matplotlib')


# 自动执行配置
suppress_matplotlib_warnings()
configure_matplotlib_for_chinese() 