import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import json
from typing import Dict, Any, Tuple, Optional

def get_data_transforms(img_size=224):
    """获取标准的数据转换操作
    
    Args:
        img_size: 图像调整大小的尺寸
        
    Returns:
        字典，包含训练和验证的数据转换操作
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def load_classification_datasets(data_dir, batch_size=32, img_size=224, num_workers=4):
    """加载分类任务的训练和验证数据集
    
    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        img_size: 图像大小
        num_workers: 数据加载的工作线程数
        
    Returns:
        dataloaders: 数据加载器字典
        dataset_sizes: 数据集大小字典
        class_names: 类名列表
        num_classes: 类别数量
    """
    # 标准化路径格式
    data_dir = os.path.normpath(data_dir).replace('\\', '/')
    
    # 检查训练和验证数据集目录是否存在
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练数据集目录不存在: {train_dir}")
            
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"验证数据集目录不存在: {val_dir}")
    
    # 获取数据转换
    data_transforms = get_data_transforms(img_size)
    
    # 加载数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    
    # 创建数据加载器
    dataloaders = {x: DataLoader(image_datasets[x],
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers)
                  for x in ['train', 'val']}
    
    # 获取数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    # 获取类名和类别数量
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    # 保存类别信息
    class_info = {
        'class_names': class_names,
        'class_to_idx': image_datasets['train'].class_to_idx
    }
    
    return dataloaders, dataset_sizes, class_names, num_classes, class_info

def save_class_info(class_info, save_dir):
    """保存类别信息到JSON文件
    
    Args:
        class_info: 类别信息字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'class_info.json'), 'w') as f:
        json.dump(class_info, f)
        
def save_training_info(training_info, save_dir):
    """保存训练信息到JSON文件
    
    Args:
        training_info: 训练信息字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=4)
        
def get_custom_transforms(config):
    """根据配置创建自定义数据转换
    
    Args:
        config: 配置字典
        
    Returns:
        字典，包含训练和验证的数据转换操作
    """
    # 从配置中获取参数
    img_size = config.get('img_size', 224)
    use_color_jitter = config.get('use_color_jitter', False)
    use_random_rotation = config.get('use_random_rotation', False)
    use_random_crop = config.get('use_random_crop', False)
    use_center_crop = config.get('use_center_crop', False)
    
    # 创建训练转换列表
    train_transforms = [
        transforms.Resize((img_size, img_size))
    ]
    
    # 添加随机裁剪
    if use_random_crop:
        train_transforms.append(transforms.RandomCrop(img_size, padding=4))
    
    # 添加随机旋转
    if use_random_rotation:
        train_transforms.append(transforms.RandomRotation(15))
    
    # 添加随机水平翻转
    train_transforms.append(transforms.RandomHorizontalFlip())
    
    # 添加颜色抖动
    if use_color_jitter:
        train_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    
    # 添加标准转换
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建验证转换列表
    val_transforms = [
        transforms.Resize((img_size, img_size))
    ]
    
    # 添加中心裁剪
    if use_center_crop:
        val_transforms.append(transforms.CenterCrop(img_size))
    
    # 添加标准转换
    val_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建最终的转换操作
    data_transforms = {
        'train': transforms.Compose(train_transforms),
        'val': transforms.Compose(val_transforms),
    }
    
    return data_transforms 