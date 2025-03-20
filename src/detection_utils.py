import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional
import torchvision.ops.boxes as box_ops

class DetectionDataset(Dataset):
    """目标检测数据集类"""
    
    def __init__(self, root_dir: str, transform=None, is_train: bool = True):
        """
        初始化数据集
        
        参数:
            root_dir: 数据集根目录，包含images和labels子目录
            transform: 数据转换函数
            is_train: 是否为训练集
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # 获取图像和标签目录
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        
        # 获取所有图像文件
        self.image_files = [f for f in os.listdir(self.image_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 验证数据集完整性
        self._validate_dataset()
    
    def _validate_dataset(self):
        """验证数据集的完整性"""
        if not os.path.exists(self.image_dir):
            raise RuntimeError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise RuntimeError(f"标签目录不存在: {self.label_dir}")
        
        # 检查每个图像是否有对应的标签文件
        for img_file in self.image_files:
            label_file = os.path.join(self.label_dir, 
                                    os.path.splitext(img_file)[0] + '.txt')
            if not os.path.exists(label_file):
                raise RuntimeError(f"图像 {img_file} 缺少对应的标签文件")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        获取数据集中的一个样本
        
        参数:
            idx: 样本索引
            
        返回:
            image: 图像张量
            target: 包含boxes和labels的字典
        """
        # 加载图像
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # 加载标签
        label_path = os.path.join(self.label_dir, 
                                os.path.splitext(self.image_files[idx])[0] + '.txt')
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # 转换YOLO格式到标准格式
                    x1 = (x_center - width/2) * image.width
                    y1 = (y_center - height/2) * image.height
                    x2 = (x_center + width/2) * image.width
                    y2 = (y_center + height/2) * image.height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(class_id) + 1)  # +1是因为0是背景类
        
        # 转换为张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        # 应用数据转换
        if self.transform:
            image = self.transform(image)
        
        return image, target

def calculate_map(predictions: List[Dict[str, torch.Tensor]], 
                 targets: List[Dict[str, torch.Tensor]], 
                 iou_threshold: float = 0.5) -> float:
    """
    计算平均精度(mAP)
    
    参数:
        predictions: 模型预测结果列表
        targets: 真实标签列表
        iou_threshold: IoU阈值
        
    返回:
        mAP: 平均精度
    """
    all_aps = []
    
    # 对每个类别计算AP
    for class_id in range(1, max(max(t['labels'].max().item() for t in targets), 1) + 1):
        # 收集当前类别的所有预测和真实标签
        class_predictions = []
        class_targets = []
        
        for pred, target in zip(predictions, targets):
            # 获取当前类别的预测
            pred_mask = pred['labels'] == class_id
            if pred_mask.any():
                class_predictions.append({
                    'boxes': pred['boxes'][pred_mask],
                    'scores': pred['scores'][pred_mask]
                })
            
            # 获取当前类别的真实标签
            target_mask = target['labels'] == class_id
            if target_mask.any():
                class_targets.append({
                    'boxes': target['boxes'][target_mask]
                })
        
        if not class_predictions or not class_targets:
            continue
        
        # 计算当前类别的AP
        class_ap = calculate_ap(class_predictions, class_targets, iou_threshold)
        all_aps.append(class_ap)
    
    # 计算mAP
    return np.mean(all_aps) if all_aps else 0.0

def calculate_ap(predictions: List[Dict[str, torch.Tensor]], 
                targets: List[Dict[str, torch.Tensor]], 
                iou_threshold: float) -> float:
    """
    计算单个类别的平均精度(AP)
    
    参数:
        predictions: 当前类别的预测结果列表
        targets: 当前类别的真实标签列表
        iou_threshold: IoU阈值
        
    返回:
        AP: 平均精度
    """
    # 收集所有预测框和分数
    all_boxes = []
    all_scores = []
    for pred in predictions:
        all_boxes.append(pred['boxes'])
        all_scores.append(pred['scores'])
    
    # 合并所有预测框和分数
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    
    # 按分数排序
    sorted_indices = torch.argsort(all_scores, descending=True)
    all_boxes = all_boxes[sorted_indices]
    all_scores = all_scores[sorted_indices]
    
    # 收集所有真实标签框
    all_target_boxes = []
    for target in targets:
        all_target_boxes.append(target['boxes'])
    all_target_boxes = torch.cat(all_target_boxes, dim=0)
    
    # 计算IoU矩阵
    iou_matrix = box_ops.box_iou(all_boxes, all_target_boxes)
    
    # 计算TP和FP
    tp = torch.zeros(len(all_boxes), dtype=torch.bool)
    fp = torch.zeros(len(all_boxes), dtype=torch.bool)
    
    for i in range(len(all_boxes)):
        max_iou, max_idx = iou_matrix[i].max()
        if max_iou >= iou_threshold:
            tp[i] = True
        else:
            fp[i] = True
    
    # 计算累积TP和FP
    tp_cumsum = torch.cumsum(tp.float(), dim=0)
    fp_cumsum = torch.cumsum(fp.float(), dim=0)
    
    # 计算召回率和精确率
    recalls = tp_cumsum / len(all_target_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
    
    # 计算AP
    ap = 0.0
    for t in torch.arange(0, 1.1, 0.1):
        if torch.sum(recalls >= t) == 0:
            p = 0
        else:
            p = torch.max(precisions[recalls >= t])
        ap = ap + p / 11.0
    
    return ap.item()

def evaluate_model(model: torch.nn.Module, 
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    评估模型性能
    
    参数:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 计算设备
        iou_threshold: IoU阈值
        
    返回:
        包含评估指标的字典
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            
            # 收集预测结果和真实标签
            all_predictions.extend(outputs)
            all_targets.extend(targets)
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    
    # 计算mAP
    mean_ap = calculate_map(all_predictions, all_targets, iou_threshold)
    
    return {
        'loss': avg_loss,
        'map': mean_ap
    } 