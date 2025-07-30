import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import logging

# 导入matplotlib配置
from src.utils.matplotlib_config import normalize_image_for_matplotlib, normalize_feature_map

logger = logging.getLogger(__name__)


def display_image(image, label_or_viewer):
    """显示图片到QLabel或ZoomableImageViewer"""
    try:
        # 获取控件的大小，用于自适应缩放
        if hasattr(label_or_viewer, 'size'):
            # QLabel
            control_size = label_or_viewer.size()
            max_width = max(200, control_size.width() - 20)
            max_height = max(200, control_size.height() - 20)
        else:
            # ZoomableImageViewer
            max_width = 800
            max_height = 600
        
        # 保持比例调整图片大小
        original_width, original_height = image.size
        scale = min(max_width / original_width, max_height / original_height)
        
        # 确保缩放后的尺寸至少为1像素
        new_width = max(1, int(original_width * scale))
        new_height = max(1, int(original_height * scale))
        
        image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为QPixmap
        image_array = np.array(image_resized)
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 根据控件类型设置图像
        if hasattr(label_or_viewer, 'setPixmap'):
            # QLabel
            label_or_viewer.setPixmap(pixmap)
        elif hasattr(label_or_viewer, 'set_image'):
            # ZoomableImageViewer
            label_or_viewer.set_image(pixmap)
        
    except Exception as e:
        logger.error(f"显示图片失败: {str(e)}")
        import traceback
        traceback.print_exc()


def display_feature_visualization(features, viewer):
    """显示特征可视化结果"""
    try:
        # 创建特征图网格
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('特征可视化')
        
        feature_items = list(features.items())
        for i, (module, feature) in enumerate(feature_items[:16]):
            row, col = i // 4, i % 4
            
            # 取第一个特征图并标准化
            feature_map = feature[0, 0].cpu().numpy()
            
            # 使用统一的特征图标准化函数
            normalized_feature = normalize_feature_map(feature_map)
            
            axes[row, col].imshow(normalized_feature, cmap='viridis', vmin=0, vmax=1)
            axes[row, col].set_title(f'Layer {i+1}')
            axes[row, col].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(feature_items), 16):
            row, col = i // 4, i % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # 转换为QPixmap并显示
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # 直接从buffer创建QPixmap，避免数据转换问题
        pixmap = QPixmap()
        if not pixmap.loadFromData(buffer.getvalue()):
            logger.error("无法加载特征可视化图像数据")
            return
        
        # 设置图像到查看器
        viewer.set_image(pixmap)
        
        plt.close()
        buffer.close()
        
    except Exception as e:
        logger.error(f"显示特征可视化失败: {str(e)}")


def display_gradcam(gradcam, original_image, viewer, class_name):
    """显示GradCAM结果"""
    try:
        import cv2
        # 调整GradCAM到原图大小
        gradcam_resized = cv2.resize(gradcam, (original_image.width, original_image.height))
        
        # 标准化GradCAM数据
        normalized_gradcam = normalize_feature_map(gradcam_resized)
        
        # 创建热力图
        plt.figure(figsize=(12, 6))
        
        # 原图 - 标准化图像数据
        plt.subplot(1, 2, 1)
        normalized_image = normalize_image_for_matplotlib(original_image)
        # 确保图像数据在0-1范围内
        normalized_image = np.clip(normalized_image, 0, 1)
        plt.imshow(normalized_image, vmin=0, vmax=1)
        plt.title('原始图片')
        plt.axis('off')
        
        # GradCAM叠加
        plt.subplot(1, 2, 2)
        plt.imshow(normalized_image, vmin=0, vmax=1)
        plt.imshow(normalized_gradcam, alpha=0.4, cmap='jet', vmin=0, vmax=1)
        plt.title(f'GradCAM - {class_name}')
        plt.axis('off')
        
        plt.tight_layout()
        
        # 转换为QPixmap并显示
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # 直接从buffer创建QPixmap，避免数据转换问题
        pixmap = QPixmap()
        if not pixmap.loadFromData(buffer.getvalue()):
            logger.error("无法加载GradCAM图像数据")
            return
        
        # 设置图像到查看器
        viewer.set_image(pixmap)
        
        plt.close()
        buffer.close()
        
    except Exception as e:
        logger.error(f"显示GradCAM失败: {str(e)}")


def display_lime_explanation(explanation, original_image, viewer, target_class, class_names):
    """显示LIME解释结果"""
    try:
        # 获取可用的标签
        # LIME的ImageExplanation对象中没有available_labels属性
        # 而是通过top_labels方法获取预测结果
        top_labels = explanation.top_labels
        
        if not top_labels:
            # 如果没有预测结果，使用目标类别
            top_labels = [target_class]
        
        if target_class not in top_labels:
            # 如果目标类别不在预测结果中，使用第一个预测结果
            target_class = top_labels[0]
        
        temp, mask = explanation.get_image_and_mask(
            target_class, 
            positive_only=False,  # 显示正面和负面影响
            num_features=10,      # 显示更多特征
            hide_rest=True        # 隐藏不重要的区域
        )
        
        # 创建图像显示
        plt.figure(figsize=(12, 6))
        
        # 原图 - 标准化图像数据
        plt.subplot(1, 2, 1)
        normalized_original = normalize_image_for_matplotlib(original_image)
        # 确保图像数据在0-1范围内
        normalized_original = np.clip(normalized_original, 0, 1)
        plt.imshow(normalized_original, vmin=0, vmax=1)
        plt.title('原始图片')
        plt.axis('off')
        
        # LIME解释 - 标准化解释图像数据
        plt.subplot(1, 2, 2)
        normalized_temp = normalize_image_for_matplotlib(temp)
        # 确保图像数据在0-1范围内
        normalized_temp = np.clip(normalized_temp, 0, 1)
        plt.imshow(normalized_temp, vmin=0, vmax=1)
        # 使用实际解释的类别名称
        class_name = class_names[target_class] if target_class < len(class_names) else f"类别{target_class}"
        plt.title(f'LIME解释 - {class_name}')
        plt.axis('off')
        
        plt.tight_layout()
        
        # 转换为QPixmap并显示
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # 直接从buffer创建QPixmap，避免数据转换问题
        pixmap = QPixmap()
        if not pixmap.loadFromData(buffer.getvalue()):
            logger.error("无法加载LIME图像数据")
            return
        
        # 设置图像到查看器
        viewer.set_image(pixmap)
        
        plt.close()
        buffer.close()
        
    except Exception as e:
        logger.error(f"显示LIME解释失败: {str(e)}")


def display_integrated_gradients(gradients, original_image, viewer, class_name):
    """显示Integrated Gradients结果"""
    try:
        logger.info(f"Integrated Gradients形状: {gradients.shape}")
        
        # 转换梯度为显示格式
        # 如果是3通道，计算L2范数
        if len(gradients.shape) == 3 and gradients.shape[0] == 3:
            attribution = np.sqrt(np.sum(gradients ** 2, axis=0))
        elif len(gradients.shape) == 3:
            attribution = np.abs(gradients).sum(axis=0)
        elif len(gradients.shape) == 2:
            attribution = np.abs(gradients)
        else:
            logger.error(f"不支持的Integrated Gradients形状: {gradients.shape}")
            return
        
        # 标准化
        if attribution.max() > 0:
            attribution = attribution / attribution.max()
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        normalized_image = normalize_image_for_matplotlib(original_image)
        # 确保图像数据在0-1范围内
        normalized_image = np.clip(normalized_image, 0, 1)
        axes[0].imshow(normalized_image, vmin=0, vmax=1)
        axes[0].set_title('原始图片')
        axes[0].axis('off')
        
        # 积分梯度热力图
        im = axes[1].imshow(attribution, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'Integrated Gradients - {class_name}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # 叠加图
        axes[2].imshow(normalized_image, vmin=0, vmax=1)
        axes[2].imshow(attribution, alpha=0.5, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('叠加显示')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 转换为QPixmap并显示
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        pixmap = QPixmap()
        if not pixmap.loadFromData(buffer.getvalue()):
            logger.error("无法加载Integrated Gradients图像数据")
            return
        
        viewer.set_image(pixmap)
        
        plt.close()
        buffer.close()
        
    except Exception as e:
        logger.error(f"显示Integrated Gradients失败: {str(e)}")


def display_shap_explanation(shap_result, original_image, viewer, class_name):
    """显示SHAP解释结果"""
    try:
        shap_values = shap_result.get('shap_values')
        method = shap_result.get('method', 'partition')
        
        if shap_values is None:
            logger.error("SHAP值为空")
            return
        
        # 处理SHAP值 - 修复多维数据问题
        logger.info(f"SHAP值形状: {shap_values.shape}")
        
        # 如果SHAP返回4维数据 (C, H, W, num_classes)，取第一个类别或指定类别
        if len(shap_values.shape) == 4:
            # 形状为 (C, H, W, num_classes)，取最后一维的第一个
            shap_values = shap_values[:, :, :, 0]
            logger.info(f"处理后SHAP值形状: {shap_values.shape}")
        
        # 如果是3D并且第一个维度是通道数
        if len(shap_values.shape) == 3 and shap_values.shape[0] == 3:
            # RGB图像，计算总重要性
            attribution = np.sqrt(np.sum(shap_values ** 2, axis=0))
        elif len(shap_values.shape) == 3:
            # 如果是其他3D情况，沿第一个轴求和
            attribution = np.abs(shap_values).sum(axis=0)
        elif len(shap_values.shape) == 2:
            # 如果是2D，直接使用
            attribution = np.abs(shap_values)
        else:
            logger.error(f"不支持的SHAP值形状: {shap_values.shape}")
            return
        
        # 标准化
        if attribution.max() > 0:
            attribution = attribution / attribution.max()
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        normalized_image = normalize_image_for_matplotlib(original_image)
        # 确保图像数据在0-1范围内
        normalized_image = np.clip(normalized_image, 0, 1)
        axes[0].imshow(normalized_image, vmin=0, vmax=1)
        axes[0].set_title('原始图片')
        axes[0].axis('off')
        
        # SHAP热力图
        im = axes[1].imshow(attribution, cmap='RdBu_r', vmin=0, vmax=1)
        axes[1].set_title(f'SHAP解释 ({method}) - {class_name}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # 叠加图
        axes[2].imshow(normalized_image, vmin=0, vmax=1)
        axes[2].imshow(attribution, alpha=0.5, cmap='RdBu_r', vmin=0, vmax=1)
        axes[2].set_title('叠加显示')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 转换为QPixmap并显示
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        pixmap = QPixmap()
        if not pixmap.loadFromData(buffer.getvalue()):
            logger.error("无法加载SHAP图像数据")
            return
        
        viewer.set_image(pixmap)
        
        plt.close()
        buffer.close()
        
    except Exception as e:
        logger.error(f"显示SHAP解释失败: {str(e)}")


def display_smoothgrad(gradients, original_image, viewer, class_name):
    """显示SmoothGrad结果"""
    try:
        logger.info(f"SmoothGrad形状: {gradients.shape}")
        
        # 转换梯度为显示格式
        if len(gradients.shape) == 3 and gradients.shape[0] == 3:
            attribution = np.sqrt(np.sum(gradients ** 2, axis=0))
        elif len(gradients.shape) == 3:
            attribution = np.abs(gradients).sum(axis=0)
        elif len(gradients.shape) == 2:
            attribution = np.abs(gradients)
        else:
            logger.error(f"不支持的SmoothGrad形状: {gradients.shape}")
            return
        
        # 标准化
        if attribution.max() > 0:
            attribution = attribution / attribution.max()
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        normalized_image = normalize_image_for_matplotlib(original_image)
        # 确保图像数据在0-1范围内
        normalized_image = np.clip(normalized_image, 0, 1)
        axes[0].imshow(normalized_image, vmin=0, vmax=1)
        axes[0].set_title('原始图片')
        axes[0].axis('off')
        
        # SmoothGrad热力图
        im = axes[1].imshow(attribution, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title(f'SmoothGrad - {class_name}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # 叠加图
        axes[2].imshow(normalized_image, vmin=0, vmax=1)
        axes[2].imshow(attribution, alpha=0.5, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('叠加显示')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 转换为QPixmap并显示
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        pixmap = QPixmap()
        if not pixmap.loadFromData(buffer.getvalue()):
            logger.error("无法加载SmoothGrad图像数据")
            return
        
        viewer.set_image(pixmap)
        
        plt.close()
        buffer.close()
        
    except Exception as e:
        logger.error(f"显示SmoothGrad失败: {str(e)}")


def display_sensitivity_analysis(result, viewer, class_name):
    """显示敏感性分析结果"""
    try:
        epsilons = result['epsilons']
        predictions = result['predictions']
        original_confidence = result['original_confidence']
        
        # 创建敏感性曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, predictions, 'b-', linewidth=2, label='预测置信度')
        plt.axhline(y=original_confidence, color='r', linestyle='--', label='原始置信度')
        plt.xlabel('扰动强度')
        plt.ylabel('预测置信度')
        plt.title(f'敏感性分析 - {class_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 转换为QPixmap并显示
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # 直接从buffer创建QPixmap，避免数据转换问题
        pixmap = QPixmap()
        if not pixmap.loadFromData(buffer.getvalue()):
            logger.error("无法加载敏感性分析图像数据")
            return
        
        # 设置图像到查看器
        viewer.set_image(pixmap)
        
        plt.close()
        buffer.close()
        
    except Exception as e:
        logger.error(f"显示敏感性分析失败: {str(e)}") 