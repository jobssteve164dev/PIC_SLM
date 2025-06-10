"""
高级过采样模块 - 实现基于深度学习和计算机视觉的智能过采样方法
提供Mixup、CutMix、基于特征插值、智能增强等高级过采样技术
"""

import os
import cv2
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Callable
from sklearn.neighbors import NearestNeighbors
import math


class AdvancedOversamplingManager:
    """高级过采样管理器，提供基于深度学习和计算机视觉的智能过采样方法"""
    
    def __init__(self, image_transformer=None, augmentation_manager=None):
        self.image_transformer = image_transformer
        self.augmentation_manager = augmentation_manager
        self.feature_cache = {}
        
    def mixup_oversampling(self, files: List[str], source_folder: str, 
                          output_folder: str, needed_samples: int, params: Dict,
                          class_name: str, status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        Mixup过采样 - 混合现有样本生成新样本
        通过线性组合两个样本来生成新的训练样本
        """
        if status_callback:
            status_callback(f"开始Mixup过采样，目标生成 {needed_samples} 个样本")
            
        additional_files = []
        
        for i in range(needed_samples):
            # 随机选择两个不同的样本进行混合
            if len(files) < 2:
                file1 = file2 = random.choice(files)
            else:
                file1, file2 = random.sample(files, 2)
                
            mixup_path = self._generate_mixup_sample(
                file1, file2, source_folder, output_folder, i, params, class_name
            )
            
            if mixup_path:
                additional_files.append(mixup_path)
                
            if status_callback and (i + 1) % 10 == 0:
                status_callback(f"Mixup进度: {i + 1}/{needed_samples}")
                
        if status_callback:
            status_callback(f"Mixup过采样完成，生成了 {len(additional_files)} 个样本")
            
        return additional_files
        
    def cutmix_oversampling(self, files: List[str], source_folder: str,
                           output_folder: str, needed_samples: int, params: Dict,
                           class_name: str, status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        CutMix过采样 - 裁剪和混合技术
        将一个图像的部分区域替换为另一个图像的对应区域
        """
        if status_callback:
            status_callback(f"开始CutMix过采样，目标生成 {needed_samples} 个样本")
            
        additional_files = []
        
        for i in range(needed_samples):
            if len(files) < 2:
                file1 = file2 = random.choice(files)
            else:
                file1, file2 = random.sample(files, 2)
                
            cutmix_path = self._generate_cutmix_sample(
                file1, file2, source_folder, output_folder, i, params, class_name
            )
            
            if cutmix_path:
                additional_files.append(cutmix_path)
                
            if status_callback and (i + 1) % 10 == 0:
                status_callback(f"CutMix进度: {i + 1}/{needed_samples}")
                
        if status_callback:
            status_callback(f"CutMix过采样完成，生成了 {len(additional_files)} 个样本")
            
        return additional_files
        
    def feature_interpolation_oversampling(self, files: List[str], source_folder: str,
                                         output_folder: str, needed_samples: int, params: Dict,
                                         class_name: str, status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        基于特征插值的过采样 - 在特征空间中插值生成新样本
        类似SMOTE但适用于图像数据
        """
        if status_callback:
            status_callback(f"开始特征插值过采样，目标生成 {needed_samples} 个样本")
            
        # 提取所有图像的特征
        features = []
        valid_files = []
        
        for file_path in files:
            try:
                full_path = os.path.join(source_folder, file_path)
                feature = self._extract_image_features(full_path)
                if feature is not None:
                    features.append(feature)
                    valid_files.append(file_path)
            except:
                continue
                
        if len(features) < 2:
            if status_callback:
                status_callback("特征提取失败，回退到基础增强")
            return self._fallback_augmentation(files, source_folder, output_folder, needed_samples, params, class_name)
            
        features = np.array(features)
        additional_files = []
        
        # 使用k-近邻找到相似样本进行插值
        knn = NearestNeighbors(n_neighbors=min(3, len(features)), metric='euclidean')
        knn.fit(features)
        
        for i in range(needed_samples):
            base_idx = random.randint(0, len(features) - 1)
            base_feature = features[base_idx]
            base_file = valid_files[base_idx]
            
            # 找到最近邻
            distances, indices = knn.kneighbors([base_feature])
            neighbor_idx = indices[0][1] if len(indices[0]) > 1 else indices[0][0]
            neighbor_file = valid_files[neighbor_idx]
            
            interpolated_path = self._generate_interpolated_sample(
                base_file, neighbor_file, source_folder, output_folder, i, params, class_name
            )
            
            if interpolated_path:
                additional_files.append(interpolated_path)
                
            if status_callback and (i + 1) % 10 == 0:
                status_callback(f"特征插值进度: {i + 1}/{needed_samples}")
                
        if status_callback:
            status_callback(f"特征插值过采样完成，生成了 {len(additional_files)} 个样本")
            
        return additional_files
        
    def adaptive_oversampling(self, files: List[str], source_folder: str,
                            output_folder: str, needed_samples: int, params: Dict,
                            class_name: str, status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        自适应过采样 - 根据样本特征自动选择最适合的过采样方法
        """
        if status_callback:
            status_callback(f"开始自适应过采样分析...")
            
        # 分析样本特征
        sample_analysis = self._analyze_sample_characteristics(files, source_folder, status_callback)
        
        if sample_analysis['diversity_score'] > 0.7:
            if status_callback:
                status_callback("检测到高多样性，使用特征插值方法")
            return self.feature_interpolation_oversampling(
                files, source_folder, output_folder, needed_samples, params, class_name, status_callback
            )
        elif sample_analysis['similarity_score'] > 0.8:
            if status_callback:
                status_callback("检测到高相似性，使用CutMix方法")
            return self.cutmix_oversampling(
                files, source_folder, output_folder, needed_samples, params, class_name, status_callback
            )
        else:
            if status_callback:
                status_callback("使用Mixup方法")
            return self.mixup_oversampling(
                files, source_folder, output_folder, needed_samples, params, class_name, status_callback
            )
            
    def smart_augmentation_oversampling(self, files: List[str], source_folder: str,
                                      output_folder: str, needed_samples: int, params: Dict,
                                      class_name: str, status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        智能增强过采样 - 基于样本特征选择最适合的增强方法
        """
        if status_callback:
            status_callback(f"开始智能增强过采样，目标生成 {needed_samples} 个样本")
            
        additional_files = []
        
        for i in range(needed_samples):
            source_file = files[i % len(files)]
            source_path = os.path.join(source_folder, source_file)
            
            augmentation_method = self._select_augmentation_method(source_path)
            
            aug_path = self._apply_smart_augmentation(
                source_path, output_folder, i, params, class_name, augmentation_method
            )
            
            if aug_path:
                additional_files.append(aug_path)
                
            if status_callback and (i + 1) % 10 == 0:
                status_callback(f"智能增强进度: {i + 1}/{needed_samples}")
                
                 if status_callback:
             status_callback(f"智能增强过采样完成，生成了 {len(additional_files)} 个样本")
             
         return additional_files
        
    def _generate_mixup_sample(self, file1: str, file2: str, source_folder: str,
                              output_folder: str, index: int, params: Dict, class_name: str) -> Optional[str]:
        """生成Mixup样本"""
        try:
            # 读取两个图像
            path1 = os.path.join(source_folder, file1)
            path2 = os.path.join(source_folder, file2)
            
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            
            if img1 is None or img2 is None:
                return None
                
            # 调整大小到相同尺寸
            target_size = (params['width'], params['height'])
            img1 = cv2.resize(img1, target_size)
            img2 = cv2.resize(img2, target_size)
            
            # 随机生成混合比例
            alpha = random.uniform(0.2, 0.8)
            
            # 混合图像
            mixed_img = (alpha * img1.astype(np.float32) + 
                        (1 - alpha) * img2.astype(np.float32)).astype(np.uint8)
            
            # 保存混合图像
            file_name = f"{class_name}_mixup_{index}.{params['format']}"
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, mixed_img)
            
            return output_path
            
        except Exception as e:
            return None
            
    def _generate_cutmix_sample(self, file1: str, file2: str, source_folder: str,
                               output_folder: str, index: int, params: Dict, class_name: str) -> Optional[str]:
        """生成CutMix样本"""
        try:
            # 读取两个图像
            path1 = os.path.join(source_folder, file1)
            path2 = os.path.join(source_folder, file2)
            
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            
            if img1 is None or img2 is None:
                return None
                
            # 调整大小到相同尺寸
            target_size = (params['width'], params['height'])
            img1 = cv2.resize(img1, target_size)
            img2 = cv2.resize(img2, target_size)
            
            h, w = img1.shape[:2]
            
            # 随机生成裁剪区域
            cut_ratio = random.uniform(0.2, 0.5)
            cut_w = int(w * cut_ratio)
            cut_h = int(h * cut_ratio)
            
            # 随机选择裁剪位置
            cx = random.randint(0, w - cut_w)
            cy = random.randint(0, h - cut_h)
            
            # 执行CutMix
            cutmix_img = img1.copy()
            cutmix_img[cy:cy+cut_h, cx:cx+cut_w] = img2[cy:cy+cut_h, cx:cx+cut_w]
            
            # 保存CutMix图像
            file_name = f"{class_name}_cutmix_{index}.{params['format']}"
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, cutmix_img)
            
            return output_path
            
        except Exception as e:
            return None
            
    def _generate_interpolated_sample(self, file1: str, file2: str, source_folder: str,
                                    output_folder: str, index: int, params: Dict, class_name: str) -> Optional[str]:
        """生成特征插值样本"""
        try:
            # 读取两个图像
            path1 = os.path.join(source_folder, file1)
            path2 = os.path.join(source_folder, file2)
            
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            
            if img1 is None or img2 is None:
                return None
                
            # 调整大小到相同尺寸
            target_size = (params['width'], params['height'])
            img1 = cv2.resize(img1, target_size)
            img2 = cv2.resize(img2, target_size)
            
            # 在HSV空间进行插值，保持更自然的颜色过渡
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            
            # 随机插值比例
            alpha = random.uniform(0.3, 0.7)
            
            # HSV插值
            interpolated_hsv = (alpha * hsv1.astype(np.float32) + 
                              (1 - alpha) * hsv2.astype(np.float32)).astype(np.uint8)
            
            # 转换回BGR
            interpolated_img = cv2.cvtColor(interpolated_hsv, cv2.COLOR_HSV2BGR)
            
            # 保存插值图像
            file_name = f"{class_name}_interp_{index}.{params['format']}"
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, interpolated_img)
            
            return output_path
            
        except Exception as e:
            return None
            
    def _extract_image_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取图像特征"""
        if image_path in self.feature_cache:
            return self.feature_cache[image_path]
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # 转换为RGB并调整大小
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (64, 64))
            
            # 提取颜色特征
            features = []
            for channel in range(3):
                hist = cv2.calcHist([image], [channel], None, [16], [0, 256])
                hist = hist.flatten()
                hist = hist / (hist.sum() + 1e-8)
                features.extend(hist)
                
            # 提取纹理特征
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # 计算梯度
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # 梯度直方图
            grad_hist, _ = np.histogram(grad_mag.flatten(), bins=16, range=(0, grad_mag.max()))
            grad_hist = grad_hist / (grad_hist.sum() + 1e-8)
            features.extend(grad_hist)
            
            feature_vector = np.array(features)
            self.feature_cache[image_path] = feature_vector
            
            return feature_vector
            
        except Exception as e:
            return None
            
    def _analyze_sample_characteristics(self, files: List[str], source_folder: str,
                                      status_callback: Optional[Callable[[str], None]] = None) -> Dict:
        """分析样本特征"""
        if status_callback:
            status_callback("分析样本特征...")
            
        # 提取所有样本的特征
        features = []
        for file_path in files:
            try:
                full_path = os.path.join(source_folder, file_path)
                feature = self._extract_image_features(full_path)
                if feature is not None:
                    features.append(feature)
            except:
                continue
                
        if len(features) < 2:
            return {'diversity_score': 0.5, 'similarity_score': 0.5}
            
        features = np.array(features)
        
        # 计算多样性分数
        pairwise_distances = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                distance = np.linalg.norm(features[i] - features[j])
                pairwise_distances.append(distance)
                
        diversity_score = np.mean(pairwise_distances) / np.std(features) if np.std(features) > 0 else 0.5
        
        # 计算相似性分数
        similarity_score = 1.0 / (1.0 + diversity_score)
        
        return {
            'diversity_score': min(1.0, diversity_score),
            'similarity_score': min(1.0, similarity_score)
        }
        
    def _select_augmentation_method(self, image_path: str) -> str:
        """根据图像特征选择增强方法"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 'rotation'
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 分析图像特征
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # 边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / gray.size
            
            # 根据特征选择增强方法
            if brightness < 100:
                return 'brightness'
            elif contrast < 30:
                return 'contrast'
            elif edge_density < 0.1:
                return 'sharpening'
            else:
                return random.choice(['rotation', 'flip', 'crop', 'noise'])
                
        except Exception as e:
            return 'rotation'
            
    def _apply_smart_augmentation(self, source_path: str, output_folder: str, index: int,
                                params: Dict, class_name: str, method: str) -> Optional[str]:
        """应用智能增强"""
        try:
            image = cv2.imread(source_path)
            if image is None:
                return None
                
            # 应用选择的增强方法
            if method == 'brightness':
                enhanced = self._adjust_brightness(image, factor=random.uniform(1.2, 1.8))
            elif method == 'contrast':
                enhanced = self._adjust_contrast(image, factor=random.uniform(1.2, 2.0))
            elif method == 'sharpening':
                enhanced = self._apply_sharpening(image)
            elif method == 'rotation':
                enhanced = self._apply_rotation(image, angle=random.uniform(-30, 30))
            elif method == 'flip':
                enhanced = cv2.flip(image, random.choice([0, 1, -1]))
            elif method == 'crop':
                enhanced = self._apply_random_crop(image, crop_ratio=0.8)
            elif method == 'noise':
                enhanced = self._add_gaussian_noise(image, std=random.uniform(5, 15))
            else:
                enhanced = image
                
            # 调整到目标尺寸
            enhanced = cv2.resize(enhanced, (params['width'], params['height']))
            
            # 保存增强图像
            file_name = f"{class_name}_smart_{method}_{index}.{params['format']}"
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, enhanced)
            
            return output_path
            
        except Exception as e:
            return None
            
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """调整亮度"""
        return np.clip(image * factor, 0, 255).astype(np.uint8)
        
    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """调整对比度"""
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """应用锐化"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
        
    def _apply_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """应用旋转"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
        
    def _apply_random_crop(self, image: np.ndarray, crop_ratio: float) -> np.ndarray:
        """应用随机裁剪"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        
        start_h = random.randint(0, h - new_h)
        start_w = random.randint(0, w - new_w)
        
        cropped = image[start_h:start_h+new_h, start_w:start_w+new_w]
        return cv2.resize(cropped, (w, h))
        
    def _add_gaussian_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, std, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
        
    def _fallback_augmentation(self, files: List[str], source_folder: str,
                             output_folder: str, needed_samples: int, params: Dict, class_name: str) -> List[str]:
        """备用增强方法"""
        additional_files = []
        
        for i in range(needed_samples):
            source_file = files[i % len(files)]
            source_path = os.path.join(source_folder, source_file)
            
            # 应用基础增强
            aug_path = self._apply_smart_augmentation(
                source_path, output_folder, i, params, class_name, 'rotation'
            )
            
            if aug_path:
                additional_files.append(aug_path)
                
        return additional_files
        
    def get_oversampling_methods(self) -> Dict[str, str]:
        """获取所有过采样方法的信息"""
        return {
            'augmentation': '传统数据增强',
            'duplication': '重复采样',
            'mixup': 'Mixup混合采样',
            'cutmix': 'CutMix裁剪混合',
            'interpolation': '特征插值采样',
            'adaptive': '自适应采样',
            'smart': '智能增强采样'
        }
        
    def clear_cache(self):
        """清空特征缓存"""
        self.feature_cache.clear() 