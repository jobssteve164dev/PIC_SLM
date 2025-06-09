"""
高级采样模块 - 基于图像特征的智能采样
提供真正的聚类采样、基于特征的采样等高级方法
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from PIL import Image
import random


class AdvancedSamplingManager:
    """高级采样管理器，提供基于图像特征的智能采样"""
    
    def __init__(self):
        self.feature_cache = {}  # 特征缓存
        
    def cluster_based_sampling(self, files: List[str], source_folder: str, 
                             target_samples: int, 
                             status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        基于真正聚类的采样
        
        Args:
            files: 文件列表
            source_folder: 源文件夹路径
            target_samples: 目标样本数
            status_callback: 状态回调函数
            
        Returns:
            List[str]: 选中的文件列表
        """
        if len(files) <= target_samples:
            return files
            
        if status_callback:
            status_callback(f"正在提取 {len(files)} 张图片的特征...")
            
        # 提取所有图像的特征
        features = []
        valid_files = []
        
        for i, file_path in enumerate(files):
            try:
                full_path = os.path.join(source_folder, file_path)
                feature = self._extract_image_features(full_path)
                if feature is not None:
                    features.append(feature)
                    valid_files.append(file_path)
                    
                if status_callback and (i + 1) % 50 == 0:
                    status_callback(f"已提取特征: {i + 1}/{len(files)}")
                    
            except Exception as e:
                if status_callback:
                    status_callback(f"提取特征失败: {file_path} - {str(e)}")
                continue
                
        if len(features) == 0:
            if status_callback:
                status_callback("警告：未能提取任何有效特征，使用随机采样")
            return random.sample(files, min(target_samples, len(files)))
            
        features = np.array(features)
        
        if status_callback:
            status_callback(f"开始聚类分析，目标聚类数: {target_samples}")
            
        # 执行K-means聚类
        try:
            # 如果目标样本数大于等于特征数，则聚类数等于特征数
            n_clusters = min(target_samples, len(features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            if status_callback:
                status_callback(f"聚类完成，共 {n_clusters} 个聚类")
                
            # 从每个聚类中选择最接近聚类中心的样本
            selected_files = []
            cluster_centers = kmeans.cluster_centers_
            
            for cluster_id in range(n_clusters):
                # 找到属于当前聚类的所有样本
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                
                if len(cluster_indices) == 0:
                    continue
                    
                # 计算每个样本到聚类中心的距离
                cluster_features = features[cluster_indices]
                center = cluster_centers[cluster_id]
                
                # 找到距离聚类中心最近的样本
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                
                selected_files.append(valid_files[closest_idx])
                
                if status_callback and len(selected_files) % 10 == 0:
                    status_callback(f"已选择样本: {len(selected_files)}/{target_samples}")
                    
            # 如果选择的样本数少于目标数，随机补充
            if len(selected_files) < target_samples:
                remaining_files = [f for f in valid_files if f not in selected_files]
                additional_needed = target_samples - len(selected_files)
                additional_files = random.sample(remaining_files, 
                                               min(additional_needed, len(remaining_files)))
                selected_files.extend(additional_files)
                
            if status_callback:
                status_callback(f"聚类采样完成，选择了 {len(selected_files)} 个样本")
                
            return selected_files[:target_samples]
            
        except Exception as e:
            if status_callback:
                status_callback(f"聚类失败: {str(e)}，使用随机采样")
            return random.sample(valid_files, min(target_samples, len(valid_files)))
            
    def diversity_based_sampling(self, files: List[str], source_folder: str,
                                target_samples: int,
                                status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        基于多样性的采样 - 最大化样本间的差异
        
        Args:
            files: 文件列表
            source_folder: 源文件夹路径
            target_samples: 目标样本数
            status_callback: 状态回调函数
            
        Returns:
            List[str]: 选中的文件列表
        """
        if len(files) <= target_samples:
            return files
            
        if status_callback:
            status_callback("开始基于多样性的采样...")
            
        # 提取特征
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
                
        if len(features) == 0:
            return random.sample(files, min(target_samples, len(files)))
            
        features = np.array(features)
        
        # 使用贪心算法选择最大化多样性的样本
        selected_indices = []
        remaining_indices = list(range(len(features)))
        
        # 随机选择第一个样本
        first_idx = random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # 迭代选择与已选样本差异最大的样本
        for _ in range(target_samples - 1):
            if not remaining_indices:
                break
                
            max_min_distance = -1
            best_idx = None
            
            for candidate_idx in remaining_indices:
                # 计算候选样本与所有已选样本的最小距离
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    distance = np.linalg.norm(features[candidate_idx] - features[selected_idx])
                    min_distance = min(min_distance, distance)
                    
                # 选择最小距离最大的候选样本
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = candidate_idx
                    
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                
                if status_callback and len(selected_indices) % 10 == 0:
                    status_callback(f"多样性采样进度: {len(selected_indices)}/{target_samples}")
                    
        selected_files = [valid_files[i] for i in selected_indices]
        
        if status_callback:
            status_callback(f"多样性采样完成，选择了 {len(selected_files)} 个样本")
            
        return selected_files
        
    def quality_based_sampling(self, files: List[str], source_folder: str,
                             target_samples: int,
                             status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        基于图像质量的采样 - 优先选择高质量图像
        
        Args:
            files: 文件列表
            source_folder: 源文件夹路径
            target_samples: 目标样本数
            status_callback: 状态回调函数
            
        Returns:
            List[str]: 选中的文件列表
        """
        if len(files) <= target_samples:
            return files
            
        if status_callback:
            status_callback("开始基于质量的采样...")
            
        # 计算每个图像的质量分数
        quality_scores = []
        valid_files = []
        
        for i, file_path in enumerate(files):
            try:
                full_path = os.path.join(source_folder, file_path)
                score = self._calculate_image_quality(full_path)
                if score is not None:
                    quality_scores.append((score, file_path))
                    valid_files.append(file_path)
                    
                if status_callback and (i + 1) % 50 == 0:
                    status_callback(f"质量评估进度: {i + 1}/{len(files)}")
                    
            except:
                continue
                
        if not quality_scores:
            return random.sample(files, min(target_samples, len(files)))
            
        # 按质量分数排序，选择质量最高的样本
        quality_scores.sort(reverse=True, key=lambda x: x[0])
        selected_files = [item[1] for item in quality_scores[:target_samples]]
        
        if status_callback:
            status_callback(f"质量采样完成，选择了 {len(selected_files)} 个高质量样本")
            
        return selected_files
        
    def _extract_image_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        提取图像特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            np.ndarray: 特征向量，失败时返回None
        """
        # 检查缓存
        if image_path in self.feature_cache:
            return self.feature_cache[image_path]
            
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 调整大小以提高计算效率
            image = cv2.resize(image, (64, 64))
            
            # 提取多种特征
            features = []
            
            # 1. 颜色直方图特征
            hist_features = self._extract_color_histogram(image)
            features.extend(hist_features)
            
            # 2. 纹理特征 (LBP)
            texture_features = self._extract_texture_features(image)
            features.extend(texture_features)
            
            # 3. 统计特征
            stat_features = self._extract_statistical_features(image)
            features.extend(stat_features)
            
            feature_vector = np.array(features)
            
            # 缓存特征
            self.feature_cache[image_path] = feature_vector
            
            return feature_vector
            
        except Exception as e:
            return None
            
    def _extract_color_histogram(self, image: np.ndarray) -> List[float]:
        """提取颜色直方图特征"""
        features = []
        
        # 分别计算RGB三个通道的直方图
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, [16], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-8)  # 归一化
            features.extend(hist)
            
        return features
        
    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """提取纹理特征（简化版LBP）"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 计算局部方差作为纹理特征
        kernel = np.ones((3, 3), np.float32) / 9
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        
        # 统计方差的分布
        var_hist, _ = np.histogram(variance.flatten(), bins=16, range=(0, variance.max()))
        var_hist = var_hist / (var_hist.sum() + 1e-8)
        
        return var_hist.tolist()
        
    def _extract_statistical_features(self, image: np.ndarray) -> List[float]:
        """提取统计特征"""
        features = []
        
        # 对每个颜色通道计算统计特征
        for channel in range(3):
            channel_data = image[:, :, channel].flatten()
            
            # 基本统计量
            features.append(np.mean(channel_data))      # 均值
            features.append(np.std(channel_data))       # 标准差
            features.append(np.min(channel_data))       # 最小值
            features.append(np.max(channel_data))       # 最大值
            features.append(np.median(channel_data))    # 中位数
            
        return features
        
    def _calculate_image_quality(self, image_path: str) -> Optional[float]:
        """
        计算图像质量分数
        
        Args:
            image_path: 图像路径
            
        Returns:
            float: 质量分数，越高表示质量越好
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 质量评估指标
            
            # 1. 清晰度 (基于拉普拉斯算子)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. 对比度
            contrast = gray.std()
            
            # 3. 亮度分布 (避免过暗或过亮)
            mean_brightness = gray.mean()
            brightness_penalty = abs(mean_brightness - 127.5) / 127.5
            brightness_score = 1 - brightness_penalty
            
            # 4. 边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # 综合质量分数
            quality_score = (
                laplacian_var * 0.4 +      # 清晰度权重40%
                contrast * 0.3 +           # 对比度权重30%
                brightness_score * 100 * 0.2 +  # 亮度权重20%
                edge_density * 1000 * 0.1  # 边缘密度权重10%
            )
            
            return quality_score
            
        except Exception as e:
            return None
            
    def get_sampling_method_info(self) -> Dict[str, str]:
        """获取所有采样方法的信息"""
        return {
            'cluster': '聚类采样 - 基于K-means聚类选择代表性样本',
            'diversity': '多样性采样 - 最大化样本间差异',
            'quality': '质量采样 - 优先选择高质量图像',
            'random': '随机采样 - 随机选择样本'
        }
        
    def clear_cache(self):
        """清空特征缓存"""
        self.feature_cache.clear() 