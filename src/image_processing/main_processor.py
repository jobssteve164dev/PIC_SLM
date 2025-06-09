"""
主图像处理器
协调所有图像处理组件，提供统一的处理接口
"""

from typing import Dict, List
from .base_processor import BaseImageProcessor
from .image_transformer import ImageTransformer
from .augmentation_manager import AugmentationManager
from .dataset_creator import DatasetCreator
from .class_balancer import ClassBalancer
from .sampling_manager import SamplingManager
from .file_manager import FileManager


class ImagePreprocessor(BaseImageProcessor):
    """
    主图像预处理器类
    协调所有组件，提供完整的图像预处理功能
    """

    def __init__(self):
        super().__init__()
        
        # 初始化所有组件
        self.image_transformer = ImageTransformer()
        self.augmentation_manager = AugmentationManager()
        self.dataset_creator = DatasetCreator(self.augmentation_manager)
        self.class_balancer = ClassBalancer(self.image_transformer, self.augmentation_manager)
        self.sampling_manager = SamplingManager(self.image_transformer, self.augmentation_manager)
        self.file_manager = FileManager()

    def preprocess_images(self, params: Dict) -> None:
        """
        预处理图片并创建数据集
        
        参数:
            params: 包含预处理参数的字典
                - source_folder: 源文件夹路径
                - target_folder: 预处理目标文件夹路径
                - width: 目标宽度
                - height: 目标高度
                - format: 目标格式 (jpg, png, bmp)
                - brightness_value: 亮度调整值 (-50 到 50)
                - contrast_value: 对比度调整值 (-50 到 50)
                - train_ratio: 训练集比例 (0.1 到 0.9)
                - augmentation_level: 数据增强级别 ('基础', '中等', '强化')
                - dataset_folder: 数据集输出文件夹
                - brightness: 是否启用亮度增强 (布尔值)
                - contrast: 是否启用对比度增强 (布尔值)
                - noise: 是否启用噪声增强 (布尔值)
                - blur: 是否启用模糊增强 (布尔值)
                - hue: 是否启用色相增强 (布尔值)
                - balance_classes: 是否保持类别平衡 (布尔值)
                - class_names: 类别名称列表
        """
        try:
            # 重置停止标志
            self._stop_preprocessing = False
            
            # 参数预处理和验证
            self._prepare_parameters(params)
            
            # 用于调试的打印
            self._log_parameters(params)
            
            # 检查处理模式
            use_class_balance = params.get('balance_classes', False)
            use_sampling = params.get('use_sampling', False)
            class_names = params.get('class_names', [])
            
            # 根据处理模式选择不同的处理方式
            if use_sampling and class_names:
                self.status_updated.emit('使用采样平衡预处理模式')
                self._preprocess_with_sampling(params)
            elif use_class_balance and class_names:
                self.status_updated.emit('使用类别平衡预处理模式')
                self._preprocess_with_class_balance(params)
            else:
                self._preprocess_standard_mode(params)
            
            # 发出完成信号
            self._emit_completion_signal(use_class_balance and class_names)
            
        except Exception as e:
            self.emit_error(f'预处理过程中出错: {str(e)}')

    def _prepare_parameters(self, params: Dict) -> None:
        """准备和验证参数"""
        # 处理亮度和对比度参数的命名冲突
        if isinstance(params.get('brightness'), bool):
            brightness_value = params.get('brightness_value', 0)
            params['brightness_value'] = brightness_value
        elif isinstance(params.get('brightness'), (int, float)):
            brightness_value = params.get('brightness', 0)
            params['brightness_value'] = brightness_value
        
        if isinstance(params.get('contrast'), bool):
            contrast_value = params.get('contrast_value', 0)
            params['contrast_value'] = contrast_value
        elif isinstance(params.get('contrast'), (int, float)):
            contrast_value = params.get('contrast', 0)
            params['contrast_value'] = contrast_value

    def _log_parameters(self, params: Dict) -> None:
        """记录参数信息用于调试"""
        self.status_updated.emit(f"增强模式: {params.get('augmentation_mode')}")
        self.status_updated.emit(f"亮度调整: {params.get('brightness')}")
        self.status_updated.emit(f"对比度调整: {params.get('contrast')}")
        self.status_updated.emit(f"噪声: {params.get('noise')}")
        self.status_updated.emit(f"模糊: {params.get('blur')}")
        self.status_updated.emit(f"色相: {params.get('hue')}")
        self.status_updated.emit(f"类别平衡: {params.get('balance_classes', False)}")

    def _preprocess_with_sampling(self, params: Dict) -> None:
        """使用采样平衡模式预处理"""
        sampling_results = self.sampling_manager.balance_dataset_with_sampling(
            params,
            self.file_manager.get_image_files,
            progress_callback=self.progress_updated.emit,
            status_callback=self.status_updated.emit,
            stop_check=lambda: self._stop_preprocessing
        )
        
        # 输出采样结果统计
        if sampling_results:
            self.status_updated.emit("\n=== 采样结果统计 ===")
            for class_name in sampling_results['original_distribution']:
                original = sampling_results['original_distribution'][class_name]
                final = sampling_results['final_distribution'][class_name]
                method = sampling_results['sampling_methods'][class_name]
                self.status_updated.emit(f"{class_name}: {original} -> {final} ({method})")

    def _preprocess_with_class_balance(self, params: Dict) -> None:
        """使用类别平衡模式预处理"""
        self.class_balancer.preprocess_with_class_balance(
            params,
            self.file_manager.get_image_files,
            progress_callback=self.progress_updated.emit,
            status_callback=self.status_updated.emit,
            stop_check=lambda: self._stop_preprocessing
        )

    def _preprocess_standard_mode(self, params: Dict) -> None:
        """使用标准模式预处理"""
        # 第一步：预处理图片
        self.status_updated.emit('第1步：开始预处理图片...')
        self._preprocess_raw_images(params)
        
        # 如果处理被停止，则退出
        if self._stop_preprocessing:
            self.status_updated.emit('预处理已停止')
            return
        
        # 第二步：创建数据集
        self.status_updated.emit('第2步：开始创建数据集...')
        self._create_dataset(params)
        
        # 如果处理被停止，则退出
        if self._stop_preprocessing:
            self.status_updated.emit('预处理已停止')
            return

    def _preprocess_raw_images(self, params: Dict) -> None:
        """预处理原始图片"""
        source_folder = params['source_folder']
        target_folder = params['target_folder']
        width = params['width']
        height = params['height']
        img_format = params['format']
        brightness = params['brightness_value']
        contrast = params['contrast_value']
        
        # 确保目标文件夹存在
        self.file_manager.ensure_directory_exists(target_folder)
        
        # 获取所有图片文件
        image_files = self.file_manager.get_image_files(source_folder)
        total_files = len(image_files)
        
        if total_files == 0:
            self.emit_error('未找到图片文件')
            return
            
        self.status_updated.emit(f'找到 {total_files} 个图片文件')
        
        # 处理每个图片
        for i, img_file in enumerate(image_files):
            # 检查是否需要停止处理
            if self._stop_preprocessing:
                self.status_updated.emit('图片预处理已停止')
                return
                
            try:
                # 构建源文件和目标文件路径
                source_path = self.file_manager.join_path(source_folder, img_file)
                file_name = self.file_manager.get_file_name_without_extension(img_file)
                target_path = self.file_manager.join_path(target_folder, f"{file_name}.{img_format}")
                
                # 处理图片
                self.image_transformer.process_single_image(
                    source_path, target_path, width, height, brightness, contrast
                )
                
                # 更新进度
                progress = int(((i + 1) / total_files) * 50)  # 预处理占总进度的50%
                self.progress_updated.emit(progress)
                
            except Exception as e:
                self.status_updated.emit(f'处理图片 {img_file} 时出错: {str(e)}')
        
        self.status_updated.emit('图片预处理完成')

    def _create_dataset(self, params: Dict) -> None:
        """创建训练和验证数据集"""
        preprocessed_folder = params['target_folder']
        dataset_folder = params['dataset_folder']
        train_ratio = params['train_ratio']
        augmentation_level = params['augmentation_level']
        
        # 检查是否需要停止处理
        if self._stop_preprocessing:
            return
        
        # 获取所有预处理后的图片
        image_files = self.file_manager.get_image_files(preprocessed_folder)
        
        if not image_files:
            self.emit_error('未找到预处理后的图片')
            return
        
        # 使用数据集创建器创建数据集
        self.dataset_creator.create_dataset(
            preprocessed_folder, dataset_folder, train_ratio, 
            augmentation_level, image_files, params,
            progress_callback=self.progress_updated.emit,
            status_callback=self.status_updated.emit,
            stop_check=lambda: self._stop_preprocessing
        )

    def _emit_completion_signal(self, use_class_balance: bool) -> None:
        """发出完成信号"""
        if not self._stop_preprocessing:
            if use_class_balance:
                self.status_updated.emit('类别平衡预处理全部完成')
            else:
                self.status_updated.emit('标准预处理全部完成')
        
        # 无论如何，都发出预处理完成信号 - 确保UI始终得到更新
        self.preprocessing_finished.emit()
        print("ImagePreprocessor: 已发出 preprocessing_finished 信号")

    def create_class_folders(self, base_folder: str, class_names: List[str]) -> None:
        """为每个类别创建文件夹"""
        try:
            self.file_manager.create_class_folders(
                base_folder, class_names, self.status_updated.emit
            )
        except Exception as e:
            self.emit_error(f'创建类别文件夹时出错: {str(e)}')

    # 保持向后兼容的方法
    def _get_image_files(self, folder_path: str) -> List[str]:
        """获取文件夹中的所有图片文件（向后兼容）"""
        return self.file_manager.get_image_files(folder_path)

    def _process_single_image(self, source_path: str, target_path: str, 
                             width: int, height: int, brightness: int, contrast: int) -> None:
        """处理单个图片（向后兼容）"""
        self.image_transformer.process_single_image(
            source_path, target_path, width, height, brightness, contrast
        ) 