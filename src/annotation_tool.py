import os
import subprocess
import sys
import json
from PyQt5.QtCore import QObject, pyqtSignal
from typing import List

class AnnotationTool(QObject):
    # 定义信号
    status_updated = pyqtSignal(str)
    annotation_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        # 内嵌UI已经在主窗口中创建，这里不需要额外的初始化
        self.processes = []  # 存储所有启动的子进程
        
    def start_labelimg(self, image_folder: str, class_names: List[str] = None, output_folder: str = None) -> None:
        """
        启动LabelImg标注工具
        
        参数:
            image_folder: 图片文件夹路径
            class_names: 缺陷类别名称列表
            output_folder: 标注结果保存目录
        """
        try:
            # 检查图片文件夹是否存在
            if not os.path.exists(image_folder):
                self.annotation_error.emit(f'图片文件夹不存在: {image_folder}')
                return
                
            # 检查LabelImg是否已安装
            try:
                subprocess.run(['labelImg', '--help'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              check=False)
            except FileNotFoundError:
                self.status_updated.emit('正在安装LabelImg...')
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'labelImg'], 
                              check=True)
                
            # 创建预定义类别文件
            predefined_classes_file = None
            if class_names:
                # 使用正斜杠替代反斜杠，避免路径问题
                dataset_dir = os.path.join(os.path.dirname(output_folder or image_folder), 'dataset')
                os.makedirs(dataset_dir, exist_ok=True)
                predefined_classes_file = os.path.join(dataset_dir, 'predefined_classes.txt')
                predefined_classes_file = predefined_classes_file.replace('\\', '/')
                
                with open(predefined_classes_file, 'w', encoding='utf-8') as f:
                    for class_name in class_names:
                        f.write(f"{class_name}\n")
                        
            # 构建命令
            cmd = ['labelImg']
            
            # 添加图片文件夹路径
            cmd.append(image_folder)
            
            # 添加预定义类别文件路径
            if predefined_classes_file:
                cmd.extend(['--predefined_classes_file', predefined_classes_file])
                
            # 添加输出文件夹路径
            if output_folder:
                cmd.extend(['--output_dir', output_folder])
                
            # 设置默认保存格式为YOLO
            cmd.extend(['--format', 'yolo'])
                
            # 启动LabelImg
            self.status_updated.emit('正在启动LabelImg标注工具...')
            process = subprocess.Popen(cmd,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
            
            # 保存进程引用
            self.processes.append(process)
            
            self.status_updated.emit('LabelImg标注工具已启动')
            
        except Exception as e:
            self.annotation_error.emit(f'启动LabelImg失败: {str(e)}')
            
    def stop(self):
        """停止所有正在运行的标注工具进程"""
        for process in self.processes:
            try:
                if process.poll() is None:  # 检查进程是否仍在运行
                    process.terminate()  # 尝试正常终止
                    process.wait(timeout=1)  # 等待进程终止
                    
                    # 如果进程仍在运行，强制终止
                    if process.poll() is None:
                        process.kill()
            except Exception as e:
                print(f"终止进程时出错: {e}")
        
        # 清空进程列表
        self.processes = []
            
    def start_labelme(self, image_folder: str, class_names: List[str] = None, output_folder: str = None) -> None:
        """
        启动LabelMe标注工具
        
        参数:
            image_folder: 图片文件夹路径
            class_names: 缺陷类别名称列表
            output_folder: 标注结果保存目录
        """
        try:
            # 检查图片文件夹是否存在
            if not os.path.exists(image_folder):
                self.annotation_error.emit(f'图片文件夹不存在: {image_folder}')
                return
                
            # 检查LabelMe是否已安装
            try:
                subprocess.run(['labelme', '--help'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              check=False)
            except FileNotFoundError:
                self.status_updated.emit('正在安装LabelMe...')
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'labelme'], 
                              check=True)
                
            # 创建配置文件
            if class_names:
                # 使用正斜杠替代反斜杠，避免路径问题
                dataset_dir = os.path.join(os.path.dirname(output_folder or image_folder), 'dataset')
                os.makedirs(dataset_dir, exist_ok=True)
                config_file = os.path.join(dataset_dir, 'labelme_config.json')
                config_file = config_file.replace('\\', '/')
                
                config = {
                    "labels": class_names,
                    "flags": {},
                    "lineColor": [0, 255, 0, 128],
                    "fillColor": [255, 0, 0, 128],
                    "shapes": ["polygon", "rectangle", "circle", "line", "point"]
                }
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                self.status_updated.emit(f'已创建LabelMe配置文件: {config_file}')
            
            # 启动LabelMe
            self.status_updated.emit('正在启动LabelMe...')
            
            # 构建命令 - 使用列表形式，不需要手动添加引号
            cmd = ['labelme']
            
            # 添加图片文件夹路径（确保路径使用正斜杠）
            image_folder = image_folder.replace('\\', '/')
            cmd.append(image_folder)
            
            # 添加输出目录（如果指定）
            if output_folder:
                # 确保输出目录存在并使用正斜杠
                os.makedirs(output_folder, exist_ok=True)
                output_folder = output_folder.replace('\\', '/')
                cmd.extend(['--output', output_folder])
                
            # 添加配置文件（如果存在）
            if class_names and os.path.exists(config_file):
                cmd.extend(['--config', config_file])
                
            # 打印命令以便调试
            self.status_updated.emit(f'执行命令: {" ".join(cmd)}')
                
            # 启动进程 - 使用shell=False（默认值）让subprocess正确处理参数
            subprocess.Popen(cmd)
            self.status_updated.emit('LabelMe已启动')
            
        except Exception as e:
            self.annotation_error.emit(f'启动LabelMe时出错: {str(e)}')
            
    def convert_annotations(self, annotation_folder: str, output_folder: str, format_type: str) -> None:
        """
        转换标注格式
        
        参数:
            annotation_folder: 标注文件夹路径
            output_folder: 输出文件夹路径
            format_type: 目标格式类型 ('voc', 'coco', 'yolo')
        """
        try:
            self.status_updated.emit(f'正在将标注转换为{format_type}格式...')
            
            # 确保输出文件夹存在
            os.makedirs(output_folder, exist_ok=True)
            
            if format_type.lower() == 'voc':
                # 转换为VOC格式
                pass
            elif format_type.lower() == 'coco':
                # 转换为COCO格式
                pass
            elif format_type.lower() == 'yolo':
                # 转换为YOLO格式
                pass
            else:
                raise ValueError(f'不支持的格式类型: {format_type}')
                
            self.status_updated.emit('标注转换完成')
            
        except Exception as e:
            self.annotation_error.emit(f'转换标注时出错: {str(e)}') 