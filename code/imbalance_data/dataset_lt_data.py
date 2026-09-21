# import torch
# import random
# import numpy as np
# import os, sys
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Dataset, Sampler
# from PIL import Image


# def read_class_names(file_path):
#     class_names = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         # 读取所有行并跳过第一行（索引0）
#         lines = f.readlines()[1:]  
#         for line in lines:
#             # 去除行末的换行符并添加到列表
#             class_names.append(line.strip())  
#     return class_names
# class LT_Dataset(Dataset):
    
#     def __init__(self, root, txt, transform=None,labelist=None):
#         self.img_path = []
#         self.labels = []
#         self.transform = transform
#         self.classes = read_class_names(labelist)
#         with open(txt) as f:
#             for line in f:
#                 self.img_path.append(os.path.join(root, line.split()[0]))
#                 self.labels.append(int(line.split()[1]))
#         self.targets = self.labels # Sampler needs to use targets
        
#     def __len__(self):
#         return len(self.labels)
        
#     def __getitem__(self, index):

#         path = self.img_path[index]
#         label = self.labels[index]
        
#         with open(path, 'rb') as f:
#             sample = Image.open(f).convert('RGB')
        
#         if self.transform is not None:
#             sample = self.transform(sample)

#         return sample, label

#     def get_per_class_num(self):
#         num_classes = len(np.unique(self.targets))
#         cls_num_list = [0] * num_classes
#         for label in self.targets:
#             cls_num_list[label] += 1
#         return cls_num_list

import torch
import random
import numpy as np
import os
import json
from torch.utils.data import Dataset
from PIL import Image
from typing import List
# --------------------------------------------------------------------------------
# 1. 为 iNaturalist 设计的类别名称生成器 (辅助类)
# --------------------------------------------------------------------------------
class iNat_ClassName_Generator:
    """
    解析 iNaturalist 的 categories.json 文件并以多种格式生成类别名称。
    """
    def __init__(self, json_file_path: str):
        """
        通过加载和解析JSON文件来初始化生成器。
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                self.categories_data = json.load(f)
            # 按ID排序以确保顺序一致
            self.categories_data.sort(key=lambda x: x['id'])
        except Exception as e:
            print(f"加载或解析 iNaturalist 类别文件时出错: {e}")
            self.categories_data = []

    # --- 修改点 ---
    def get_scientific_names(self) -> List[str]: # 使用大写的 List
        """
        返回科学名称列表 (例如, "Hermodice carunculata")。
        """
        if not self.categories_data:
            return []
        return [category.get('name', 'Unknown') for category in self.categories_data]

    # --- 修改点 ---
    def get_custom_format_names(self, format_string: str) -> List[str]: # 使用大写的 List
        """
        根据自定义格式字符串生成类别名，支持分类学占位符。
        """
        if not self.categories_data:
            return []
            
        custom_names = []
        for category in self.categories_data:
            default_category = {
                'kingdom': 'unknown', 'phylum': 'unknown', 'class': 'unknown',
                'order': 'unknown', 'family': 'unknown', 'genus': 'unknown',
                'name': 'unknown', 'id': -1, 'supercategory': 'unknown'
            }
            default_category.update(category)
            custom_names.append(format_string.format_map(default_category))
            
        return custom_names

# --------------------------------------------------------------------------------
# 2. 统一的类别名称加载函数 (智能判断文件类型)
# --------------------------------------------------------------------------------
def load_class_names(label_file_path: str, inat_template: str = None) -> List[str]: # 使用大写的 List
    """
    从给定的文件路径加载类别名称。
    """
    class_names = []
    
    if label_file_path.endswith('.json'):
        print("检测到 iNaturalist 标签文件 (json)，正在使用 iNat 解析器...")
        generator = iNat_ClassName_Generator(label_file_path)
        if inat_template:
            print(f"使用自定义模板生成类别名称: \"{inat_template}\"")
            class_names = generator.get_custom_format_names(inat_template)
        else:
            print("未提供模板，默认使用科学名称。")
            class_names = generator.get_scientific_names()
            
    else:
        print("检测到 ImageNet-LT 风格的标签文件 (txt)，正在使用行解析器...")
        with open(label_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  
            for line in lines:
                class_names.append(line.strip())
                
    if not class_names:
        raise ValueError(f"无法从文件 {label_file_path} 加载任何类别名称。请检查文件内容和路径。")
        
    return class_names
# --------------------------------------------------------------------------------
# 3. 修改后的 LT_Dataset 类
# --------------------------------------------------------------------------------
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None, labelist=None, inat_template='"a photo of a {name}, a species of {genus} in the family {family}."'):
        self.img_path = []
        self.labels = []
        self.transform = transform
        
        self.classes = load_class_names(labelist, inat_template)
        
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
        self.targets = self.labels

    def __len__(self):
        return len(self.labels)
        

    def get_original_item(self,index):
        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        return sample, label
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

    def get_per_class_num(self):
        num_classes = len(np.unique(self.targets))
        cls_num_list = [0] * num_classes
        for label in self.targets:
            cls_num_list[label] += 1
        return cls_num_list
