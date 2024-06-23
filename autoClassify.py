import os
import shutil
import cv2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

# 加载训练好的YOLOv8分类模型
model_path = 'models/train11_ep20.pt'  # 替换为你的模型路径
model = YOLO(model_path)

# 类别名
class_names = ["Manga", "Normal", "Sex"]

# 输入和输出文件夹路径
input_folder = '/Users/oplin/OpDocuments/VscodeProjects/PythonLessonWorks/PixivCrawler/download_images/2024_6_10__16'  # 替换为你的输入文件夹路径
output_folder = '/Users/oplin/OpDocuments/VscodeProjects/PythonLessonWorks/PixivCrawler/download_images/testImgOut'  # 替换为你的输出文件夹路径

# 创建输出目录
for class_name in class_names:
    os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

# 模型推理
print("Start Classify images...")
results = model(source=input_folder, save=False, save_txt=False)

# 处理并移动图片
print("Start moving images...")
for result in tqdm(results, desc="move images"):
    # 获取预测的类别
    pred_class = result.probs.top1  # 使用 top1 属性获取最高概率的类别索引
    pred_class_name = class_names[pred_class]
    
    # 获取图片路径
    img_path = result.path
    
    # 目标路径
    dest_folder = os.path.join(output_folder, pred_class_name)
    shutil.copy(img_path, os.path.join(dest_folder, os.path.basename(img_path)))
    print(f"Image {os.path.basename(img_path)} copied to {pred_class_name}")

print("All images processed and copied to corresponding folders.")