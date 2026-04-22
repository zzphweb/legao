# 注释掉所有torch相关的导入和推理代码
# import torch
# import torchvision.transforms as transforms
from PIL import Image
import numpy as np
# from model.model_def import load_model, CLASS_NAMES, NORMALIZE_MEAN, NORMALIZE_STD, INPUT_SIZE

# 模拟CLASS_NAMES
CLASS_NAMES = ['1×1', '1×2', '1×3', '1×4', '1×6', '2×2', '2×3', '2×4']

def predict(image_path):
    # 临时占位，返回一个假的结果
    return {
        'predicted_class': '1×2',
        'confidence': 0.85,
        'probabilities': [0.05, 0.85, 0.03, 0.02, 0.01, 0.02, 0.01, 0.01],
        'class_names': CLASS_NAMES
    }

def predict_from_bytes(image_bytes):
    # 临时占位，返回一个假的结果
    return {
        'predicted_class': '1×2',
        'confidence': 0.85,
        'probabilities': [0.05, 0.85, 0.03, 0.02, 0.01, 0.02, 0.01, 0.01],
        'class_names': CLASS_NAMES
    }