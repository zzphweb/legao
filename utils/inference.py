import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model.model_def import load_model, CLASS_NAMES, NORMALIZE_MEAN, NORMALIZE_STD, INPUT_SIZE

# 数据变换
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# 加载模型
model = load_model()

def predict(image_path):
    """
    预测图片
    
    Args:
        image_path (str): 图片路径
        
    Returns:
        dict: 预测结果
    """
    try:
        # 读取图片
        img = Image.open(image_path)
        
        # 转换为RGB模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 预处理
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
        
        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 构建结果
        result = {
            'predicted_class': CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].tolist(),
            'class_names': CLASS_NAMES
        }
        
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def predict_from_bytes(image_bytes):
    """
    从字节数据预测图片
    
    Args:
        image_bytes (bytes): 图片字节数据
        
    Returns:
        dict: 预测结果
    """
    try:
        # 从字节数据创建图片
        img = Image.open(image_bytes)
        
        # 转换为RGB模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 预处理
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
        
        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 构建结果
        result = {
            'predicted_class': CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].tolist(),
            'class_names': CLASS_NAMES
        }
        
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None