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
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        return {
            'predicted_class': CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].tolist(),
            'class_names': CLASS_NAMES
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def predict_from_bytes(image_bytes):
    try:
        img = Image.open(image_bytes)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        return {
            'predicted_class': CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].tolist(),
            'class_names': CLASS_NAMES
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None