import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import sys
# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.lego_dataset import LegoDataset
from model.lego_cnn import LegoCNN

# 计算数据的 mean 和 std
def calculate_normalize_params(dataset):
    """
    计算数据集的均值和标准差
    
    Args:
        dataset: 数据集对象
        
    Returns:
        tuple: (mean, std)
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    mean = 0.0
    std = 0.0
    total_images = 0
    
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_size
    
    mean /= total_images
    std /= total_images
    
    return mean.numpy(), std.numpy()

# 测试模型
def test_model(model, test_loader):
    """
    在测试集上评估模型
    
    Args:
        model: 模型对象
        test_loader: 测试数据加载器
        
    Returns:
        float: 测试准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    # 确保 experiments 目录存在
    os.makedirs('../experiments', exist_ok=True)
    
    # 加载数据集
    train_dataset = LegoDataset(root_dir='d:\\乐高\\data\\processed\\train', transform=transforms.ToTensor())
    test_dataset = LegoDataset(root_dir='d:\\乐高\\data\\processed\\test', transform=transforms.ToTensor())
    
    # 计算 Normalize 参数
    print("Calculating normalize parameters...")
    mean, std = calculate_normalize_params(train_dataset)
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    # 保存 Normalize 参数
    np.save('../experiments/normalize_mean.npy', mean)
    np.save('../experiments/normalize_std.npy', std)
    print("Normalize parameters saved to ../experiments/")
    
    # 测试变换
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 重新加载测试集
    test_dataset = LegoDataset(root_dir='d:\\乐高\\data\\processed\\test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载最佳模型
    model = LegoCNN(num_classes=len(test_dataset.get_classes()), dropout=0.5)
    
    # 选择最佳模型文件
    best_model_files = [f for f in os.listdir('../experiments') if f.endswith('best_model.pth')]
    if best_model_files:
        # 简单选择第一个模型文件
        best_model_file = best_model_files[0]
        model_path = os.path.join('../experiments', best_model_file)
        print(f"Loading best model: {model_path}")
        model.load_state_dict(torch.load(model_path))
        
        # 测试模型
        test_accuracy = test_model(model, test_loader)
        
        # 保存测试结果
        with open('../experiments/test_results.txt', 'w') as f:
            f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
            f.write(f"Mean: {mean}\n")
            f.write(f"Std: {std}\n")
        print("Test results saved to ../experiments/test_results.txt")
    else:
        print("No best model found. Please run training first.")

if __name__ == '__main__':
    main()