import torch
import sys
import os
# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from dataset.lego_dataset import LegoDataset
from model.lego_cnn import LegoCNN

# 数据变换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载训练集
train_dataset = LegoDataset(root_dir='d:\\乐高\\data\\processed\\train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 加载验证集
val_dataset = LegoDataset(root_dir='d:\\乐高\\data\\processed\\val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 可视化一个 batch 的图片
def visualize_batch(loader, classes):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(images.size(0)):
        img = images[i].numpy().transpose(1, 2, 0)  # 转换为 (H, W, C)
        img = img * 0.5 + 0.5  # 反归一化
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {classes[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    # 使用绝对路径保存图片
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'batch_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Batch visualization saved to {save_path}")

# 验证模型前向传播
def test_model():
    model = LegoCNN(num_classes=len(train_dataset.get_classes()))
    print(f"Model parameters: {model.count_parameters()}")
    
    # 测试 dummy input
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

if __name__ == '__main__':
    # 确保 experiments 目录存在
    import os
    os.makedirs('../experiments', exist_ok=True)
    
    # 可视化 batch
    visualize_batch(train_loader, train_dataset.get_classes())
    
    # 测试模型
    test_model()
    
    print("Data loading and model testing completed successfully!")