import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.lego_dataset import LegoDataset
from model.lego_cnn import LegoCNN

# 无数据增强的变换
no_aug_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 有数据增强的变换
aug_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5, experiment_name="no_aug"):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    no_improve = 0
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            # 保存最佳模型 - 使用绝对路径
            experiments_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
            model_path = os.path.join(experiments_dir, f'{experiment_name}_best_model.pth')
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return best_val_acc, train_losses, val_losses, train_accs, val_accs

def run_augmentation_experiment():
    # 确保 experiments 目录存在 - 使用绝对路径
    experiments_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    
    # 加载数据集
    print("\n=== No Augmentation ===")
    no_aug_train_dataset = LegoDataset(root_dir='d:\\乐高\\data\\processed\\train', transform=no_aug_transform)
    val_dataset = LegoDataset(root_dir='d:\\乐高\\data\\processed\\val', transform=no_aug_transform)
    
    no_aug_train_loader = DataLoader(no_aug_train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    no_aug_model = LegoCNN(num_classes=len(no_aug_train_dataset.get_classes()), dropout=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(no_aug_model.parameters(), lr=0.001)
    
    no_aug_acc, no_aug_train_losses, no_aug_val_losses, no_aug_train_accs, no_aug_val_accs = train_model(
        no_aug_model, no_aug_train_loader, val_loader, criterion, optimizer, 
        num_epochs=50, patience=5, experiment_name="no_augmentation"
    )
    
    print("\n=== With Augmentation ===")
    aug_train_dataset = LegoDataset(root_dir='d:\\乐高\\data\\processed\\train', transform=aug_transform)
    aug_train_loader = DataLoader(aug_train_dataset, batch_size=32, shuffle=True)
    
    aug_model = LegoCNN(num_classes=len(aug_train_dataset.get_classes()), dropout=0.5)
    optimizer = optim.Adam(aug_model.parameters(), lr=0.001)
    
    aug_acc, aug_train_losses, aug_val_losses, aug_train_accs, aug_val_accs = train_model(
        aug_model, aug_train_loader, val_loader, criterion, optimizer, 
        num_epochs=50, patience=5, experiment_name="with_augmentation"
    )
    
    # 绘制对比图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(no_aug_train_losses)+1), no_aug_train_losses, label='No Aug (Train)')
    plt.plot(range(1, len(no_aug_val_losses)+1), no_aug_val_losses, label='No Aug (Val)')
    plt.plot(range(1, len(aug_train_losses)+1), aug_train_losses, label='With Aug (Train)')
    plt.plot(range(1, len(aug_val_losses)+1), aug_val_losses, label='With Aug (Val)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves Comparison')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(no_aug_train_accs)+1), no_aug_train_accs, label='No Aug (Train)')
    plt.plot(range(1, len(no_aug_val_accs)+1), no_aug_val_accs, label='No Aug (Val)')
    plt.plot(range(1, len(aug_train_accs)+1), aug_train_accs, label='With Aug (Train)')
    plt.plot(range(1, len(aug_val_accs)+1), aug_val_accs, label='With Aug (Val)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves Comparison')
    
    plt.tight_layout()
    # 使用绝对路径保存图片
    comparison_path = os.path.join(experiments_dir, 'augmentation_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    
    print("\n=== Augmentation Experiment Results ===")
    print(f"No Augmentation Accuracy: {no_aug_acc:.2f}%")
    print(f"With Augmentation Accuracy: {aug_acc:.2f}%")
    print(f"Improvement: {aug_acc - no_aug_acc:.2f}%")

if __name__ == '__main__':
    run_augmentation_experiment()