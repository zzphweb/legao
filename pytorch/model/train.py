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

# 数据增强变换
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5, experiment_name="baseline"):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_epoch = 0
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
            best_epoch = epoch
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
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
    plt.plot(range(1, len(val_accs)+1), val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    # 使用绝对路径保存图片
    curves_path = os.path.join(experiments_dir, f'{experiment_name}_curves.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {curves_path}")
    
    return best_val_acc, best_epoch

# 超参数对比实验
def run_experiments():
    # 确保 experiments 目录存在
    os.makedirs('../experiments', exist_ok=True)
    
    # 加载数据集
    train_dataset = LegoDataset(root_dir='d:\\乐高\\data\\processed\\train', transform=train_transform)
    val_dataset = LegoDataset(root_dir='d:\\乐高\\data\\processed\\val', transform=val_transform)
    
    # 1. 基线模型
    print("\n=== Baseline Model ===")
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = LegoCNN(num_classes=len(train_dataset.get_classes()), dropout=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    baseline_acc, _ = train_model(model, train_loader, val_loader, criterion, optimizer, 
                                  num_epochs=50, patience=5, experiment_name="baseline")
    
    # 2. Batch Size 对比
    print("\n=== Batch Size Comparison ===")
    batch_sizes = [8, 16, 32, 64]
    batch_accs = []
    
    for bs in batch_sizes:
        print(f"\nBatch Size: {bs}")
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        
        model = LegoCNN(num_classes=len(train_dataset.get_classes()), dropout=0.5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        acc, _ = train_model(model, train_loader, val_loader, criterion, optimizer, 
                             num_epochs=50, patience=5, experiment_name=f"batch_size_{bs}")
        batch_accs.append(acc)
    
    # 绘制 Batch Size 对比图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(batch_sizes)), batch_accs, tick_label=batch_sizes)
    plt.xlabel('Batch Size')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Batch Size Comparison')
    experiments_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
    plt.savefig(os.path.join(experiments_dir, 'batch_size_comparison.png'), dpi=150, bbox_inches='tight')
    
    # 3. Learning Rate 对比
    print("\n=== Learning Rate Comparison ===")
    learning_rates = [0.01, 0.001, 0.0001]
    lr_accs = []
    
    for lr in learning_rates:
        print(f"\nLearning Rate: {lr}")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = LegoCNN(num_classes=len(train_dataset.get_classes()), dropout=0.5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        acc, _ = train_model(model, train_loader, val_loader, criterion, optimizer, 
                             num_epochs=50, patience=5, experiment_name=f"lr_{lr}")
        lr_accs.append(acc)
    
    # 绘制 Learning Rate 对比图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(learning_rates)), lr_accs, tick_label=learning_rates)
    plt.xlabel('Learning Rate')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Learning Rate Comparison')
    plt.savefig(os.path.join(experiments_dir, 'learning_rate_comparison.png'), dpi=150, bbox_inches='tight')
    
    # 4. Dropout 对比
    print("\n=== Dropout Comparison ===")
    dropouts = [0.0, 0.3, 0.5]
    dropout_accs = []
    
    for dropout in dropouts:
        print(f"\nDropout: {dropout}")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = LegoCNN(num_classes=len(train_dataset.get_classes()), dropout=dropout)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        acc, _ = train_model(model, train_loader, val_loader, criterion, optimizer, 
                             num_epochs=50, patience=5, experiment_name=f"dropout_{dropout}")
        dropout_accs.append(acc)
    
    # 绘制 Dropout 对比图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(dropouts)), dropout_accs, tick_label=dropouts)
    plt.xlabel('Dropout Rate')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Dropout Comparison')
    plt.savefig(os.path.join(experiments_dir, 'dropout_comparison.png'), dpi=150, bbox_inches='tight')
    
    # 5. 优化器对比
    print("\n=== Optimizer Comparison ===")
    optimizers = ['SGD', 'Adam']
    optimizer_accs = []
    
    for opt_name in optimizers:
        print(f"\nOptimizer: {opt_name}")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = LegoCNN(num_classes=len(train_dataset.get_classes()), dropout=0.5)
        criterion = nn.CrossEntropyLoss()
        
        if opt_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        acc, _ = train_model(model, train_loader, val_loader, criterion, optimizer, 
                             num_epochs=50, patience=5, experiment_name=f"optimizer_{opt_name}")
        optimizer_accs.append(acc)
    
    # 绘制 Optimizer 对比图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(optimizers)), optimizer_accs, tick_label=optimizers)
    plt.xlabel('Optimizer')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Optimizer Comparison')
    plt.savefig(os.path.join(experiments_dir, 'optimizer_comparison.png'), dpi=150, bbox_inches='tight')
    
    # 6. 网络宽度对比
    print("\n=== Network Width Comparison ===")
    widths = ['narrow', 'medium', 'wide']
    width_accs = []
    
    for width in widths:
        print(f"\nNetwork Width: {width}")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = LegoCNN(num_classes=len(train_dataset.get_classes()), dropout=0.5, width=width, depth='medium')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        acc, _ = train_model(model, train_loader, val_loader, criterion, optimizer, 
                             num_epochs=50, patience=5, experiment_name=f"width_{width}")
        width_accs.append(acc)
    
    # 绘制 Network Width 对比图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(widths)), width_accs, tick_label=widths)
    plt.xlabel('Network Width')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Network Width Comparison')
    plt.savefig(os.path.join(experiments_dir, 'width_comparison.png'), dpi=150, bbox_inches='tight')
    
    # 7. 网络深度对比
    print("\n=== Network Depth Comparison ===")
    depths = ['shallow', 'medium', 'deep']
    depth_accs = []
    
    for depth in depths:
        print(f"\nNetwork Depth: {depth}")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = LegoCNN(num_classes=len(train_dataset.get_classes()), dropout=0.5, width='medium', depth=depth)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        acc, _ = train_model(model, train_loader, val_loader, criterion, optimizer, 
                             num_epochs=50, patience=5, experiment_name=f"depth_{depth}")
        depth_accs.append(acc)
    
    # 绘制 Network Depth 对比图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(depths)), depth_accs, tick_label=depths)
    plt.xlabel('Network Depth')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Network Depth Comparison')
    plt.savefig(os.path.join(experiments_dir, 'depth_comparison.png'), dpi=150, bbox_inches='tight')
    
    print("\n=== All Experiments Completed ===")
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"Batch Size Best: {batch_sizes[np.argmax(batch_accs)]} with {max(batch_accs):.2f}%")
    print(f"Learning Rate Best: {learning_rates[np.argmax(lr_accs)]} with {max(lr_accs):.2f}%")
    print(f"Dropout Best: {dropouts[np.argmax(dropout_accs)]} with {max(dropout_accs):.2f}%")
    print(f"Optimizer Best: {optimizers[np.argmax(optimizer_accs)]} with {max(optimizer_accs):.2f}%")
    print(f"Network Width Best: {widths[np.argmax(width_accs)]} with {max(width_accs):.2f}%")
    print(f"Network Depth Best: {depths[np.argmax(depth_accs)]} with {max(depth_accs):.2f}%")

if __name__ == '__main__':
    run_experiments()