import torch
import torch.nn as nn
import os

# 模型配置
MODEL_NAME = "LegoCNN"
CLASS_NAMES = ['1x1', '1x2', '1x3', '1x4', '1x6', '2x2', '2x3', '2x4']
INPUT_SIZE = (64, 64)
NORMALIZE_MEAN = [0.7519301, 0.75730944, 0.73740506]
NORMALIZE_STD = [0.09922784, 0.10341568, 0.12011352]

# 模型路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")

class LegoCNN(nn.Module):
    def __init__(self, num_classes=8, dropout=0.5, width='medium', depth='medium'):
        """
        初始化模型
        
        Args:
            num_classes (int): 类别数量
            dropout (float): dropout 比例
            width (str): 网络宽度 ['narrow', 'medium', 'wide']
            depth (str): 网络深度 ['shallow', 'medium', 'deep']
        """
        super(LegoCNN, self).__init__()
        
        # 根据宽度设置通道数
        if width == 'narrow':
            channels = [16, 32, 64, 128]
        elif width == 'medium':
            channels = [32, 64, 128, 256]
        elif width == 'wide':
            channels = [64, 128, 256, 512]
        else:
            channels = [32, 64, 128, 256]
        
        # 根据深度设置卷积层数量
        if depth == 'shallow':
            num_layers = 2
        elif depth == 'medium':
            num_layers = 4
        elif depth == 'deep':
            num_layers = 6
        else:
            num_layers = 4
        
        # 构建卷积层
        conv_layers = []
        in_channels = 3
        
        for i in range(num_layers):
            if i < len(channels):
                out_channels = channels[i]
            else:
                out_channels = channels[-1]
            
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 自适应平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = self.fc_layers(x)
        return x
    
    def count_parameters(self):
        """
        统计模型参数量
        
        Returns:
            int: 模型参数总数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def load_model():
    """
    加载模型
    
    Returns:
        LegoCNN: 加载好权重的模型
    """
    model = LegoCNN(num_classes=len(CLASS_NAMES))
    
    # 尝试加载最佳模型
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        # 尝试从pytorch experiments目录加载
        pytorch_experiments = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pytorch", "experiments")
        if os.path.exists(pytorch_experiments):
            # 找到最佳模型文件
            model_files = [f for f in os.listdir(pytorch_experiments) if f.endswith('best_model.pth')]
            if model_files:
                # 选择第一个模型文件
                best_model_file = model_files[0]
                model_path = os.path.join(pytorch_experiments, best_model_file)
                model.load_state_dict(torch.load(model_path))
                print(f"Model loaded from {model_path}")
            else:
                print("No model files found")
        else:
            print("No model files found")
    
    model.eval()
    return model
