import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LegoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        
        Args:
            root_dir (str): 数据集根目录
            transform (callable, optional): 数据增强变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = []
        
        # 遍历目录，收集图片和标签
        for class_idx, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.classes.append(class_name)
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(img_path)
                        self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        根据索引获取数据
        
        Args:
            idx (int): 数据索引
            
        Returns:
            tuple: (image, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_classes(self):
        """
        获取类别名称列表
        """
        return self.classes