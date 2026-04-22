# 数据预处理脚本

import os
import PIL
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# 配置参数
INPUT_DIR = 'data/raw'
OUTPUT_DIR = 'data/processed'
IMAGE_SIZE = (64, 64)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 创建输出目录
def create_output_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'test'), exist_ok=True)

# 检查图片质量
def check_image_quality(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False

# 调整图片大小
def resize_image(image_path, output_path):
    img = Image.open(image_path)
    img = img.resize(IMAGE_SIZE)
    img.save(output_path)

# 划分数据集
def split_dataset(images, labels):
    # 先将数据分为训练集和临时集
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=VAL_RATIO + TEST_RATIO, stratify=labels, random_state=42
    )
    
    # 检查临时集中每个类别的样本数量
    temp_label_counts = pd.Series(temp_labels).value_counts()
    min_temp_count = temp_label_counts.min()
    
    # 根据样本数量决定是否使用分层划分
    if min_temp_count >= 2:
        # 样本足够，使用分层划分
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO), stratify=temp_labels, random_state=42
        )
    else:
        # 样本不足，使用随机划分
        print(f"警告：临时集中某些类别样本数量不足（最少 {min_temp_count} 个），使用随机划分")
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO), random_state=42
        )
    
    return train_images, val_images, test_images, train_labels, val_labels, test_labels

# 主函数
def main():
    create_output_dirs()
    
    # 收集所有图片
    images = []
    labels = []
    
    for brick_type in os.listdir(INPUT_DIR):
        brick_dir = os.path.join(INPUT_DIR, brick_type)
        if os.path.isdir(brick_dir):
            for img_file in os.listdir(brick_dir):
                if img_file.endswith('.png') or img_file.endswith('.jpg') or img_file.endswith('.jpeg'):
                    img_path = os.path.join(brick_dir, img_file)
                    if check_image_quality(img_path):
                        images.append(img_path)
                        labels.append(brick_type)
    
    # 划分数据集
    train_images, val_images, test_images, train_labels, val_labels, test_labels = split_dataset(images, labels)
    
    # 处理并保存图片
    def process_and_save(image_list, label_list, output_subdir):
        for img_path, label in zip(image_list, label_list):
            img_name = os.path.basename(img_path)
            output_dir = os.path.join(OUTPUT_DIR, output_subdir, label)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, img_name)
            resize_image(img_path, output_path)
    
    process_and_save(train_images, train_labels, 'train')
    process_and_save(val_images, val_labels, 'val')
    process_and_save(test_images, test_labels, 'test')
    
    # 生成数据统计报告
    stats = {
        'total': len(images),
        'train': len(train_images),
        'val': len(val_images),
        'test': len(test_images)
    }
    
    # 类别分布
    label_counts = pd.Series(labels).value_counts()
    train_label_counts = pd.Series(train_labels).value_counts()
    val_label_counts = pd.Series(val_labels).value_counts()
    test_label_counts = pd.Series(test_labels).value_counts()
    
    print('数据统计报告:')
    print(f'总图片数: {stats["total"]}')
    print(f'训练集: {stats["train"]} ({stats["train"]/stats["total"]*100:.1f}%)')
    print(f'验证集: {stats["val"]} ({stats["val"]/stats["total"]*100:.1f}%)')
    print(f'测试集: {stats["test"]} ({stats["test"]/stats["total"]*100:.1f}%)')
    print('\n类别分布:')
    print(label_counts)
    print('\n训练集类别分布:')
    print(train_label_counts)
    print('\n验证集类别分布:')
    print(val_label_counts)
    print('\n测试集类别分布:')
    print(test_label_counts)

if __name__ == '__main__':
    main()
