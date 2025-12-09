import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sns
import torch
import numpy as np

def visualize_samples(base_path='./malaria_dataset/train', num_samples=3):
    classes = ['positive', 'negative']
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(12, 8))
    
    for i, cls in enumerate(classes):
        cls_path = os.path.join(base_path, cls)
        if not os.path.exists(cls_path):
            continue
            
        all_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        selected_files = random.sample(all_files, min(len(all_files), num_samples))
        
        for j, file_name in enumerate(selected_files):
            img_path = os.path.join(cls_path, file_name)
            img = Image.open(img_path)
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_title(f"Class: {cls}\n{img.size}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_training_metrics(log_dir):
    metrics_path = f"{log_dir}/metrics.csv"
    if not os.path.exists(metrics_path):
        print("Metrics file not found.")
        return

    metrics = pd.read_csv(metrics_path)
    metrics_epoch = metrics.groupby("epoch").mean(numeric_only=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Loss
    if 'train_loss' in metrics_epoch.columns and 'val_loss' in metrics_epoch.columns:
        axes[0].plot(metrics_epoch.index, metrics_epoch['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(metrics_epoch.index, metrics_epoch['val_loss'], label='Val Loss', marker='o')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    # Plot Accuracy
    if 'train_acc' in metrics_epoch.columns and 'val_acc' in metrics_epoch.columns:
        axes[1].plot(metrics_epoch.index, metrics_epoch['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(metrics_epoch.index, metrics_epoch['val_acc'], label='Val Acc', marker='o')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def save_confusion_matrix(cm_tensor, class_names, save_path="confusion_matrix.png"):
    """
    Drawing and saving confusion matrix as a heatmap.
    """
    # Convert tensor to numpy array
    cm_numpy = cm_tensor.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_numpy, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.title('Confusion Matrix')

    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")