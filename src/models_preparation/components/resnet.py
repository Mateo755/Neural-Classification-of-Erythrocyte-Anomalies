import torch.nn as nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super().__init__()
        
        # 1. Load Pre-trained ResNet18
        # We use 'DEFAULT' weights which correspond to the best available ImageNet weights
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Freeze Backbone (Optional strategy)
        # If True, we stop gradient calculation for feature extraction layers.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 3. Replace the Classification Head
        num_filters = self.backbone.fc.in_features
        # We replace it with a new Linear layer for our binary classification (2 classes)
        self.backbone.fc = nn.Linear(num_filters, num_classes)
        
    def forward(self, x):
        return self.backbone(x)