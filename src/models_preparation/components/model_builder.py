from unittest.mock import DEFAULT

import torch
import torch.nn as nn
from torchvision import models

class CustomResnet(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True,
                 n_layers=4, hidden_dim=128, apply_dropout=False, dropout_rate=0.5):
        super().__init__()

        # 1. Load Pre-trained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # 2. Freeze Backbone (Optional)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. Build Custom Classification Head

        # CRITICAL STEP: Get input dimension
        # ResNet50 outputs 2048 features from its convolutional layers.
        # We MUST capture this value dynamically to connect our new layers correctly.
        # If we didn't do this, we wouldn't know the input size for the first Linear layer.
        in_features = self.backbone.fc.in_features  # Value is 2048 for ResNet50

        layers = []
        current_dim = in_features  # Start with 2048

        # Loop to create the hidden layers (Depth = 4)
        for _ in range(n_layers):
            # Connect current dimension to the hidden dimension (e.g., 2048 -> 128, then 128 -> 128)
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Logic for Dropout (disabled based on your Optuna results)
            if apply_dropout:
                layers.append(nn.Dropout(dropout_rate))

            # Update current_dim so the next layer knows what to expect
            current_dim = hidden_dim

        # 4. Final Output Layer
        # Maps the last hidden layer (128) to the number of classes (2)
        layers.append(nn.Linear(current_dim, num_classes))

        # 5. Replace the original ResNet head with our new sequence
        self.backbone.fc = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the entire network (Backbone + New Head)
        return self.backbone(x)


class CustomEfficientnet(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True,
                 n_layers=4, hidden_dim=128, apply_dropout=False, dropout_rate=0.5):
        super().__init__()


        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[-1].in_features

        layers = []
        current_dim = in_features


        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if apply_dropout:
                layers.append(nn.Dropout(dropout_rate))

            current_dim = hidden_dim


        layers.append(nn.Linear(current_dim, num_classes))


        self.backbone.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)


class ViT(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True,
                 n_layers=4, hidden_dim=128, apply_dropout=False, dropout_rate=0.5):
        super().__init__()


        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.heads[-1].in_features

        layers = []
        current_dim = in_features


        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if apply_dropout:
                layers.append(nn.Dropout(dropout_rate))

            current_dim = hidden_dim


        layers.append(nn.Linear(current_dim, num_classes))


        self.backbone.heads = nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)