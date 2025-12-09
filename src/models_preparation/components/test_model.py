from unittest.mock import DEFAULT

import torch
import torch.nn as nn
from torchvision import models

class BestClassifier(nn.Module):
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
        in_features = self.backbone.fc.in_features

        layers = []
        current_dim = in_features  # Start with 2048

        # Loop to create the hidden layers (Depth = 4)
        for _ in range(n_layers):
            # Connect current dimension to the hidden dimension
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Logic for Dropout (disabled based on your Optuna results)
            if apply_dropout:
                layers.append(nn.Dropout(dropout_rate))

            # Update current_dim so the next layer knows what to expect
            current_dim = hidden_dim

        # 4. Final Output Layer
        layers.append(nn.Linear(current_dim, num_classes))


        self.backbone.fc = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the entire network (Backbone + New Head)
        return self.backbone(x)