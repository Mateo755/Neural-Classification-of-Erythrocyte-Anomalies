import torch
import torch.nn as nn
from torchvision import models


class UniversalModel(nn.Module):
    def __init__(self, base_model_name="resnet50", num_classes=2, freeze_backbone=True,
                 n_layers=4, hidden_dim=128, apply_dropout=False, dropout_rate=0.5):
        super().__init__()

        # 1. Load the Backbone dynamically based on the name
        self.backbone = self._load_backbone(base_model_name)

        # 2. Freeze Backbone (Optional)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. Handle different naming conventions (fc vs classifier vs heads)
        # We need to find:
        # a) How many input features go into the last layer?
        # b) What is the name of the last layer module so we can replace it?

        if hasattr(self.backbone, 'fc'):
            # For ResNet, RegNet, Inception, GoogLeNet
            in_features = self.backbone.fc.in_features
            layer_name = 'fc'

        elif hasattr(self.backbone, 'classifier'):
            # For EfficientNet, MobileNet, VGG, ConvNeXt, MaxViT
            # Note: 'classifier' is usually a Sequential block.
            # We take the input features of the last linear layer in that block.
            if isinstance(self.backbone.classifier, nn.Sequential):
                # Usually the last layer is the linear one
                in_features = self.backbone.classifier[-1].in_features
            else:
                # Sometimes it's just a single layer
                in_features = self.backbone.classifier.in_features
            layer_name = 'classifier'

        elif hasattr(self.backbone, 'heads'):
            # For ViT (Vision Transformer)
            # ViT structure: self.heads.head
            if isinstance(self.backbone.heads, nn.Sequential):
                in_features = self.backbone.heads[-1].in_features
            else:
                in_features = self.backbone.heads.head.in_features  # Often nested
            layer_name = 'heads'

        else:
            raise AttributeError(f"Model {base_model_name} has unknown last layer type.")

        # 4. Build Custom Head (The logic remains the same)
        layers = []
        current_dim = in_features

        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if apply_dropout:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        new_head = nn.Sequential(*layers)

        # 5. Replace the layer dynamically using setattr
        # This effectively does: self.backbone.fc = new_head  OR  self.backbone.classifier = new_head
        setattr(self.backbone, layer_name, new_head)

    def forward(self, x):
        return self.backbone(x)

    def _load_backbone(self, name):
        """Helper to load weights based on string name"""
        name = name.lower()
        if name == "resnet50":
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif name == "efficientnet_b0":
            return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        elif name == "efficientnet_v2_s":
            return models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        elif name == "convnext_tiny":
            return models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        elif name == "maxvit_t":
            return models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)
        elif name == "regnet_y_16gf":
            return models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.DEFAULT)
        elif name == "vit_b_16":
            # WARNING: ViT usually requires image size 224x224
            return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        else:
            raise ValueError(f"Model {name} not supported yet.")