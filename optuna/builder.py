import torch
import torch.nn as nn
from torchvision import models

class ModelBuilder(nn.Module):
    def __init__(self, base_model_name, num_classes=2, dropout_rate=0.0, use_dropout=False, freeze_backbone=True,
                 num_hidden_layers=1, hidden_dim=128):
        super().__init__()
        
        # 1. Load the Backbone (Pre-trained on ImageNet)
        # We need to handle different model types because they have different structures
        if base_model_name == "resnet18":
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # ResNet stores features in layers before 'fc'
            # We get the number of input features for the last layer
            in_features = base_model.fc.in_features
            # Remove the last layer (fc) to keep only feature extractor
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            
        elif base_model_name == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = base_model.fc.in_features
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            
        elif base_model_name == "mobilenet_v2":
            base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            in_features = base_model.classifier[1].in_features
            # MobileNet features are in 'features' attribute
            # We also need Global Average Pooling because MobileNet returns [Batch, C, 7, 7]
            self.feature_extractor = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d(1) 
            )
            
        elif base_model_name == "vgg16":
            base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            in_features = 512 # VGG feature output after pooling
            self.feature_extractor = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            raise ValueError(f"Unknown model name: {base_model_name}")

        # 2. Freeze Backbone (Optional)
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # 3. Build Dynamic Custom Head
        layers = []
        layers.append(nn.Flatten())
        
        # Variable to track the input dimension for the next layer
        current_dim = in_features
        
        # LOOP: Add hidden layers dynamically
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            if use_dropout:
                layers.append(nn.Dropout(p=dropout_rate))
            
            # Update current_dim so the next layer connects correctly
            current_dim = hidden_dim
            
        # Final Output Layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # Extract features
        x = self.feature_extractor(x)
        # Classify
        x = self.classifier(x)
        return x