"""
Define and Return Model
"""

import torch.nn as nn
import torch.optim as optim
from torchvision import models


def build_resnet50_basic(
        num_classes=8, hidden_units1=100, 
        dropout=0.1,freeze_backbone=False ):
    """
    Builds and returns mode;.
    Returns: nn.Module: PyTorch model.
    """
    # Load base architecture (e.g., ResNet50)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

     # Optionally freeze the backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Modify the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(2048, hidden_units1), 
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_units1, num_classes)
    )
    
    return model





