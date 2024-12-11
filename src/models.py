"""
Define and Return Model
"""

import torch.nn as nn
import torch.optim as optim
from torchvision import models
from transformers import DeiTFeatureExtractor, DeiTForImageClassification, AutoFeatureExtractor, AutoModelForImageClassification


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

    # Check which params are trainable
    # print("freeze_backbone: ", freeze_backbone)
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    return model

def build_efficientnet_b0_basic(
        num_classes=8, 
        hidden_units1=100, 
        dropout=0.1, 
        freeze_backbone=False):
    """
    Builds and returns an EfficientNet model.
    Returns: nn.Module: PyTorch model.
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    # Load the base EfficientNet model
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Optionally freeze the backbone
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Modify the classifier head
    in_features = model.classifier[1].in_features  # Get input size of the final layer
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units1), 
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_units1, num_classes)
    )
    
    return model

def build_efficientnet_v2_basic(
        num_classes=8, hidden_units1=100,
        dropout=0.1, freeze_backbone=False):
    
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        for param in model.features.parameters(): # backbone aka feature extractor
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # Get input size of the final layer
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes)
    )

    return model


def build_deit_model(
    num_classes=8, 
    dropout=0.1, 
    hidden_units1=256,
    freeze_backbone=False
):
    """
    Builds and returns a DeiT model with feature extractor for image classification.
    Args:
        num_classes (int): Number of output classes
        hidden_units1 (int): Number of hidden units in the additional layer
        dropout (float): Dropout rate
        freeze_backbone (bool): Whether to freeze the backbone layers
    Returns: tuple: (feature_extractor, model)
    "facebook/deit-small-patch16-224"
    """
    # Load pre-trained DeiT feature extractor
    feature_extractor = DeiTFeatureExtractor.from_pretrained(
        'facebook/deit-base-distilled-patch16-224'
    )
    
    # Load pre-trained DeiT model
    model = DeiTForImageClassification.from_pretrained(
        'facebook/deit-base-distilled-patch16-224', 
        num_labels=num_classes
    )
    
    # Optionally freeze the backbone
    if freeze_backbone:
        for param in model.parameters(): # Freeze everything except the head
            param.requires_grad = False
    
     # Modify the classification head
    in_features = model.classifier.in_features  # Number of input features to the classifier
    # model.classifier = nn.Sequential(
    #     nn.Dropout(p=dropout),  # Add dropout for regularization
    #     nn.Linear(in_features, num_classes)  # Final classification layer
    # )
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),                   # First dropout for regularization
        nn.Linear(in_features, hidden_units1),    # First layer to project features
        nn.ReLU(inplace=True),                   # Activation for non-linearity
        nn.Dropout(p=dropout),                   # Second dropout for regularization
        nn.Linear(hidden_units1, num_classes)     # Final classification layer
    )
    
    return feature_extractor, model

def build_swin_model(
    num_classes=8,
    dropout=0.1,
    freeze_backbone=False
):
    """
    Builds and returns a Swin Transformer model for image classification.
    Args:
        num_classes (int): Number of output classes.
        dropout (float): Dropout rate for the classification head.
        freeze_backbone (bool): Whether to freeze the backbone layers.
    Returns: tuple: (feature_extractor, model)
    """
    # Load the feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # Ignore size mismatch for classifier layer
    )
    
    # Optionally freeze the backbone
    if freeze_backbone:
        for param in model.swin.parameters():
            param.requires_grad = False
    
    # Modify the classification head (if needed)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, num_classes),
        nn.Dropout(p=dropout),
    )
    
    return feature_extractor, model


"""
Notes: The freeze_backbone parameter controls whether the parameters (weights) of
backbone (i.e., the pretrained ResNet-50 architecture) are updated during training

In ResNet-50, the backbone refers to the conv layers that learn hierarchical 
features from the input images. These layers have been pretrained on ImageNet 
and contain useful feature representations.

When we freeze the backbone, we prevent gradients in the backbone layers from
being computed during the backward pass. So the backbone weights remain unchanged
during training.

So in our code we freeze the backbone by default, so all params are trainable.
And whether the backbone is frozen or not, the final fully connected layer (model.fc)
if replaces with our custom head. This head is always trainable since it is
initialized in our code.

Freezing is often used in transfer learning when we want to leverage the 
pretrained features without altering them. Good if dataset is small or similar
to the dataset the model was pretrained on, can avoid overfit.
Freezing saves computation and focuses
training only on the newly added classifier (model.fc).

If dataset is large or very different from ImageNet, you may want to fine-tune 
the entire model. In this case, you set freeze_backbone=False to allow the 
backbone weights to be updated. Allows model to adapt better to specific task.
"""
