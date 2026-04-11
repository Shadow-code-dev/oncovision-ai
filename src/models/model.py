import torch.nn as nn
from torchvision import models

def get_efficientnet(num_classes=3, freeze=True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze feature extractor
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

        # Unfreeze last block
        for param in model.features[-1].parameters():
            param.requires_grad = True

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

def get_resnet(num_classes=3, freeze=True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model