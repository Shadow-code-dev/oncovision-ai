import torch.nn as nn
from torchvision import models

def get_efficientnet(num_classes=3):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

def get_resnet(num_classes=3):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model