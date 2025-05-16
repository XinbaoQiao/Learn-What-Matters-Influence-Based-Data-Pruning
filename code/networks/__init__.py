import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, SqueezeNet1_0_Weights, mobilenet_v2, MobileNet_V2_Weights, shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

def load_model(model_name, num_classes, weights=None):
    if model_name == "SqueezeNet":
        model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1 if weights else None)
    elif model_name == "ResNet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if weights else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet34":
        model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if weights else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if weights else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ShuffleNet":
        model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Model not supported")
    # Print model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    return model 