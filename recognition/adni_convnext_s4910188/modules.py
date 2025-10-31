import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

class ConvNeXtAD(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = convnext_tiny(weights="IMAGENET1K_V1")
        self.model.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.model(x)
