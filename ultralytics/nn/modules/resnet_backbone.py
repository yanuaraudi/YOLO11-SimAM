import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Stage(nn.Module):
    _shared_backbone = None

    def __init__(self, stage: int):
        super().__init__()
        if ResNet18Stage._shared_backbone is None:
            base = resnet18(weights=ResNet18_Weights.DEFAULT)
            ResNet18Stage._shared_backbone = nn.ModuleList([
                nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool), 
                base.layer1,  # stride 4
                base.layer2,  # stride 8  → P3
                base.layer3,  # stride 16 → P4
                base.layer4,  # stride 32 → P5
            ])
        self.stage = ResNet18Stage._shared_backbone[stage]

    def forward(self, x):
        return self.stage(x)