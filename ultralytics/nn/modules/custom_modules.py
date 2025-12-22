import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics.nn.modules.head import Classify


# SimAM
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)
    
#CBAM
# ---------- ChannelAttention (lazy) ----------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int | None = None, reduction: int = 16):
        super().__init__()
        self.reduction = reduction
        self.mlp = None  # dibangun saat forward pertama
        self._c = channels  # hint opsional

    def _build(self, c: int):
        hidden = max(c // self.reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(c, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        if self.mlp is None or self.mlp[0].in_features != c:
            self._build(c)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        out = self.mlp(avg_pool) + self.mlp(max_pool)
        return x * torch.sigmoid(out).view(b, c, 1, 1)

# ---------- SpatialAttention (tetap) ----------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn

# ---------- CBAMv2 (abaikan argumen 'c') ----------
class CBAMv2(nn.Module):
    def __init__(self, c: int | None = None, reduction: int = 16, kernel_size: int = 7, *_, **__):
        super().__init__()
        self.ca = ChannelAttention(None, reduction)   # c autodetect
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x))
    
    
#ECAnet

class ECA(nn.Module):
    def __init__(self, c: int, gamma: float = 2, b: float = 1, k_size: int | None = None):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(c) + b) / gamma))
            k_size = t if t % 2 else t + 1
            k_size = max(3, k_size)  # minimal 3 dan selalu ganjil
        else:
            if k_size % 2 == 0 or k_size < 3:
                raise ValueError("k_size must be odd and >= 3")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        y = y.unsqueeze(1)                             # [B, 1, C]
        y = self.conv(y)                               # [B, 1, C]
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y

class ResNet18_ECA_Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4

        self.eca4 = ECA(c=512)

        # 🔴 ADD THESE LINES:
        self.out_channels = 512
        self.c2 = 512

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.eca4(x)
        return x

class ECAClassifyHead(nn.Module):
    """
    Simple wrapper around Ultralytics Classify head that forces c1=512
    so it matches ResNet18_ECA_Backbone output channels.
    """
    def __init__(self, nc: int, c1: int = 512):
        super().__init__()
        # use official Ultralytics classification head but with correct input channels
        self.head = Classify(c1, nc)

    def forward(self, x):
        return self.head(x)

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = torch.clamp(x, min=self.eps)
        x = x ** self.p
        x = torch.mean(x, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        x = x ** (1.0 / self.p)
        return x


class GeMHead512(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = GeM()

    def forward(self, x):
        return self.pool(x)  # still 4D
    
class BottleneckHead512(nn.Module):
    def __init__(self, nc: int, c1: int = 512, hidden: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c1, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, nc),
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.fc(x)
        return x

class SE(nn.Module):
    """
    Squeeze-and-Excitation block
    Args:
        c (int): number of input channels
        r (int): reduction ratio (default 16)
    """

    def __init__(self, c, r=16):
        super().__init__()
        c_ = max(c // r, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c_, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class ConvProj1x1(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DualPool(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.c2 = c1 * 2  # tell YOLO output channels

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1)
        gmp = F.adaptive_max_pool2d(x, 1)
        return torch.cat([gap, gmp], dim=1)
    
class DualPoolClassifyHead(nn.Module):
    def __init__(self, nc, c1=512):
        super().__init__()
        self.pool = DualPool(c1)
        self.classifier = nn.Linear(c1 * 2, nc)

    def forward(self, x):
        x = self.pool(x)          # [B, 1024, 1, 1]
        x = x.flatten(1)          # [B, 1024]
        return self.classifier(x)

class MLPClassifyHead(nn.Module):
    def __init__(self, nc, c1=512, hidden=1024, dropout=0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c1, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, nc),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)

class ConvProj1x1Drop(nn.Module):
    def __init__(self, c, p=0.1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(p)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))

class ConvProjBN(nn.Module):
    def __init__(self, c1=512):
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvProjLowRank(nn.Module):
    def __init__(self, c1=512, c_mid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c_mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)

class ConvProjResidual(nn.Module):
    def __init__(self, c1=512):
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, 1, bias=False)

    def forward(self, x):
        return x + self.conv(x)


