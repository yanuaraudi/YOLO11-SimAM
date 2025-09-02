import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        out = self.mlp(avg_pool) + self.mlp(max_pool)
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out


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
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        return x * attn


class CBAM(nn.Module):
    """
    CBAM block.
    Args:
        c (int): input channels
        reduction (int): reduction ratio in ChannelAttention
        kernel_size (int): 3 or 7 for SpatialAttention
    """
    def __init__(self, c: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(c, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
    
    
#ECAnet

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA-Net).
    - Tanpa MLP/reduction: pakai GAP (B,C,1,1) -> Conv1d(1,1,k) di dimensi channel -> sigmoid -> scale.
    Args:
        c (int): jumlah channel input
        gamma (int|float): faktor skala untuk menghitung k
        b (int|float): bias untuk menghitung k
        k_size (int|None): kalau None, dihitung adaptif dari c; kalau diberikan, harus bilangan ganjil >= 3
    """
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