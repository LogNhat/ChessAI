"""
AlphaZeroNet V1 — kiến trúc gốc khớp với best_model.pth (20M moves).
- 10 ResBlocks (không có SE), 192 channels
- Policy Head: 2-channel conv → FC(128) → 4272
- Value Head: 1-channel conv → FC(256) → FC(1) → tanh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlockV1(nn.Module):
    """Plain residual block (không SE), khớp với best_model.pth."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class AlphaZeroNetV1(nn.Module):
    """
    Kiến trúc khớp 100% với best_model.pth (trained 20M moves).
    ResNet 10 blocks, 192 channels — KHÔNG có SE attention.
    Policy: 2-ch conv → FC(4272)
    Value : 1-ch conv → FC(256) → FC(1) → tanh
    """
    def __init__(self, num_blocks: int = 10, num_channels: int = 192,
                 num_input_channels: int = 18, num_moves: int = 4272):
        super().__init__()

        # Stem
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_input_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlockV1(num_channels) for _ in range(num_blocks)
        ])

        # Policy head: 2-channel conv, then FC(128) → FC(4272)
        # actual checkpoint: policy_conv(2,192,1,1), policy_fc(4272,128)
        policy_flat = 2 * 8 * 8  # 128
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(policy_flat, num_moves)

        # Value head: 1-ch conv → FC(256) → FC(1)
        self.value_conv  = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2   = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)           # (B, 4272)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)               # (B, 64)
        v = F.relu(self.value_fc1(v))              # (B, 256)
        value = torch.tanh(self.value_fc2(v))      # (B, 1)

        return policy_logits, value
