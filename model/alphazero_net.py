import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-Excitation Block như trong Leela Chess Zero (LCZero).
    Cho phép mạng học cách trọng số hóa từng channel dựa trên nội dung toàn bộ bàn cờ.
    
    Kiến trúc: GlobalAvgPool → FC(channels → channels//se_ratio) → ReLU
                → FC(channels//se_ratio → channels*2) → split → Sigmoid scale + Linear bias
    """
    def __init__(self, channels: int, se_ratio: int = 4):
        super().__init__()
        se_hidden = max(1, channels // se_ratio)
        # Compress global information
        self.fc1 = nn.Linear(channels, se_hidden)
        # Expand to scale AND bias (LCZero-style dual output)
        self.fc2 = nn.Linear(se_hidden, channels * 2)

    def forward(self, x):
        # x: (B, C, 8, 8)
        # Squeeze: global average pooling → (B, C)
        s = x.mean(dim=[2, 3])
        # Excitation
        s = F.relu(self.fc1(s))
        s = self.fc2(s)                          # (B, C*2)
        # Split into scale and bias
        scale, bias = s.chunk(2, dim=1)          # each (B, C)
        scale = torch.sigmoid(scale)
        bias = bias.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias


class ResBlock(nn.Module):
    """
    Pre-activation Residual Block tích hợp SE Attention.
    Kiến trúc: Conv → BN → ReLU → Conv → BN → SE → Residual → ReLU
    """
    def __init__(self, channels: int, se_ratio: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SEBlock(channels, se_ratio)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)        # Apply SE attention trước khi cộng residual
        out = out + residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """
    SE-ResNet AlphaZero-style Chess Network.
    
    Architecture (tuned for 6M elite positions on Tesla M40 24GB):
    - 12 SE-ResBlocks × 192 channels
    - Policy Head: 4-channel conv + hidden FC → 4272 moves
    - Value Head: 1-channel conv → FC(256) → FC(128) → tanh scalar
    
    Total params: ~27M (vs ~13M trước đây)
    Estimated VRAM with batch_size=4096: ~7-8 GB (safe for 24GB M40)
    """
    def __init__(self, num_blocks: int = 12, num_channels: int = 192,
                 num_input_channels: int = 18, num_moves: int = 4272,
                 se_ratio: int = 4):
        super().__init__()

        # ── Initial convolutional stem ─────────────────────────────────────
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_input_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )

        # ── Residual tower with SE Attention ──────────────────────────────
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels, se_ratio) for _ in range(num_blocks)
        ])

        # ── Policy Head ───────────────────────────────────────────────────
        # 4-channel conv để nắm bắt nhiều đặc trưng không gian hơn
        policy_mid = 4 * 8 * 8   # 256
        self.policy_conv = nn.Conv2d(num_channels, 4, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(4)
        self.policy_fc1  = nn.Linear(policy_mid, 512)
        self.policy_fc2  = nn.Linear(512, num_moves)

        # ── Value Head ────────────────────────────────────────────────────
        # 1-channel conv → FC(256) → FC(128) → scalar
        self.value_conv  = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2   = nn.Linear(256, 128)
        self.value_fc3   = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, 18, 8, 8) board state tensor
        Returns:
            policy_logits: (batch_size, 4272) — raw logits (không softmax)
            value:         (batch_size, 1)    — giá trị trong [-1, 1]
        """
        # Stem
        x = self.initial_conv(x)

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # ── Policy head ───────────────────────────────────────────────────
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)            # (B, 4*8*8=256)
        p = F.relu(self.policy_fc1(p))
        policy_logits = self.policy_fc2(p)      # (B, 4272)

        # ── Value head ────────────────────────────────────────────────────
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)            # (B, 64)
        v = F.relu(self.value_fc1(v))           # (B, 256)
        v = F.relu(self.value_fc2(v))           # (B, 128)
        value = torch.tanh(self.value_fc3(v))   # (B, 1)

        return policy_logits, value


if __name__ == "__main__":
    net = AlphaZeroNet(num_blocks=12, num_channels=192)
    net.eval()

    dummy_input = torch.randn(4, 18, 8, 8)
    with torch.no_grad():
        policy, value = net(dummy_input)

    print("Policy shape:", policy.shape)  # [4, 4272]
    print("Value shape :", value.shape)   # [4, 1]
    print("Value range :", value.min().item(), "→", value.max().item())

    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:,}")

    # Ước tính VRAM
    param_mb = params * 4 / (1024 ** 2)
    print(f"Param memory (FP32): {param_mb:.1f} MB")
