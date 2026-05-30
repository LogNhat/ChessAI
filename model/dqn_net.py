"""
Chess Deep Q-Network (ChessDQN)
Uses a Residual Convolutional Neural Network to estimate Q-values for all possible actions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlockDQN(nn.Module):
    """
    Standard residual block for ChessDQN.
    Consists of two Conv2D layers with Batch Normalization and ReLU activation.
    """
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

class ChessDQN(nn.Module):
    """
    Chess DQN Architecture.
    Inputs: (Batch, 18, 8, 8) representing the board state.
    Outputs: (Batch, 4272) representing Q-values for all possible moves.
    """
    def __init__(self, num_blocks: int = 6, num_channels: int = 128, 
                 num_input_channels: int = 18, num_moves: int = 4272):
        super().__init__()
        
        # Stem: Conv2D -> BatchNorm -> ReLU
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_input_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlockDQN(num_channels) for _ in range(num_blocks)
        ])
        
        # Q-value fully connected head
        self.q_conv = nn.Conv2d(num_channels, 4, kernel_size=1, bias=False)
        self.q_bn   = nn.BatchNorm2d(4)
        
        policy_flat = 4 * 8 * 8 # 256
        self.q_fc1  = nn.Linear(policy_flat, 256)
        self.q_fc2  = nn.Linear(256, num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes raw Q-value predictions for all actions.
        Args:
            x: (B, 18, 8, 8) state tensor
        Returns:
            q_values: (B, 4272) tensor of Q-values
        """
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)
            
        # Q-head
        x_q = F.relu(self.q_bn(self.q_conv(x)))
        x_q = x_q.reshape(x_q.size(0), -1)
        x_q = F.relu(self.q_fc1(x_q))
        q_values = self.q_fc2(x_q)
        
        return q_values

if __name__ == "__main__":
    net = ChessDQN(num_blocks=6, num_channels=128)
    net.eval()
    
    dummy_input = torch.randn(2, 18, 8, 8)
    with torch.no_grad():
        q_vals = net(dummy_input)
        
    print("Q-values shape:", q_vals.shape) # Should be [2, 4272]
    print("Parameter count:", sum(p.numel() for p in net.parameters() if p.requires_grad))
