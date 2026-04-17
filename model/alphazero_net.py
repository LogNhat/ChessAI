import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class AlphaZeroNet(nn.Module):
    def __init__(self, num_blocks=10, num_channels=192, num_input_channels=18, num_moves=4272):
        super().__init__()
        
        # Initial convolutional block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_input_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_blocks)])
        
        # Policy Head
        # AlphaZero uses 2 channels of 1x1 conv
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_moves) # 2 * 64 = 128
        
        # Value Head
        # AlphaZero uses 1 channel of 1x1 conv, followed by 256-d hidden layer
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, 18, 8, 8)
        Returns:
            policy: Logits of shape (batch_size, num_moves) # CrossEntropyLoss expects logits, not softmax
            value: Continuous value from -1 to 1 of shape (batch_size, 1)
        """
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)
            
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1) # Flatten
        policy_logits = self.policy_fc(p)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1) # Flatten
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value

if __name__ == "__main__":
    # Test the network
    net = AlphaZeroNet(num_blocks=10) # Using 10 to keep it manageable for local testing
    
    # Create fake batch of 4 inputs
    dummy_input = torch.randn(4, 18, 8, 8)
    policy, value = net(dummy_input)
    
    print("Policy shape:", policy.shape) # Should be [4, 4272]
    print("Value shape:", value.shape)   # Should be [4, 1]
    
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:,}")
