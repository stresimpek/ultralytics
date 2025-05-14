import torch
import torch.nn as nn

class SE(nn.Module):
    """Squeeze-and-Excitation block - lightweight attention mechanism.
    
    Args:
        c (int): Number of input/output channels
        r (int): Reduction ratio for channels in the SE block (default=16)
    """
    def __init__(self, c, r=16):
        super().__init__()
        # Ensure minimum of at least 1 channel after reduction
        c_reduced = max(1, c // r)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c_reduced, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)  # SiLU activation for better performance
        self.fc2 = nn.Conv2d(c_reduced, c, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
