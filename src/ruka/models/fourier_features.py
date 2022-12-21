import torch 
import torch.nn as nn

class FourierFeatures(nn.Module):
    def __init__(self, inp_size: int, out_size: int = 128, sigma: float = 10.):
        super().__init__()
        assert out_size % 2 == 0
        self.features_dim = out_size
        self.weights = nn.Parameter(torch.randn(inp_size, out_size // 2) * sigma, requires_grad=False)
    
    def forward(self, input):
        """
        input:  [..., inp_size]
        -----------------------
        return: [..., out_size]
        """
        proj = input @ self.weights
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
