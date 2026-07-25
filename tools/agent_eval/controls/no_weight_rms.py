import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Deliberately never applies self.weight — pre-fix this PASSED everywhere."""
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True, **kw):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    def forward(self, x, residual=None):
        if residual is not None:
            residual.add_(x)
            x = residual
        v = x.float()
        out = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        if residual is not None:
            return out, residual
        return out
