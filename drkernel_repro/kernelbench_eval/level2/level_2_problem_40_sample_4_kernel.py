import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    """
    Optimized and numerically-consistent version of the original Model.

    Original did:
      y  = matmul(x)            # y = x @ W^T + b
      ox = y.clone().detach()   # redundant copy
      y  = y * s
      out= y + ox               # out = y * (1 + s) = (x @ W^T + b) * (1 + s)

    We compute:
      1) y = addmm(bias, x, W^T)  -> y = x @ W^T + b  (single GEMM+bias call)
      2) y = y * (1 + s)          -> single elementwise pass

    This removes the redundant clone and collapses scale+add to a single scale,
    while keeping GEMM+bias numerics very close to the original.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.linear.weight   # [out_features, in_features] = [N, K]
        b = self.linear.bias     # [out_features] or None

        # Compute y = x @ W^T + b using a single cuBLAS addmm call.
        # torch.addmm(input=b, mat1=x, mat2=W.t())
        # Shapes:
        #  x:  [M, K]
        #  W^T:[K, N]
        #  b:  [N]
        y = torch.addmm(b, x, W.t()) if b is not None else torch.mm(x, W.t())

        # Apply the final scaling: out = y * (1 + scaling_factor)
        alpha = 1.0 + self.scaling_factor
        y = y * alpha
        return y
