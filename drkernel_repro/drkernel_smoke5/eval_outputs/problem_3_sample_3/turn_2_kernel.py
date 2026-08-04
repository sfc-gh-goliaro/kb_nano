import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # parameters, buffers, etc. go here

    def forward(self, x):
        # elementwise operation: y = 2*x
        return x * 2
