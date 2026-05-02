import torch
import torch.nn.functional as F
import torch.nn as nn

# FINISHED TYPING FOR NOW, TESTED

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.relu(x)

class ReLU_Squared(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.relu(x) ** 2

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.gelu(x)
