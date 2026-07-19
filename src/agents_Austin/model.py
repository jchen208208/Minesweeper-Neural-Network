import torch

class ResidualBlock(torch.nn.Module):
    def __init__(self,channels:int):
        super().__init__()