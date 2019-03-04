import torch
from .base_block import conv_block

class FFMv2(torch.nn.Module):
    
    def __init__(self, in_channels=768, out_channels=128):
        super(FFMv2, self).__init__()
        self.block = conv_block(in_channels, out_channels, 1, 1)
        
    def forward(self, input1, input2):
        return torch.cat((self.block(input1), input2), dim=1)

