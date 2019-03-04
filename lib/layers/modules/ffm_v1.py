import torch
from .base_block import conv_block

class FFMv1(torch.nn.Module):
    
    def __init__(self, in_channels=[1024,512], out_channels=[512, 256]):
        super(FFMv1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block1 = conv_block(self.in_channels[0], self.out_channels[0], 1, 1)
        self.block2 = conv_block(self.in_channels[1], self.out_channels[1], 1, 1)
                       
    def forward(self, input1, input2):
        return torch.cat(
            (torch.nn.functional.interpolate(
                self.block1(input1), 
                scale_factor=2, 
                mode="bilinear", 
                align_corners=True),
            self.block2(input2)), dim=1)

