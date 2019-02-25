import torch

def conv_block(in_channels, out_channels, kernel_size, stride, padding=0):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    )
        
class FFMv2(torch.nn.Module):
    
    def __init__(self, in_channels=768, out_channels=128):
        super(FFMv2, self).__init__()
        self.block = conv_block(in_channels, out_channels, 1, 1)
        
    def forward(self, input1, input2):
        return torch.cat((self.block(input1), input2), dim=1)

