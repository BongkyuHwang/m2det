import torch

def conv_block(in_channels, out_channels, kernel_size, stride, padding=0):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    )
        
class FFMv3(torch.nn.Module):

    def __init__(self, in_channels=[4320, 2160, 1080], out_channels=[540, 270, 135]):
        super(FFMv3, self).__init__()
        self.block1 = conv_block(in_channels[0], out_channels[0], 1, 1)
        self.block2 = conv_block(in_channels[1], out_channels[1], 3, 1, 1)
        self.block3 = conv_block(in_channels[2], out_channels[2], 3, 1, 1)

    def forward(self, deep, mid, shallow):
        return torch.cat([
            torch.nn.functional.interpolate(
                self.block1(deep), scale_factor=4, mode="bilinear", align_corners=True
            ),
            torch.nn.functional.interpolate(
                self.block2(mid), scale_factor=2, mode="bilinear", align_corners=True
            ),
            self.block3(shallow)
        ], dim=1
        )
