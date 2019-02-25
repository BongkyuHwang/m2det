import torch

def conv_block(in_channels, out_channels, kernel_size, stride, padding=0):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    )
        
class TUM(torch.nn.Module):
    
    def __init__(self, in_channels=256, out_channels=128, num_scales=6):
        super(TUM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales

        self.block_list1 = torch.nn.ModuleList(
            [conv_block(self.in_channels, self.in_channels, 3, 2, 1 if i != self.num_scales -2 else 0) for i in range(self.num_scales - 1)])
        self.block_list2 = torch.nn.ModuleList(
            [conv_block(self.in_channels, self.in_channels, 3, 1, 1) for i in range(self.num_scales - 1)])
        self.block_list3 = torch.nn.ModuleList(
            [conv_block(self.in_channels, self.out_channels, 1, 1) for i in range(self.num_scales)])
            
    def forward(self, input):
        output = [input.clone()]
        for l in self.block_list1:
            input = l(input)
            output.append(input.clone())
            #output.append(l(input.clone()))
        
        for idx, l in enumerate(reversed(self.block_list2)):
            output[-idx-2].add_(torch.nn.functional.interpolate(l(output[-idx-1]), size=output[-idx-2].shape[-1], mode="bilinear", align_corners=True))
        for idx, l in enumerate(self.block_list3):
            output[idx] = l(output[idx])
        return output
       
