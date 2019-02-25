import torch
import collections

__all__ = ["FFMv1", "FFMv2", "FFMv3", "TUM", "MultiBox", "TextBox"]

def conv_block(in_channels, out_channels, kernel_size, stride, padding=0):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    )

class SFAM(torch.nn.Module):
    
    def __init__(self, in_channels=1024, mid_channels=64, out_channels=1024, num_levels=8):
        super(SFAM, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.pyramid_levels = num_levels
        
        self.SE = torch.nn.ModuleList(
                [torch.nn.Sequential( 
                    torch.nn.Linear(self.in_channels, self.mid_channels),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(self.mid_channels, self.out_channels),
                    torch.nn.ReLU(inplace=True)) 
                for i in range(self.pyramid_levels)]
                )
        
    def forward(self, feature_pyramids):
        aggregated_feature_pyramids = []
        for idx in range(len(feature_pyramids[0])):
            aggregated_feature = torch.cat(
                    [feature[idx] for feature in feature_pyramids], 
                    dim=1)
            width = aggregated_feature.size(2)
            height = aggregated_feature.size(3)

            attention = torch.nn.functional.avg_pool2d(
                    aggregated_feature.clone(),
                    (width, height))
            attention = attention.squeeze(-1).squeeze(-1)

            attention = self.SE[idx](attention).unsqueeze(-1).unsqueeze(-1)

            aggregated_feature_pyramids.append(aggregated_feature.mul_(attention))

        return aggregated_feature_pyramids

