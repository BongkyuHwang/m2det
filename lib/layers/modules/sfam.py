import torch
import collections
from .base_block import conv_block

class SFAM(torch.nn.Module):
    
    def __init__(self, in_channels=1024, mid_channels=64, out_channels=1024, num_levels=8):
        super(SFAM, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.pyramid_levels = num_levels
        
        self.SE = torch.nn.ModuleList(
                [torch.nn.Sequential( 
                    torch.nn.Conv2d(self.in_channels, self.mid_channels, 1, 1, 0),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(self.mid_channels, self.out_channels, 1, 1, 0),
                    torch.nn.Sigmoid()) 
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
            #attention = attention.squeeze(-1).squeeze(-1)
            #attention = attention.squeeze(-1)


            #attention = self.SE[idx](attention).unsqueeze(-1).unsqueeze(-1)
            attention = self.SE[idx](attention)


            aggregated_feature_pyramids.append(aggregated_feature.mul_(attention))

        return aggregated_feature_pyramids

