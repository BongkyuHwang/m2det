import torch

class MultiBox(torch.nn.Module):

    def __init__(self, in_channels, num_classes, mbox=[4, 6, 6, 6, 4, 4]):
        super(MultiBox, self).__init__()
        self.num_classes = num_classes
        self.loc_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels, num_box * 4, 3, padding=1) for num_box in mbox
        ])
        self.conf_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels, num_box * num_classes, 3, padding=1) for num_box in mbox
        ])

    def forward(self, feature_pyramids):

        locations = []
        confidences = []

        for (x, l, c) in zip(feature_pyramids, self.loc_layers, self.conf_layers):
            locations.append(l(x).permute(0, 2, 3, 1).contiguous())
            confidences.append(c(x).permute(0, 2, 3 ,1).contiguous())
        
        locations = torch.cat([loc.view(loc.shape[0], -1) for loc in locations], 1)
        confidences = torch.cat([con.view(con.shape[0], -1) for con in confidences], 1)

        return locations.view(locations.shape[0], -1, 4), confidences.view(confidences.shape[0], -1, self.num_classes)


