import torch

class TextBox(torch.nn.Module):
    
    def __init__(self, in_channels, mbox=[4, 6, 6, 6, 4, 4]):
        super(TextBox, self).__init__()
        # text + background
        self.num_classes = 2
        # (x, y, w, h)
        self.loc_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels, num_box * 4, (3, 5), padding=(1, 2)) for num_box in mbox
        ])
        # (x1, y1, x2, y2, x3, y3, x4, y4)
        self.quad_layers = torch.nn.ModuleList([
            torch.nn.Conv2(in_chnnels, num_box * 8, (3, 5), padding=(1, 2)) for num_box in mbox
        ])
        # (x1, y1, x2, y2, h)
        self.rot_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels, num_box * 5 (3, 5), padding=(1, 2)) for num_box in mbox
        ])
        self.conf_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels, num_box * self.num_classes, (3, 5), padding=(1, 2)) for num_box in mbox
        ])

    def forward(self, feature_pyramids):
        locations = []
        quadrilaterals = []
        rotates = []
        confidences = []

        for (x, l, q, r, c) in zip(feature_pyramids, self.loc_layers, self.quad_layers, self.rot_layers, self.conf_layers):
            locations.append(l(x).permute(0, 2, 3, 1).contiguous())
            quadrilateral.append(q(x).permute(0, 2, 3, 1).contignous())
            rotates.append(r(x).permute(0, 2, 3, 1).contignous())
            confidences.append(c(x).permute(0, 2, 3 ,1).contiguous())
        
        locations = torch.cat([loc.view(loc.shape[0], -1) for loc in locations], 1)
        quadrilaterals = torch.cat([quad.view(quad.shape[0], -1) for quad in quadrilaterals], 1)
        rotates = torch.cat([rot.view(rot.shape[0], -1) for rot in rotates], 1)
        confidences = torch.cat([con.view(con.shape[0], -1) for con in confidences], 1)
        
        return (torch.cat([locations.view(locations.shape[0], -1, 4), 
                            quadrilaterals.view(quadrilaterals.shape[0], -1, 8),
                            rotates.view(rotates.shape[0], -1, 5)],
                            dim=2),
                confidences.view(confidences.shape[0], -1, self.num_classes))

