import torch
import torch.utils.model_zoo as model_zoo
import pretrainedmodels.models.pnasnet as pnasnet
import collections
import layers.functions.detection as detection
import layers.functions.prior_box as prior_box
import data

def conv_block(in_channels, out_channels, kernel_size, stride, padding=0):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    )
        
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
        
class FFMv2(torch.nn.Module):
    
    def __init__(self, in_channels=768, out_channels=128):
        super(FFMv2, self).__init__()
        self.block = conv_block(in_channels, out_channels, 1, 1)
        
    def forward(self, input1, input2):
        return torch.cat((self.block(input1), input2), dim=1)

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
        
        return locations.view(locations.shape[0], -1, 4), 
            quadrilaterals.view(quadrilaterals.shape[0], -1, 8),
            rotates.view(rotates.shape[0], -1, 8), 
            confidences.view(confidences.shape[0], -1, self.num_classes)

class PNASNetFE(torch.nn.Module):

    def __init__(self):
        super(PNASNetFE, self).__init__()
        self.conv_0 = torch.nn.Sequential(collections.OrderedDict([
            ('conv', torch.nn.Conv2d(3, 96, kernel_size=3, stride=2, bias=False)),
            ('bn', torch.nn.BatchNorm2d(96, eps=0.001))
        ]))
        self.cell_stem_0 = pnasnet.CellStem0(in_channels_left=96, out_channels_left=54,
                in_channels_right=96, out_channels_right=54)
        self.cell_stem_1 = pnasnet.Cell(in_channels_left=96, out_channels_left=108,
                in_channels_right=270, out_channels_right=108,
                match_prev_layer_dimensions=True,
                is_reduction=True)
        self.cell_0 = pnasnet.Cell(in_channels_left=270, out_channels_left=216,
                in_channels_right=540, out_channels_right=216,
                match_prev_layer_dimensions=True)
        self.cell_1 = pnasnet.Cell(in_channels_left=540, out_channels_left=216,
                in_channels_right=1080, out_channels_right=216)
        self.cell_2 = pnasnet.Cell(in_channels_left=1080, out_channels_left=216,
                in_channels_right=1080, out_channels_right=216)
        self.cell_3 = pnasnet.Cell(in_channels_left=1080, out_channels_left=216,
                in_channels_right=1080, out_channels_right=216)
        self.cell_4 = pnasnet.Cell(in_channels_left=1080, out_channels_left=432,
                in_channels_right=1080, out_channels_right=432,
                is_reduction=True, zero_pad=True)
        self.cell_5 = pnasnet.Cell(in_channels_left=1080, out_channels_left=432,
                in_channels_right=2160, out_channels_right=432,
                match_prev_layer_dimensions=True)
        self.cell_6 = pnasnet.Cell(in_channels_left=2160, out_channels_left=432,
                in_channels_right=2160, out_channels_right=432)
        self.cell_7 = pnasnet.Cell(in_channels_left=2160, out_channels_left=432,
                in_channels_right=2160, out_channels_right=432)
        self.cell_8 = pnasnet.Cell(in_channels_left=2160, out_channels_left=864,
                in_channels_right=2160, out_channels_right=864,
                is_reduction=True)
        self.cell_9 = pnasnet.Cell(in_channels_left=2160, out_channels_left=864,
                in_channels_right=4320, out_channels_right=864,
                match_prev_layer_dimensions=True)
        self.cell_10 = pnasnet.Cell(in_channels_left=4320, out_channels_left=864,
                in_channels_right=4320, out_channels_right=864)
        self.cell_11 = pnasnet.Cell(in_channels_left=4320, out_channels_left=864,
                in_channels_right=4320, out_channels_right=864)

    def features(self, x):
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        return x_cell_3, x_cell_7, x_cell_11
            
    def forward(self, input):
        return self.features(input)
                    
class M2Det(torch.nn.Module):

    def __init__(self, phase="train", num_scales=6, num_levels=8, num_classes=21, backborn="pnasnet5large"):
        super(M2Det, self).__init__()
        self.phase = phase
        self.num_scales = num_scales
        self.num_levels = num_levels
        self.num_classes = num_classes

        self.cfg = (data.coco, data.voc)[num_classes == 21]
        self.prior_boxes = prior_box.PriorBox(self.cfg).forward()
        self.settings = pnasnet.pretrained_settings[backborn]["imagenet"]
        
        self.backborn = PNASNetFE()
        #self.backborn.load_state_dict(model_zoo.load_url(self.settings["url"]), strict=False)
        
        self.ffm_v1 = FFMv3(in_channels=[4320, 2160, 1080], out_channels=[460, 230, 120])
        self.ffm_v2s = torch.nn.ModuleList([
            FFMv2(in_channels=810, out_channels=135) for i in  range(self.num_levels)])
        self.tums = torch.nn.ModuleList([
            TUM(in_channels=(810 if i == 0 else 270), out_channels=135, num_scales=self.num_scales) for i in range(self.num_levels)])
        self.sfam = SFAM(in_channels=1080, mid_channels=68, out_channels=1080, num_levels=self.num_levels)
        self.mb = MultiBox(in_channels=1080, num_classes=self.num_classes)
        self.detect = detection.Detect(self.num_classes, 0, 200, 0.05, 0.45)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        self.backborn.load_state_dict(model_zoo.load_url(self.settings["url"]), strict=False)

    def forward(self, x):
        shallow, mid, deep = self.backborn(x)
        x = self.ffm_v1(deep, mid, shallow)

        feature_pyramids = []
        feature_pyramids.append(self.tums[0](x))
        for idx, (ffm_v2, tum) in enumerate(zip(self.ffm_v2s, self.tums[1:])):
            feature_pyramids.append(tum(ffm_v2(x, feature_pyramids[idx][0])))

        feature_pyramids = self.sfam(feature_pyramids)
        locations, confidences = self.mb(feature_pyramids)
        if self.phase == "test":
            return self.detect(
                    locations,
                    torch.nn.functional.softmax(confidences, dim=-1),
                    self.prior_boxes.type(type(x.data))
            )
        else:
            return (locations, confidences, self.prior_boxes)




