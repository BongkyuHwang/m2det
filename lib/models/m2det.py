import types

import torch
import torch.utils.model_zoo as model_zoo

import pretrainedmodels
import pretrainedmodels.models.pnasnet as pnasnet
import pretrainedmodels.models.senet as senet

from .. import data
from .. import layers

model_names = ["pnasnet5large", "se_resnext101_32x4d"]

# chnage stride 2 to 1 for increasing feature map sizes
# ex stride = 2) 3, 320, 320 -> layer3 : 1024, 20, 20 | layer4 : 2048, 10, 10
# ex stride = 1) 3, 320, 320 -> layer3 : 1024, 40, 40 | layer4 : 2048, 20, 20
def modify_senet(model):
    del model.avg_pool
    del model.last_linear

    blocks = len(model.layer3)
    groups = model.layer3[0].conv2.groups
    downsample_kernel_size = model.layer3[0].downsample[0].kernel_size[0]
    downsample_padding = model.layer3[0].downsample[0].padding[0]
    del model.layer3

    model.inplanes = 512
    model.layer3 = model._make_layer(block=senet.SEResNeXtBottleneck, planes=256, blocks=blocks, 
                                    stride=1, groups=groups, reduction=16, 
                                    downsample_kernel_size=downsample_kernel_size, 
                                    downsample_padding=downsample_padding)
    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x3, x4

    model.forward = types.MethodType(forward, model)

    return model

def modify_pnasnet(model):
    del model.relu
    del model.avg_pool
    del model.dropout
    del model.last_linear
    
    def forward(self, x):
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

    model.forward = types.MethodType(forward, model)
    return model


class M2Det(torch.nn.Module):
    def __init__(self, phase="train", num_scales=6, num_levels=8, num_classes=21, model_name="pnasnet5large"):
        super(M2Det, self).__init__()
        self.phase = phase
        self.num_scales = num_scales
        self.num_levels = num_levels
        self.num_classes = num_classes

        self.cfg = (data.coco, data.voc)[num_classes == 21]
        self.prior_boxes = layers.PriorBox(self.cfg).forward()                       
        self.model_name = model_name
        self.settings = pretrainedmodels.pretrained_settings[self.model_name]["imagenet"]

        if self.model_name == "pnasnet5large":
            self.fe = modify_pnasnet(pnasnet.pnasnet5large(self.settings["num_classes"]))
            self.ffm1 = layers.FFMv3(in_channels=[4320, 2160, 1080], out_channels=[460, 230, 120])
            self.ffm2 = torch.nn.ModuleList([
                layers.FFMv2(in_channels=810, out_channels=135) for i in  range(self.num_levels)]) 
            self.tums = torch.nn.ModuleList([
                layers.TUM(in_channels=(810 if i == 0 else 270), out_channels=135, num_scales=self.num_scales) for i in range(self.num_levels)])
            self.sfam = layers.SFAM(in_channels=1080, mid_channels=68, out_channels=1080, num_levels=self.num_levels)
            self.mb = layers.MultiBox(in_channels=1080, num_classes=self.num_classes)

        elif self.model_name == "se_resnext101_32x4d":
            self.fe = models.modify_senet(senet.se_resnext101_32x4d(self.settings["num_classes"]))
            self.ffm1 = layers.FFMv1(in_channels=[2048, 1024], out_channels=[512, 256])
            self.ffm2 = torch.nn.ModuleList([
                layers.FFMv2(in_channels=768, out_channels=128) for i in  range(self.num_levels)])
            self.tums = torch.nn.ModuleList([
                layers.TUM(in_channels=(768 if i == 0 else 256), out_channels=128, num_scales=self.num_scales) for i in range(self.num_levels)])
            self.sfam = layers.SFAM(in_channels=1024, mid_channels=64, out_channels=1024, num_levels=self.num_levels)
            self.mb = layers.MultiBox(in_channels=1024, num_classes=self.num_classes)
        self.detect = layers.Detect(self.num_classes, 0, 200, 0.05, 0.45)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        self.fe.load_state_dict(model_zoo.load_url(self.settings["url"]), strict=False)

    def forward(self, x):
        if self.model_name == "pnasnet5large":
            shallow, mid, deep = self.fe(x)
            x = self.ffm1(deep, mid, shallow)
        else:
            shallow, deep = self.fe(x)
            x = self.ffm1(deep, shallow)

        feature_pyramids = []
        feature_pyramids.append(self.tums[0](x))
        for idx, (ffm_v2, tum) in enumerate(zip(self.ffm_v2s, self.tums[1:])):
            feature_pyramids.append(tum(ffm_v2(x, feature_pyramids[idx][0])))

        feature_pyramids = self.sfam(feature_pyramids)
        locations, confidences = self.mb(feature_pyramids)
        
        if self.phase == "test":
            return self.detect(locations,
                                torch.nn.functional.softmax(confidences, dim=-1),
                                self.prior_boxes.type(type(x.data))
            )
        else:
            return (locations, confidences, self.prior_boxes)

