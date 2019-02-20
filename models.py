import types
import pretrainedmodels.models.pnasnet as pnasnet
import pretrainedmodels.models.senet as senet

__all__ = ["modify_senet"]
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

