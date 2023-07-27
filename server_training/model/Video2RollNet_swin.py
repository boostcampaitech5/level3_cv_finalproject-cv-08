from .Video2RollNet import conv3x3, FTB, FRB, BasicBlock, Bottleneck
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from .swin_backbone import SwinTransformer


class Video2RollNet_swin(nn.Module):
    # original default params of top_channel_nums, reduced_channel_nums are 2048, 256
    def __init__(
        self,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        top_channel_nums=512,
        reduced_channel_nums=64,
        num_classes=51,
        scale=1,
    ):
        self.inplanes = 64
        super(Video2RollNet_swin, self).__init__()

        self.FTB2_1 = FTB(192, 192)
        self.FTB2_2 = FTB(192, 192)
        self.FRB2 = FRB(192, 192)

        self.FTB3 = FTB(384, 192)
        self.FRB3 = FRB(192, 192)

        self.FTB4 = FTB(768, 192)
        self.FRB4 = FRB(96, 192)

        # FPN PARTS
        # Top layer
        self.toplayer = nn.Conv2d(
            top_channel_nums, reduced_channel_nums, kernel_size=1, stride=1, padding=0
        )  # Reduce channels,
        self.toplayer_bn = nn.BatchNorm2d(reduced_channel_nums)
        self.toplayer_relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(192, 192, kernel_size=1)
        self.fc = nn.Linear(192, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.backbone = SwinTransformer(in_chans=3, use_checkpoint=True)
        self.backbone.init_weights(
            pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth"
        )

        self.backbone.patch_embed.proj = nn.Conv2d(5, 96, kernel_size=(4, 4), stride=(4, 4))

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode="bilinear")

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode="bilinear") + y

    def forward(self, x):
        h = x
        x1, x2, x3, x4 = self.backbone(h)

        # Top-down
        x5 = self.toplayer(x4)
        x5 = self.toplayer_relu(self.toplayer_bn(x5))

        x2_ = self.FTB2_1(x2)

        x2_ = self.FTB2_2(x2_)

        x3_ = self.FTB3(x3)

        x4_ = self.FTB4(x4, avg=False)

        p4 = self.FRB4(x4_, x5)

        p3 = self.FRB3(x3_, p4)

        p2 = self.FRB2(x2_, p3)

        out1 = p2 * p3

        out1_ = F.softmax(out1.view(*out1.size()[:2], -1), dim=2).view_as(out1)

        out2 = out1_ * p4

        out2 = self.conv2(out2)

        out = out2 + p4

        out = F.avg_pool2d(out, kernel_size=out.size()[2:])

        out = torch.flatten(out, 1)

        out = self.fc(out)

        return out
