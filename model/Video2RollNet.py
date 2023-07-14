import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from torchvision.models import resnet50, vgg16

__all__ = ['ResNet', 'resnet18']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class FTB(nn.Module):
    def __init__(self,in_planes, out_planes=512, stride=1):
        super(FTB,self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=1,bias=False)
        self.conv1 = conv3x3(out_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.avgpool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(3, 3), stride=1)
    def forward(self, x, avg=True):
        x1 = self.conv0(x)
        residual = x1
        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        if avg:
            out = self.avgpool1(out)
        else:
            out = self.avgpool2(out)
        return out

class FRB(nn.Module):
    def __init__(self,in_planes1,in_planes2):
        super(FRB,self).__init__()
        self.fc1 = nn.Linear(in_planes1+in_planes2, in_planes2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_planes2, in_planes2)
    def forward(self, xl, xh):
        xc = torch.cat([xl,xh],dim=1)
        zc = F.avg_pool2d(xc, kernel_size=xc.size()[2:]) # C x 1 x 1
        zc = torch.flatten(zc, 1)
        out = self.fc1(zc)
        out = self.relu(out)
        out = self.fc2(out)
        zc_ = torch.sigmoid(out)
        zc_ = torch.unsqueeze(zc_,dim=2)
        zc_ = zc_.repeat(1, 1, xl.shape[2] * xl.shape[3]).view(-1,xl.shape[1],xl.shape[2],xl.shape[3])
        xl_ = zc_ * xl #n,c,h,w
        return xl_

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Video2RollNet(nn.Module):
    # original default params of top_channel_nums, reduced_channel_nums are 2048, 256
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], top_channel_nums=512, reduced_channel_nums=64, num_classes=51, scale=1):
        self.inplanes = 64
        super(Video2RollNet, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=(11, 11), stride=(2, 2), padding=(4, 4),bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.FTB2_1 = FTB(128, 128)
        self.FTB2_2 = FTB(128, 128)
        self.FRB2 = FRB(128, 128)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.FTB3 = FTB(256, 128)
        self.FRB3 = FRB(128, 128)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.FTB4 = FTB(512, 128)
        self.FRB4 = FRB(64, 128)


        #FPN PARTS
        # Top layer
        self.toplayer = nn.Conv2d(top_channel_nums, reduced_channel_nums, kernel_size=1, stride=1, padding=0)  # Reduce channels,
        self.toplayer_bn = nn.BatchNorm2d(reduced_channel_nums)
        self.toplayer_relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.maxpool(h)

        h = self.layer1(h)
        x1 = h

        h = self.layer2(h)
        x2 = h

        h = self.layer3(h)

        x3 = h

        h = self.layer4(h)
        x4 = h

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

        out1 = p2*p3

        out1_ = F.softmax(out1.view(*out1.size()[:2], -1),dim=2).view_as(out1)

        out2 = out1_*p4

        out2 = self.conv2(out2)

        out = out2 + p4

        out = F.avg_pool2d(out, kernel_size=out.size()[2:])

        out = torch.flatten(out, 1)

        out = self.fc(out)

        return out
    
class Video2RollNet_vgg16(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], top_channel_nums=512, reduced_channel_nums=64, num_classes=51, scale=1):
        super(Video2RollNet_vgg16, self).__init__()
        self.inplanes = 64

        self.FTB2_1 = FTB(256, 256)
        self.FTB2_2 = FTB(256, 256)
        self.FRB2 = FRB(256, 256)

        self.FTB3 = FTB(512, 256)
        self.FRB3 = FRB(256, 256)

        self.FTB4 = FTB(512, 256)
        self.FRB4 = FRB(64, 256)

        #FPN PARTS
        # Top layer
        self.toplayer = nn.Conv2d(top_channel_nums, reduced_channel_nums, kernel_size=1, stride=1, padding=0)  # Reduce channels,
        self.toplayer_bn = nn.BatchNorm2d(reduced_channel_nums)
        self.toplayer_relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=1)
        self.fc = nn.Linear(256, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        self.backbone = vgg16(pretrained=False)
        self.vgg16_features = self.backbone.features
        
        self.conv1_1 = nn.Conv2d(5, 64, (3, 3), (1, 1), (1, 1))
        self.relu1_1 = self.vgg16_features[1]
        self.conv1_2 = self.vgg16_features[2]
        self.relu1_2 = self.vgg16_features[3]
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        
        self.layer1 = nn.Sequential(
            self.vgg16_features[5],
            self.vgg16_features[6],
            self.vgg16_features[7],
            self.vgg16_features[8],
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        )
        self.layer2 = nn.Sequential(
            self.vgg16_features[10],
            self.vgg16_features[11],
            self.vgg16_features[12],
            self.vgg16_features[13],
            self.vgg16_features[14],
            self.vgg16_features[15],
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        )
        self.layer3 = nn.Sequential(
            self.vgg16_features[17],
            self.vgg16_features[18],
            self.vgg16_features[19],
            self.vgg16_features[20],
            self.vgg16_features[21],
            self.vgg16_features[22],
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        )
        self.layer4 = nn.Sequential(
            self.vgg16_features[24],
            self.vgg16_features[25],
            self.vgg16_features[26],
            self.vgg16_features[27],
            self.vgg16_features[28],
            self.vgg16_features[29],
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        )
    
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        h = x # (5, 100, 900)
        h = self.conv1_1(h)
        h = self.relu1_1(h)
        h = self.conv1_2(h)
        h = self.relu1_2(h)
        h = self.maxpool(h) # (64, 50, 450)

        h = self.layer1(h) # (128, 25, 225)
        x1 = h
        h = self.layer2(h) # (256, 12, 112)
        x2 = h
        h = self.layer3(h) # (512, 6, 56)
        x3 = h
        h = self.layer4(h) # (512, 3, 28)
        x4 = h
        # Top-down
        x5 = self.toplayer(x4) # (64, 3, 28)
        x5 = self.toplayer_relu(self.toplayer_bn(x5))

        x2_ = self.FTB2_1(x2)
        x2_ = self.FTB2_2(x2_)
        x3_ = self.FTB3(x3)
        x4_ = self.FTB4(x4, avg=False)

        p4 = self.FRB4(x4_, x5)
        p3 = self.FRB3(x3_, p4)
        p2 = self.FRB2(x2_, p3)

        out1 = p2*p3
        out1_ = F.softmax(out1.view(*out1.size()[:2], -1),dim=2).view_as(out1)
        out2 = out1_*p4
        out2 = self.conv2(out2)
        out = out2 + p4
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
    
class Video2RollNet_resnet50(nn.Module):
    # original default params of top_channel_nums, reduced_channel_nums are 2048, 256
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], top_channel_nums=2048, reduced_channel_nums=256, num_classes=51, scale=1):
        self.inplanes = 64
        super(Video2RollNet_resnet50, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=(11, 11), stride=(2, 2), padding=(4, 4),bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.FTB2_1 = FTB(512, 512)
        self.FTB2_2 = FTB(512, 512)
        self.FRB2 = FRB(512, 512)

        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.FTB3 = FTB(1024, 512)
        self.FRB3 = FRB(512, 512)

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.FTB4 = FTB(2048, 512)
        self.FRB4 = FRB(256, 512)


        #FPN PARTS
        # Top layer
        self.toplayer = nn.Conv2d(top_channel_nums, reduced_channel_nums, kernel_size=1, stride=1, padding=0)  # Reduce channels,
        self.toplayer_bn = nn.BatchNorm2d(reduced_channel_nums)
        self.toplayer_relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        self.backbone = resnet50(pretrained=True)
        
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.maxpool(h)

        h = self.layer1(h)
        x1 = h
        h = self.layer2(h)
        x2 = h
        h = self.layer3(h)
        x3 = h

        h = self.layer4(h)
        x4 = h
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

        out1 = p2*p3

        out1_ = F.softmax(out1.view(*out1.size()[:2], -1),dim=2).view_as(out1)

        out2 = out1_*p4

        out2 = self.conv2(out2)

        out = out2 + p4

        out = F.avg_pool2d(out, kernel_size=out.size()[2:])

        out = torch.flatten(out, 1)

        out = self.fc(out)

        return out


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, layers=[2, 2, 2, 2], top_channel_nums=512, reduced_channel_nums=64, **kwargs)
    return model

if __name__ == "__main__":
    net = resnet18()
    print(net)
    imgs = torch.rand((2, 5, 100,900))
    logits = net(imgs)
    print(logits.shape)