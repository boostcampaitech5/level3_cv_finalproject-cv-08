import cv2
import math
import ffmpeg
import numpy as np
from PIL import Image
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from roll_to_wav import MIDISynth
from ultralytics import YOLO


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
        zc_ = F.sigmoid(out)
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

class ResNet(nn.Module):

    def __init__(self, block, layers, top_channel_nums=2048, reduced_channel_nums=256, num_classes=51, scale=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
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

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, layers=[2, 2, 2, 2], top_channel_nums=512, reduced_channel_nums=64, **kwargs)
    return model

def get_video2roll_model(model_path: str="/opt/ml/data/models/video_to_roll.pth") -> resnet18:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_pianodetection_model(model_path: str="/opt/ml/data/models/piano_detection.pt"):
    model = YOLO(model_path)
    return model

def preprocess(video_info, video_range) -> List:
    model = get_pianodetection_model()
    
    out, _ = (
        ffmpeg
        .input("./video/01.mp4", ss=video_range[0], t=video_range[1]-video_range[0])
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="quiet")
        .run(capture_stdout=True)
    )

    frames = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, video_info['height'], video_info['width'], 3])
    )
    
    key_detected = False
    
    for i, frame in enumerate(frames):
        # Piano Detection
        if not key_detected:
            pred = model.predict(source=frame, device='0', verbose=False)
            if pred[0].boxes:
                if pred[0].boxes.conf.item() > 0.8:
                    xmin, ymin, xmax, ymax = tuple(np.array(pred[0].boxes.xyxy.detach().cpu()[0], dtype=int))
                    cv2.imwrite("02.jpg", cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY))
                    start_idx = i
                    key_detected = True
                    break
    
    if key_detected:
        frames = np.mean(frames[start_idx:, ymin:ymax, xmin:xmax, ...], axis=3)
        frames = np.stack([cv2.resize(f, (900 ,100), interpolation=cv2.INTER_LINEAR) for f in frames], axis=0) / 255.
    else:
        # 영상 전체에서 Top View 피아노가 없을 경우 None 반환
        return None
        
    # 5 frame 씩
    frames_with5 = []
    for i in range(len(frames)):
        if i >= 2 and i < len(frames)-2:
            file_list = [frames[i-2], frames[i-1], frames[i], frames[i+1], frames[i+2]]
        elif i < 2:
            file_list = [frames[i], frames[i], frames[i], frames[i+1], frames[i+2]]
        else:
            file_list = [frames[i-2], frames[i-1], frames[i], frames[i], frames[i]]
        frames_with5.append(file_list)
    
    frames_with5 = torch.Tensor(np.array(frames_with5)).float().cuda()
    
    return frames_with5

def predict(frames_with5, frame_range):
    model = get_video2roll_model()
    
    min_key, max_key = 15, 65
    threshold = 0.7

    batch_size = 32
    preds_roll = []
    for idx in range(0, len(frames_with5), batch_size):
        batch_frames = torch.stack([frames_with5[i] for i in range(idx, min(len(frames_with5), idx+batch_size))])
        pred_logits = model(batch_frames)
        pred_roll = torch.sigmoid(pred_logits) >= threshold   
        numpy_pred_roll = pred_roll.cpu().detach().numpy().astype(np.int_)
        
        for pred in numpy_pred_roll:
            preds_roll.append(pred)

    preds_roll = np.asarray(preds_roll).squeeze()
    if preds_roll.shape[0] != frame_range:
        temp = np.zeros((frame_range, max_key-min_key+1))
        temp[:preds_roll.shape[0], :] = preds_roll[:frame_range, :]
        preds_roll = temp

    roll = np.zeros((frame_range, 88))
    roll[:, min_key:max_key+1] = preds_roll
    wav, pm = MIDISynth(roll, frame_range).process_roll()

    return roll, wav


if __name__ == "__main__":
    net = resnet18()
    print(net)
    imgs = torch.rand((2, 5, 100,900))
    logits = net(imgs)
    print(logits.shape)