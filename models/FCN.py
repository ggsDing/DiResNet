import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np

import functools
import sys, os

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class FCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, :3, :, :].copy_(resnet.conv1.weight.data)
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        self.head = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128, momentum=0.95),
                                  nn.ReLU())
        
        self.final = nn.Sequential(
            conv3x3(128, 128),
            # nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128, momentum=0.95),
            nn.ReLU(),
            conv3x3(128, 128),
            nn.BatchNorm2d(128, momentum=0.95),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, num_classes, kernel_size=1))

    def forward(self, x):
        x_size = x.size()
        
        x0 = self.layer0(x)
        x = self.maxpool(x0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        x = self.final(x)
        
        return F.upsample(x, x_size[2:], mode='bilinear') #, x0 #F.upsample(x, x_size[2:], mode='bilinear'), F.upsample(x0, x_size[2:], mode='bilinear')#
        
        
class FCN_res18(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, :3, :, :].copy_(resnet.conv1.weight.data)
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.head = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(64, momentum=0.95),
                                  nn.ReLU())
        
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128, momentum=0.95),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x_size = x.size()
        
        x0 = self.layer0(x)
        x = self.maxpool(x0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        x = self.classifier(x)
        
        return F.upsample(x, x_size[2:], mode='bilinear') 