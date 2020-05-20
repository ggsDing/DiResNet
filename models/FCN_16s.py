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
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

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
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        self.head = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128, momentum=0.95),
                                  nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128, momentum=0.95),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x_size = x.size()
        
        x = self.layer0(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        x = self.classifier(x)
        
        return F.upsample(x, x_size[2:], mode='bilinear') #, F.upsample(x0, x_size[2:], mode='bilinear')#

class FCN_res18(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128, momentum=0.95),
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

class FCN_res34(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res34, self).__init__()
        resnet = models.resnet34(pretrained=False)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128, momentum=0.95),
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

class FCN_res18a(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=False):
        super(FCN_res18a, self).__init__()
        resnet = models.resnet18(pretrained=False)
        newconv1 = conv3x3(in_channels, 64, stride=2)
        
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.head = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(64, momentum=0.95),
                                  nn.ReLU())
        
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        x_size = x.size()
        
        x = self.layer0(x) #scale:1/2, 32
        x = self.layer1(x) #scale:1/2, 64
        x = self.layer2(x) #scale:1/4, 128
        x = self.layer3(x) #scale:1/8, 256
        x = self.layer4(x) #scale:1/8, 512
        x = self.head(x)
        x = self.classifier(x)
        
        out = F.upsample(x, x_size[2:], mode='bilinear') 
        
        return out

class FCN_res34a(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res34a, self).__init__()
        resnet = models.resnet34(pretrained=False)
        newconv1 = conv3x3(in_channels, 64, stride=2)
        
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.head = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(64, momentum=0.95),
                                  nn.ReLU())
        
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        x_size = x.size()
        
        x = self.layer0(x) #scale:1/2, 32
        x = self.layer1(x) #scale:1/2, 64
        x = self.layer2(x) #scale:1/4, 128
        x = self.layer3(x) #scale:1/8, 256
        x = self.layer4(x) #scale:1/8, 512
        x = self.head(x)
        x = self.classifier(x)
        
        out = F.upsample(x, x_size[2:], mode='bilinear') 
        
        return out