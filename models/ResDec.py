import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from utils import initialize_weights
#from models.FCN_8s import FCN
from models.FCN_8s import FCN_res18a as FCN
#from models.FCN_16s import FCN_res34a as FCN

import functools
import sys, os

BN_MOMENTUM = 0.01

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN_Dec(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True):
        super(FCN_Dec, self).__init__()
        self.FCN = FCN(in_channels, num_classes)
        
        self.Dec = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(32, momentum=BN_MOMENTUM), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(16, momentum=BN_MOMENTUM), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(16, momentum=BN_MOMENTUM), nn.ReLU())
                                                                
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)
                
        initialize_weights(self.Dec, self.classifier)
        
    def forward(self, x):
        x_size = x.size()
        
        x = self.FCN.layer0(x)
        x = self.FCN.layer1(x)
        x = self.FCN.layer2(x)
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
                
        out = self.Dec(x)
        out = self.classifier(out)
        
        return out
