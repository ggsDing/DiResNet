#An implementation of ResUNet in 'Road extraction by deep residual unet'
import torch
import torch.nn as nn
import torchvision

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.relu = nn.ReLU()
        
        self.identity = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                conv1x1(in_channels, out_channels))

    def forward(self, x, low_feat):
        x = self.up(x)
        x = torch.cat((x, low_feat), dim=1)
        
        identity = self.identity(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + identity

class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.relu = nn.ReLU()
        
        self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                conv1x1(in_channels, out_channels, stride))

    def forward(self, x):
        identity = self.downsample(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + identity

class Enc0(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.relu = nn.ReLU()
        
        self.identity = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + identity


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True):
        super(ResUNet, self).__init__()
        
        self.Enc0 = Enc0(in_channels, 64)
        self.Enc1 = EncBlock(64,128)
        self.Enc2 = EncBlock(128,256)
        self.bridge = EncBlock(256, 512)
        
        self.Dec2 = DecBlock(512+256, 256)
        self.Dec1 = DecBlock(256+128, 128)
        self.Dec0 = DecBlock(128+64, 64)
        
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1))        

    def forward(self, x):
        x_size = x.size()
                
        enc0 = self.Enc0(x)
        enc1 = self.Enc1(enc0)
        enc2 = self.Enc2(enc1)
        enc3 = self.bridge(enc2)
        dec2 = self.Dec2(enc3, enc2)
        dec1 = self.Dec1(dec2, enc1)
        dec0 = self.Dec0(dec1, enc0)
        
        out = self.classifier(dec0) 
        
        return out
