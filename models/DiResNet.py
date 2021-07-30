#The direction reference data should be generated using the DirectionNet.
import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
#from models.FCN_8s import FCN
from models.FCN_16s import FCN_res18a as FCN
#from models.FCN_16s import FCN_res34a as FCN
from utils import initialize_weights

BN_MOMENTUM = 0.01

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class RefUnet(nn.Module):
    def __init__(self,in_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,in_ch,3,padding=1)

        self.conv1 = nn.Conv2d(in_ch,16,3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####
        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(64+32,32,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(32)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(32+16,16,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(16)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(16,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx))) #scale:1
        hx = self.pool1(hx1) #scale:1/2

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2) #scale:1/4

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3) #scale:1/8

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.upscore2(hx4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual

class FCN_Ref(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_Ref, self).__init__()
        self.FCN = FCN(in_channels, num_classes)
        
        self.Dec = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(32, momentum=BN_MOMENTUM), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(16, momentum=BN_MOMENTUM), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(16, momentum=BN_MOMENTUM), nn.ReLU())
                                 
        self.DConv = nn.Conv2d(16, 5, kernel_size=1)
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.Ref = RefUnet(1)
        
        initialize_weights(self.classifier, self.DConv, self.Dec, self.Ref)
        
    def forward(self, x):
        x_size = x.size()
        
        x = self.FCN.layer0(x)
        x = self.FCN.layer1(x)
        x = self.FCN.layer2(x)
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
        aux_s = self.FCN.classifier(x)
        
        x = self.Dec(x)
        
        aux_d = self.DConv(x)
        out = self.classifier(x)
        
        #DS = aux_d[:,1:2,:,:] + aux_d[:,2:3,:,:] + aux_d[:,3:4,:,:] + aux_d[:,4:5,:,:]
        aux_rf = self.Ref(out)
        
        return out, aux_s, aux_d, aux_rf
