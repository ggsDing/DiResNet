#An implementation of CasNet in 'Automatic road detection and centerline extraction via cascaded end-to-end convolutional neural network'
import torch
from torch import nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    
class CasNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True, freeze_bn=False):
        super(CasNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        self.conv1_1 = conv3x3(in_channels, 32)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = conv3x3(32, 32)
        self.bn1_2 = nn.BatchNorm2d(32)
        
        self.conv2_1 = conv3x3(32, 64)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = conv3x3(64, 64)
        self.bn2_2 = nn.BatchNorm2d(64)
        
        self.conv3_1 = conv3x3(64, 128)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = conv3x3(128, 128)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.conv3_3 = conv3x3(128, 128)
        self.bn3_3 = nn.BatchNorm2d(128)
        
        self.conv4_1 = conv3x3(128, 256)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = conv3x3(256, 256)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.conv4_3 = conv3x3(256, 256)
        self.bn4_3 = nn.BatchNorm2d(256)
        
        self.dconv4_1 = conv3x3(256, 256)
        self.dbn4_1 = nn.BatchNorm2d(256)
        self.dconv4_2 = conv3x3(256, 256)
        self.dbn4_2 = nn.BatchNorm2d(256)
        self.dconv4_3 = conv3x3(256, 128)
        self.dbn4_3 = nn.BatchNorm2d(128)
        
        self.dconv3_1 = conv3x3(128, 128)
        self.dbn3_1 = nn.BatchNorm2d(128)
        self.dconv3_2 = conv3x3(128, 128)
        self.dbn3_2 = nn.BatchNorm2d(128)
        self.dconv3_3 = conv3x3(128, 64)
        self.dbn3_3 = nn.BatchNorm2d(64)
        
        self.dconv2_1 = conv3x3(64, 64)
        self.dbn2_1 = nn.BatchNorm2d(64)
        self.dconv2_2 = conv3x3(64, 32)
        self.dbn2_2 = nn.BatchNorm2d(32)
        
        self.dconv1_1 = conv3x3(32, 32)
        self.dbn1_1 = nn.BatchNorm2d(32)
        self.dconv1_2 = conv3x3(32, 32)
        self.dbn1_2 = nn.BatchNorm2d(32)
        
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x, indices1 = self.pool(x)
        
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x, indices2 = self.pool(x)
        
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x, indices3 = self.pool(x)
        
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.relu(self.bn4_3(self.conv4_3(x)))
        x, indices4 = self.pool(x)
                
        x = self.unpool(x, indices=indices4)
        x = self.relu(self.dbn4_1(self.dconv4_1(x)))
        x = self.relu(self.dbn4_2(self.dconv4_2(x)))
        x = self.relu(self.dbn4_3(self.dconv4_3(x)))
        
        x = self.unpool(x, indices=indices3)
        x = self.relu(self.dbn3_1(self.dconv3_1(x)))
        x = self.relu(self.dbn3_2(self.dconv3_2(x)))
        x = self.relu(self.dbn3_3(self.dconv3_3(x)))
        
        x = self.unpool(x, indices=indices2)
        x = self.relu(self.dbn2_1(self.dconv2_1(x)))
        x = self.relu(self.dbn2_2(self.dconv2_2(x)))
        
        x = self.unpool(x, indices=indices1)
        x = self.relu(self.dbn1_1(self.dconv1_1(x)))
        x = self.relu(self.dbn1_2(self.dconv1_2(x)))
        
        out = self.classifier(x)
        
        return out
        
        
        
        
        
        
        
        
        
        
