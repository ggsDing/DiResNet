import torch.nn as nn
from torch.nn import functional as F
import math
import torch
import numpy as np
from torch.nn.modules.utils import _pair, _quadruple

def GenerateTemplate(ksize, rwidth):
    r = int((ksize-1)/2)
    step = np.arctan(rwidth/r)
    Direction_list = [-10]
    Index_list = [[[r, r]]]
    for theta in np.arange(-0.7853981633974483, 0.7853981633974483, step):
        Curr_DIndex = []
        for col in range(0, ksize):
            x = col-r
            y = np.around(x*np.tan(theta))
            row = y+r
            Curr_DIndex.append([int(ksize-1-row), int(col)])
        Index_list.append(Curr_DIndex)
        Direction_list.append(theta)
    for theta in np.arange(0.7853981633974483, 2.35619449019, step):
        beta = 1.57079632679-theta
        Curr_DIndex = []
        for row in range(0, ksize):
            y = row-r
            x = np.around(y*np.tan(beta))
            col = x+r
            Curr_DIndex.append([int(ksize-1-row), int(col)])
        Index_list.append(Curr_DIndex)
        Direction_list.append(theta)
    return Index_list, Direction_list
    
def Generate_kernels(ksize, Index_list):
    directions = len(Index_list)
    print(directions)
    kernels = np.zeros((directions, ksize, ksize))
    for i in range(directions):
        for idx in Index_list[i]:
            kernels[i, idx[0], idx[1]] = 1
        print(kernels[i])
        print('\n')
    return kernels

class DirectionNet(nn.Module):
    def __init__(self, in_channels=1, rwidth=1, rescale=True):
        super(DirectionNet, self).__init__()
        self.rescale = rescale
        ksize = rwidth*9
        r = int((ksize-1)/2)
        Index_list, Direction_list = GenerateTemplate(ksize, rwidth)
        print('Direction nums: %d'%len(Direction_list))
        self.Direction_list = Direction_list
        self.DConv = nn.Conv2d(1, len(Direction_list), kernel_size=ksize, stride=1, padding=r, bias=False)
        self.mask_weight(Index_list)
        self.avg = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.DConv.weight.data = self.mask
    
    def mask_weight(self, Index_List):
        out_c, in_c, h, w = self.DConv.weight.data.shape
        mask = torch.zeros((out_c, in_c, h, w), requires_grad=False)
        for i in range(in_c):
            for j in range(out_c):
                for idx in Index_List[j]:
                    mask[j, i, idx[0], idx[1]] = 1
        self.DConv.weight.data = mask
    
    def reduce_direction(self, direction_map, num_directions):
        b, c, h, w = direction_map.size()
        pi = 3.14159265359
        step = pi/num_directions
        reduced_map = torch.zeros(b, num_directions+1, h, w).cuda()
        
        reduced_map[:, 0, :, :] = direction_map[:, 0, :, :]        
        for idx, d in enumerate(self.Direction_list):
            if d>=-pi/8 and d<step-pi/8:
                reduced_map[:, 1, :, :]+=direction_map[:, idx, :, :]
            elif d>=step-pi/8 and d<step*2-pi/8:
                reduced_map[:, 2, :, :]+=direction_map[:, idx, :, :]
            elif d>=step*2-pi/8 and d<step*3-pi/8:
                reduced_map[:, 3, :, :]+=direction_map[:, idx, :, :]
            elif d>=step*3-pi/8 or d<-pi/8:
                reduced_map[:, 4, :, :]+=direction_map[:, idx, :, :]
                
        return reduced_map
    
    def forward(self, x):        
        if self.rescale:
            x = F.interpolate(x, scale_factor=1/8, mode='area')
        out = self.DConv(x)
        out = self.reduce_direction(out, num_directions=4)
        out = torch.round(self.avg(out))
        return out*x
        