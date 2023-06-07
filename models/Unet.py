#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import wfdb
from random import shuffle
from config import config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F


from IPython.display import Image


# ## Model Structure : U-net

import torch

import torchvision


def conv1d_bn(in_channels, filters, kernel_size, stride=1,padding=4, activation='relu'):

    affine = False if activation == 'relu' or activation == 'sigmoid' else True
    sequence = []
    sequence += [torch.nn.Conv1d(in_channels=in_channels,out_channels=filters,kernel_size=kernel_size,stride=stride,bias=False,padding=padding)]
    sequence += [torch.nn.BatchNorm1d(filters, affine=affine)]
    if activation == "relu":
        sequence += [torch.nn.ReLU()]
    elif activation == "sigmoid":
        sequence += [torch.nn.Sigmoid()]
    elif activation == 'tanh':
        sequence += [torch.nn.Tanh()]
    return torch.nn.Sequential(*sequence)


class MultiResBlock(torch.nn.Module):
    def __init__(self, in_channels, u, alpha=1.67, use_dropout=False):
        super().__init__()
        w = alpha * u
        self.out_channel = int(w * 0.167) + int(w * 0.333) + int(w * 0.5)
        self.conv1d_bn = conv1d_bn(in_channels, self.out_channel, 1, activation=None,padding=0)
        self.conv3x3 = conv1d_bn(in_channels, int(w * 0.167), 9, activation='relu')
        self.conv5x5 = conv1d_bn(int(w * 0.167), int(w * 0.333), 9, activation='relu')
        self.conv7x7 = conv1d_bn(int(w * 0.333), int(w * 0.5), 9, activation='relu')
        self.bn_1 = torch.nn.BatchNorm1d(self.out_channel)
        self.relu = torch.nn.ReLU()
        self.bn_2 = torch.nn.BatchNorm1d(self.out_channel)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = torch.nn.Dropout(0.5)

    def forward(self, inp):
        if self.use_dropout:
            x = self.dropout(inp)
        else:
            x = inp

        shortcut = self.conv1d_bn(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(conv3x3)
        conv7x7 = self.conv7x7(conv5x5)

        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = self.bn_1(out)

        out = torch.add(shortcut, out)
        out = self.relu(out)
        out = self.bn_2(out)
        return out


class ResPathBlock(torch.nn.Module):
    def __init__(self, in_channels, filters):
        super(ResPathBlock, self).__init__()
        self.conv1d_bn1 = conv1d_bn(in_channels, filters, 1, activation=None,padding=0)
        self.conv1d_bn2 = conv1d_bn(in_channels, filters, 9, activation='relu')
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(filters)

    def forward(self, inp):
        shortcut = self.conv1d_bn1(inp)
        out = self.conv1d_bn2(inp)
        out = torch.add(shortcut, out)

        out = self.relu(out)
        out = self.bn(out)
        return out


class ResPath(torch.nn.Module):
    def __init__(self, in_channels, filters, length):
        super(ResPath, self).__init__()
        self.first_block = ResPathBlock(in_channels, filters)
        self.blocks = torch.nn.Sequential(*[ResPathBlock(filters, filters) for i in range(length - 1)])

    def forward(self, inp):
        out = self.first_block(inp)
        out = self.blocks(out)
        return out


class MultiResUnet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, nf=16, use_dropout=False):
        super(MultiResUnet, self).__init__()
        self.mres_block1 = MultiResBlock(in_channels, u=nf)
        self.pool = torch.nn.MaxPool1d(kernel_size=2)
        self.res_path1 = ResPath(self.mres_block1.out_channel, nf, 4)

        self.mres_block2 = MultiResBlock(self.mres_block1.out_channel, u=nf * 2)
        # self.pool2 = torch.nn.MaxPool1d(kernel_size=2)
        self.res_path2 = ResPath(self.mres_block2.out_channel, nf * 2, 3)

        self.mres_block3 = MultiResBlock(self.mres_block2.out_channel, u=nf * 4)
        # self.pool3 = torch.nn.MaxPool1d(kernel_size=2)
        self.res_path3 = ResPath(self.mres_block3.out_channel, nf * 4, 2)

        self.mres_block4 = MultiResBlock(self.mres_block3.out_channel, u=nf * 8)
        # self.pool4 = torch.nn.MaxPool1d(kernel_size=2)
        self.res_path4 = ResPath(self.mres_block4.out_channel, nf * 8, 1)

        self.mres_block5 = MultiResBlock(self.mres_block4.out_channel, u=nf * 16)

        self.deconv1 = torch.nn.ConvTranspose1d(self.mres_block5.out_channel, nf * 8,kernel_size=8,stride=2,padding=3)
        self.mres_block6 = MultiResBlock(nf * 8 + nf * 8, u=nf * 8, use_dropout=use_dropout)
        # MultiResBlock(nf * 8 + self.mres_block4.out_channel, u=nf * 8)

        self.deconv2 = torch.nn.ConvTranspose1d(self.mres_block6.out_channel, nf * 4, kernel_size=8,stride=2,padding=3)
        self.mres_block7 = MultiResBlock(nf * 4 + nf * 4, u=nf * 4, use_dropout=use_dropout)
        # MultiResBlock(nf * 4 + self.mres_block3.out_channel, u=nf * 4)

        self.deconv3 = torch.nn.ConvTranspose1d(self.mres_block7.out_channel, nf * 2, kernel_size=8,stride=2,padding=3)
        self.mres_block8 = MultiResBlock(nf * 2 + nf * 2, u=nf * 2, use_dropout=use_dropout)
        # MultiResBlock(nf * 2 + self.mres_block2.out_channel, u=nf * 2)

        self.deconv4 = torch.nn.ConvTranspose1d(self.mres_block8.out_channel, nf, kernel_size=8,stride=2,padding=3)
        self.mres_block9 = MultiResBlock(nf + nf, u=nf)
        # MultiResBlock(nf + self.mres_block1.out_channel, u=nf)

        self.conv10 = conv1d_bn(self.mres_block9.out_channel, out_channels, 9, activation='tanh')

    def forward(self, inp):
        mresblock1 = self.mres_block1(inp)
        pool = self.pool(mresblock1)
        mresblock1 = self.res_path1(mresblock1)

        mresblock2 = self.mres_block2(pool)
        pool = self.pool(mresblock2)
        mresblock2 = self.res_path2(mresblock2)

        mresblock3 = self.mres_block3(pool)
        pool = self.pool(mresblock3)
        mresblock3 = self.res_path3(mresblock3)

        mresblock4 = self.mres_block4(pool)
        pool = self.pool(mresblock4)
        mresblock4 = self.res_path4(mresblock4)

        mresblock = self.mres_block5(pool)
        # print("mresblock:{}".format(mresblock.size()))
        # print("mresblock4:{}".format(mresblock4.size()))
        # print("deconv1:{}".format(self.deconv1(mresblock).size()))
        up = torch.cat([self.deconv1(mresblock), mresblock4], dim=1)
        mresblock = self.mres_block6(up)

        up = torch.cat([self.deconv2(mresblock), mresblock3], dim=1)
        mresblock = self.mres_block7(up)

        up = torch.cat([self.deconv3(mresblock), mresblock2], dim=1)
        mresblock = self.mres_block8(up)

        up = torch.cat([self.deconv4(mresblock), mresblock1], dim=1)
        mresblock = self.mres_block9(up)

        conv10 = self.conv10(mresblock)
        # print("outsize:{}".format(conv10.size()))
        return conv10


class CBR_1D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel=9,stride=1,padding=4):
        super().__init__()
        self.seq_list = [
        nn.Conv1d(in_channels,out_channels,kernel,stride,padding,bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()]
        
        self.seq = nn.Sequential(*self.seq_list)
        
    def forward(self,x):
        return self.seq(x)


# In[16]:


class Unet_1D(nn.Module):
    def __init__(self,class_n,layer_n):
        super().__init__()
        
        ### ------- encoder -----------
        self.enc1_1 = CBR_1D(1,layer_n)
        self.enc1_2 = CBR_1D(layer_n,layer_n)
        self.enc1_3 = CBR_1D(layer_n,layer_n)
        
        self.enc2_1 = CBR_1D(layer_n,layer_n*2)
        self.enc2_2 = CBR_1D(layer_n*2,layer_n*2)
        
        self.enc3_1 = CBR_1D(layer_n*2,layer_n*4)
        self.enc3_2 = CBR_1D(layer_n*4,layer_n*4)
        
        self.enc4_1 = CBR_1D(layer_n*4,layer_n*8)
        self.enc4_2 = CBR_1D(layer_n*8,layer_n*8)
        
#         self.enc5_1 = CBR_1D(32,64)
#         self.enc5_2 = CBR_1D(64,64)
        
#         self.upsample_1 = nn.ConvTranspose1d(kernel_size=8,stride=2,padding=3)
#         self.dec4_1 = CBR_1D(32+64,32)
#         self.dec4_2 = CBR_1D(32,32)
        
        ### ------- decoder -----------
        self.upsample_3 = nn.ConvTranspose1d(layer_n*8,layer_n*8,kernel_size=8,stride=2,padding=3)
        self.dec3_1 = CBR_1D(layer_n*4+layer_n*8,layer_n*4)
        self.dec3_2 = CBR_1D(layer_n*4,layer_n*4)
        
        self.upsample_2 = nn.ConvTranspose1d(layer_n*4,layer_n*4,kernel_size=8,stride=2,padding=3)
        self.dec2_1 = CBR_1D(layer_n*2+layer_n*4,layer_n*2)
        self.dec2_2 = CBR_1D(layer_n*2,layer_n*2)
        
        self.upsample_1 = nn.ConvTranspose1d(layer_n*2,layer_n*2,kernel_size=8,stride=2,padding=3)
        self.dec1_1 = CBR_1D(layer_n*1+layer_n*2,layer_n*1)
        self.dec1_2 = CBR_1D(layer_n*1,layer_n*1)
        self.dec1_3 = CBR_1D(layer_n*1,class_n)
        self.dec1_4 = CBR_1D(class_n,class_n)
        
    def forward(self,x):
        
        enc1 = self.enc1_1(x)
        enc1 = self.enc1_2(enc1)
        enc1 = self.enc1_3(enc1)
        
        enc2 = nn.functional.max_pool1d(enc1,2)
        enc2 = self.enc2_1(enc2)
        enc2 = self.enc2_2(enc2)
        
        enc3 = nn.functional.max_pool1d(enc2,2)
        enc3 = self.enc3_1(enc3)
        enc3 = self.enc3_2(enc3)
        
        enc4 = nn.functional.max_pool1d(enc3,2)        
        enc4 = self.enc4_1(enc4)
        enc4 = self.enc4_2(enc4)
        
        dec3 = self.upsample_3(enc4)
        dec3 = self.dec3_1(torch.cat([enc3,dec3],dim=1)) ##concat
        dec3 = self.dec3_2(dec3)
        
        dec2 = self.upsample_2(dec3)
        dec2 = self.dec2_1(torch.cat([enc2,dec2],dim=1)) ##concat
        dec2 = self.dec2_2(dec2)
        
        dec1 = self.upsample_1(dec2)
        dec1 = self.dec1_1(torch.cat([enc1,dec1],dim=1)) ##concat
        dec1 = self.dec1_2(dec1)
        dec1 = self.dec1_3(dec1)
        out = self.dec1_4(dec1)
        
        return out