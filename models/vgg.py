#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      https://github.com/kazuto1011
# Created:  2017-04-11

import math
from collections import OrderedDict

import torch.nn as nn


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, mean=0, std=1e-2)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        ch = [3, 32, 32, 64, 64, 128, 128]
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        self.features = nn.Sequential(OrderedDict([
            # Layer1
            ('conv1', nn.Conv2d(ch[0], ch[1], **kwargs)),
            ('norm1', nn.BatchNorm2d(ch[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2)),
            # Layer2
            ('conv2', nn.Conv2d(ch[1], ch[2], **kwargs)),
            ('norm2', nn.BatchNorm2d(ch[2])),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2)),
            # Layer3
            ('conv3', nn.Conv2d(ch[2], ch[3], **kwargs)),
            ('norm3', nn.BatchNorm2d(ch[3])),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(kernel_size=2)),
            # Layer4
            ('conv4', nn.Conv2d(ch[3], ch[4], **kwargs)),
            ('norm4', nn.BatchNorm2d(ch[4])),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2)),
            # Layer5
            ('conv5', nn.Conv2d(ch[4], ch[5], **kwargs)),
            ('norm5', nn.BatchNorm2d(ch[5])),
            ('relu5', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d(kernel_size=2)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(ch[5], ch[6], bias=False)),
            ('norm1', nn.BatchNorm1d(ch[6])),
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(ch[6], 10, bias=False)),
        ]))
        _init_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
