#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-11-30 0:10
# @Author  : YangLs
# @File    : test.py

from model import *
import os
import torch
from Unet.utils import keep_image_size_open
from data import *
from torchvision.utils import save_image

net = Unet().cuda()

weights = 'params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('failed')

_input=  input('please input image path:')
# E:\电科\Coding\Unet\test_image\2007_000187.jpg

img = keep_image_size_open(_input)
img_data = transform(img).cuda()
img_data = torch.unsqueeze(img_data, dim=0)
out = net(img_data)
save_image(out, 'test_result/result.jpg')