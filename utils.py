#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-11-28 11:36
# @Author  : YangLs
# @File    : utils.py

"""
    实现不同输入大小的图片的等比缩放，防止变形
"""

from PIL import Image

def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask