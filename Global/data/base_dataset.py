# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_crop_pos(image_size, resize_or_crop: str, load_size: int, fine_size: int):
    w, h = image_size
    new_h = h
    new_w = w
    if resize_or_crop == 'resize_and_crop':
        new_h = new_w = load_size
    elif resize_or_crop == 'scale_width_and_crop': # we scale the shorter side into 256
        if w<h:
            new_w = load_size
            new_h = load_size * h // w
        else:
            new_h = load_size
            new_w = load_size * w // h
    elif resize_or_crop == 'crop_only':
        pass
    else:
        raise NotImplementedError("Unknown type of 'resize_or_crop' value!")

    x = random.randint(0, np.maximum(0, new_w - fine_size))
    y = random.randint(0, np.maximum(0, new_h - fine_size))
    return (x, y)

def get_random_flip() -> bool:
    return random.random() > 0.5

def get_transform(
    resize_or_crop: str, 

    image_size, 
    load_size: int, 
    fine_size: int, 
    test_random_crop: bool, 

    is_train: bool, 
    no_flip: bool, 
    flip: bool, 

    n_downsample_global: int, 
    netG: str, 
    n_local_enhancers: int, 

    method=Image.BICUBIC, 
    normalize=True
):
    transform_list = []
    if 'resize' in resize_or_crop:
        osize = [load_size, load_size]
        transform_list.append(transforms.Scale(osize, method))
    elif 'scale_width' in resize_or_crop:
        # transform_list.append(transforms.Lambda(lambda img: __scale_width(img, loadSize, method)))  ## Here , We want the shorter side to match 256, and Scale will finish it.
        transform_list.append(transforms.Scale(256,method))

    if 'crop' in resize_or_crop:
        if is_train:
            crop_pos = get_crop_pos(image_size, resize_or_crop, load_size, fine_size)
            transform_list.append(transforms.Lambda(lambda img: __crop(img, crop_pos, fine_size)))
        else:
            if test_random_crop:
                transform_list.append(transforms.RandomCrop(fine_size))
            else:
                transform_list.append(transforms.CenterCrop(fine_size))

    ## when testing, for ablation study, choose center_crop directly.

    if resize_or_crop == 'none':
        base = float(2 ** n_downsample_global)
        if netG == 'local':
            base *= (2 ** n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if is_train and not no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, flip)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),  (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
