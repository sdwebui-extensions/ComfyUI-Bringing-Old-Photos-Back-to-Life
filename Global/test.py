# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import gc
from collections import OrderedDict
from torch.autograd import Variable

try:
    from .options.test_options import TestOptions
    from .models.models import create_model
    from .models.mapping_model import Pix2PixHDModel_Mapping
    from . import util
except ImportError:
    from options.test_options import TestOptions
    from models.models import create_model
    from models.mapping_model import Pix2PixHDModel_Mapping
    from util import util

from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import cv2

def data_transforms(img, method=Image.BILINEAR, scale=False):

    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img

    return img.resize((w, h), method)


def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Scale(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)


def irregular_hole_synthesize(img, mask):

    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

    return hole_img


def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "./checkpoints/restoration"
    ##

    if opt.Quality_restore:
        opt.name = "mapping_quality"
        if opt.test_vae_a == "":
            opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        if opt.test_vae_b == "":
            opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        if opt.test_vae_a == "":
            opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        if opt.test_vae_b == "":
            opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
        if opt.HR:
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = "mapping_Patch_Attention"


def load_model(opt):
    model = Pix2PixHDModel_Mapping()
    model.initialize(opt)
    model.eval()
    return model


def get_transforms():
    img_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.ToTensor()
    return (img_transform, mask_transform)


def transform_image(input, img_transform, test_mode="Full"):
    if test_mode == "Scale":
        input = data_transforms(input, scale=True)
    elif test_mode == "Full":
        input = data_transforms(input, scale=False)
    elif test_mode == "Crop":
        input = data_transforms_rgb_old(input)
    origin = input
    input = img_transform(input)
    input = input.unsqueeze(0)
    mask = torch.zeros_like(input)
    return (input, mask, origin)

def transform_image_and_mask(input, img_transform, mask, mask_transform, mask_dilation=0):
    if mask_dilation != 0:
        kernel = np.ones((3,3),np.uint8)
        mask = np.array(mask)
        mask = cv2.dilate(mask,kernel,iterations = mask_dilation)
        mask = Image.fromarray(mask.astype('uint8'))
    origin = input
    input = irregular_hole_synthesize(input, mask)
    mask = mask_transform(mask)
    mask = mask[:1, :, :]  ## Convert to single channel
    mask = mask.unsqueeze(0)
    input = img_transform(input)
    input = input.unsqueeze(0)
    return (input, mask, origin)


def main(opt):
    parameter_set(opt)

    # output directories
    output_input_dir = opt.outputs_dir + os.path.sep + "input_image"
    output_restored_dir = opt.outputs_dir + os.path.sep + "restored_image"
    output_origin_dir = opt.outputs_dir + os.path.sep + "origin"

    if not os.path.exists(output_input_dir):
        os.makedirs(output_input_dir)
    if not os.path.exists(output_restored_dir):
        os.makedirs(output_restored_dir)
    if not os.path.exists(output_origin_dir):
        os.makedirs(output_origin_dir)

    # images
    dataset_size = 0

    input_loader = os.listdir(opt.test_input)
    dataset_size = len(input_loader)
    input_loader.sort()

    if opt.test_mask != "":
        mask_loader = os.listdir(opt.test_mask)
        dataset_size = len(os.listdir(opt.test_mask))
        mask_loader.sort()

    # load model
    model = load_model(opt)

    # image transforms
    img_transform, mask_transform = get_transforms()

    for i in range(dataset_size):
        # load image
        input_name = input_loader[i]
        input_file = os.path.join(opt.test_input, input_name)
        if not os.path.isfile(input_file):
            print("Skipping non-file %s" % input_name)
            continue
        input = Image.open(input_file).convert("RGB")

        # load mask
        mask = None
        if opt.NL_use_mask:
            mask_name = mask_loader[i]
            mask = Image.open(os.path.join(opt.test_mask, mask_name)).convert("RGB")

        print("Now you are processing %s" % (input_name))

        # transform image
        if mask is None:
            input, mask, origin = transform_image(input, img_transform, opt.test_mode)
        else:
            input, mask, origin = transform_image_and_mask(input, img_transform, mask, mask_transform, opt.mask_dilation)

        # restore image
        with torch.no_grad():
            generated = model.inference(input, mask)

        # save image
        if input_name.endswith(".jpg"):
            input_name = input_name[:-4] + ".png"

        image_grid = vutils.save_image(
            (input + 1.0) / 2.0,
            output_input_dir + os.path.sep + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )
        image_grid = vutils.save_image(
            (generated.data.cpu() + 1.0) / 2.0,
            output_restored_dir + os.path.sep + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )
        origin.save(output_origin_dir + os.path.sep + input_name)

        # clean up
        gc.collect()
        torch.cuda.empty_cache()



if __name__ == "__main__":
    opt = TestOptions().parse(save=False)
    main(opt)