# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import gc
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageFile

try:
    from .detection_models import networks
    from .detection_util.util import *
except ImportError:
    from detection_models import networks
    from detection_util.util import *

warnings.filterwarnings("ignore", category=UserWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def data_transforms(
    img: Image.Image, 
    input_size: str, 
    resize_method: Image.Resampling=Image.Resampling.BICUBIC, 
):
    if input_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), resize_method)

    elif input_size == "scale_256":
        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), resize_method)


def scale_tensor(img_tensor, default_scale=256) -> torch.Tensor:
    _, _, w, h = img_tensor.shape
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")


def blend_mask(img, mask):
    np_img = np.array(img).astype("float")
    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")


def load_model(
    device_ids, # str | int | torch.device
    checkpoint_path: str,
):
    device_ids = str(device_ids)
    model = networks.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=len(device_ids) > 0 and "-1" not in device_ids and "cpu" != device_ids and torch.cuda.is_available(),
        antialiasing=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    try:
        device_ids = int(device_ids)
    except:
        pass
    if device_ids == "cpu" or (type(device_ids) is int and device_ids < 0):
        model.cpu()
    else:
        model.to(device_ids)
    model.eval()
    return model


def detect_scratches(
    image: Image.Image, 
    model: networks.UNet, 
    device_ids, # str | int
    input_size: str, 
    resize_method: Image.Resampling=Image.Resampling.BICUBIC, 
) -> torch.Tensor:
    image = data_transforms(image, input_size, resize_method).convert("L")
    image = torchvision.transforms.ToTensor()(image)
    image = torchvision.transforms.Normalize([0.5], [0.5])(image)
    image = torch.unsqueeze(image, 0)
    _, _, ow, oh = image.shape
    scaled_image = scale_tensor(image)
    try:
        device_ids = int(device_ids)
    except:
        pass
    if type(device_ids) is int and device_ids < 0:
        scaled_image = scaled_image.cpu()
    else:
        scaled_image = scaled_image.to(device_ids)

    with torch.no_grad():
        mask = torch.sigmoid(model(scaled_image))
    mask = mask.data.cpu()
    mask = F.interpolate(mask, [ow, oh], mode="nearest")
    mask: torch.Tensor = (mask >= 0.4).float()
    return mask[0]


def main(config):
    if not os.path.isdir(config.test_path):
        raise RuntimeError("Image directory does not exist!")
    if config.test_path == config.output_dir:
        raise RuntimeError("Input and output directories cannot be the same!")

    # load model
    model = load_model(
        device_ids=config.GPU, 
        checkpoint_path=config.checkpoint_name
    )

    for file in os.listdir(config.test_path):
        file_path = os.path.join(config.test_path, file)
        if not os.path.isfile(file_path):
            continue

        # load image
        try:
            image: Image.Image = Image.open(file_path).convert("RGB")
        except:
            continue

        # compute mask
        mask = detect_scratches(
            image=image, 
            model=model, 
            device_ids=config.GPU, 
            input_size=config.input_size
        )

        # save mask
        filename = os.path.split(file_path)[1]
        mask_path = os.path.join(config.output_dir, os.path.splitext(filename)[0] + ".png")
        torchvision.utils.save_image(
            mask,
            mask_path,
            nrow=1,
            padding=0,
            normalize=True,
        )

        # clean up
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_name', type=str, default="./checkpoints/detection/FT_Epoch_latest.pt", help='Checkpoint Path')
    parser.add_argument("--GPU", type=str, default=0, help='Default gpu_id=0, cpu_id=-1, multiple gpus=\"2,3\"')
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--input_size", type=str, default="scale_256", help="resize_256|full_size|scale_256")
    config = parser.parse_args()
    main(config)
