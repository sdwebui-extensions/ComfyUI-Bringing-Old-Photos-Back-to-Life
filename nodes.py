import os
import glob

import torch
import torchvision
from PIL import Image

from .Global import detection

import comfy.model_management

scratch_models_paths = "." + os.sep + "models" + os.sep + "scratch_models" + os.sep

def get_scratch_model_list():
    if os.path.isdir(scratch_models_paths):
        models = glob.glob(scratch_models_paths + "*.pt")
        models = [path[len(scratch_models_paths):] for path in models]
    return models

class LoadScratchMaskModel:
    RETURN_TYPES = ("SCRATCH_MODEL",)
    RETURN_NAMES = ("scratch_model",)
    FUNCTION = "load_model"
    OUTPUT_NODE = True

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "scratch_model": (get_scratch_model_list(),),
            },
        }

    def load_model(self, scratch_model: str):
        model_path = scratch_models_paths + scratch_model
        model = detection.load_model(
            device_ids=comfy.model_management.get_torch_device(), 
            checkpoint_path=model_path, 
        )
        return (model,)

class ScratchMask:
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "detect_scratches"
    OUTPUT_NODE = True
    CATEGORY = "image"

    INPUT_SIZE_METHODS = ["full_size", "resize_256", "scale_256"]
    UPSCALE_METHODS = {
        "nearest-exact": Image.Resampling.NEAREST, 
        "bilinear": Image.Resampling.BILINEAR, 
        "area" : Image.Resampling.BOX, 
        "bicubic" : Image.Resampling.BICUBIC, 
        "lanczos" : Image.Resampling.LANCZOS, 
    }

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "scratch_model": ("SCRATCH_MODEL",),
                "image": ("IMAGE",),
                "input_size": (
                    self.INPUT_SIZE_METHODS, {
                    "default": self.INPUT_SIZE_METHODS[0],
                }),
                "resize_method": (
                    list(self.UPSCALE_METHODS.keys()), {
                    "default": "bilinear",
                }),
            },
        }

    def detect_scratches(
        self, 
        scratch_model,
        image: torch.Tensor, 
        input_size: str, 
        resize_method: str
    ):
        image = image.permute(0, 3, 1, 2)
        masks = []
        for i in range(image.size()[0]):
            masks.append(detection.detect_scratches(
                image=torchvision.transforms.ToPILImage()(image[i]), 
                model=scratch_model,
                device_ids=comfy.model_management.get_torch_device(), 
                input_size=input_size, 
                resize_method=self.UPSCALE_METHODS[resize_method], 
            ))
        masks = torch.stack(masks)
        masks = masks.permute(1, 0, 2, 3)[0]
        
        return (masks,)

NODE_CLASS_MAPPINGS = {
    "BOPBTL_ScratchMask": ScratchMask,
    "BOPBTL_LoadScratchMaskModel": LoadScratchMaskModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BOPBTL_ScratchMask": "Scratch Mask",
    "BOPBTL_LoadScratchMaskModel": "Load Scratch Mask Model",
}
