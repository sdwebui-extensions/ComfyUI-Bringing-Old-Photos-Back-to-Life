import os
import gc
import glob

import torch
import torchvision
from PIL import Image

from .Global import detection as ScratchDetector
from .Global import test as Restorer
from .Global.options.test_options import TestOptions

try: # TODO: remove after debugging
    import comfy.model_management
    import folder_paths
except:
    pass

class LoadScratchMaskModel:
    RETURN_TYPES = ("SCRATCH_MODEL",)
    RETURN_NAMES = ("scratch_model",)
    FUNCTION = "load_model"
    OUTPUT_NODE = True

    SCRATCH_MODELS_PATH = "." + os.sep + "models" + os.sep + "scratch_models" + os.sep

    def __init__(self):
        pass

    @classmethod
    def get_scratch_model_list(self):
        models = []
        if os.path.isdir(self.SCRATCH_MODELS_PATH):
            models = glob.glob(self.SCRATCH_MODELS_PATH + os.sep + "**" + os.sep + "*.pt", recursive=True)
            models = [path[len(self.SCRATCH_MODELS_PATH):] for path in models]
        return models

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "scratch_model": (self.get_scratch_model_list(),),
            },
        }

    def load_model(self, scratch_model: str):
        model_path = self.SCRATCH_MODELS_PATH + scratch_model
        model = ScratchDetector.load_model(
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
            masks.append(ScratchDetector.detect_scratches(
                image=torchvision.transforms.ToPILImage()(image[i]), 
                model=scratch_model,
                device_ids=comfy.model_management.get_torch_device(), 
                input_size=input_size, 
                resize_method=self.UPSCALE_METHODS[resize_method], 
            ))
        masks = torch.stack(masks)
        masks = masks.permute(1, 0, 2, 3)[0]

        return (masks,)

class LoadRestoreOldPhotosModel:
    RETURN_TYPES = ("BOPBTL_MODELS",)
    RETURN_NAMES = ("bopbtl_models",)
    FUNCTION = "load_models"
    OUTPUT_NODE = True

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "device_ids": ("STRING", {"default": "0"}), # TODO: can this be automated away?
                "use_scratch_detection": ( # TODO: is this needed this early?
                    ["True", "False"], {
                    "default": "False",
                }),
                "mapping_patch_attention": (
                    ["True", "False"], {
                    "default": "False",
                }),
                "mapping_net": (folder_paths.get_filename_list("checkpoints"),),
                "vae_b": (folder_paths.get_filename_list("vae"),),
                "vae_a": (folder_paths.get_filename_list("vae"),),
            },
        }

    def load_models(
        self, 
        device_ids: str, 
        use_scratch_detection: str, 
        mapping_patch_attention: str, 
        mapping_net: str, 
        vae_b: str, 
        vae_a: str, 
        test_mode: str = "Full", 
    ):
        opt = TestOptions()
        opt.initialize()
        opt = opt.parser.parse_args("")
        opt.isTrain = False
        opt.test_mode = test_mode

        opt.Quality_restore = not use_scratch_detection
        opt.Scratch_and_Quality_restore = use_scratch_detection

        opt.test_vae_a = folder_paths.get_full_path("vae", vae_a)
        opt.test_vae_b = folder_paths.get_full_path("vae", vae_b)
        opt.test_mapping_net = folder_paths.get_full_path("checkpoints", mapping_net)
        opt.HR = True if mapping_patch_attention == "True" else False
        opt.gpu_ids = [int(n) for n in device_ids.split(",")]

        #opt.test_vae_a = "./checkpoints/restoration/VAE_A_quality/latest_net_G.pth"
        #if opt.Quality_restore:
        #    opt.test_vae_b = "./checkpoints/restoration/VAE_B_quality/latest_net_G.pth"
        #    opt.test_mapping_net = "./checkpoints/restoration/mapping_quality/latest_net_mapping_net.pth"
        #if opt.Scratch_and_Quality_restore:
        #    opt.test_vae_b = "./checkpoints/restoration/VAE_B_scratch/latest_net_G.pth"
        #    opt.test_mapping_net = "./checkpoints/restoration/mapping_scratch/latest_net_mapping_net.pth"
        #if opt.HR:
        #    opt.test_mapping_net = "./checkpoints/restoration/mapping_Patch_Attention/latest_net_mapping_net.pth"

        Restorer.parameter_set(opt)
        model = Restorer.load_model(opt)
        image_transform, mask_transform = Restorer.get_transforms()
        return((opt, model, image_transform, mask_transform),)

class RestoreOldPhotos:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore_old_photos"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "bopbtl_models": ("BOPBTL_MODELS",),
                "image": ("IMAGE",),
            },
            "optional": {
                "scratch_mask": ("MASK",),
            },
        }

    def restore_old_photos(
        self, 
        image: torch.Tensor, 
        bopbtl_models, 
        scratch_mask: torch.Tensor = None, 
    ):
        (opt, model, image_transform, mask_transform) = bopbtl_models

        print(image.size())
        image = image.permute(0, 3, 1, 2)
        restored_images = []
        for i in range(image.size()[0]):
            pil_image = torchvision.transforms.ToPILImage()(image[i]).convert("RGB")
            if scratch_mask is None:
                transformed_image, transformed_mask, _ = Restorer.transform_image(
                    pil_image, 
                    image_transform, 
                    opt.test_mode, 
                )
            else:
                pil_mask = torchvision.transforms.ToPILImage()(torch.stack([scratch_mask[i]])).convert("RGB")
                transformed_image, transformed_mask, _ = Restorer.transform_image_and_mask(
                    pil_image, 
                    image_transform, 
                    pil_mask, 
                    mask_transform, 
                    opt.mask_dilation, 
                )
            with torch.no_grad():
                restored_image = model.inference(transformed_image, transformed_mask)[0]
                restored_image = (restored_image + 1.0) / 2.0
                restored_images.append(restored_image)
        restored_images = torch.stack(restored_images)
        restored_images = restored_images.permute(0, 2, 3, 1)
        return (restored_images,)

NODE_CLASS_MAPPINGS = {
    "BOPBTL_ScratchMask": ScratchMask,
    "BOPBTL_LoadScratchMaskModel": LoadScratchMaskModel,
    "BOPBTL_LoadRestoreOldPhotosModel": LoadRestoreOldPhotosModel,
    "BOPBTL_RestoreOldPhotos": RestoreOldPhotos,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BOPBTL_ScratchMask": "Scratch Mask",
    "BOPBTL_LoadScratchMaskModel": "Load Scratch Mask Model",
    "BOPBTL_LoadRestoreOldPhotosModel": "Load Restore Old Photos Model",
    "BOPBTL_RestoreOldPhotos": "Restore Old Photos",
}
