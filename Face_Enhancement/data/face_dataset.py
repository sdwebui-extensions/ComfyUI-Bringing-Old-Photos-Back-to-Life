# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

try:
    from .base_dataset import BaseDataset, get_random_flip, get_transform
    from ..util import util
except:
    from data.base_dataset import BaseDataset, get_random_flip, get_transform
    import util

import os
import torch
from PIL import Image


class FaceTestDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument("--no_pairing_check", action="store_true", help="If specified, skip sanity check of correct label-image file pairing")
        # parser.set_defaults(contain_dontcare_label=False)
        # parser.set_defaults(no_instance=True)
        return parser

    @staticmethod
    def get_parts():
        return [
            "skin",
            "hair",
            "l_brow",
            "r_brow",
            "l_eye",
            "r_eye",
            "eye_g",
            "l_ear",
            "r_ear",
            "ear_r",
            "nose",
            "mouth",
            "u_lip",
            "l_lip",
            "neck",
            "neck_l",
            "cloth",
            "hat",
        ]

    def initialize(self, opt):
        self.preprocess_mode = opt.preprocess_mode
        self.load_size = opt.load_size
        self.crop_size = opt.crop_size
        self.aspect_ratio = opt.aspect_ratio
        self.is_train = opt.isTrain
        self.no_flip = opt.no_flip

        self.image_dir = os.path.join(opt.dataroot, opt.old_face_folder)
        self.label_dir = os.path.join(opt.dataroot, opt.old_face_label_folder)

        image_list = os.listdir(self.image_dir)
        image_list = sorted(image_list)
        self.image_list = image_list

        self.parts = FaceTestDataset.get_parts()

    def get_image_transform(self, image_size, flip: bool):
        return get_transform(
            self.preprocess_mode, 
            image_size, 
            self.load_size, 
            self.crop_size, 
            self.aspect_ratio, 
            self.is_train, 
            self.no_flip, 
            flip, 
        )

    def get_label_transform(self, image_size, flip: bool):
        return get_transform(
            self.preprocess_mode, 
            image_size, 
            self.load_size, 
            self.crop_size, 
            self.aspect_ratio, 
            self.is_train, 
            self.no_flip, 
            flip, 
            method=Image.NEAREST, 
            normalize=False, 
        )

    def __getitem__(self, index):
        image_size = (-1, -1)
        flip = get_random_flip()
        transform_image = self.get_image_transform(image_size, flip)
        transform_label = self.get_label_transform(image_size, flip)

        # load image
        rel_image_path = self.image_list[index]
        image_path = os.path.join(self.image_dir, rel_image_path)
        image_name = os.path.splitext(rel_image_path)[0]
        image = Image.open(image_path).convert("RGB")

        # transform image
        image_tensor = transform_image(image)

        # load labels
        labels = []
        for part in self.parts:
            part_name = image_name + "_" + part + ".png"
            part_path = os.path.join(self.label_dir, part_name)

            if os.path.exists(part_path):
                # load label
                label = Image.open(part_path).convert("RGB")

                # transform label
                label_tensor = transform_label(label)[0]  ## 3 channels and pixel [0,1]
            else:
                # create empty label
                label_tensor = torch.zeros((self.load_size, self.load_size))
            labels.append(label_tensor)
        labels_tensor = torch.stack(labels, 0)

        return {
            "label": labels_tensor,
            "image": image_tensor,
            "path": image_path,
        }

    def __len__(self):
        return len(self.image_list)


class FaceTensorDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @staticmethod
    def get_parts():
        return [
            "skin",
            "hair",
            "l_brow",
            "r_brow",
            "l_eye",
            "r_eye",
            "eye_g",
            "l_ear",
            "r_ear",
            "ear_r",
            "nose",
            "mouth",
            "u_lip",
            "l_lip",
            "neck",
            "neck_l",
            "cloth",
            "hat",
        ]

    def initialize(
        self, 
        preprocess_mode: str, 
        load_size: int, 
        crop_size: int, 
        aspect_ratio: float, 
        is_train: bool, 
        no_flip: bool, 
        image_list, 
        parts_list, 
    ):
        self.preprocess_mode = preprocess_mode
        self.load_size = load_size
        self.crop_size = crop_size
        self.aspect_ratio = aspect_ratio
        self.is_train = is_train
        self.no_flip = no_flip

        self.image_list = image_list
        self.parts_list = parts_list

    def get_image_transform(self, image_size, flip: bool):
        return get_transform(
            self.preprocess_mode, 
            image_size, 
            self.load_size, 
            self.crop_size, 
            self.aspect_ratio, 
            self.is_train, 
            self.no_flip, 
            flip, 
        )

    def get_label_transform(self, image_size, flip: bool):
        return get_transform(
            self.preprocess_mode, 
            image_size, 
            self.load_size, 
            self.crop_size, 
            self.aspect_ratio, 
            self.is_train, 
            self.no_flip, 
            flip, 
            method=Image.NEAREST, 
            normalize=False, 
        )

    def __getitem__(self, index):
        image_size = (-1, -1)
        flip = get_random_flip()
        transform_image = self.get_image_transform(image_size, flip)
        transform_label = self.get_label_transform(image_size, flip)

        # load image
        image = self.image_list[index]

        # transform image
        image_tensor = transform_image(image)

        # load labels
        labels = []
        for part in self.parts_list:
            if part is not None:
                label_tensor = transform_label(part)[0]  ## 3 channels and pixel [0,1]
            else:
                label_tensor = torch.zeros((self.load_size, self.load_size))
            labels.append(label_tensor)
        labels_tensor = torch.stack(labels, 0)

        return {
            "label": labels_tensor,
            "image": image_tensor,
        }

    def __len__(self):
        return len(self.image_list)

