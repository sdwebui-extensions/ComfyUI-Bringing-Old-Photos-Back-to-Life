# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

try:
    from .options.test_options import TestOptions
    from .models.pix2pix_model import Pix2PixModel
    from .data.face_dataset import FaceTestDataset
except:
    from options.test_options import TestOptions
    from models.pix2pix_model import Pix2PixModel
    from data.face_dataset import FaceTestDataset

import torchvision.utils as vutils
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from torch.utils.data import DataLoader


def create_directory_dataloader(opt):
    instance = FaceTestDataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" % (type(instance).__name__, len(instance)))
    dataloader = DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain,
    )
    return dataloader


def load_model(opt):
    model = Pix2PixModel(opt)
    model.eval()
    return model


def main(model: Pix2PixModel, dataloader: DataLoader, output_dir: str, batch_size: int, max_process_count: int):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, batch in enumerate(dataloader):
        if i * batch_size >= max_process_count:
            break

        # enhance faces
        enhanced_faces = model(batch, mode="inference")
        for i in range(len(enhanced_faces)):
            enhanced_faces[i] = (enhanced_faces[i] + 1) / 2

        # save image
        for path, enhanced_face in zip(batch["path"], enhanced_faces):
            image_name = os.path.split(path)[1]
            image_path = os.path.join(output_dir, image_name)
            vutils.save_image(enhanced_face, image_path)


if __name__ == "__main__":
    opt = TestOptions().parse()
    model = load_model(opt)
    dataloader = create_directory_dataloader(opt)
    main(model, dataloader, opt.results_dir, opt.batchSize, opt.how_many)