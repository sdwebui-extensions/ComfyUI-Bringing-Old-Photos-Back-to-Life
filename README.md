# comfy-bringing-old-photos-back-to-life

Restore images in ComfyUI. Optionally use automatic scratch removal and face enhancement. Built on [microsoft/Bringing-Old-Photos-Back-to-Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life).

TODO: teaser

## 1. Installation Requirements

### Detection Models

Bash:

```bash
cd Face_Enhancement/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch.git
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
```

```bash
cd Global/detection_models/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch.git
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
```

Powershell:

```powershell
cd Face_Enhancement/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch.git
Copy-Item -Path "Synchronized-BatchNorm-PyTorch/sync_batchnorm" -Destination . -Recurse -Force
```

```powershell
cd Global/detection_models/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch.git
Copy-Item -Path "Synchronized-BatchNorm-PyTorch/sync_batchnorm" -Destination . -Recurse -Force
```

### Dlib

Dlib is required for Stages 2-4 for face detection.

Make sure to install dlib to your python.

```python
pip install dlib
```

## 2. Models

Download the models.

### Load Restore Old Photos (Stage 1)

[Download - Global Models](https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip)

#### vae_a models

Place in `models/vae/`.

- VAE_A_quality/latest_net_G.pth

#### vae_b models

Place in `models/vae/`.

- VAE_B_quality/latest_net_G.pth
- VAE_B_scratch/latest_net_G.pth (scratch detection)

#### mapping_net models

Place in `models/checkpoints/`.

- mapping_quality/latest_net_mapping_net.pth
- mapping_scratch/latest_net_mapping_net.pth (scratch detection)
- mapping_Patch_Attention/latest_net_mapping_net.pth (mapping patch attention)

### Load Face Detector (Dlib) (Stage 2)

[Download - shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Place in `models/facedetection/` (custom directory).

- shape_predictor_68_face_landmarks.dat

### Load Face Enhancer (Stage 3)

[Download - Face Enhancement Models](https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip)

Place in `models/checkpoints/`.

- Setting_9_epoch_100/latest_net_G.pth (256)
- FaceSR_512/latest_net_G.pth (512)

## 3. Workflows

### Simple Workflow

TODO:

### Simple Workflow with Scratch Detection

TODO:

### Stage 1 with Scratch Detection

TODO:

### Stage 2-4 Advanced

TODO:

## Citation

If you find this work useful for your research, please consider citing the following papers.

```bibtex
@inproceedings{wan2020bringing,
title={Bringing Old Photos Back to Life},
author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={2747--2757},
year={2020}
}
```

```bibtex
@article{wan2020old,
  title={Old Photo Restoration via Deep Latent Space Translation},
  author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
  journal={arXiv preprint arXiv:2009.07047},
  year={2020}
}
```

If you are also interested in the legacy photo/video colorization, please refer to [this work](https://github.com/zhangmozhe/video-colorization).

## License

The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
