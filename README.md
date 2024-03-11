# ComfyUI-Bringing-Old-Photos-Back-to-Life

Enhance images in [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Optional features include automatic scratch removal and face enhancement. Based on [microsoft/Bringing-Old-Photos-Back-to-Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life).

TODO: [Add Teaser]

## 1. Requirements

### Synchronized BatchNorm Models

This repo has to be downloaded and extracted in two separate locations.

TODO: [Can this be simplified somehow? Ideally so it can be added to ComfyUI node manager.]

UNIX Shell:

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

Dlib is only required for Stages 2-4 for face detection, so it is optional.

Make sure to install dlib to the python and virtual environment for ComfyUI.

```python
pip install dlib==19.24.1
```

## 2. Models

### BOPBTL Models (Stage 1)

[Download - BOPBTL Models](https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip)

#### Load Restore Old Photos Model

Set `device_ids` as a comma separated list of device ids (i.e. `0` or `1,2`). Use `-1` for cpu.

##### vae_a

Place in `models/vae/`.

- VAE_A_quality/latest_net_G.pth

##### vae_b

Extract the following models and place them inside `models/vae/`.

- VAE_B_quality/latest_net_G.pth
- VAE_B_scratch/latest_net_G.pth (scratch_detection)

##### mapping_net

Extract the following models and place them inside `models/checkpoints/`.

- mapping_quality/latest_net_mapping_net.pth
- mapping_scratch/latest_net_mapping_net.pth (scratch_detection)
- mapping_Patch_Attention/latest_net_mapping_net.pth (mapping_patch_attention)

#### Load Scratch Mask Model

##### scratch_model

Extract the following models and place them inside `models/checkpoints/`.

- detection/FT_Epoch_latest.pt

### Face Detection Models (Stages 2-4)

#### Load Face Detector Model (Dlib)

[Download - shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Extract the following models and place them inside `models/facedetection/` (custom directory).

##### shape_predictor_68_face_landmarks

- shape_predictor_68_face_landmarks.dat

#### Load Face Enhancer Model

[Download - Face Enhancement Models](https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip)

Extract the following models and place them inside `models/checkpoints/`.

Set `device_ids` as a comma separated list of device ids (i.e. `0` or `1,2`). Use `-1` for cpu.

##### face_enhance_model

- Setting_9_epoch_100/latest_net_G.pth (256x256)
- FaceSR_512/latest_net_G.pth (512x512)

## 3. Workflows

TODO: [Finalize node interfaces.]

### BOPBTL and Face Restoration (Stages 1-4)

TODO: [Add workflow image.]

### BOPBTL + Scratch Detection and Face Restoration (Stages 1-4)

TODO: [Add workflow image.]

### BOPBTL + Scratch Detection (Stage 1)

TODO: [Add workflow image.]

### Face Restoration (Advanced) (Stages 2-4)

TODO: [Add workflow image.]

## License

The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
