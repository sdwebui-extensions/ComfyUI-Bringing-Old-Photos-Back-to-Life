from .nodes import BOPBTL_ScratchMask, BOPBTL_LoadScratchMaskModel, BOPBTL_LoadRestoreOldPhotosModel, BOPBTL_RestoreOldPhotos, BOPBTL_LoadFaceDetectorModel, BOPBTL_DetectFaces, BOPBTL_LoadFaceEnhancerModel, BOPBTL_EnhanceFaces, BOPBTL_EnhanceFacesAdvanced, BOPBTL_BlendFaces, BOPBTL_DetectEnhanceBlendFaces

NODE_CLASS_MAPPINGS = {
    "BOPBTL_ScratchMask": BOPBTL_ScratchMask,
    "BOPBTL_LoadScratchMaskModel": BOPBTL_LoadScratchMaskModel,
    "BOPBTL_LoadRestoreOldPhotosModel": BOPBTL_LoadRestoreOldPhotosModel,
    "BOPBTL_RestoreOldPhotos": BOPBTL_RestoreOldPhotos,
    "BOPBTL_LoadFaceDetectorModel": BOPBTL_LoadFaceDetectorModel,
    "BOPBTL_DetectFaces": BOPBTL_DetectFaces,
    "BOPBTL_LoadFaceEnhancerModel": BOPBTL_LoadFaceEnhancerModel,
    "BOPBTL_EnhanceFaces": BOPBTL_EnhanceFaces,
    "BOPBTL_EnhanceFacesAdvanced": BOPBTL_EnhanceFacesAdvanced,
    "BOPBTL_BlendFaces": BOPBTL_BlendFaces,
    "BOPBTL_DetectEnhanceBlendFaces": BOPBTL_DetectEnhanceBlendFaces,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BOPBTL_ScratchMask": "Scratch Mask",
    "BOPBTL_LoadScratchMaskModel": "Load Scratch Mask Model",
    "BOPBTL_LoadRestoreOldPhotosModel": "Load Restore Old Photos Model",
    "BOPBTL_RestoreOldPhotos": "Restore Old Photos",
    "BOPBTL_LoadFaceDetectorModel": "Load Face Detector Model (dlib)",
    "BOPBTL_DetectFaces": "Detect Faces (dlib)",
    "BOPBTL_LoadFaceEnhancerModel": "Load Face Enhancer Model",
    "BOPBTL_EnhanceFaces": "Enhance Faces",
    "BOPBTL_EnhanceFacesAdvanced": "Enhance Faces (Advanced)",
    "BOPBTL_BlendFaces": "Blend Faces (dlib)",
    "BOPBTL_DetectEnhanceBlendFaces": "Detect-Enhance-Blend Faces (dlib)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']