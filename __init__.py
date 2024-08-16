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
    "BOPBTL_ScratchMask": "Scratch Mask (Bringing Old Photos Back to Life)",
    "BOPBTL_LoadScratchMaskModel": "Load Scratch Mask Model (Bringing Old Photos Back to Life)",
    "BOPBTL_LoadRestoreOldPhotosModel": "Load Restore Old Photos Model (Bringing Old Photos Back to Life)",
    "BOPBTL_RestoreOldPhotos": "Restore Old Photos (Bringing Old Photos Back to Life)",
    "BOPBTL_LoadFaceDetectorModel": "Load Face Detector Model (dlib) (Bringing Old Photos Back to Life)",
    "BOPBTL_DetectFaces": "Detect Faces (dlib) (Bringing Old Photos Back to Life)",
    "BOPBTL_LoadFaceEnhancerModel": "Load Face Enhancer Model (Bringing Old Photos Back to Life)",
    "BOPBTL_EnhanceFaces": "Enhance Faces (Bringing Old Photos Back to Life)",
    "BOPBTL_EnhanceFacesAdvanced": "Enhance Faces (Advanced) (Bringing Old Photos Back to Life)",
    "BOPBTL_BlendFaces": "Blend Faces (dlib) (Bringing Old Photos Back to Life)",
    "BOPBTL_DetectEnhanceBlendFaces": "Detect-Enhance-Blend Faces (dlib) (Bringing Old Photos Back to Life)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']