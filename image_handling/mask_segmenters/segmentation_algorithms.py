from cellpose import models, core
from typing import Dict, List, Callable, Any
import numpy as np
from image_handling.mask_segmenters.segmentation_class import SegmentationClass


@SegmentationClass.register_algorithm("cellpose")
def cellpose_segmentation(image: np.ndarray, model: str = 'cyto3', use_gpu: bool = True, channels: List[int] = [0, 0], diams: float = 60.0) -> np.ndarray:
    """Cellpose segmentation method."""
    print("Running Cellpose segmentation...")  # Debugging

    use_gpu = use_gpu and core.use_gpu()
    
    try:
        model_instance = models.Cellpose(gpu=use_gpu, model_type=model)
        masks, flows, styles, diams = model_instance.eval(image, diameter=diams, channels=channels)
        return masks
    except Exception as e:
        print(f"Error in Cellpose segmentation: {e}")
        return np.zeros_like(image, dtype=np.uint8)


@SegmentationClass.register_algorithm("unknown")
def algorithm_segmentation(image: np.ndarray) -> np.ndarray:
    """Placeholder for unknown segmentation algorithm."""
    return np.zeros_like(image, dtype=np.uint8)


@SegmentationClass.register_algorithm("other")
def intensity_cutoff_segmentation(image: np.ndarray) -> np.ndarray:
    """Placeholder for an intensity-based segmentation method."""
    return np.zeros_like(image, dtype=np.uint8)

