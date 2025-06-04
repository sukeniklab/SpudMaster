from cellpose import models, core
from typing import Dict, List, Callable, Any
import numpy as np

class SegmentationClass:
    """Class for handling segmentation methods."""
    _segmentation_methods: Dict[str, Callable[..., np.ndarray]] = {}

    def __init__(self, algorithm: str = None, **algorithm_kwargs):
        """Initialize segmentation with a chosen algorithm (optional)."""
        self.segmentation_function: Callable[..., np.ndarray] = None
        self.algorithm_kwargs: Dict[str, Any] = {}

        if algorithm:
            self.set_algorithm(algorithm, **algorithm_kwargs)

    @classmethod
    def register_algorithm(cls, name: str):
        """Decorator to register a segmentation algorithm."""
        def decorator(func: Callable[..., np.ndarray]):
            cls._segmentation_methods[name] = func
            print(f"Registered segmentation method: {name}")  # Debugging print
            return func  # Ensure the function remains callable
        return decorator

    def set_algorithm(self, method: str, **algorithm_kwargs):
        """Set the segmentation method dynamically."""
        if method not in self._segmentation_methods:
            raise ValueError(f"Segmentation method '{method}' not registered. Available: {list(self._segmentation_methods.keys())}")

        self.segmentation_function = self._segmentation_methods[method]
        self.algorithm_kwargs = algorithm_kwargs

    def generate_mask(self, image: Any, **kwargs) -> np.ndarray:
        """Generate a segmentation mask with the selected algorithm."""
        if self.segmentation_function is None:
            raise ValueError("No segmentation algorithm set. Call set_algorithm() first.")

        # Merge stored algorithm parameters with new ones
        all_kwargs = {**self.algorithm_kwargs, **kwargs}
        return self.segmentation_function(image, **all_kwargs)

    @classmethod
    def list_registered_algorithms(cls):
        """Return a list of available segmentation algorithms."""
        return list(cls._segmentation_methods.keys())


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