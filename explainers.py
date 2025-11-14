# explainers.py
#
# This file is the 'factory' for creating our XAI explainers.
# It imports methods from both the 'pytorch-grad-cam' library (which we've
# added as source) and the 'captum' library, and wraps them in a 
# single, consistent API for the main notebook to use.
#

import torch
import torch.nn as nn
import numpy as np

# --- Library Imports ---

# 1. From 'captum' (for Integrated Gradients)
try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("Captum library not found. Please: pip install captum")

# 2. From 'pytorch-grad-cam' (for all CAM methods)
#    We assume the 'pytorch_grad_cam' folder is in our project root.
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, EigenCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    print("Error: 'pytorch_grad_cam' source folder not found.")
    print("Please make sure the library folder is in the same directory.")


# --- Base Class ---
# We still use a base class for a consistent API.
class BaseExplainer:
    """
    Base class for all explainers.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval() # Always set to eval mode

    def explain(self, input_tensor, target_class_idx, target_layer):
        """
        Main explanation method.
        
        Args:
            input_tensor (torch.Tensor): The preprocessed input image tensor (B, C, H, W).
            target_class_idx (int): The index of the target class.
            target_layer (nn.Module): The model layer to target.

        Returns:
            np.ndarray: A 2D heatmap (H, W) normalized to [0, 1].
        """
        raise NotImplementedError("Subclass must implement explain method")

    def _postprocess_heatmap(self, heatmap_np):
        """
        Common postprocessing: make sure it's 2D and normalized.
        """
        if heatmap_np.ndim > 2:
            # e.g., (1, 1, H, W) -> (H, W)
            heatmap_np = np.squeeze(heatmap_np)
            
        # Check for NaN/Inf
        heatmap_np[np.isnan(heatmap_np)] = 0
        heatmap_np[np.isinf(heatmap_np)] = 0
        
        # Normalize to [0, 1]
        if np.max(heatmap_np) > 0:
            heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np))
            
        return heatmap_np

# --- Wrapper Implementations ---

class CaptumIntegratedGradients(BaseExplainer):
    """
    Wrapper for Captum's Integrated Gradients.
    """
    def __init__(self, model):
        super().__init__(model)
        self.ig = IntegratedGradients(self.model)

    def explain(self, input_tensor, target_class_idx, target_layer=None):
        # IG doesn't need a target_layer
        if target_layer is not None:
            # print("Info: IntegratedGradients does not use a target_layer.")
            pass # supress spammy log
            
        attribution = self.ig.attribute(input_tensor, 
                                        target=target_class_idx, 
                                        n_steps=50, 
                                        internal_batch_size=1)
        
        # Sum over channels to get a (B, H, W) heatmap
        heatmap = attribution.sum(dim=1, keepdim=True)
        heatmap_np = heatmap.squeeze().cpu().detach().numpy()
        
        return self._postprocess_heatmap(heatmap_np)


class PytorchGradCAMWrapper(BaseExplainer):
    """
    A single wrapper for all methods in the 'pytorch-grad-cam' library.
    """
    def __init__(self, model, cam_method_class, use_cuda=False):
        super().__init__(model)
        self.use_cuda = use_cuda
        self.cam_method_class = cam_method_class
        # We will instantiate the 'cam' object in the explain method
        # because it needs the target_layer
        self.cam = None
        self.last_target_layer = None

    def explain(self, input_tensor, target_class_idx, target_layer):
        # Re-create the cam object if the target layer changes
        if self.cam is None or self.last_target_layer != target_layer:
            self.cam = self.cam_method_class(model=self.model, 
                                             target_layers=[target_layer], 
                                            #  use_cuda=self.use_cuda
                                             )
            self.last_target_layer = target_layer

        # 'pytorch-grad-cam' requires a special "target" object
        targets = [ClassifierOutputTarget(target_class_idx)]
        targets = target_class_idx
        # The library's main call
        # It returns a (B, H, W) numpy array
        grayscale_cam = self.cam(input_tensor=input_tensor, 
                                 targets=targets,
                                 eigen_smooth=False, # keep it simple
                                 aug_smooth=False)
        
        # Get the first (and only) heatmap
        heatmap_np = grayscale_cam[0, :]
        
        # We don't need the library's postprocessing, just the raw map
        return self._postprocess_heatmap(heatmap_np)

# --- The Main Factory Function ---

def get_explainer(method_name: str, model: nn.Module):
    """
    This is the main factory function our notebook will call.
    
    It returns an initialized explainer object with a consistent .explain() API.
    """
    print(f"Initializing explainer for: {method_name}")
    
    # Check for CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("...CUDA is available.")
        model = model.to('cuda')
    
    if method_name.lower() == 'gradcam':
        return PytorchGradCAMWrapper(model, GradCAM, use_cuda)
        
    elif method_name.lower() == 'gradcam++':
        return PytorchGradCAMWrapper(model, GradCAMPlusPlus, use_cuda)

    elif method_name.lower() == 'scorecam':
        return PytorchGradCAMWrapper(model, ScoreCAM, use_cuda)
        
    # --- Commented out: AblationCAM is too slow for a quick demo ---
    # elif method_name.lower() == 'ablationcam':
    #     return PytorchGradCAMWrapper(model, AblationCAM, use_cuda)
        
    elif method_name.lower() == 'eigencam':
        return PytorchGradCAMWrapper(model, EigenCAM, use_cuda)
        
    elif method_name.lower() == 'integratedgradients':
        return CaptumIntegratedGradients(model) # Captum handles its own CUDA
        
    else:
        print(f"Warning: Explainer method '{method_name}' not recognized.")
        print("Defaulting to GradCAM.")
        return PytorchGradCAMWrapper(model, GradCAM, use_cuda)
