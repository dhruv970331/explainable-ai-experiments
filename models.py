# models.py
#
# This file handles loading the pretrained models for the experiments.
# The main function `get_model` is what the notebook will call.
# It returns the model and the *target layer* for CAM.
#

import torch
import torch.nn as nn
from torchvision import models

# --- Model Loading ---

def get_model(model_name: str, use_pretrained=True):
    """
    Loads a specified pretrained model from torchvision and returns the model
    along with the name of the target layer for CAM-based methods.
    
    This is tricky because each architecture has a different name for its
    last convolutional block.
    
    Args:
        model_name (str): The name of the model (e.g., 'resnet50', 'densenet121')
        use_pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        (nn.Module, str): A tuple of (model, target_layer_name)
        Returns (None, None) if the model name is not recognized.
    """
    
    if model_name == 'resnet50':
        print("Loading pretrained ResNet-50...")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
        # The target layer for resnet is the final bottleneck block in 'layer4'
        # We target the whole 'layer4' block.
        target_layer = model.layer4 
        return model, target_layer

    elif model_name == 'densenet121':
        print("Loading pretrained DenseNet-121...")
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else None)
        # For densenet, the last conv block is part of the 'features'
        # We can target the final 'norm5' or the whole denseblock.
        # Let's target the final normalization layer which comes after the last convs
        target_layer = model.features.norm5 
        # alt: target_layer = model.features.denseblock4
        return model, target_layer

    elif model_name == 'vgg16':
        print("Loading pretrained VGG-16...")
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if use_pretrained else None)
        # VGG is simple, its all in 'features'. The last conv is index 28.
        # The last maxpool is 30. Lets use 28.
        # Actually, let's just target the last layer in the features module.
        target_layer = model.features[-1] # This is a MaxPool2d, but CAM can handle it.
        # alt: target_layer = model.features[28] # Conv3d
        return model, target_layer

    elif model_name == 'resnet18':
        # Added this for a quick test, might be useful for faster runs
        print("Loading pretrained ResNet-18...")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
        target_layer = model.layer4
        return model, target_layer

    # --- Commented out: Custom Model Loading ---
    # elif model_name == 'my_custom_model':
    #     print("Loading custom model from checkpoint...")
    #     model = MyCustomNet() # <-- define this somewhere else
    #     try:
    #         checkpoint_path = './checkpoints/best_model.pth'
    #         model.load_state_dict(torch.load(checkpoint_path))
    #         print(f"Loaded checkpoint from {checkpoint_path}")
    #     except FileNotFoundError:
    #         print(f"WARN: Could not find checkpoint at {checkpoint_path}. Using random weights.")
    #     
    #     # This would need to be known for a custom model
    #     target_layer = model.my_final_conv_block 
    #     return model, target_layer
    # ---

    else:
        print(f"Error: Model '{model_name}' not recognized in get_model function.")
        return None, None


# --- Model Helpers (extra stuff) ---

def set_model_to_eval_mode(model: nn.Module):
    """
    Sets a model to evaluation mode (turns off dropout, batchnorm updates).
    This is crusial for consistent explanations.
    """
    model.eval()
    return model


def get_model_with_random_weights(model_name: str):
    """
    Loads a model architecture but with *random* weights.
    This is for the sanity check experiment.
    """
    print(f"Loading {model_name} with RANDOM weights...")
    model, target_layer = get_model(model_name, use_pretrained=False)
    
    # We should re-init the weights just to be sure
    # (even though use_pretrained=False should do it)
    if model is not None:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # just a simple init
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        print("...Model weights have been re-initialized.")
        return model, target_layer
    else:
        return None, None

def test_model_forward_pass(model: nn.Module, device='cpu'):
    """
    Runs a single dummy tensor through the model to make sure it works.
    Helpful for debugging.
    """
    print("Testing model with a dummy input...")
    try:
        # A random dummy tensor, shape (B, C, H, W)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        model = model.to(device)
        
        output = model(dummy_input)
        
        print(f"...Success. Output shape: {output.shape}")
        return True, output.shape
    except Exception as e:
        print(f"!!! Model forward pass FAILED: {e}")
        return False, None


def find_target_layer_automatically(model: nn.Module):
    """
    Tries to find the last nn.Conv2d layer in a model.
    This is kinda experimental and might not always work.
    
    (Probably better to specify manually like we do in get_model)
    """
    last_conv_layer = None
    last_conv_name = ""

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module
            last_conv_name = name
            
    if last_conv_layer is not None:
        print(f"Found last Conv2d layer: '{last_conv_name}'")
        return last_conv_layer
    else:
        print("Could not automatically find a Conv2d layer.")
        return None
