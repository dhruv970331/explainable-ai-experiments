# utils.py
#
# Helper functions for the XAI seminar project.
# Includes image loading, preprocessing, metrics (IoU, SSIM), and plotting.
# This file contains all the core utilities needed by the main notebooks.
#

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2  # Using opencv for colormaps and overlay
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_score # this is IoU
import os
import glob
import time # for timing experiments?

# These are standard for ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- 1. Image Loading and Preprocessing ---

def load_image_from_url(url: str) -> Image.Image:
    """
    Loads a PIL image from a given URL.
    
    Args:
        url (str): The web URL of the image.

    Returns:
        Image.Image: A PIL image object in RGB mode. None on failure.
    """
    print(f"Attempting to download image from: {url}")
    try:
        response = requests.get(url, timeout=10) # add timeout
        response.raise_for_status() # raise error if bad response
        img_bytes = BytesIO(response.content)
        img_pil = Image.open(img_bytes).convert('RGB')
        print("...Success.")
        return img_pil
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from {url}: {e}")
        return None

def load_image_from_path(path: str) -> Image.Image:
    """
    Loads a PIL image from a local file path.
    (This is the one we uncommented, its useful for local testing)
    
    Args:
        path (str): The local filepath of the image.

    Returns:
        Image.Image: A PIL image object in RGB mode. None on failure.
    """
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None
        
    try:
        img_pil = Image.open(path).convert('RGB')
        return img_pil
    except Exception as e:
        print(f"Error loading local image {path}: {e}")
        return None


def preprocess_image(pil_img: Image.Image, resize_shape=(224, 224)) -> torch.Tensor:
    """
    Converts a PIL image into a normalized torch tensor for model input.
    
    Args:
        pil_img (Image.Image): The input PIL image.
        resize_shape (tuple, optional): The target size (H, W). Defaults to (224, 224).

    Returns:
        torch.Tensor: A 4D tensor (B, C, H, W) ready for the model.
    """
    # Standard ImageNet normalization and resizing
    preprocess_transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    img_tensor = preprocess_transform(pil_img)
    
    # Add batch dimension (B, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

# ---
# class ImagePreprocessor:
#     """
#     A class-based preprocessor. Maybe more 'pro' but function is fine for now.
#     """
#     def __init__(self, resize_shape=(224, 224)):
#         self.resize_shape = resize_shape
#         self.transform = transforms.Compose([
#             transforms.Resize(resize_shape),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#         ])

#     def process(self, pil_img):
#         tensor = self.transform(pil_img)
#         return tensor.unsqueeze(0)
# ---

def de_normalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized image tensor back to a displayable numpy array (H, W, C).
    
    Args:
        tensor (torch.Tensor): A (B, C, H, W) or (C, H, W) normalized tensor.

    Returns:
        np.ndarray: A (H, W, C) numpy array with pixel values [0, 255].
    """
    # remove batch dim if it exists
    if tensor.dim() == 4:
        img = tensor.squeeze(0).cpu().detach().numpy()
    elif tensor.dim() == 3:
        img = tensor.cpu().detach().numpy()
    else:
        raise ValueError(f"Input tensor must have 3 or 4 dimensions, got {tensor.dim()}")
        
    # move channel axis (C, H, W) -> (H, W, C)
    img = np.transpose(img, (1, 2, 0))
    
    # De-normalize: (img * std) + mean
    img = IMAGENET_STD * img + IMAGENET_MEAN
    
    # Clip values to be safe (0, 1)
    img = np.clip(img, 0, 1)
    
    return (img * 255).astype(np.uint8)


def save_image(img_arr: np.ndarray, path: str):
    """
    Saves a numpy image array to a file.
    Assumes img_arr is (H, W, C) and RGB.
    """
    try:
        pil_img = Image.fromarray(img_arr)
        pil_img.save(path)
        print(f"Image saved to {path}")
    except Exception as e:
        print(f"Error saving image: {e}")

# --- 2. Heatmap and Visualization ---

def normalize_to_0_1(arr: np.ndarray) -> np.ndarray:
    """
    Simple min-max normalization for a numpy array.
    """
    if arr.max() == arr.min():
        # Handle constant array (e.g., all zeros) to avoid division by zero
        if arr.max() == 0:
            return arr
        else:
            return arr / arr.max() # just make it all 1s
    return (arr - arr.min()) / (arr.max() - arr.min())


def overlay_heatmap(img_arr: np.ndarray, heatmap: np.ndarray, alpha=0.5, colormap_name=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Applies a colormap to a heatmap and overlays it on the original image.
    
    Args:
        img_arr (np.ndarray): Original image, shape (H, W, C), dtype uint8 [0, 255]
        heatmap (np.ndarray): 2D heatmap, shape (H, W), values [0, 1]
        alpha (float): Opacity of the heatmap overlay
        colormap_name (int): OpenCV colormap (e.g., cv2.COLORMAP_JET)
        
    Returns:
        np.ndarray: The overlaid image.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, but got shape {heatmap.shape}")
        
    # Apply colormap to heatmap
    # Heatmap needs to be [0, 255] uint8 for applyColorMap
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_u8, colormap_name)
    
    # OpenCV uses BGR, convert to RGB for matplotlib/PIL
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image (just in case)
    if heatmap_colored.shape[:2] != img_arr.shape[:2]:
        # Use INTER_LINEAR for smoother resizing
        heatmap_colored = cv2.resize(heatmap_colored, (img_arr.shape[1], img_arr.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Blend image and heatmap
    overlaid_img = cv2.addWeighted(img_arr, 1 - alpha, heatmap_colored, alpha, 0)
    
    # --- Manual overlay method (buggy? keep for reference) ---
    # beta = (1.0 - alpha)
    # overlaid_img_manual = (img_arr * beta + heatmap_colored * alpha)
    # overlaid_img_manual = np.clip(overlaid_img_manual, 0, 255).astype(np.uint8)
    # ---
    
    return overlaid_img


# --- 3. Quantatative Metrics ---

def binarize_map(heatmap: np.ndarray, threshold=0.5) -> np.ndarray:
    """
    Normalizes and binarizes a heatmap.
    
    Args:
        heatmap (np.ndarray): The 2D heatmap.
        threshold (float): Value between 0 and 1 to threshold at.
        
    Returns:
        np.ndarray: A 2D boolean array.
    """
    norm_map = normalize_to_0_1(heatmap)
    return norm_map > threshold

    # --- Alternative thresholding (Otsu) ---
    # if threshold == 'otsu':
    #     norm_map_u8 = (normalize_to_0_1(heatmap) * 255).astype(np.uint8)
    #     _, bin_map = cv2.threshold(norm_map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     return bin_map.astype(bool)
    # else:
    #     return normalize_to_0_1(heatmap) > threshold
    # ---

def compute_ssim(map1: np.ndarray, map2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two heatmaps.
    Assumes maps are 2D and will be normalized.
    
    Args:
        map1 (np.ndarray): First heatmap.
        map2 (np.ndarray): Second heatmap.

    Returns:
        float: SSIM score.
    """
    # Normalize maps to [0, 1] for SSIM
    map1_norm = normalize_to_0_1(map1)
    map2_norm = normalize_to_0_1(map2)
    
    # Ensure maps are same size
    if map1_norm.shape != map2_norm.shape:
        map2_norm = cv2.resize(map2_norm, (map1_norm.shape[1], map1_norm.shape[0]), interpolation=cv2.INTER_LINEAR)

    # data_range is the dynamic range of the data. 
    return ssim(map1_norm, map2_norm, data_range=1.0)


def compute_iou(map1: np.ndarray, map2: np.ndarray, threshold=0.5) -> float:
    """
    Computes Intersection over Union (Jaccard Index) for binarized maps.
    Threshholding is important for good results.
    
    Args:
        map1 (np.ndarray): First heatmap.
        map2 (np.ndarray): Second heatmap.
        threshold (float): Threshold for binarization.

    Returns:
        float: IoU score.
    """
    # Binarize maps using our helper
    bin_map1 = binarize_map(map1, threshold=threshold)
    bin_map2 = binarize_map(map2, threshold=threshold)

    # Ensure maps are same size
    if bin_map1.shape != bin_map2.shape:
        bin_map2_u8 = bin_map2.astype(np.uint8)
        bin_map2_resized = cv2.resize(bin_map2_u8, (bin_map1.shape[1], bin_map1.shape[0]), interpolation=cv2.INTER_NEAREST)
        bin_map2 = bin_map2_resized.astype(bool)

    # Flatten for sklearn
    bin_map1_flat = bin_map1.flatten()
    bin_map2_flat = bin_map2.flatten()
    
    # using zero_division=1.0 means if both are empty, they are 100% overlap
    return jaccard_score(bin_map1_flat, bin_map2_flat, zero_division=1.0)


# --- 4. Plotting ---

def plot_single_image(img: np.ndarray, title="", figsize=(5, 5)):
    """A simple wrapper to plot one image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_comparison_grid(images: list, titles: list, grid_shape: tuple, figsize=(15, 5), filename=None, main_title=None):
    """
    Plots a grid of images with titles.
    
    Args:
        images (list): List of np.ndarray images to plot.
        titles (list): List of string titles for each image.
        grid_shape (tuple): (rows, cols) for the subplot grid.
        filename (str, optional): If provided, saves the figure to this path.
        main_title (str, optional): An overall suptitle for the figure.
    """
    rows, cols = grid_shape
    if len(images) != len(titles) or len(images) > rows * cols:
        print(f"Warning: Mismatch in items. Got {len(images)} images and {len(titles)} titles for a {rows}x{cols} grid. May not plot correctly.")
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Make sure axes is always an array for easy iteration
    if not isinstance(axes, (np.ndarray)):
        axes = np.array([axes]) # make it a 1-element array
        
    # Flatten axes for easy 1D iteration
    axes_flat = axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes_flat[i].imshow(img)
        axes_flat[i].set_title(title, fontsize=10)
        axes_flat[i].axis('off')
        
    # Fill any unused subplots
    for i in range(len(images), len(axes_flat)):
        axes_flat[i].axis('off')
        
    if main_title:
        fig.suptitle(main_title, fontsize=16, y=1.03) # y > 1 to avoid overlap
        
    plt.tight_layout()
    
    if filename:
        # TODO: check if path exists, create dirs if not
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")
    
    plt.show()

# --- 5. Model Output and Class Helpers ---

# Store class names globally so we don't re-download
_IMAGENET_CLASS_NAMES = None

def get_imagenet_class_names():
    """
    Downloads and returns a list of ImageNet class names.
    Caches the result globally to avoid repeated downloads.
    """
    global _IMAGENET_CLASS_NAMES
    
    # if already downloaded, return cached version
    if _IMAGENET_CLASS_NAMES is not None:
        return _IMAGENET_CLASS_NAMES
        
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    print("Downloading ImageNet class names...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        class_names = response.text.split('\n')
        
        # Store in cache
        _IMAGENET_CLASS_NAMES = [name.strip() for name in class_names if name.strip()]
        return _IMAGENET_CLASS_NAMES
    except Exception as e:
        print(f"Could not download class names: {e}")
        return None

def get_top_n_predictions(model_output_logits, n=5):
    """
    Gets the top N class indices, names, and confidences from model's raw output.
    
    Args:
        model_output_logits (torch.Tensor): The raw output from the model (B, NumClasses).
        n (int): The number of top predictions to return.

    Returns:
        list[tuple]: A list of (class_name, confidence) tuples.
    """
    class_names = get_imagenet_class_names()
    if class_names is None:
        print("Cannot get class names. Aborting prediction.")
        return []
        
    probabilities = F.softmax(model_output_logits, dim=1)
    top_n_probs, top_n_indices = torch.topk(probabilities, n, dim=1)
    
    # Squeeze to 1D arrays
    top_n_probs = top_n_probs.squeeze().tolist()
    top_n_indices = top_n_indices.squeeze().tolist()
    
    results = []
    for i in range(n):
        class_idx = top_n_indices[i]
        class_name = class_names[class_idx]
        confidence = top_n_probs[i]
        results.append((class_name, confidence))
        
    return results

def get_top_prediction(model_output_logits):
    """
    Gets just the top class index and confidence.
    (This is a simpler version, keep for compatibility)
    """
    probabilities = F.softmax(model_output_logits, dim=1)
    top_prob, top_class_idx = torch.max(probabilities, 1)
    
    return top_class_idx.item(), top_prob.item()
