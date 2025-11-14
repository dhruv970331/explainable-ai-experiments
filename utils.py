# utils.py
#
# Helper functions for the XAI seminar project.
# Includes image loading, preprocessing, *ADVANCED* metrics, and plotting.
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
from scipy.stats import pearsonr
from scipy.ndimage import center_of_mass
import os
import time

# These are standard for ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- 1. Image Loading and Preprocessing ---

def load_image_from_url(url: str, save_path=None) -> Image.Image:
    """
    Loads a PIL image from a given URL.
    Optionally saves it locally to avoid re-downloading.
    """
    if save_path and os.path.exists(save_path):
        # print(f"Loading cached image from {save_path}")
        return Image.open(save_path).convert('RGB')
        
    print(f"Downloading image from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        img_bytes = BytesIO(response.content)
        img_pil = Image.open(img_bytes).convert('RGB')
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            # print(f"Cached image to {save_path}")
            
        return img_pil
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from {url}: {e}")
        return None

def load_image_from_path(path: str) -> Image.Image:
    """
    Loads a PIL image from a local file path.
    (This is the one we need for the wget files)
    """
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None
        
    try:
        img_pil = Image.open(path).convert('RGB')
        # print(f"Loaded local image from {path}") # too spammy
        return img_pil
    except Exception as e:
        print(f"Error loading local image {path}: {e}")
        return None


def preprocess_image(pil_img: Image.Image, resize_shape=(224, 224)) -> torch.Tensor:
    """
    Converts a PIL image into a normalized torch tensor for model input.
    This is the standard ImageNet pipeline.
    """
    # Standard ImageNet normalization and resizing
    # We use resize(256) and center_crop(224) as is standard
    preprocess_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    img_tensor = preprocess_transform(pil_img)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
    return img_tensor


def de_normalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized image tensor back to a displayable numpy array (H, W, C).
    """
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = IMAGENET_STD * img + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# --- 2. Heatmap and Visualization ---

def normalize_to_0_1(arr: np.ndarray) -> np.ndarray:
    """
    Simple min-max normalization.
    """
    if arr.max() == arr.min():
        return np.zeros_like(arr) # avoid division by zero
    return (arr - arr.min()) / (arr.max() - arr.min())


def overlay_heatmap(img_arr: np.ndarray, heatmap: np.ndarray, alpha=0.5, colormap_name=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Applies a colormap to a heatmap and overlays it on the original image.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, but got shape {heatmap.shape}")
        
    # Normalize heatmap first!
    heatmap_norm = normalize_to_0_1(heatmap)
    
    heatmap_u8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_u8, colormap_name)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    if heatmap_colored.shape[:2] != img_arr.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (img_arr.shape[1], img_arr.shape[0]), interpolation=cv2.INTER_LINEAR)

    overlaid_img = cv2.addWeighted(img_arr, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid_img


# --- 3. Quantatative Metrics (NEW & IMPROVED) ---

def _resize_maps(map1, map2):
    """Internal helper to make sure maps are same size."""
    if map1.shape != map2.shape:
        map2 = cv2.resize(map2, (map1.shape[1], map1.shape[0]), interpolation=cv2.INTER_LINEAR)
    return map1, map2

def compute_ssim(map1: np.ndarray, map2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two heatmaps.
    """
    map1_norm = normalize_to_0_1(map1)
    map2_norm = normalize_to_0_1(map2)
    map1_norm, map2_norm = _resize_maps(map1_norm, map2_norm)
    return ssim(map1_norm, map2_norm, data_range=1.0)


def compute_pearson_correlation(map1: np.ndarray, map2: np.ndarray) -> float:
    """
    Computes the Pearson correlation co-efficient between two heatmaps.
    This measures linear correlation of intensities.
    """
    map1_norm = normalize_to_0_1(map1)
    map2_norm = normalize_to_0_1(map2)
    map1_norm, map2_norm = _resize_maps(map1_norm, map2_norm)
    
    # Flatten to 1D arrays
    map1_flat = map1_norm.flatten()
    map2_flat = map2_norm.flatten()
    
    # Use scipy.stats.pearsonr, which returns (correlation, p-value)
    corr, _ = pearsonr(map1_flat, map2_flat)
    return corr


def compute_iou_at_k(map1: np.ndarray, map2: np.ndarray, k=0.2) -> float:
    """
    Computes IoU on the *top k percent* most-activated pixels.
    This is a much better metric for 'focus' overlap.
    
    Args:
        k (float): Percentile threshold (0.0 to 1.0). e.g., 0.2 = top 20%.
    """
    map1_norm = normalize_to_0_1(map1)
    map2_norm = normalize_to_0_1(map2)
    map1_norm, map2_norm = _resize_maps(map1_norm, map2_norm)

    # Find the threshold value for the top k% pixels for *each map*
    # (this is more robust than a global threshold)
    thresh1 = np.quantile(map1_norm, 1.0 - k)
    thresh2 = np.quantile(map2_norm, 1.0 - k)
    
    # Binarize
    bin_map1 = (map1_norm >= thresh1)
    bin_map2 = (map2_norm >= thresh2)
    
    # Flatten for jaccard_score
    bin_map1_flat = bin_map1.flatten()
    bin_map2_flat = bin_map2.flatten()
    
    return jaccard_score(bin_map1_flat, bin_map2_flat, zero_division=1.0)


def compute_center_of_mass_distance(map1: np.ndarray, map2: np.ndarray) -> float:
    """
    Computes the Euclidean distance between the 'center of mass'
    of two heatmaps.
    """
    map1_norm = normalize_to_0_1(map1)
    map2_norm = normalize_to_0_1(map2)
    map1_norm, map2_norm = _resize_maps(map1_norm, map2_norm)

    # Compute center of mass (y, x)
    com1 = center_of_mass(map1_norm)
    com2 = center_of_mass(map2_norm)
    
    # com1 and com2 are (y, x) tuples.
    # Calculate Euclidean distance
    distance = np.sqrt((com1[0] - com2[0])**2 + (com1[1] - com2[1])**2)
    
    # Normalize by image diagonal to make it a relative measure?
    # max_dist = np.sqrt(map1_norm.shape[0]**2 + map1_norm.shape[1]**2)
    # return distance / max_dist
    #
    # Let's just return raw pixel distance. It's easier to interpret.
    return distance

# --- 4. Plotting ---

def plot_comparison_grid(images: list, titles: list, grid_shape: tuple, figsize=(15, 5), filename=None, main_title=None):
    """
    Plots a grid of images with titles.
    """
    rows, cols = grid_shape
    if len(images) != len(titles) or len(images) > rows * cols:
        print(f"Warning: Mismatch in items. Got {len(images)} images and {len(titles)} titles for a {rows}x{cols} grid.")
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if not isinstance(axes, (np.ndarray)):
        axes = np.array([axes])
        
    axes_flat = axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes_flat[i].imshow(img)
        axes_flat[i].set_title(title, fontsize=10)
        axes_flat[i].axis('off')
        
    for i in range(len(images), len(axes_flat)):
        axes_flat[i].axis('off')
        
    if main_title:
        fig.suptitle(main_title, fontsize=16, y=1.03)
        
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")
    
    plt.show()

# --- 5. Model Output and Class Helpers ---

_IMAGENET_CLASS_NAMES = None
def get_imagenet_class_names():
    """
    Downloads and returns a list of ImageNet class names.
    Caches the result globally.
    """
    global _IMAGENET_CLASS_NAMES
    if _IMAGENET_CLASS_NAMES is not None:
        return _IMAGENET_CLASS_NAMES
        
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    print("Downloading ImageNet class names...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        class_names = response.text.split('\n')
        _IMAGENET_CLASS_NAMES = [name.strip() for name in class_names if name.strip()]
        return _IMAGENET_CLASS_NAMES
    except Exception as e:
        print(f"Could not download class names: {e}")
        return None

def get_top_prediction(model_output_logits):
    """
    Gets just the top class index and confidence.
    """
    probabilities = F.softmax(model_output_logits, dim=1)
    top_prob, top_class_idx = torch.max(probabilities, 1)
    return top_class_idx.item(), top_prob.item()

# --- End of file ---