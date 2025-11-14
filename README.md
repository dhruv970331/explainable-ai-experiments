# Code for *Understanding Machine Learning: The “Black Box” Problem*

This repository contains the Python code and Jupyter notebooks used to generate the illustrative experiments for the seminar paper:

**“Understanding Machine Learning: The ‘Black Box’ Problem and the Imperative of Interpretability.”**

The experiments demonstrate the fundamental instability of post-hoc explanation methods (such as Grad-CAM) and support the paper’s thesis that these methods are unsuitable for high-stakes accountability.

---

## Summary of Experiments

The repository includes three main experiments:

### **Experiment 1 — Architectural Instability**

Shows that two different SOTA models (**ResNet-50** vs. **DenseNet-121**) produce contradictory heatmaps for the *same* image and *same* predicted class (e.g., focusing on the head vs. hindquarters of a dog).

### **Experiment 2 — Method Instability**

Using one model and one decision, four popular XAI methods
(**Grad-CAM**, **Grad-CAM++**, **EigenCAM**, **Integrated Gradients**)
yield four different, contradictory “reasons” (cab vs. windshield vs. side door of a truck).

### **Experiment 3 — Sanity Check**

Verifies that heatmaps reflect learned features:

* a **trained model** produces a focused, meaningful heatmap
* a **random-weight model** produces meaningless noise

---

## Repository Structure

```
.
├── New Main Runner Wikimedia Cleaned.ipynb   # Final main notebook (used in paper)
├── utils.py                                  # Metrics + preprocessing + plotting
├── models.py                                 # Pretrained models + target layers for CAM
├── explainers.py                             # Unified wrapper for pytorch-grad-cam & captum
├── images/                                   # Original high-resolution experiment images
├── Main Experiment Runner Wikimedia.ipynb    # (Archive) Earlier, messy version
└── Main Experiment Runner (Original CIFAR).ipynb  # (Archive) Prototype using CIFAR-10
```

---

## How to Run

1. Upload the entire repository (**all .py files, .ipynb files, and the `images/` folder**) to a single directory in your Google Drive.
2. Open **`Main_Experiment_Runner_wikimedia_cleaned.ipynb`** in Google Colab.
3. In the second cell, update the `HELPER_PATH` variable to point to your Drive project directory.
4. Run all cells.

---

## Dependencies

The main notebook installs these automatically:

```bash
pip install torch torchvision
pip install captum
pip install pytorch-grad-cam
pip install scikit-image scipy pandas matplotlib
```