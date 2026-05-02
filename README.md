# Adaptive Multi-Modal Flood Prediction & Inundation Analysis

**Author:** Sehrish Kazmi,Maleeha Ahmad,Rabia Amin Khan  
**Course:** Machine Learning  
**Submission Date:** 02-05-2026  

---

## 0. Development Environment

- **IDE:** Pycharm  
- **Python Environment:** Anaconda / Conda virtual environment  
- **Python Version:** 3.10  
---

## 1. Overview
AMFI (Adaptive Multi-Modal Flood
Inundation) by modeling a synchronous system for prediction of
floods and accurate flood mapping from SAR imagery. Through
our experiments, we designed a model trained on Mekong dataset
and synthetic sensor stream, and tested it on Bolivia samples,
garnering a Global IoU of 0.5163, an F1-Score of 0.6814, Precision
0.6609, and Recall 0.7033 showcasing multi-modality and domain
adaptation.


---

# Core Deep Learning (Blackwell/RTX 5050 optimized)

This repository contains the environment configuration and framework setup optimized for **NVIDIA Blackwell (RTX 5050)** architecture using **CUDA 12.8**.

##  Installation

To install the core deep learning stack with Blackwell optimization, run:

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://pytorch.org
```

##  Dependencies

### Satellite & Data Processing
Tools for handling geospatial data and visualization:
*   `rasterio==1.4.4`
*   `numpy`
*   `matplotlib`
*   `tifffile`

### Computer Vision Framework
Implementation of the **EfficientNet-B4** backbone for segmentation:
*   `segmentation-models-pytorch==0.5.0`

### Machine Learning Helpers
Data analysis and evaluation tools:
*   `scikit-learn`
*   `seaborn`

##  Usage
This setup is designed for high-performance satellite image segmentation. Ensure your drivers are updated to support **cu128** to take full advantage of the RTX 5050 hardware acceleration.

##  Experiment Trials
### sen1flood11 datasets: Bolivia and Mekong (SAR images and labels)
### Trial 1
*   **Training:** `train.py` (50 epochs).
*   **Evaluation:** `eval_metrics1.py` for test dataset predictions.
*   **Artifacts:** 
    *   `models/`: Checkpoint weights.
    *   `trial1_final_results/`: Output flood masks, ground truth comparisons, and confusion matrices.

### Trial 2
*   **Training:** `train_v2.py` (New optimization applied, 50 epochs).
*   **Evaluation:** `eval_metrics2.py` (Demonstrates improved metrics from best epoch).
*   **Artifacts:** 
    *   `models/`: Updated checkpoints for the 2nd trial.
    *   `Trial2_Analysis_Report_20260501_150655/`: Detailed analysis including flood masks and performance reports.
