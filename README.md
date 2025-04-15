# Hybrid Capsule Network for Malaria Parasite Detection

This repository implements a Hybrid Capsule Network (Hybrid CapNet) for accurate and interpretable malaria parasite detection from blood smear images. The model combines CNN feature extraction with capsule-based spatial reasoning and is validated across four benchmark datasets.

## 🚀 Key Features
- Combines CNN with Capsule Layers (dynamic routing)
- Composite loss: Margin, Focal, Offset Regression, and Reconstruction
- Grad-CAM-based interpretability
- Lightweight: 1.35M parameters, 0.26 GFLOPs
- Supports cross-dataset evaluation

## 📁 Repository Structure
```
Malaria-Parasite-Detection/
├── cnn_backbone.py       # CNN-based feature extractor
├── capsule_layers.py     # PrimaryCaps + DigitCaps with routing
├── hybrid_capnet.py      # Full model integration
├── data_loader.py        # Data loader with augmentation
├── loss_functions.py     # Margin, Focal, Offset, Recon loss
├── train.py              # Model training and validation
├── evaluate.py           # Model evaluation and confusion matrix
├── visualization.py      # Grad-CAM visualizer
└── README.md             # This file
```

## 📦 Datasets
You may use the following:
- MP-IDB
- MP-IDB2
- IML-Malaria
- Malaria-Detection-2019

Ensure you adjust paths to `images/` and `annotations.json`.

## 🛠️ Usage
### 1. Install dependencies
```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

### 2. Train the model
```bash
python train.py
```

### 3. Evaluate on test set
```bash
python evaluate.py
```

### 4. Generate Grad-CAM
```bash
python visualization.py
```

## 🔓 License
This project is licensed under CC-BY 4.0.
