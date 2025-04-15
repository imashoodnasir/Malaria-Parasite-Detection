import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from hybrid_capnet import HybridCapNet
from data_loader import MalariaDataset
from PIL import Image
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_and_evaluate(model_path, img_dir, ann_path):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = MalariaDataset(img_dir, ann_path, transform=tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = HybridCapNet(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            v_length, _ = model(images)
            preds = v_length.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print(classification_report(y_true, y_pred, target_names=["Gametocyte", "Ring", "Schizont", "Trophozoite"]))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

test_and_evaluate("best_model.pth", "images/", "annotations.json")
