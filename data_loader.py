import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import random

class MalariaDataset(Dataset):
    def __init__(self, root, annotation_file, transform=None):
        self.root = root
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.annotations[idx]["image_name"])
        img = Image.open(img_path).convert("RGB")
        label = 0
        for obj in self.annotations[idx]["objects"]:
            if obj["type"].lower() in ["gametocyte", "ring", "schizont", "trophozoite"]:
                label = ["gametocyte", "ring", "schizont", "trophozoite"].index(obj["type"].lower())
                break
        if self.transform:
            img = self.transform(img)
        return img, label

def get_loaders(data_dir, annotation_path, batch_size=32):
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = MalariaDataset(data_dir, annotation_path, transform=tf)
    n = len(dataset)
    train_size = int(n * 0.6)
    val_size = int(n * 0.2)
    test_size = n - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
