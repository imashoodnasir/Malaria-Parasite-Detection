import torch
import torch.nn as nn
import torch.optim as optim
from hybrid_capnet import HybridCapNet
from loss_functions import total_loss
from data_loader import get_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = get_loaders("images/", "annotations.json", batch_size=16)

model = HybridCapNet(num_classes=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

best_val_acc = 0

for epoch in range(1, 101):
    model.train()
    total_train_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        labels_onehot = torch.eye(4)[labels].to(device)
        optimizer.zero_grad()
        v_length, recon = model(images)
        loss = total_loss(v_length, labels_onehot, images, recon)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        pred = v_length.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels_onehot = torch.eye(4)[labels].to(device)
            v_length, recon = model(images)
            pred = v_length.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    scheduler.step()
