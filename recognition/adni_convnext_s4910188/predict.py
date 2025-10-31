"""
predict.py
Model inference for ADNI Alzheimer’s Disease Classification
Author: s4910188
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import ADNIDataset, get_transforms
from modules import ConvNeXtAD

# ====== Config ======
TEST_ROOT = "ADNI/AD_NC/test"
MODEL_PATH = "checkpoints/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load Test Data ======
print("Loading test data...")
test_ds = ADNIDataset(TEST_ROOT, transform=get_transforms(train=False))
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

# ====== Load Model ======
model = ConvNeXtAD(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ====== Inference ======
correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

acc = correct / total
print(f"\n Test Accuracy: {acc:.4f}")

# ====== Visualize Sample Predictions ======
classes = ["NC", "AD"]
num_show = 6
indices = np.random.choice(len(test_ds), num_show, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    img, label = test_ds[idx]
    pred = classes[all_preds[idx]]
    true = classes[all_labels[idx]]
    img = (img.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # unnormalize
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"True: {true}, Pred: {pred}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("checkpoints/sample_predictions.png")
plt.show()
