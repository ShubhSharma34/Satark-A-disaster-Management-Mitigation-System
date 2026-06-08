import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, preprocess_pipes
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from tqdm import tqdm
from PIL import ImageFile

# ==============================
# FIX CORRUPTED IMAGES ISSUE
# ==============================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================
# DEVICE
# ==============================
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", compute_device)

# ==============================
# PARAMETERS
# ==============================
BATCH_SIZE = 16   # If GPU memory error → change to 8
EPOCHS = 5
LR = 2e-5

# ==============================
# TRANSFORMS
# ==============================
preprocess_pipe = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ==============================
# LOAD AND SPLIT DATASET
# ==============================
# The cyclone binary data is in "cyclone/binary_data/binary_data"
dataset_path = "cyclone/binary_data/binary_data"
full_dataset = datasets.ImageFolder(dataset_path, preprocess_pipe=transform)

# 80% Train, 10% Validation, 10% Test
training_count = int(0.8 * len(full_dataset))
validation_count = int(0.1 * len(full_dataset))
testing_count = len(full_dataset) - training_count - val_size

train_dataset, val_subset, test_subset = random_split(
    full_dataset, [train_size, validation_count, testing_count], generator=torch.Generator().manual_seed(42)
)

train_generator = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_generator = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_generator = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

print("Total images:", len(full_dataset))
print("Train images:", len(train_subset))
print("Validation images:", len(val_subset))
print("Test images:", len(test_subset))

# ==============================
# LOAD PRETRAINED VIT
# ==============================
vit_network = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)

vit_network.to(compute_device)

# ==============================
# LOSS & OPTIMIZER
# ==============================
loss_calculator = nn.CrossEntropyLoss()
weight_updater = optim.AdamW(vit_network.parameters(), lr=LR)

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    # TRAINING
    vit_network.train()
    train_correct = 0
    train_total = 0

    train_loop = tqdm(train_generator)

    for images, labels in train_loop:
        images = images.to(compute_device)
        labels = labels.to(compute_device)

        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        weight_updater.zero_grad()
        loss.backward()
        weight_updater.step()

        _, predicted = torch.max(outputs.logits, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

        train_loop.set_postfix(loss=loss.item())

    train_acc = 100 * train_correct / train_total
    print(f"Train Accuracy: {train_acc:.2f}%")

    # VALIDATION
    vit_network.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(compute_device)
            labels = labels.to(compute_device)

            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")

# ==============================
# TESTING LOOP
# ==============================
print("\n--- Testing Model ---")
vit_network.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(compute_device)
        labels = labels.to(compute_device)

        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)

        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")

# ==============================
# SAVE MODEL
# ==============================
torch.save(vit_network.state_dict(), "vit_cyclone_model.pth")
print("\nModel saved successfully as 'vit_cyclone_model.pth'!")
