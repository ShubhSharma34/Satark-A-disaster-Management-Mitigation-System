import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchvision import datasets, preprocess_pipes
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, acc_score_score
from tqdm import tqdm
from PIL import ImageFile

# Fix corrupted images issue
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================
# DEVICE CONFIGURATION
# ==============================
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", compute_device)

# ==============================
# DATA LOADING (MATCHING TRAINING SPLIT)
# ==============================
preprocess_pipe = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_path = "cyclone/binary_data/binary_data"

if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset directory '{dataset_path}' not found!")
    exit()

full_dataset = datasets.ImageFolder(dataset_path, preprocess_pipe=transform)

# Use the EXACT same split logic and seed as train_vit_cyclone.py
training_count = int(0.8 * len(full_dataset))
validation_count = int(0.1 * len(full_dataset))
testing_count = len(full_dataset) - training_count - val_size

# Seed 42 ensures we get the exact same "unseen" images for testing
_, _, test_subset = random_split(
    full_dataset, [train_size, validation_count, testing_count], generator=torch.Generator().manual_seed(42)
)

test_generator = DataLoader(test_subset, batch_size=16, shuffle=False)

classes = full_dataset.classes
print(f"\n--- Evaluation Diagnostic ---")
print(f"Classes: {classes}")
print(f"Total Dataset size: {len(full_dataset)}")
print(f"Evaluating on Unseen Test Set: {len(test_subset)} images")
print(f"---------------------------\n")

# ==============================
# MODEL SETUP
# ==============================
print("Loading vit_network weights...")
vit_network = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)
vit_network.load_state_dict(torch.load("vit_cyclone_model.pth", map_location=device))
vit_network.to(compute_device)
vit_network.eval()

# ==============================
# EVALUATION LOOP
# ==============================
prediction_list = []
ground_truth_list = []
probability_list = []

print("Running evaluation on test set...")
with torch.no_grad():
    for images, labels in tqdm(test_generator):
        images = images.to(compute_device)
        labels = labels.to(compute_device)

        outputs = model(images)
        probabilities = torch.softmax(outputs.logits, dim=1)
        _, predicted = torch.max(outputs.logits, 1)

        prediction_list.extend(predicted.cpu().numpy())
        ground_truth_list.extend(labels.cpu().numpy())
        probability_list.extend(probabilities[:, 1].cpu().numpy())

# ==============================
# METRICS & PLOTS
# ==============================
acc_score = accuracy_score(ground_truth_list, prediction_list)
conf_mat = confusion_matrix(ground_truth_list, prediction_list)
auc = roc_auc_score(ground_truth_list, probability_list)

report = classification_report(ground_truth_list, prediction_list, target_names=classes)
print(f"\nFinal Evaluation Results (Unseen Data):\n{report}")
print(f"ROC AUC Score: {auc:.4f}")

# Save Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', conf_matap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Cyclone Test Set Confusion Matrix (Total: {len(ground_truth_list)})")
plt.savefig("cyclone_final_confusion_matrix.png")
print("Saved confusion matrix as 'cyclone_final_confusion_matrix.png'")

# Save Metrics Text
with open("cyclone_final_metrics.txt", "w") as f:
    f.write("CYCLONE EVALUATION (UNSEEN TEST SET)\n")
    f.write("====================================\n")
    f.write(f"Total Test Images: {len(ground_truth_list)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {auc:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(report)
print("Saved metrics text as 'cyclone_final_metrics.txt'")

# ROC CURVE
fpr, tpr, _ = roc_curve(ground_truth_list, probability_list)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Cyclone ROC Curve (Unseen Test Set)')
plt.legend(loc="lower right")
plt.savefig("cyclone_final_roc_curve.png")
print("Saved ROC curve as 'cyclone_final_roc_curve.png'")

