import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score
from tqdm import tqdm
from PIL import Image

# ==============================
# DEVICE CONFIGURATION
# ==============================
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", compute_device)

# ==============================
# DATASET DEFINITION (Same as Training)
# ==============================
class FloodNetDataset(Dataset):
    def __init__(self, preprocess_pipe=None):
        self.preprocess_pipe = transform
        self.data = []
        
        print("Loading dataset metadata from both Tracks...")
        
        # 1. LOAD TRACK 1 DATA
        track1_path = os.path.join("FloodNet Challenge", "FloodNet Challenge - Track 1", "Train", "Labeled")
        flooded_dir = os.path.join(track1_path, 'Flooded', 'image')
        non_flooded_dir = os.path.join(track1_path, 'Non-Flooded', 'image')
        
        if os.path.exists(flooded_dir):
            for filename in os.listdir(flooded_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.data.append({
                        'filepath': os.path.join(flooded_dir, filename),
                        'label': 1
                    })
        
        if os.path.exists(non_flooded_dir):
            for filename in os.listdir(non_flooded_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.data.append({
                        'filepath': os.path.join(non_flooded_dir, filename),
                        'label': 0
                    })

        # 2. LOAD TRACK 2 DATA
        track2_base = os.path.join("FloodNet Challenge", "FloodNet Challenge - Track 2")
        track2_subsets = [
            (
                os.path.join(track2_base, "Images", "Train_Image"),
                os.path.join(track2_base, "Questions", "Training Question.json")
            ),
            (
                os.path.join(track2_base, "Images", "Valid_Image"),
                os.path.join(track2_base, "Questions", "Valid Question.json")
            )
        ]

        for img_dir, json_path in track2_subsets:
            if os.path.exists(json_path) and os.path.exists(img_dir):
                try:
                    with open(json_path, 'r') as f:
                        qa_data = json.load(f)
                    
                    labeled_t2 = set() 
                    for key, item in qa_data.items():
                        if item.get("Question_Type") == "Condition_Recognition":
                            gt = str(item.get("Ground_Truth", "")).strip().lower()
                            img_id = item.get("Image_ID")
                            
                            if img_id and img_id not in labeled_t2:
                                img_path = os.path.join(img_dir, img_id)
                                if os.path.exists(img_path):  
                                    if gt == "flooded":
                                        self.data.append({'filepath': img_path, 'label': 1})
                                        labeled_t2.add(img_id)
                                    elif gt == "non flooded":
                                        self.data.append({'filepath': img_path, 'label': 0})
                                        labeled_t2.add(img_id)
                except Exception as e:
                    pass

        print(f"Total merged images available for splitting: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['filepath']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# ==============================
# DATA SETUP
# ==============================
preprocess_pipe = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = FloodNetDataset(preprocess_pipe=transform)

# Using a fixed seed ensures we evaluate on the exact same 20% validation split 
# that the master vit_network used.
torch.manual_seed(42)
training_count = int(0.8 * len(dataset))
validation_count = len(dataset) - train_size
_, val_subset = torch.utils.data.random_split(dataset, [train_size, validation_count])

eval_generator = DataLoader(val_subset, batch_size=16, shuffle=False)
classes = ["No Flood", "Flood"]

# ==============================
# MODEL CONFIGURATION
# ==============================
print("\nLoading vit_network...")
vit_network = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    id2label={0: "No Flood", 1: "Flood"},
    label2id={"No Flood": 0, "Flood": 1}
)

model_path = "vit_floodnet_model.pth"
if os.path.exists(vit_network_path):
    print(f"Loading weights from {model_path}...")
    vit_network.load_state_dict(torch.load(vit_network_path, map_location=device))
else:
    print(f"Warning: {model_path} not found. Running evaluation with uninitialized random state.")

vit_network.to(compute_device)
vit_network.eval()

# ==============================
# EVALUATION LOOP
# ==============================
prediction_list = []
ground_truth_list = []
probability_list = []

print("\nRunning Master Track 1+2 Evaluation...")
with torch.no_grad():
    for images, labels in tqdm(eval_loader):
        images = images.to(compute_device)
        labels = labels.to(compute_device)

        outputs = model(images)
        probabilities = torch.softmax(outputs.logits, dim=1)
        
        _, predicted = torch.max(outputs.logits, 1)

        prediction_list.extend(predicted.cpu().numpy())
        ground_truth_list.extend(labels.cpu().numpy())
        probability_list.extend(probabilities[:, 1].cpu().numpy())

# ==============================
# METRICS CALCULATION
# ==============================
f1 = f1_score(ground_truth_list, prediction_list, zero_division=0)
precision = precision_score(ground_truth_list, prediction_list, zero_division=0)
try:
    auc = roc_auc_score(ground_truth_list, probability_list)
except ValueError:
    auc = 0.0  

print("\n" + "="*30)
print("EVALUATION METRICS: MASTER FLOODNET")
print("="*30)
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"ROC AUC:   {auc:.4f}")
print("="*30)

print("\nClassification Report:")
print(classification_report(ground_truth_list, prediction_list, target_names=classes, zero_division=0))

# ==============================
# CONFUSION MATRIX
# ==============================
conf_mat = confusion_matrix(ground_truth_list, prediction_list)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', conf_matap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix (Master FloodNet)")
plt.savefig("floodnet_confusion_matrix.png")
print("\nSaved confusion matrix plot as 'floodnet_confusion_matrix.png'")

# ==============================
# ROC CURVE
# ==============================
try:
    fpr, tpr, _ = roc_curve(ground_truth_list, probability_list)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Master FloodNet)')
    plt.legend(loc="lower right")
    plt.savefig("floodnet_roc_curve.png")
    print("Saved ROC curve plot as 'floodnet_roc_curve.png'")
except Exception as e:
    print("Could not generate ROC curve:", e)
