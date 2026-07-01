import os
import random
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ==========================================
# 1. SEED SETTING & DEVICE CONFIG
# ==========================================
SEED = 29
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

from tqdm import tqdm



# Device configuration (Replaces GPU memory growth logic)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. HYPERPARAMETERS & PATHS
# ==========================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
DATASET_PATH = "../Mendeley_Dataset"
RESULTS_DIR = "results_vgg"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 3. DATA SPLITTING & STRATIFICATION
# ==========================================
DATASET_PATH = Path(DATASET_PATH)

image_paths = []
labels = []

for class_name in os.listdir(DATASET_PATH):
    class_dir = DATASET_PATH / class_name
    if class_dir.is_dir():
        for img in class_dir.iterdir():
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(str(img))
                labels.append(class_name)

# Create label encoding map
CLASS_NAMES = sorted(list(set(labels)))
NUM_CLASSES = len(CLASS_NAMES)
class_to_idx = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}

# Stratified split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=SEED,
    shuffle=True
)

train_df = pd.DataFrame({'filename': train_paths, 'class': train_labels})
val_df = pd.DataFrame({'filename': val_paths, 'class': val_labels})


# 2. Save the DataFrame to a pickle file
# train_df.to_pickle("train_df_mendeley_dataframe.pkl")
# val_df.to_pickle("val_df_mendeley_dataframe.pkl")

# 3. Load it back later
train_df = pd.read_pickle("train_df_mendeley_dataframe.pkl")
val_df = pd.read_pickle("val_df_mendeley_dataframe.pkl")

print("Classes:", CLASS_NAMES)
print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))

# ==========================================
# 4. CUSTOM DATASET & AUGMENTATIONS
# ==========================================
class CustomMinMaxNormalize(object):
    """Custom PyTorch transform for image-specific min-max normalization"""
    def __call__(self, tensor):
        img_min = tensor.min()
        img_max = tensor.max()
        if img_max - img_min == 0:
            return tensor - img_min
        return (tensor - img_min) / (img_max - img_min)

# Transforms mimicking your original Keras ImageDataGenerator
train_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomRotation(degrees=40),
    # Keras zoom_range=0.3 translates to cropping/scaling 70% to 100% of the image
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.0)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),              # Converts PIL Image to PyTorch Tensor (0.0 to 1.0 automatically)
    CustomMinMaxNormalize()             # Enforces clean per-image min-max tracking
])

val_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    CustomMinMaxNormalize()
])

class MendeleyDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'filename']
        label_name = self.df.loc[idx, 'class']
        label = self.class_to_idx[label_name]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

train_dataset = MendeleyDataset(train_df, class_to_idx, transform=train_transforms)
val_dataset = MendeleyDataset(val_df, class_to_idx, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


# 1. Define a wrapper module that applies Multi-Head Attention with Bottleneck to VGG16
class MidBlockMHA_VGG16(nn.Module):
    def __init__(self, base_model, num_classes, num_heads=8, reduction_factor=4):
        super(MidBlockMHA_VGG16, self).__init__()
        
        # Determine if base_model uses Batch Normalization to set the correct layer index split point
        # For standard vgg16, layer 23 is Conv4_3 and layer 24 is MaxPool4 -> Split at [:24] and [24:]
        # For vgg16_bn, layer 33 is Conv4_3 and layer 34 is MaxPool4 -> Split at [:34] and [34:]
        is_bn = isinstance(base_model.features[1], nn.BatchNorm2d)
        split_idx = 34 if is_bn else 24
        
        # 1. Split the flat VGG features container right after MaxPool 4
        # For a 224x224 input, this stage outputs a shape of [B, 512, 14, 14]
        self.stage1 = base_model.features[:split_idx] 
        
        mid_features = 512 
        
        # Ensure the reduced embedding dimension is strictly divisible by num_heads
        raw_reduced = mid_features // reduction_factor # 512 // 4 = 128
        self.reduced_dim = (raw_reduced // num_heads) * num_heads
        if self.reduced_dim == 0:
            self.reduced_dim = num_heads
            
        # 2. Bottleneck MHA Components
        self.project_down = nn.Conv2d(mid_features, self.reduced_dim, kernel_size=1)
        self.mha = nn.MultiheadAttention(embed_dim=self.reduced_dim, num_heads=num_heads)
        self.project_up = nn.Conv2d(self.reduced_dim, mid_features, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Start training as standard VGG16
        
        # 3. Remaining Backbone stage (Conv5 block + final MaxPool 5 down to 7x7)
        self.stage2 = base_model.features[split_idx:]
        
        # 4. Standard VGG Avg Pooling & Classifier Structure
        self.pool = base_model.avgpool
        
        # Reconstruct the original VGG linear classification block
        self.classifier = nn.Sequential(
            base_model.classifier[0],  # Linear(512 * 7 * 7, 4096)
            base_model.classifier[1],  # ReLU
            base_model.classifier[2],  # Dropout
            base_model.classifier[3],  # Linear(4096, 4096)
            base_model.classifier[4],  # ReLU
            base_model.classifier[5],  # Dropout
            nn.Linear(4096, num_classes) # Adjusted target leaf disease head
        )

    def forward(self, x):
        # Forward through first half of network (up to MaxPool 4): [B, 512, 14, 14]
        identity = self.stage1(x) 
        
        # Compress channels for attention
        x_reduced = self.project_down(identity) 
        
        # Reshape for MHA: (B, C_red, H, W) -> (H*W, B, C_red)
        B, C_red, H, W = x_reduced.shape
        x_reshaped = x_reduced.view(B, C_red, H * W).permute(2, 0, 1)
        
        # Self-Attention
        attn_output, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)
        
        # Reshape back to CNN format
        x_attn = attn_output.permute(1, 2, 0).view(B, C_red, H, W)
        
        # Expand back and apply residual connection scaled by gamma
        x_expanded = self.project_up(x_attn)
        mid_features = identity + self.gamma * x_expanded
        
        # Forward through remaining layers (Conv5 block + MaxPool5): Outputs [B, 512, 7, 7]
        final_features = self.stage2(mid_features) 
        
        # Global Pooling, flattening, and Dense Classification
        pooled = self.pool(final_features)
        flat = torch.flatten(pooled, 1)
        out = self.classifier(flat)
        
        return out


# ==========================================
# 5. MODEL ARCHITECTURE (DenseNet169)
# ==========================================
# Fetch weights cleanly in modern torchvision
# densenet_weights = models.DenseNet169_Weights.IMAGENET1K_V1#models.DenseNet169_Weights.DEFAULT
base_model = models.vgg16(pretrained=True)#weights=densenet_weights)

# Replace the original classification head with your custom class size
# num_features = base_model.fc.in_features
# base_model.classifier = nn.Linear(num_features, NUM_CLASSES)
model = MidBlockMHA_VGG16(base_model, NUM_CLASSES, num_heads=8)
model = model.to(device)

def count_parameters(model):
    # Sum up the elements of each parameter tensor if it requires gradients
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calculate the parameters
# total_trainable_params_base = count_parameters(base_model)
total_trainable_params = count_parameters(model)

print("=" * 40)
print(f"Total Trainable Parameters: {total_trainable_params:,}")
print("=" * 40)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss() # Combines LogSoftmax and NLLLoss (handles categorical implicitly)
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Keras ReduceLROnPlateau equivalent
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.2, 
    patience=3, 
    # verbose=True
)
# 2. Evaluation function containing comprehensive metrics evaluation
def evaluate_and_print_metrics(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Core Scikit-Learn Metric Computations
    overall_acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    avg_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) # Recall == Sensitivity
    avg_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # Specificity Calculation via Confusion Matrix (One-vs-All approach)
    cm = confusion_matrix(all_labels, all_preds)
    num_classes = cm.shape[0]
    class_specificities = []
    
    for i in range(num_classes):
        # True Positives, False Positives, False Negatives, True Negatives per class
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        class_specificities.append(specificity)
        
    avg_specificity = np.mean(class_specificities)

    # Print Results nicely formatted
    print("\n================ Validation Metrics ================")
    print(f"Overall Accuracy:                  {overall_acc:.4f}")
    print(f"Balanced Accuracy:                 {balanced_acc:.4f}")
    print(f"Average Precision (Macro):         {avg_precision:.4f}")
    print(f"Average Recall / Sensitivity:      {avg_recall:.4f}")
    print(f"Average Specificity (Macro):       {avg_specificity:.4f}")
    print(f"Average F1-Score (Macro):          {avg_f1:.4f}")
    print(f"Matthews Correlation Coefficient:  {mcc:.4f}")
    print("====================================================\n")
# ==========================================
# 6. TRAINING LOOP WITH CALLBACKS
# ==========================================
# Tracking objects mimicking EarlyStopping, ModelCheckpoint, and CSVLogger
best_val_loss = float('inf')
best_val_acc = 0
best_val_ep =0
early_stop_patience = 30
early_stop_counter = 0
min_delta = 0.001

log_history = []
log_csv_path = os.path.join(RESULTS_DIR, "training_log.csv")

for epoch in tqdm(range(1, EPOCHS + 1)):
    # --- Training Phase ---
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()
        
    epoch_train_loss = running_train_loss / total_train
    epoch_train_acc = correct_train / total_train
    
    # --- Validation Phase ---
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_val += targets.size(0)
            correct_val += predicted.eq(targets).sum().item()
            
    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = correct_val / total_val

    if(epoch_val_acc > best_val_acc):
        best_val_acc = epoch_val_acc
        best_val_ep = epoch

    
    # Update Learning Rate Scheduler
    scheduler.step(epoch_val_loss)
    
    # Print progress
    print(f"Epoch {epoch}/{EPOCHS} - loss: {epoch_train_loss:.4f} - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc:.4f} - best: {best_val_acc:.4f} ep {best_val_ep} {early_stop_counter}")
    evaluate_and_print_metrics(model, val_loader, device)
    # Log to CSV
    log_history.append({
        'epoch': epoch, 'loss': epoch_train_loss, 'accuracy': epoch_train_acc, 
        'val_loss': epoch_val_loss, 'val_accuracy': epoch_val_acc
    })
    pd.DataFrame(log_history).to_csv(log_csv_path, index=False)
    
    # ModelCheckpoint & EarlyStopping logic
    if epoch_val_loss < (best_val_loss - min_delta):
        best_val_loss = epoch_val_loss
        early_stop_counter = 0
        # Save best weights
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best.pth"))
        print(f"--> Epoch {epoch}: val_loss improved, saving model to best.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}. Restoring best weights.")
            # Load best weights back to model
            # model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best.pth")))
            break