import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms
import torch.serialization
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# TRANSFORM FOR IMAGES
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

class CustomTransform:
    def __init__(self, mean, std):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            img_tensor = self.to_tensor(img)
        if img_tensor.dim() == 3 and img_tensor.shape[0] not in [1, 3]:
            img_tensor = img_tensor.permute(2, 0, 1)
        return self.normalize(img_tensor)

transform = CustomTransform(mean=mean, std=std)

# MODEL SETUP
model = resnet18(weights=None)
model.fc = torch.nn.Linear(512, 44)
ckpt = torch.load("./01_MIA.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt)
model.eval()

# DATASET CLASSES
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None, is_private=False):
        super().__init__(transform)
        self.membership = []
        self.is_private = is_private

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        is_private = getattr(self, 'is_private', False)
        if is_private:
            return id_, img, label
        return id_, img, label, self.membership[index]

torch.serialization.add_safe_globals({"MembershipDataset": MembershipDataset})

# LOAD DATA
try:
    public_data = torch.load("./pub.pt", weights_only=False)
    priv_data = torch.load("./priv_out.pt", weights_only=False)
    public_data.is_private = False
    priv_data.is_private = True
    public_data.transform = transform
    priv_data.transform = transform
    print(f"[DEBUG] Type of first public image: {type(public_data[0][1])}")
    print(f"[DEBUG] Shape of first public image: {public_data[0][1].shape}")
    print(f"[INFO] Public dataset size: {len(public_data)}")
    print(f"[INFO] Private dataset size: {len(priv_data)}")
except FileNotFoundError as e:
    print(f"[ERROR] Dataset file not found: {e}")
    exit(1)

# ENHANCED FEATURE EXTRACTION
def extract_features(model, imgs, labels, augmentations=3):
    model.eval()
    
    # First forward pass for basic features
    with torch.no_grad():
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        confidences = torch.max(probs, dim=1)[0]
        losses = F.cross_entropy(outputs, labels, reduction='none')
        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        topk_probs, _ = torch.topk(probs, k=3, dim=1)
        margins = topk_probs[:, 0] - topk_probs[:, 1]
        true_logits = outputs[torch.arange(len(labels)), labels]
        
    # Second forward pass with gradient computation
    imgs.requires_grad_(True)
    outputs = model(imgs)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    gradients = imgs.grad
    gradient_norm = torch.norm(gradients, p=2, dim=(1, 2, 3))
    imgs.requires_grad_(False)
    
    # Additional features
    correct = (torch.argmax(outputs, dim=1) == labels).float()
    prob_std = torch.std(probs, dim=1)
    prob_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

     # Add: robustness to augmentations
    with torch.no_grad():
        consistency_scores = []
        base_preds = torch.argmax(probs, dim=1)

        for _ in range(augmentations):
            aug_imgs = imgs + 0.02 * torch.randn_like(imgs)  # small noise
            aug_imgs = torch.clamp(aug_imgs, 0, 1)
            aug_out = model(aug_imgs)
            aug_pred = torch.argmax(aug_out, dim=1)
            consistency_scores.append((aug_pred == base_preds).float())

        consistency = torch.stack(consistency_scores).mean(dim=0)
    
    return {
        'confidence': confidences,
        'loss': losses,
        'entropy': entropies,
        'margin': margins,
        'true_logit': true_logits,
        'gradient_norm': gradient_norm,
        'correct': correct,
        'prob_std': prob_std,
        'prob_entropy': prob_entropy,
        'top1_prob': topk_probs[:, 0],
        'top2_prob': topk_probs[:, 1],
        'robustness': consistency  # NEW FEATURE
    }

# EXTRACT FEATURES FROM PUBLIC DATA
public_loader = DataLoader(public_data, batch_size=32, shuffle=False)
X_train, y_train, ids_train = [], [], []

for i, (ids, imgs, labels, memberships) in enumerate(tqdm(public_loader, desc="Processing public data")):
    features = extract_features(model, imgs, labels)
    
    for j in range(len(ids)):
        X_train.append([
            features['loss'][j].item(),
            features['margin'][j].item(),
            features['true_logit'][j].item(),
            features['gradient_norm'][j].item(),
            features['confidence'][j].item(),
            features['entropy'][j].item(),
            features['correct'][j].item(),
            features['prob_std'][j].item(),
            features['prob_entropy'][j].item(),
            features['top1_prob'][j].item(),
            features['top2_prob'][j].item()
        ])
        y_train.append(memberships[j].item())
        ids_train.append(ids[j].item())

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"\n[INFO] Proportion of members: {np.mean(y_train):.4f}")

# Feature importance analysis (using simple correlation)
df_public = pd.DataFrame(X_train, columns=[
    "loss", "margin", "true_logit", "gradient_norm", 
    "confidence", "entropy", "correct", "prob_std",
    "prob_entropy", "top1_prob", "top2_prob"
])
df_public["membership"] = y_train

correlations = df_public.corr()["membership"].abs().sort_values(ascending=False)
print("\n[INFO] Feature correlations with membership:")
print(correlations)

# Keep top 7 most correlated features
selected_features = correlations.index[1:8]  # skip membership itself
print(f"\n[INFO] Selected features: {list(selected_features)}")
X_train = df_public[selected_features].values

# TRAIN ATTACK MODEL WITH CROSS-VALIDATION
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Use stratified K-fold for better validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_auc = 0
best_model = None
best_scaler = None

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_train)):
    print(f"\n[INFO] Training fold {fold + 1}/5")
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # XGBoost with balanced class weights
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_tr) - sum(y_tr)) / sum(y_tr),  # handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    # Calibrate the classifier to get better probabilities
    calibrated_xgb = CalibratedClassifierCV(xgb, method='isotonic', cv=3)
    calibrated_xgb.fit(X_tr, y_tr)
    
    # Evaluate
    y_score = calibrated_xgb.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_score)
    fpr, tpr, thresholds = roc_curve(y_val, y_score)
    tpr_at_fpr_005 = tpr[np.argmin(np.abs(fpr - 0.05))]
    
    print(f"[Fold {fold + 1}] AUC: {auc:.4f}, TPR@FPR=0.05: {tpr_at_fpr_005:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_model = calibrated_xgb
        best_scaler = scaler

# Retrain on full data with best configuration
print("\n[INFO] Retraining on full public data with best configuration...")
final_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    random_state=42,
    n_jobs=-1
)
final_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
final_model.fit(X_scaled, y_train)

# Final evaluation on full data (for reference)
y_score_full = final_model.predict_proba(X_scaled)[:, 1]
auc_full = roc_auc_score(y_train, y_score_full)
fpr_full, tpr_full, thresholds_full = roc_curve(y_train, y_score_full)
tpr_at_fpr_005_full = tpr_full[np.argmin(np.abs(fpr_full - 0.05))]
print(f"[FINAL] Full data AUC: {auc_full:.4f}")
print(f"[FINAL] Full data TPR@FPR=0.05: {tpr_at_fpr_005_full:.4f}")

# Save models
joblib.dump(final_model, "attack_model_xgb.pkl")
joblib.dump(scaler, "scaler_xgb.pkl")
joblib.dump(selected_features, "selected_features.pkl")
print("\n[INFO] Saved models and scaler")

# PROCESS PRIVATE DATA
priv_loader = DataLoader(priv_data, batch_size=32, shuffle=False)
ids_test, features_test = [], []

for i, (ids, imgs, labels) in enumerate(tqdm(priv_loader, desc="Processing private data")):
    features = extract_features(model, imgs, labels)
    
    for j in range(len(ids)):
        features_test.append([
            features['loss'][j].item(),
            features['margin'][j].item(),
            features['true_logit'][j].item(),
            features['gradient_norm'][j].item(),
            features['confidence'][j].item(),
            features['entropy'][j].item(),
            features['correct'][j].item(),
            features['prob_std'][j].item(),
            features['prob_entropy'][j].item(),
            features['top1_prob'][j].item(),
            features['top2_prob'][j].item()
        ])
        ids_test.append(ids[j].item())

# Prepare private features
df_private = pd.DataFrame(features_test, columns=[
    "loss", "margin", "true_logit", "gradient_norm", 
    "confidence", "entropy", "correct", "prob_std",
    "prob_entropy", "top1_prob", "top2_prob"
])
X_test = df_private[selected_features].values
X_test_scaled = scaler.transform(X_test)

# Predict membership scores
membership_scores = final_model.predict_proba(X_test_scaled)[:, 1]

# Create submission
submission_df = pd.DataFrame({
    "ids": ids_test,
    "score": membership_scores
})

# Ensure scores are in [0, 1] and properly distributed
submission_df['score'] = submission_df['score'].clip(0.001, 0.999)  # avoid 0 and 1 exactly
print("\n[INFO] Score distribution in submission:")
print(submission_df['score'].describe())

submission_df.to_csv("test4_xgb.csv", index=False)
print("\n[INFO] Saved submission file to test4_xgb.csv")
