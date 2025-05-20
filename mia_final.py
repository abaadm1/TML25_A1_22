import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet18
from torchvision import transforms
import torch.serialization

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import requests

# ===============================
# TRANSFORM FOR IMAGES
# ===============================
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ===============================
# MODEL SETUP
# ===============================
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)
ckpt = torch.load("./01_MIA.pt", map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

# ===============================
# DATASET CLASSES
# ===============================
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
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

# Register custom class for torch.load
torch.serialization.add_safe_globals({"MembershipDataset": MembershipDataset})

# ===============================
# LOAD DATA
# ===============================
public_data: MembershipDataset = torch.load("./pub.pt", weights_only=False)
priv_data: MembershipDataset = torch.load("./priv_out.pt", weights_only=False)
print(f"[INFO] Public dataset size: {len(public_data)}")
print(f"[INFO] Private dataset size: {len(priv_data)}")

# ===============================
# FEATURE EXTRACTION FUNCTION
# ===============================
def extract_features(model, img_tensor, label):
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        probs = F.softmax(output, dim=1).squeeze()
        confidence = torch.max(probs).item()
        loss = F.cross_entropy(output, torch.tensor([label])).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    return [confidence, loss, entropy]

# ===============================
# EXTRACT FEATURES FROM PUBLIC DATA
# ===============================
X_train, y_train, ids_train = [], [], []

for i in range(len(public_data)):
    id_, img, label, membership = public_data[i]
    features = extract_features(model, img, label)
    X_train.append(features)
    y_train.append(membership)
    ids_train.append(id_)
    if i % 100 == 0:
        print(f"[Public] Processed {i}/{len(public_data)}")

X_train = np.array(X_train)
y_train = np.array(y_train)

# Save to CSV (optional)
df_public = pd.DataFrame(X_train, columns=["confidence", "loss", "entropy"])
df_public["membership"] = y_train
df_public["id"] = ids_train
df_public.to_csv("public_features.csv", index=False)
print("[INFO] Saved public features to public_features.csv")

# ===============================
# TRAIN ATTACK MODEL
# ===============================
X = df_public[["confidence", "loss", "entropy"]].values
y = df_public["membership"].values

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = LogisticRegression()
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_val)
y_score = clf.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_score)
print(f"[RESULT] Validation Accuracy: {acc:.4f}")
print(f"[RESULT] Validation AUC: {auc:.4f}")

# Save model
joblib.dump(clf, "attack_model.pkl")
print("[INFO] Saved attack model to attack_model.pkl")

# ===============================
# TEST ON PRIVATE DATA (NO GROUND TRUTH)
# ===============================
X_test, ids_test = [], []

for i in range(len(priv_data)):
    id_, img, label, _ = priv_data[i]
    features = extract_features(model, img, label)
    X_test.append(features)
    ids_test.append(id_)
    if i % 100 == 0:
        print(f"[Private] Processed {i}/{len(priv_data)}")

X_test = np.array(X_test)
y_score = clf.predict_proba(X_test)[:, 1]

print("[INFO] Skipped accuracy/AUC computation — private dataset has no ground-truth membership labels.")

# ===============================
# SUBMISSION
# ===============================
submission_df = pd.DataFrame({
    "ids": ids_test,
    "score": y_score
})
submission_df.to_csv("test.csv", index=False)
print("[INFO] Saved submission file to test.csv")

# # Submit
# response = requests.post(
#     "http://34.122.51.94:9090/mia",
#     files={"file": open("test.csv", "rb")},
#     headers={"token": "12910150"}
# )
# print("[INFO] Submission response:", response.json())
