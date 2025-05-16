import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd
import torch.serialization
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms

#### TRANSFORM FOR IMAGES
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

#### LOAD MODEL
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)
ckpt = torch.load("./01_MIA.pt", map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

#### DATASETS

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

#### LOAD DATASETS
torch.serialization.add_safe_globals({"MembershipDataset": MembershipDataset})

data: MembershipDataset = torch.load("./priv_out.pt", weights_only=False)
public_data: MembershipDataset = torch.load("./pub.pt", weights_only=False)
print("Loaded public dataset with", len(public_data), "samples.")

#### FEATURE EXTRACTION FUNCTION
def extract_features(model, img_tensor, label):
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        probs = F.softmax(output, dim=1).squeeze()
        confidence = torch.max(probs).item()
        loss = F.cross_entropy(output, torch.tensor([label])).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    return [confidence, loss, entropy]

#### EXTRACT FEATURES FROM PUBLIC DATASET
X_train = []
y_train = []
ids = []

for i in range(len(public_data)):
    id_, img, label, membership = public_data[i]
    features = extract_features(model, img, label)
    X_train.append(features)
    y_train.append(membership)
    ids.append(id_)

    if i % 100 == 0:
        print(f"[INFO] Processed sample {i}/{len(public_data)} - ID: {id_}, Label: {label}, Membership: {membership}")

X_train = np.array(X_train)
y_train = np.array(y_train)

print("Finished extracting features.")
print("Feature shape:", X_train.shape)
print("First sample features:", X_train[0])
print("First sample membership:", y_train[0])

#### SAVE FEATURES TO CSV
df_features = pd.DataFrame(X_train, columns=["confidence", "loss", "entropy"])
df_features["membership"] = y_train
df_features["id"] = ids
df_features.to_csv("public_features.csv", index=False)
print("Saved extracted features to public_features.csv")
