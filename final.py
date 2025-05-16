import torch
from torch.utils.data import Dataset
from typing import Tuple
import pandas as pd
import numpy as np
import torch.serialization
import torch.nn.functional as F
import joblib
from torchvision.models import resnet18
from torchvision import transforms

#### CONFIGURATION
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

#### MODEL SETUP
resnet = resnet18(pretrained=False)
resnet.fc = torch.nn.Linear(512, 44)
ckpt = torch.load("01_MIA.pt", map_location="cpu")
resnet.load_state_dict(ckpt)
resnet.eval()

#### DATASET CLASSES

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
        return id_, img, label, self.membership[index]  # membership will be None for private

#### LOAD PRIVATE DATASET
torch.serialization.add_safe_globals({"MembershipDataset": MembershipDataset})
private_data: MembershipDataset = torch.load("priv_out.pt", weights_only=False)
print("Loaded private dataset with", len(private_data), "samples.")

#### FEATURE EXTRACTION FUNCTION
def extract_features(model, img_tensor, label):
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        probs = F.softmax(output, dim=1).squeeze()
        confidence = torch.max(probs).item()
        loss = F.cross_entropy(output, torch.tensor([label])).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    return [confidence, loss, entropy]

#### LOAD ATTACK MODEL
clf = joblib.load("attack_model.pkl")
print("Loaded attack model.")

#### EXTRACT FEATURES AND PREDICT
ids = []
features = []

for i in range(len(private_data)):
    id_, img, label, _ = private_data[i]
    feat = extract_features(resnet, img, label)
    features.append(feat)
    ids.append(id_)
    
    if i % 100 == 0:
        print(f"[INFO] Processed {i}/{len(private_data)}")

X_priv = np.array(features)
scores = clf.predict_proba(X_priv)[:, 1]  # membership confidence

#### SAVE TO CSV FOR SUBMISSION
df = pd.DataFrame({
    "ids": ids,
    "score": scores
})
df.to_csv("test.csv", index=False)
print("Saved predictions to test.csv")
