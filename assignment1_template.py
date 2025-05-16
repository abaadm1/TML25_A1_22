import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd
import torch.serialization

#### LOADING THE MODEL

from torchvision.models import resnet18

### Add this as a transofrmation to pre-process the images
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("./01_MIA.pt", map_location="cpu")

model.load_state_dict(ckpt)

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
        if not self.transform is None:
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


torch.serialization.add_safe_globals({"MembershipDataset": MembershipDataset})
data: MembershipDataset = torch.load("./priv_out.pt", weights_only=False)

    
#### EXAMPLE SUBMISSION

df = pd.DataFrame(
    {
        "ids": data.ids,
        "score": np.random.randn(len(data.ids)),
    }
)
df.to_csv("test.csv", index=None)
# response = requests.post("http://34.122.51.94:9090/mia", files={"file": open("test.csv", "rb")}, headers={"token": "TOKEN"})
# print(response.json())


# print("Number of samples:", len(data))

# id_, img, label, _ = data[0]
# print("ID:", id_)
# print("Image shape:", img.shape)
# print("Label:", label)

# for i in range(3):
#     id_, img, label, _ = data[i]
#     print(f"Sample {i} → ID: {id_}, Label: {label}, Image Shape: {img.shape}")
