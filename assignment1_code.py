import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd
from torchvision.models import resnet18
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ===== 1. MODEL LOADING =====
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

transform = transforms.Compose([
    transforms.Normalize(mean=mean, std=std)
])

model = resnet18(weights=None)
model.fc = torch.nn.Linear(512, 44)
ckpt = torch.load("./01_MIA.pt", map_location="cpu")
model.load_state_dict(ckpt)

# ===== 2. FEATURE EXTRACTION =====
def get_membership_features(model, dataset):
    model.eval()
    features = []
    with torch.no_grad():
        for item in dataset:
            if len(item) == 4:
                _, img, true_label, _ = item
            else:
                _, img, true_label = item

            img = img.unsqueeze(0)
            output = model(img)
            prob = torch.softmax(output, dim=1)
            prob = torch.clamp(prob, min=1e-10)

            entropy = -(prob * torch.log(prob)).sum().item()
            topk = torch.topk(prob, k=3, dim=1).values.squeeze()

            pred_label = torch.argmax(prob).item()
            is_correct = int(pred_label == true_label)

            logits = output.squeeze().numpy()
            logits_mean = logits.mean()
            logits_std = logits.std()

            features.append([
                prob.max().item(),
                entropy,
                topk[0].item(), topk[1].item(), topk[2].item(),
                (topk[0] - topk[1]).item(),
                (topk[1] - topk[2]).item(),
                logits_mean,
                logits_std,
                is_correct,
                pred_label
            ])
    return np.array(features)

# ===== 3. DATASET CLASSES =====
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
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

# ===== 4. LOAD DATA =====
public_data: MembershipDataset = torch.load("./pub.pt", weights_only=False)
private_data: MembershipDataset = torch.load("./priv_out.pt", weights_only=False)
public_data.transform = transform
private_data.transform = transform

# ===== 5. FEATURE EXTRACTION =====
public_features = get_membership_features(model, public_data)
public_labels = torch.tensor(public_data.membership).numpy()
assert not np.isnan(public_features).any(), "NaNs in public features!"
assert not np.isnan(public_labels).any(), "NaNs in public labels!"

# ===== 6. SPLIT & TRAIN ATTACK MODEL =====
X_train, X_val, y_train, y_val = train_test_split(public_features, public_labels, test_size=0.2, random_state=42)
attack_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
attack_model.fit(X_train, y_train)
print("Validation AUC:", roc_auc_score(y_val, attack_model.predict_proba(X_val)[:, 1]))

# ===== 7. PREDICT ON PRIVATE DATA =====
private_features = get_membership_features(model, private_data)
assert not np.isnan(private_features).any(), "NaNs in private features!"
scores = attack_model.predict_proba(private_features)[:, 1]

# ===== 8. SUBMISSION =====
df = pd.DataFrame({
    "ids": private_data.ids,
    "score": scores
})
df.to_csv("submission.csv", index=False)
response = requests.post(
    "http://34.122.51.94:9090/mia",
    files={"file": open("submission.csv", "rb")},
    headers={"token": "25359591"}  # Replace with your team token
)
print("Submission Results:", response.json())
