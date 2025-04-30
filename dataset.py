import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class ModelNet40Dataset(Dataset):
    def __init__(self, h5_dir, split="train"):
        files = [f for f in os.listdir(h5_dir) if f.startswith(split)]
        data, labels = [], []
        for f in files:
            with h5py.File(os.path.join(h5_dir, f), "r") as hf:
                data.append(hf['data'][:])
                labels.append(hf['label'][:])
        self.data = np.concatenate(data, axis=0)
        self.labels = np.concatenate(labels, axis=0).squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pts = self.data[idx][:1024]  # Take first 1024 points
        label = self.labels[idx]
        return torch.tensor(pts, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
