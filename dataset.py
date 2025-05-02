import os
import h5py
import glob
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset

def random_transform(points):
        # Zufällige Rotation um Z-Achse
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = points @ rot.T
        # Leichtes Rauschen
        points += np.random.normal(0, 0.02, points.shape)
        return points

class ModelNetDataset(Dataset):
    def __init__(self, root_dir, split="train", num_points=1024, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.files = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls, split)
            for off_file in glob.glob(os.path.join(cls_dir, '*.off')):
                self.files.append((off_file, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        mesh = trimesh.load(path)
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)

        if self.transform:
            points = self.transform(points)
        
        points = torch.tensor(points, dtype=torch.float32)
        # Normalize the points prevent division by zero
        points = points - points.min(dim=0, keepdim=True)[0]
        points = points / (points.max(dim=0, keepdim=True)[0] - points.min(dim=0, keepdim=True)[0])

        label = torch.tensor(label, dtype=torch.long)
        return points, label

