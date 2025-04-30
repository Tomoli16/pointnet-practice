import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import PointNet
from dataset import ModelNet40Dataset
from utils import accuracy
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--data_dir', type=str, default="./modelnet40_h5")
args = parser.parse_args()

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = ModelNet40Dataset(args.data_dir, split="train")
test_set = ModelNet40Dataset(args.data_dir, split="test")
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size)

model = PointNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for pts, lbls in train_loader:
        pts, lbls = pts.to(device), lbls.to(device)
        out = model(pts)
        loss = criterion(out, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    writer.add_scalar("Loss/train", total_loss/len(train_loader), epoch)

    model.eval()
    acc = 0
    with torch.no_grad():
        for pts, lbls in test_loader:
            pts, lbls = pts.to(device), lbls.to(device)
            pred = model(pts).argmax(dim=1)
            acc += accuracy(pred, lbls)
    acc /= len(test_loader)
    writer.add_scalar("Accuracy/test", acc, epoch)
    print(f"[Epoch {epoch}] Test Acc: {acc:.4f}")
