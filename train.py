import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import ModelNetDataset
from utils import accuracy
import argparse
import os
from my_pointnet import SimplePointNet
from dataset import random_transform

import wandb



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--data_dir', type=str, default="./data/ModelNet40")
parser.add_argument('--num_classes', type=int, default=10)
args = parser.parse_args()

wandb.init(
    project="pointnet-modelnet10",
    config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_classes": args.num_classes,
        "lr": 1e-3,
        "dataset": "ModelNet10"
    }
)

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = ModelNetDataset(args.data_dir, split="train", transform=random_transform)
test_set = ModelNetDataset(args.data_dir, split="test", transform=random_transform)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size)


model = SimplePointNet(num_classes=args.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

print("Using device:", device)

def debug_batch(pts, lbls, out, pred, num_classes):
    import torch

    # Geräte check
    if pts.device != torch.device("cpu"):
        pts, lbls, out, pred = pts.cpu(), lbls.cpu(), out.cpu(), pred.cpu()

    # Label-Werte prüfen
    print("\n🔎 Labels im Batch:", torch.unique(lbls).tolist())
    if lbls.min() < 0 or lbls.max() >= num_classes:
        print("❌ Fehler: Ungültige Label-Werte!")
        print("   Min:", lbls.min().item(), "Max:", lbls.max().item())
        exit()

    # Prediction-Verteilung
    pred_classes = torch.argmax(out, dim=1)
    print("🧠 Prediction class distribution:", torch.bincount(pred_classes))

    # Beispieldruck
    print("✅ Beispiel Vorhersagen:", pred_classes[:5].tolist())
    print("🎯 Ground Truth:", lbls[:5].tolist())

    # Logit-Check
    print("📊 Logits (erster Punkt):", out[0].tolist())
    print("🔁 Softmax-Summe:", torch.softmax(out[0], dim=0).sum().item())

    # Unterschiedliche Klassen im Output?
    if len(torch.unique(pred_classes)) < 2:
        print("⚠️ Warnung: Modell prediktet nur eine Klasse in diesem Batch.")



for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}/{args.epochs}")
    model.train()
    total_loss = 0
    for i, (pts, lbls) in enumerate(train_loader):
        print(f"Batch {i+1}/{len(train_loader)} - Loss: {total_loss/((i+1)*args.batch_size):.4f}", end="\r")
        pts, lbls = pts.to(device), lbls.to(device)
        out = model(pts)
        loss = criterion(out, lbls)
        optimizer.zero_grad()

        # Debugging every 200 batches
        if i % 200 == 0:
            print("Debugging batch...")
            debug_batch(pts, lbls, out, out.argmax(dim=1), args.num_classes)
            print("Debugging batch done.")
         

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        

        if torch.isnan(loss):
            print("NaN detected!")
            print("Logits:", out[0])
            print("Labels:", lbls[0])
            break
    wandb.log({"train_loss": total_loss / len(train_loader)}, step=epoch)
    writer.add_scalar("Loss/train", total_loss/len(train_loader), epoch)

    model.eval()
    acc = 0
    with torch.no_grad():
        print("Evaluating on test set...")
        for pts, lbls in test_loader:
            pts, lbls = pts.to(device), lbls.to(device)
            pred = model(pts).argmax(dim=1)
            acc += accuracy(pred, lbls)
    acc /= len(test_loader)
    writer.add_scalar("Accuracy/test", acc, epoch)
    print(f"[Epoch {epoch}] Test Acc: {acc:.4f}")
    wandb.log({"test_accuracy": acc}, step=epoch)


