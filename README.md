# PointNet Practice Project

A lightweight PyTorch project to help you understand and implement PointNet on the ModelNet40 dataset.

## ✅ What you'll learn
- How to load and process point cloud data
- How to implement PointNet from scratch
- How to train and evaluate a 3D classification model
- How to use PyTorch features like custom Dataset, nn.Module, and mixed precision (AMP)

## 🧱 Structure
- `dataset.py` – Data loading and pre-processing
- `model.py` – PointNet model
- `train.py` – Training + evaluation loop
- `utils.py` – Logging, checkpointing

## 🚀 Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and unzip ModelNet40 (link in `dataset.py`)

3. Train:
```bash
python train.py --epochs 50 --batch_size 32
```

4. Visualize with TensorBoard:
```bash
tensorboard --logdir runs/
```
