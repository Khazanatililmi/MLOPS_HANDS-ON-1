import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("breast-cancer-clf")

cfg = {"lr": 3e-4, "batch_size": 64, "epochs": 20, "arch": "resnet18", "seed": 42}

X_dummy = torch.randn(200, 10)
y_dummy = torch.randint(0, 2, (200,)).float()
dataset = TensorDataset(X_dummy, y_dummy)
train_loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
val_loader = DataLoader(dataset, batch_size=cfg["batch_size"])

model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
criterion = nn.BCELoss()

with mlflow.start_run(run_name=f"{cfg['arch']}-lr{cfg['lr']}"):
    mlflow.log_params(cfg)
    mlflow.log_param("data_md5", "dummy_hash_12345")
    
    if os.path.exists("requirements.txt"):
        mlflow.log_artifact("requirements.txt")
    
    best_f1 = 0.0
    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx).squeeze()
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx).squeeze()
                loss = criterion(out, by)
                val_loss += loss.item()
                preds = (out > 0.5).float()
                correct += (preds == by).sum().item()
                total += len(by)
        val_loss /= len(val_loader)
        val_f1 = correct / total
        
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "lr": cfg["lr"]
        }, step=epoch)
        
        if val_f1 >= best_f1:
            best_f1 = val_f1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best.pt")
            mlflow.log_artifact("checkpoints/best.pt", artifact_path="checkpoints")
            
    mlflow.log_metric("best_val_f1", best_f1)
    mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
    print(f"[02_pytorch_logging] PyTorch run finished successfully. best_val_f1: {best_f1:.4f}")
