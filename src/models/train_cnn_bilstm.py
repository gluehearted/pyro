from pathlib import Path
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.models.cnn_bilstm import CNNBiLSTMFireMap


from src.utils.experiment import (
    CNN_DIR,
    CNN_MODEL_DIR,
    REPORT_DIR,
    TARGET_COL,
    EXP_NAME,
    HORIZON_DAYS,
    ensure_experiment_dirs,
    print_experiment_info,
)

MODEL_DIR = CNN_MODEL_DIR

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 3
THRESHOLD = 0.5

CNN_CHANNELS = 16
LSTM_HIDDEN = 32
DROPOUT = 0.2


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class FireMapDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = np.array(self.X[idx], dtype=np.float32, copy=True)
        y = np.array(self.y[idx], dtype=np.float32, copy=True)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y


def load_datasets():
    train_ds = FireMapDataset(
        CNN_DIR / "X_train.npy",
        CNN_DIR / "y_train.npy"
    )

    val_ds = FireMapDataset(
        CNN_DIR / "X_val.npy",
        CNN_DIR / "y_val.npy"
    )

    test_ds = FireMapDataset(
        CNN_DIR / "X_test.npy",
        CNN_DIR / "y_test.npy"
    )

    return train_ds, val_ds, test_ds


def compute_pos_weight(y_path):
    y = np.load(y_path, mmap_mode="r")

    pos = float(y.sum())
    total = float(y.size)
    neg = total - pos

    raw_weight = neg / max(pos, 1.0)

    # Biar training tidak terlalu agresif.
    pos_weight = min(raw_weight, 30.0)

    print("\n========== CLASS BALANCE ==========")
    print(f"Positive pixels : {pos:,.0f}")
    print(f"Negative pixels : {neg:,.0f}")
    print(f"Raw pos_weight  : {raw_weight:.4f}")
    print(f"Used pos_weight : {pos_weight:.4f}")

    return pos_weight


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for x, y in tqdm(loader, desc="Training"):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()

    total_loss = 0.0

    all_probs = []
    all_targets = []

    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu().numpy().reshape(-1))
        all_targets.append(y.detach().cpu().numpy().reshape(-1))

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets).astype(np.int8)

    y_pred = (y_prob >= threshold).astype(np.int8)

    metrics = {
        "loss": float(total_loss / max(len(loader), 1)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    return metrics, y_prob, y_true


@torch.no_grad()
def save_probability_maps(model, dataset, device, split_name):
    model.eval()

    probs = []

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for x, _ in tqdm(loader, desc=f"Saving {split_name} probabilities"):
        x = x.to(device)

        logits = model(x)
        prob = torch.sigmoid(logits)

        probs.append(prob.detach().cpu().numpy()[0].astype(np.float32))

    probs = np.stack(probs, axis=0)

    out_path = REPORT_DIR / f"cnn_{split_name}_prob.npy"

    np.save(out_path, probs)

    print(f"Saved probability map: {out_path}")


def main():
    ensure_experiment_dirs()
    print_experiment_info()
    device = get_device()

    print("\n========== DEVICE ==========")
    print(device)

    train_ds, val_ds, test_ds = load_datasets()

    sample_x, sample_y = train_ds[0]

    sequence_length = sample_x.shape[0]
    in_channels = sample_x.shape[1]
    height = sample_x.shape[2]
    width = sample_x.shape[3]

    print("\n========== DATA SHAPE ==========")
    print("Sequence length:", sequence_length)
    print("Channels       :", in_channels)
    print("Height         :", height)
    print("Width          :", width)
    print("Train samples  :", len(train_ds))
    print("Val samples    :", len(val_ds))
    print("Test samples   :", len(test_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    pos_weight_value = compute_pos_weight(CNN_DIR / "y_train.npy")

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    model = CNNBiLSTMFireMap(
        in_channels=in_channels,
        cnn_channels=CNN_CHANNELS,
        lstm_hidden=LSTM_HIDDEN,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    best_val_prauc = -1.0
    best_epoch = -1
    patience_counter = 0

    best_model_path = MODEL_DIR / "cnn_bilstm_best.pt"

    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{EPOCHS} ==========")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=THRESHOLD,
        )

        scheduler.step(val_metrics["pr_auc"])

        print("\nTrain loss:", train_loss)
        print("Val metrics:")
        print(json.dumps(val_metrics, indent=2))

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
        }

        history.append(record)

        if val_metrics["pr_auc"] > best_val_prauc:
            best_val_prauc = val_metrics["pr_auc"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "cnn_channels": CNN_CHANNELS,
                    "lstm_hidden": LSTM_HIDDEN,
                    "dropout": DROPOUT,
                    "epoch": epoch,
                    "best_val_prauc": best_val_prauc,
                },
                best_model_path,
            )

            print(f"Saved best model: {best_model_path}")

        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\n========== LOAD BEST MODEL ==========")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Best epoch:", best_epoch)
    print("Best val PR-AUC:", best_val_prauc)

    print("\n========== FINAL TEST EVALUATION ==========")

    test_metrics, _, _ = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=THRESHOLD,
    )

    print(json.dumps(test_metrics, indent=2))

    report = {
        "experiment": EXP_NAME,
        "target": TARGET_COL,
        "horizon_days": HORIZON_DAYS,
        "best_epoch": best_epoch,
        "best_val_prauc": best_val_prauc,
        "threshold": THRESHOLD,
        "history": history,
        "test": test_metrics,
        "model_config": {
            "sequence_length": sequence_length,
            "in_channels": in_channels,
            "height": height,
            "width": width,
            "cnn_channels": CNN_CHANNELS,
            "lstm_hidden": LSTM_HIDDEN,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "pos_weight": pos_weight_value,
        },
    }

    report_path = REPORT_DIR / "cnn_bilstm_metrics.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved report: {report_path}")

    print("\n========== SAVE PROBABILITY MAPS ==========")

    save_probability_maps(model, val_ds, device, "val")
    save_probability_maps(model, test_ds, device, "test")

    print("\n========== CNN-BiLSTM TRAINING COMPLETE ==========")


if __name__ == "__main__":
    main()