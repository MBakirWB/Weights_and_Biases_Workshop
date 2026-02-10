"""
Sweep training script for CLI-based sweeps.

This is the standalone version of the sweep_train() function from the notebook.
It's designed to be called by `wandb agent` via the sweep_config.yaml file.

Usage:
  1. Create the sweep:
     wandb sweep sweep_config.yaml

  2. Launch one agent:
     wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>

  3. (Optional) Open more terminals to run agents in parallel:
     wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>

  Each agent picks up a different config from the sweep controller automatically.
  On a multi-GPU machine, pin each agent to a GPU:
     CUDA_VISIBLE_DEVICES=0 wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
     CUDA_VISIBLE_DEVICES=1 wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import wandb
from dotenv import load_dotenv

from workshop_utils import (
    CLASS_NAMES, NUM_CLASSES, DEVICE,
    get_transforms, create_model,
    train_one_epoch, evaluate,
    AquaticDataset,
)

# ── Configuration ─────────────────────────────────────────────────────────────
# Reads WANDB_ENTITY and WANDB_PROJECT from .env (same file as the notebook)
load_dotenv()
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")

# Artifact paths for lineage tracking
ARTIFACT_PROJECT = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
TRAIN_ARTIFACT = f"{ARTIFACT_PROJECT}/aqua-train:v0"
VAL_ARTIFACT = f"{ARTIFACT_PROJECT}/aqua-val:v0"

# Local data paths (pre-loaded in your workshop environment)
LOCAL_TRAIN_DIR = "./data/train"
LOCAL_VAL_DIR = "./data/val"
LOCAL_WEIGHTS_DIR = "./pretrained_weights"


def main():
    """Training function called by the sweep agent."""
    with wandb.init() as run:
        # Mark as preemptible — auto-requeues if killed
        run.mark_preempting()

        cfg = wandb.config

        # Define custom x-axis
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")

        # Declare artifact usage for lineage
        run.use_artifact(TRAIN_ARTIFACT, type="dataset")
        run.use_artifact(VAL_ARTIFACT, type="dataset")

        # Create datasets from pre-loaded local data
        train_dataset = AquaticDataset(
            LOCAL_TRAIN_DIR,
            transform=get_transforms(cfg.image_size, is_training=True),
            class_names=CLASS_NAMES,
            max_samples=cfg.max_samples,
        )
        val_dataset = AquaticDataset(
            LOCAL_VAL_DIR,
            transform=get_transforms(cfg.image_size, is_training=False),
            class_names=CLASS_NAMES,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )

        # Resolve weights artifact dynamically based on which model the sweep picked
        weights_artifact = f"{ARTIFACT_PROJECT}/pretrained-{cfg.model_name}:latest"

        # Create model from local pretrained weights
        model = create_model(
            cfg.model_name, NUM_CLASSES, pretrained=True,
            weights_artifact=weights_artifact, run=run,
            local_weights_dir=LOCAL_WEIGHTS_DIR,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scaler = GradScaler(enabled=cfg.use_amp)

        # Training loop
        best_val_acc = 0.0
        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, DEVICE,
                epoch, log_interval=5, run=run,
            )
            val_loss, val_acc, _, _, _ = evaluate(
                model, val_loader, criterion, DEVICE, desc=f"Epoch {epoch+1}",
            )

            run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        run.summary["best_val_accuracy"] = best_val_acc
        print(f"Done — best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
