"""
CLI ベースの Sweep 用の学習スクリプト。

これはノートブック内の sweep_train() 関数をスタンドアロン化したものです。
sweep_config.yaml を介して `wandb agent` から呼び出されることを想定しています。

使い方:
  1. Sweep を作成:
     wandb sweep sweep_config.yaml

  2. agent を 1 つ起動:
     wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>

  3. （オプション）別のターミナルを開いて agent を並列実行:
     wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>

  各 agent は Sweep controller から自動的に異なる config を受け取ります。
  マルチ GPU マシンでは、各 agent を GPU にピン留めしてください:
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

# ── 設定 ─────────────────────────────────────────────────────────────
# .env から WANDB_ENTITY と WANDB_PROJECT を読み込み（ノートブックと同じファイル）
load_dotenv()
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")

YOUR_NAME = os.environ.get("YOUR_NAME")

# リネージ追跡のための Artifact のパス
ARTIFACT_PROJECT = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
TRAIN_ARTIFACT = f"{ARTIFACT_PROJECT}/aqua-train:v0"
VAL_ARTIFACT = f"{ARTIFACT_PROJECT}/aqua-val:v0"

# ローカルデータのパス（ワークショップ環境にプリロード済み）
LOCAL_TRAIN_DIR = "./data/train"
LOCAL_VAL_DIR = "./data/val"
LOCAL_WEIGHTS_DIR = "./pretrained_weights"


def main():
    """Sweep agent が呼び出す学習関数。"""
    init_kwargs = {}
    if YOUR_NAME:
        init_kwargs["group"] = YOUR_NAME
        init_kwargs["tags"] = [YOUR_NAME, "aqua", "sweep"]
    with wandb.init(**init_kwargs) as run:
        # プリエンプティブルとしてマーク — 中断時に自動的に再キュー
        run.mark_preempting()

        cfg = wandb.config

        # カスタム X 軸を定義
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")

        # リネージのために Artifact の使用を宣言
        run.use_artifact(TRAIN_ARTIFACT, type="dataset")
        run.use_artifact(VAL_ARTIFACT, type="dataset")

        # プリロード済みのローカルデータからデータセットを作成
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

        # Sweep が選んだモデルに応じて、重み Artifact を動的に解決
        weights_artifact = f"{ARTIFACT_PROJECT}/pretrained-{cfg.model_name}:latest"

        # ローカルの事前学習済み重みからモデルを作成
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

        # 学習ループ
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
