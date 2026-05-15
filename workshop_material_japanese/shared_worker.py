"""
Shared モードのワーカー: ノートブック（プライマリ）がモデルを学習している間、
AQUA データセット上でテストセット評価を行う 2 つ目のノードをシミュレートします。

使い方:
  1. ノートブック（プライマリノード）の学習セルを開始
  2. 別のターミナルで: python test.py
  3. 両方のプロセスが同じ W&B Run にロギングする様子を確認
"""
import os
import wandb
import time
import random
from dotenv import load_dotenv

# .env から WANDB_ENTITY と WANDB_PROJECT を読み込み（ノートブックと同じファイル）
load_dotenv()
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "SIE-Workshop-2026")

# ノートブック側のプライマリ Run の ID と一致している必要があります
PRIMARY_RUN_ID = input("Enter the run ID from your notebook (shown after wandb.init): ").strip()

if not PRIMARY_RUN_ID:
    print("No run ID provided. Exiting.")
    raise SystemExit

AQUA_CLASSES = [
    "Clams", "Corals", "Crabs", "Dolphin", "Eel", "Fish", "Jelly Fish",
    "Lobster", "Nudibranchs", "Octopus", "Otter", "Penguin", "Puffers",
    "Sea Rays", "Sea Urchins", "Seahorse", "Seal", "Sharks", "Shrimp",
    "Starfish",
]

run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    id=PRIMARY_RUN_ID,
    settings=wandb.Settings(
        mode="shared",
        x_label="worker_1",
        x_primary=False,
    ),
)

print(f"Worker attached to run {run.id}")
print(f"Simulating test-set evaluation on AQUA ({len(AQUA_CLASSES)} classes)\n")

# プライマリノードで学習が走っている間、定期的なテストセット評価をシミュレート
NUM_EVAL_ROUNDS = 6
for round_num in range(1, NUM_EVAL_ROUNDS + 1):
    # 評価にかかる時間をシミュレート（画像の読み込み、推論の実行）
    time.sleep(random.uniform(2.0, 4.0))

    # いくつかのランダムな種についてクラス別の精度をシミュレート
    sampled_classes = random.sample(AQUA_CLASSES, k=4)
    per_class_acc = {cls: random.uniform(30, 95) for cls in sampled_classes}

    test_loss = 2.8 * (0.72 ** round_num) + random.uniform(-0.03, 0.03)
    test_acc = min(92.0, 35.0 + 8.5 * round_num + random.uniform(-3, 3))
    num_images = random.randint(600, 656)

    print(f"Eval round {round_num}/{NUM_EVAL_ROUNDS}: "
          f"{num_images} images | loss {test_loss:.4f} | acc {test_acc:.1f}%")
    for cls, acc in per_class_acc.items():
        print(f"  {cls:<14s} {acc:.1f}%")

    run.log({
        "worker/test_loss": test_loss,
        "worker/test_accuracy": test_acc,
        "worker/images_evaluated": num_images,
        "worker/eval_round": round_num,
    })

print(f"\nWorker done. Final test accuracy: {test_acc:.1f}%")
run.finish()
