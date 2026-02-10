"""
Shared-mode worker: simulates a second node running test-set evaluation
on the AQUA dataset while the primary notebook trains the model.

Usage:
  1. Start the training cell in the notebook (primary node)
  2. In a separate terminal: python test.py
  3. Watch both processes log to the same run in W&B
"""
import os
import wandb
import time
import random
from dotenv import load_dotenv

# Load WANDB_ENTITY and WANDB_PROJECT from .env (same file as the notebook)
load_dotenv()
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "SIE-Workshop-2026")

# Must match the primary run's ID from the notebook
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

# Simulate periodic test-set evaluation while training runs on the primary node
NUM_EVAL_ROUNDS = 6
for round_num in range(1, NUM_EVAL_ROUNDS + 1):
    # Simulate evaluation time (loading images, running inference)
    time.sleep(random.uniform(2.0, 4.0))

    # Simulate per-class accuracy on a few random species
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
