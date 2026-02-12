"""
Workshop Utilities - AQUA Marine Species Classification

This module contains all the ML boilerplate code for the workshop.
Participants can focus on W&B concepts while these utilities handle:
- Data loading and transforms
- Model creation
- Training and evaluation loops

Usage:
    from workshop_utils import *
"""

import os
import random
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
import timm
import wandb

warnings.filterwarnings('ignore')


# AQUA class names (20 marine species)
CLASS_NAMES = [
    "coral", "crab", "diver", "eel", "fish", "fishInGroups", "flatworm", 
    "jellyfish", "marine_dolphin", "octopus", "rayfish", "seaAnemone", 
    "seaCucumber", "seaSlug", "seaUrchin", "shark", "shrimp", "squid", 
    "starfish", "turtle"
]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet normalization (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# DEVICE SETUP
def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


# REPRODUCIBILITY
def set_seed(seed: int = 42, deterministic: bool = False):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# DATA TRANSFORMS
def get_transforms(image_size: int = 224, is_training: bool = True):
    """Get image transforms optimized for underwater imagery."""
    if is_training:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


# DATASET CLASS
class AquaticDataset(Dataset):
    """PyTorch Dataset for images organized by class folders."""
    
    def __init__(self, root_dir: str, transform=None, class_names: List[str] = None, 
                 max_samples: int = None):
        """
        Args:
            root_dir: Path to directory with class subfolders
            transform: Torchvision transforms to apply
            class_names: List of class names (uses CLASS_NAMES if None)
            max_samples: Limit samples for fast training (None = use all)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        self.samples = []  # List of (image_path, class_idx)
        
        # Collect all image files
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_dir, img_name), class_idx))
        
        # Optionally limit samples for fast training
        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_raw_image(self, idx):
        """Get raw PIL image without transforms (for visualization)."""
        img_path, label = self.samples[idx]
        return Image.open(img_path).convert("RGB"), label


# MODEL CREATION
def create_model(model_name: str = "resnet50", num_classes: int = NUM_CLASSES, 
                 pretrained: bool = True, weights_artifact: str = None, run=None,
                 local_weights_dir: str = None):
    """Create a model using timm. Optionally load weights from a W&B artifact.
    
    Args:
        model_name: timm model name (e.g. 'resnet50', 'efficientnet_b0')
        num_classes: Number of output classes
        pretrained: If True and no weights_artifact, downloads from HuggingFace
        weights_artifact: W&B artifact path for pretrained weights (air-gapped mode)
        run: Active wandb run (required if weights_artifact is set, for lineage)
        local_weights_dir: Path to local directory with pretrained weights.
            When set, weights are loaded from disk and use_artifact() is called
            purely for lineage tracking (no download).
    """
    if weights_artifact and run:
        # Track lineage in W&B
        run.use_artifact(weights_artifact, type="pretrained-weights")

        # Resolve weights path: local directory or W&B artifact download
        if local_weights_dir:
            weights_path = os.path.join(local_weights_dir, f"{model_name}_imagenet.pth")
        else:
            artifact = run.use_artifact(weights_artifact, type="pretrained-weights")
            weights_path = os.path.join(artifact.download(), f"{model_name}_imagenet.pth")

        model = timm.create_model(model_name, pretrained=False, num_classes=1000)
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        # Replace classifier head for our num_classes
        if hasattr(model, 'head') and hasattr(model.head, 'in_features'):
            model.head = nn.Linear(model.head.in_features, num_classes)
        elif hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        # Normal path: timm downloads from HuggingFace
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# TRAINING UTILITIES
def train_one_epoch(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    scaler, 
    device,
    epoch: int,
    use_amp: bool = True,
    log_interval: int = 1,
    run=None
) -> Tuple[float, float]:
    """Train for one epoch with mixed precision and W&B logging."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": running_loss / (batch_idx + 1),
            "acc": 100. * correct / total
        })
        
        # Log to W&B
        if run and (batch_idx + 1) % log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            run.log({
                "train/global_step": global_step,
                "train/loss_step": loss.item(),
                "train/acc_step": 100. * correct / total,
            })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model, 
    data_loader, 
    criterion, 
    device,
    use_amp: bool = True,
    desc: str = "Eval"
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model and return metrics + predictions."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(data_loader, desc=desc)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        pbar.set_postfix({"loss": running_loss / (len(all_preds) // data_loader.batch_size + 1)})

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# W&B HELPERS
def generate_run_name(config: Dict) -> str:
    """Generate a descriptive run name from config.
    
    Format: {user}-{model}-lr{learning_rate}-bs{batch_size}-ep{epochs}
    Example: alice-resnet50-lr1e-3-bs32-ep3
    
    If config["user_name"] is set, it's prepended for easy identification
    in shared workshop projects.
    """
    user = config.get("user_name", "")
    model = config.get("model_name", "model")
    lr = config.get("learning_rate", 1e-3)
    bs = config.get("batch_size", 32)
    ep = config.get("epochs", 3)
    
    # Format learning rate nicely (1e-3 -> 1e-3, 0.001 -> 1e-3)
    lr_str = f"{lr:.0e}".replace("-0", "-")
    
    prefix = f"{user}-" if user else ""
    return f"{prefix}{model}-lr{lr_str}-bs{bs}-ep{ep}"
# VISUALIZATION HELPERS
def create_prediction_images(dataset, preds, probs, class_names, n_samples=16, image_size=224):
    """
    Create a list of wandb.Image objects with prediction captions.
    
    Args:
        dataset: Dataset with get_raw_image() method
        preds: Predicted class indices
        probs: Prediction probabilities (n_samples x n_classes)
        class_names: List of class names
        n_samples: Number of samples to create
        image_size: Resize images to this size for consistent display
        
    Returns:
        List of wandb.Image objects ready for logging
    """
    images_to_log = []
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    for idx in indices:
        image, true_label = dataset.get_raw_image(idx)
        pred_label = preds[idx]
        confidence = probs[idx][pred_label] * 100

        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)

        is_correct = "✓" if pred_label == true_label else "✗"
        caption = f"{is_correct} True: {true_name} | Pred: {pred_name} ({confidence:.1f}%)"

        # Resize for consistent display in W&B UI
        image = image.resize((image_size, image_size))
        images_to_log.append(wandb.Image(image, caption=caption))

    return images_to_log


def create_predictions_table(dataset, preds, probs, class_names, n_samples=100, image_size=224):
    """
    Create a W&B Table with predictions for detailed analysis.
    
    Features in W&B UI:
    - Group by 'truth' to see recall (false negatives per class)
    - Group by 'guess' to see precision (false positives per class)  
    - Filter by row['truth'] != row['guess'] to find all errors
    - Sort by score columns to find high-confidence mistakes
    
    Args:
        dataset: Dataset with get_raw_image() method
        preds: Predicted class indices
        probs: Prediction probabilities (n_samples x n_classes)
        class_names: List of class names
        n_samples: Number of samples to include
        image_size: Resize images to this size
        
    Returns:
        wandb.Table ready for logging
    """
    # Build columns: image, prediction info, then score for each class
    columns = ["image", "id", "guess", "truth", "correct"]
    columns.extend([f"score_{name}" for name in class_names])
    
    table = wandb.Table(columns=columns)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    for idx in indices:
        image, true_label = dataset.get_raw_image(idx)
        pred_label = preds[idx]
        is_correct = pred_label == true_label

        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)

        # Resize for consistent display
        image = image.resize((image_size, image_size))
        
        # Build row data
        row_data = [
            wandb.Image(image),
            str(idx),  # Unique ID for joining tables across runs
            pred_name,
            true_name,
            is_correct,
        ]
        
        # Add confidence scores for each class (enables histogram visualization)
        for class_idx in range(len(class_names)):
            score = float(probs[idx][class_idx]) if class_idx < len(probs[idx]) else 0.0
            row_data.append(round(score, 4))
        
        table.add_data(*row_data)

    return table
# NOTEBOOK HELPERS  (keep W&B calls visible, hide PyTorch boilerplate)
def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from already-downloaded artifact dirs.

    Call *after* ``run.use_artifact().download()`` so lineage stays visible
    in the notebook.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 32)
    max_samples = config.get("max_samples")

    train_dataset = AquaticDataset(
        train_dir,
        transform=get_transforms(image_size, is_training=True),
        class_names=CLASS_NAMES,
        max_samples=max_samples,
    )
    val_dataset = AquaticDataset(
        val_dir,
        transform=get_transforms(image_size, is_training=False),
        class_names=CLASS_NAMES,
    )
    test_dataset = AquaticDataset(
        test_dir,
        transform=get_transforms(image_size, is_training=False),
        class_names=CLASS_NAMES,
    )

    loader_kwargs = dict(num_workers=0, pin_memory=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )

    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def create_training_components(model, config: Dict):
    """Create criterion, optimizer, scheduler, and scaler from config.

    Returns:
        (criterion, optimizer, scheduler, scaler)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("epochs", 3),
    )
    scaler = GradScaler(enabled=config.get("use_amp", True))
    return criterion, optimizer, scheduler, scaler


def save_checkpoint(
    model, optimizer, config: Dict,
    epoch: int, val_acc: float, val_loss: float,
    path: str,
):
    """Save a PyTorch training checkpoint to *path*."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "config": config,
    }, path)


def log_checkpoint_artifact(
    run,
    model,
    optimizer,
    config: Dict,
    epoch: int,
    metrics: Dict,
    is_best: bool = False,
    is_last: bool = False,
    ttl_days: int = 7,
):
    """Create a versioned W&B Artifact for a model checkpoint.

    Sets a TTL so intermediate checkpoints are auto-cleaned, and applies
    ``best`` / ``latest`` aliases as appropriate.
    """
    from datetime import timedelta

    user_name = config.get("user_name", "")
    name_suffix = f"-{user_name}" if user_name else ""
    artifact = wandb.Artifact(
        name=f"model{name_suffix}-{config.get('model_name', 'model')}",
        type="model",
        metadata={**metrics, "epoch": epoch},
    )
    artifact.ttl = timedelta(days=ttl_days)

    # Save checkpoint locally, then add as a REFERENCE artifact
    # (logs metadata + checksum only, no upload — keeps workshop fast)
    ckpt_path = f"checkpoint_epoch{epoch}.pth"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, ckpt_path)
    artifact.add_reference(f"file://{os.path.abspath(ckpt_path)}")

    aliases = [f"epoch_{epoch}"]
    if is_best:
        aliases.append("best")
    if is_last:
        aliases.append("latest")

    run.log_artifact(artifact, aliases=aliases)
    print(f"  Logged checkpoint artifact (aliases: {aliases}, TTL: {ttl_days}d)")



# SWEEP HELPERS
def create_sweep_components(cfg, run, weights_artifact: str, local_weights_dir: str):
    """Create datasets, dataloaders, model, and optimizer for a sweep run.

    Reads hyperparameters from ``cfg`` (a ``wandb.config`` object) and
    returns everything needed for the training loop.

    Returns:
        (model, train_loader, val_loader, criterion, optimizer, scaler)
    """
    train_dataset = AquaticDataset(
        "./data/train",
        transform=get_transforms(cfg.image_size, is_training=True),
        class_names=CLASS_NAMES,
        max_samples=cfg.max_samples,
    )
    val_dataset = AquaticDataset(
        "./data/val",
        transform=get_transforms(cfg.image_size, is_training=False),
        class_names=CLASS_NAMES,
    )

    loader_kwargs = dict(num_workers=0, pin_memory=True)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, **loader_kwargs,
    )

    model = create_model(
        cfg.model_name, NUM_CLASSES, pretrained=True,
        weights_artifact=weights_artifact, run=run,
        local_weights_dir=local_weights_dir,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler(enabled=cfg.use_amp)

    return model, train_loader, val_loader, criterion, optimizer, scaler


# MODEL ARTIFACT HELPERS
def prepare_model_files(
    config: Dict,
    best_model_path: str,
    best_val_acc: float,
    test_acc: float,
    test_loss: float,
    f1_macro: float,
    total_params: int,
    trainable_params: int,
    train_artifact_ref,
    val_artifact_ref,
    run_id: str,
    output_dir: str = "artifacts",
) -> Dict:
    """Prepare model files and return artifact metadata.

    Creates ``output_dir/model.pth`` (copy of best checkpoint) and
    ``output_dir/model_info.json`` (full model card). Returns a flat
    metadata dict suitable for ``wandb.Artifact(metadata=...)``.
    """
    import json
    import shutil
    from datetime import datetime

    train_ref = f"{train_artifact_ref.name}:{train_artifact_ref.version}"
    val_ref = f"{val_artifact_ref.name}:{val_artifact_ref.version}"

    model_info = {
        "architecture": config["model_name"],
        "num_classes": config.get("num_classes", NUM_CLASSES),
        "class_names": CLASS_NAMES,
        "input_size": config["image_size"],
        "pretrained": True,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_config": {
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "optimizer": "AdamW",
        },
        "metrics": {
            "best_val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "f1_macro": f1_macro,
        },
        "data_artifacts": {"train": train_ref, "val": val_ref},
        "created_at": datetime.now().isoformat(),
        "training_run_id": run_id,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    shutil.copy(best_model_path, os.path.join(output_dir, "model.pth"))

    # Flat metadata for the artifact (what shows up in the W&B UI)
    return {
        "architecture": config["model_name"],
        "num_classes": config.get("num_classes", NUM_CLASSES),
        "input_size": config["image_size"],
        "final_val_accuracy": best_val_acc,
        "final_test_accuracy": test_acc,
        "f1_macro": f1_macro,
        "framework": "pytorch",
        "domain": "marine_biology",
        "train_artifact": train_ref,
        "val_artifact": val_ref,
    }


# EVALUATION HELPERS
def compute_and_log_class_metrics(
    run,
    test_labels: np.ndarray,
    test_preds: np.ndarray,
    class_names: List[str],
) -> float:
    """Compute per-class precision/recall/F1 and log to W&B.

    Logs a ``evaluation/per_class_metrics`` table and updates
    ``run.summary`` with macro-averaged precision, recall, and F1.

    Returns:
        f1_macro: The macro-averaged F1 score.
    """
    from sklearn.metrics import precision_recall_fscore_support

    unique_classes = sorted(set(test_labels) | set(test_preds))

    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, labels=unique_classes, average=None, zero_division=0
    )

    table = wandb.Table(columns=["Class", "Precision", "Recall", "F1-Score", "Support"])
    for i, class_idx in enumerate(unique_classes):
        name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
        table.add_data(name, round(precision[i], 4), round(recall[i], 4),
                       round(f1[i], 4), int(support[i]))

    run.log({"evaluation/per_class_metrics": table})

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="macro"
    )
    run.summary["precision_macro"] = precision_macro
    run.summary["recall_macro"] = recall_macro
    run.summary["f1_macro"] = f1_macro

    print(f"Per-class metrics logged. Macro F1: {f1_macro:.4f}")
    return f1_macro


# REGISTRY PROMOTION HELPERS
def promote_sweep_winner(
    user_name: str,
    config: Dict,
    entity: str,
    project: str,
    registry_name: str,
    registry_path: str,
    sweep_id: str,
    best_sweep_run,
    best_sweep_acc: float,
    baseline_acc: float,
    staged_artifact=None,
):
    """Log sweep winner as a new artifact version and link to the registry.

    Creates a placeholder weights file (in production you'd save real
    weights during the sweep) so users see v0 → v1 in the Registry.
    """
    promotion_run = wandb.init(
        entity=entity, project=project,
        name=f"promote-sweep-winner-{user_name}",
        job_type="registry-promotion",
        group=user_name,
        tags=[user_name, "registry", "promotion", "sweep-winner"],
    )

    placeholder_path = f"sweep_winner_{user_name}.pth"
    torch.save({"placeholder": True, "sweep_run_id": best_sweep_run.id}, placeholder_path)

    artifact = wandb.Artifact(
        name=f"aqua-species-classifier-{user_name}",
        type="model",
        description=f"Sweep winner — beat baseline {baseline_acc:.2f}% with {best_sweep_acc:.2f}%",
        metadata={
            "source": "sweep",
            "val_accuracy": best_sweep_acc,
            "baseline_accuracy": baseline_acc,
            "sweep_id": sweep_id,
            "sweep_run_id": best_sweep_run.id,
            "learning_rate": best_sweep_run.config.get("learning_rate"),
            "batch_size": best_sweep_run.config.get("batch_size"),
            "weight_decay": best_sweep_run.config.get("weight_decay"),
            "weights": "placeholder",
        },
    )
    artifact.add_file(placeholder_path)
    promotion_run.log_artifact(artifact)

    promotion_run.link_artifact(
        artifact=artifact,
        target_path=f"{registry_path}/{registry_name}",
        aliases=["production"],
    )
    wandb.finish()
    os.remove(placeholder_path)

    # Clean up staging alias on the old baseline
    if staged_artifact and "staging" in staged_artifact.aliases:
        staged_artifact.aliases.remove("staging")
        staged_artifact.save()

    print(f"  New version logged: aqua-species-classifier-{user_name}")
    print(f"  Linked to {registry_name} with 'production' alias")
    print(f"  → Registry: v0 = baseline (staging), v1 = sweep winner (production)")


def promote_baseline(staged_artifact):
    """Promote the existing baseline from staging to production."""
    if "staging" in staged_artifact.aliases:
        staged_artifact.aliases.remove("staging")
    if "production" not in staged_artifact.aliases:
        staged_artifact.aliases.append("production")
    staged_artifact.save()
    print(f"  Baseline promoted to production. Aliases: {staged_artifact.aliases}")


# EXPORTS
__all__ = [
    # Constants
    "CLASS_NAMES", "NUM_CLASSES", "DEVICE",
    # Functions
    "set_seed", "get_device", "get_transforms",
    "create_model", "count_parameters",
    "train_one_epoch", "evaluate",
    "generate_run_name",
    # Notebook helpers (keep W&B visible, hide boilerplate)
    "create_dataloaders", "create_training_components",
    "save_checkpoint", "log_checkpoint_artifact",
    # Visualization helpers
    "create_prediction_images", "create_predictions_table",
    # Sweep helpers
    "create_sweep_components",
    # Model artifact helpers
    "prepare_model_files",
    # Evaluation helpers
    "compute_and_log_class_metrics",
    # Registry helpers
    "promote_sweep_winner", "promote_baseline",
    # Classes
    "AquaticDataset",
]