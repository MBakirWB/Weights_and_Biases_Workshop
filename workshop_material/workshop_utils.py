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
    # Classes
    "AquaticDataset",
]