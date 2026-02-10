#!/usr/bin/env python3
"""
Upload AQUA Dataset to W&B and Link to Organization Registry

AQUA Dataset
==============
The AQUA dataset is a comprehensive benchmark dataset designed for underwater 
species classification under challenging real-world conditions. It comprises 
8,171 underwater images across 20 distinct marine species, specifically curated 
to reflect environmental complexities such as turbidity, low illumination, and 
occlusion, which commonly degrade the performance of standard vision systems. 
This dataset provides a valuable resource for advancing robust visual recognition 
in aquatic environments.

Reference:
    AQUA: A Benchmark Dataset for Underwater Species Classification 
    under Challenging Conditions

Species (20 classes):
    coral, crab, diver, eel, fish, fishInGroups, flatworm, jellyfish,
    marine_dolphin, octopus, rayfish, seaAnemone, seaCucumber, seaSlug,
    seaUrchin, shark, shrimp, squid, starfish, turtle

Source: https://huggingface.co/datasets/taufiktrf/AQUA20
"""

import os
import json
import shutil
import random
import argparse
from datetime import datetime
from PIL import Image

import numpy as np
import torch
import timm
import wandb
from wandb.automations import OnAddArtifactAlias, ArtifactEvent, SendWebhook
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION - Set these values before running
# ============================================================================

# Set via --entity and --project CLI args
WANDB_ENTITY = None
WANDB_PROJECT = "SIE-Workshop-2026"

#Uploads
DATASET_NAME = "taufiktrf/aqua20"
ARTIFACT_NAME = "aqua-raw-dataset"
ARTIFACT_TYPE = "dataset"

REGISTRY_TYPE = "dataset"
COLLECTION_NAME = "Aqua-raw-dataset"

# Model Registry automation config
MODEL_REGISTRY_COLLECTION = "aqua-classifier"
AUTOMATION_NAME = "github-model-cicd"
AUTOMATION_WEBHOOK_NAME = "github-model-cicd"

DATA_DIR = "aqua_raw_data"
SPLITS_DIR = "aqua_splits"

# Pretrained model weights to upload (for air-gapped environments)
WORKSHOP_MODELS = ["resnet50", "efficientnet_b0"]

# Split configuration
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

# EDA Table configuration
EDA_SAMPLES_PER_CLASS = 25  # Stratified sample size per class for EDA table

# Dataset description for artifact metadata
DATASET_DESCRIPTION = """The AQUA dataset is a comprehensive benchmark dataset designed for underwater species classification under challenging real-world conditions. It comprises 8,171 underwater images across 20 distinct marine species, specifically curated to reflect environmental complexities such as turbidity, low illumination, and occlusion, which commonly degrade the performance of standard vision systems. This dataset provides a valuable resource for advancing robust visual recognition in aquatic environments.

Reference: AQUA: A Benchmark Dataset for Underwater Species Classification under Challenging Conditions"""


def create_split_indices(image_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Create train/val/test split indices from a list of image files."""
    random.seed(seed)
    
    # Split indices
    indices = list(range(len(image_files)))
    
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio / (train_ratio + val_ratio), random_state=seed
    )
    
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }


def get_all_image_files(data_dir, class_names):
    """Get all image files from the data directory, organized by class."""
    image_files = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append({
                        "path": os.path.join(class_dir, filename),
                        "class_name": class_name,
                        "filename": filename
                    })
    return image_files


def create_split_directory(split_name, image_files, indices, class_names, base_dir):
    """Create a directory with images for a specific split."""
    split_dir = os.path.join(base_dir, split_name)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    
    # Create class directories
    for class_name in class_names:
        os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    
    # Copy images to split directory
    samples_per_class = {name: 0 for name in class_names}
    for idx in indices:
        img_info = image_files[idx]
        src_path = img_info["path"]
        class_name = img_info["class_name"]
        
        # Create new filename to avoid conflicts
        new_filename = f"{class_name}_{samples_per_class[class_name]:05d}.jpg"
        dst_path = os.path.join(split_dir, class_name, new_filename)
        
        shutil.copy2(src_path, dst_path)
        samples_per_class[class_name] += 1
    
    return split_dir, samples_per_class


def compute_image_stats(image):
    """
    Compute various statistics for an image useful for EDA.
    
    Returns dict with:
    - Brightness: Mean pixel value (0-255), indicates lighting
    - Contrast: Std of pixel values, indicates dynamic range
    - R/G/B means: Color channel averages, underwater images often have blue/green cast
    - Blue ratio: How much blue dominates (underwater indicator)
    - Saturation: Color intensity (low = washed out)
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Basic stats on grayscale
    gray = np.mean(img_array, axis=2)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    
    # Color channel stats
    r_mean = float(np.mean(img_array[:, :, 0]))
    g_mean = float(np.mean(img_array[:, :, 1]))
    b_mean = float(np.mean(img_array[:, :, 2]))
    
    # Blue ratio - how dominant is blue (typical of underwater)
    total_color = r_mean + g_mean + b_mean
    blue_ratio = round(b_mean / total_color, 3) if total_color > 0 else 0.33
    green_ratio = round(g_mean / total_color, 3) if total_color > 0 else 0.33
    
    # Simple saturation estimate (difference between max and min channels per pixel)
    max_channel = np.max(img_array, axis=2)
    min_channel = np.min(img_array, axis=2)
    saturation = float(np.mean(max_channel - min_channel))
    
    return {
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "r_mean": round(r_mean, 1),
        "g_mean": round(g_mean, 1),
        "b_mean": round(b_mean, 1),
        "blue_ratio": blue_ratio,
        "green_ratio": green_ratio,
        "saturation": round(saturation, 1)
    }


def create_stratified_sample(image_files, class_names, samples_per_class, seed=42):
    """Create a stratified sample with equal representation from each class."""
    random.seed(seed)
    
    # Group by class
    by_class = {name: [] for name in class_names}
    for idx, img_info in enumerate(image_files):
        by_class[img_info["class_name"]].append(idx)
    
    # Sample from each class
    sampled_indices = []
    for class_name in class_names:
        class_indices = by_class[class_name]
        n_to_sample = min(samples_per_class, len(class_indices))
        if n_to_sample > 0:
            sampled = random.sample(class_indices, n_to_sample)
            sampled_indices.extend(sampled)
    
    return sampled_indices


def main(entity=None, host=None, webhook_name=None):
    global WANDB_ENTITY
    if entity:
        WANDB_ENTITY = entity

    if not WANDB_ENTITY:
        raise ValueError("--entity is required. Usage: python setup.py --entity your-team")

    # Authenticate with W&B (handles self-hosted instances)
    wandb.login(host=host) if host else wandb.login()

    print(f"W&B Host: {host or 'https://api.wandb.ai (default)'}")
    print(f"W&B Entity: {WANDB_ENTITY}")
    print(f"W&B Project: {WANDB_PROJECT}\n")

    # Step 1: Load dataset
    print("Step 1: Loading dataset from HuggingFace...")
    dataset = load_dataset(DATASET_NAME)
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    data = dataset[split_name]
    
    if hasattr(data.features.get("label", None), "names"):
        class_names = data.features["label"].names
    else:
        class_names = [str(c) for c in sorted(set(data["label"]))]
    
    num_classes = len(class_names)

    # Step 2: Save images to disk
    print("Step 2: Saving images to disk...")
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    
    for class_name in class_names:
        os.makedirs(os.path.join(DATA_DIR, class_name), exist_ok=True)
    
    saved_counts = {name: 0 for name in class_names}
    for idx in range(len(data)):
        sample = data[idx]
        image = sample["image"]
        label = sample["label"]
        class_name = class_names[label]
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_filename = f"{class_name}_{saved_counts[class_name]:05d}.jpg"
        img_path = os.path.join(DATA_DIR, class_name, img_filename)
        image.save(img_path, "JPEG", quality=95)
        saved_counts[class_name] += 1
    
    total_saved = sum(saved_counts.values())
    
    # Save metadata
    metadata = {
        "source": DATASET_NAME,
        "description": DATASET_DESCRIPTION,
        "reference": "AQUA: A Benchmark Dataset for Underwater Species Classification under Challenging Conditions",
        "domain": "underwater_species_classification",
        "total_samples": total_saved,
        "num_classes": num_classes,
        "class_names": class_names,
        "samples_per_class": saved_counts,
        "challenges": ["turbidity", "low_illumination", "occlusion"],
        "created_at": datetime.now().isoformat()
    }
    with open(os.path.join(DATA_DIR, "dataset_info.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 3: Log artifact to W&B
    print("Step 3: Logging artifact to W&B...")
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="dataset-upload",
        job_type="data-ingestion",
        config={"dataset_name": DATASET_NAME, "total_samples": total_saved}
    )
    
    artifact = wandb.Artifact(
        name=ARTIFACT_NAME,
        type=ARTIFACT_TYPE,
        description=DATASET_DESCRIPTION,
        metadata=metadata
    )
    artifact.add_dir(DATA_DIR)
    run.log_artifact(artifact, aliases=["latest", "source-of-truth"])

    # Step 4: Link to registry
    print("Step 4: Linking artifact to W&B Registry...")
    # target_path = f"{WANDB_ORG}/wandb-registry-{REGISTRY_TYPE}/{COLLECTION_NAME}"
    target_path = f"wandb-registry-{REGISTRY_TYPE}/{COLLECTION_NAME}"
    run.link_artifact(artifact=artifact, target_path=target_path, aliases=["latest", "production"])
    
    wandb.finish()

    # Step 5: Create train/val/test splits with lineage to registry artifact
    print("\nStep 5: Creating train/val/test splits with lineage...")
    
    # Start a new run for splitting (creates lineage)
    split_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="dataset-splitting",
        job_type="data-processing",
        tags=["aqua", "data-pipeline", "splitting"],
        notes="Split raw AQUA data into train/val/test sets",
        config={
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "seed": SEED,
            "source_artifact": ARTIFACT_NAME
        }
    )
    
    # CONSUME the artifact from registry - this creates INPUT lineage!
    registry_artifact_path = f"{target_path}:latest"
    consumed_artifact = split_run.use_artifact(registry_artifact_path)
    print(f"   Fetched artifact from registry for lineage: {registry_artifact_path}")
    
    # Get all image files from local directory
    image_files = get_all_image_files(DATA_DIR, class_names)
    
    # Create split indices
    split_indices = create_split_indices(
        image_files,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED
    )
    
    
    # Create splits directory
    if os.path.exists(SPLITS_DIR):
        shutil.rmtree(SPLITS_DIR)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    # Log each split as a separate artifact
    splits_config = [
        ("train", split_indices["train"], ["latest", "training-ready"], ["train-split", "aqua"]),
        ("val", split_indices["val"], ["latest", "validation-ready"], ["val-split", "aqua"]),
        ("test", split_indices["test"], ["latest", "holdout"], ["test-split", "holdout-data", "aqua"])
    ]
    
    for split_name, indices, aliases, tags in splits_config:
        # Create split directory with actual images
        split_dir, samples_per_class = create_split_directory(
            split_name, image_files, indices, class_names, SPLITS_DIR
        )
        
        # Save split metadata
        split_metadata = {
            "split_type": split_name,
            "num_samples": len(indices),
            "percentage_of_total": round(len(indices) / len(image_files) * 100, 1),
            "parent_artifact": f"{consumed_artifact.name}:{consumed_artifact.version}",
            "seed": SEED,
            "source": DATASET_NAME,
            "samples_per_class": samples_per_class,
            "class_names": class_names,
            "num_classes": num_classes,
            "created_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(split_dir, "split_info.json"), "w") as f:
            json.dump(split_metadata, f, indent=2)
        
        # Create split artifact with actual image files
        split_artifact = wandb.Artifact(
            name=f"aqua-{split_name}",
            type=ARTIFACT_TYPE,
            description=f"AQUA {split_name} split - derived from raw dataset in registry",
            metadata=split_metadata
        )
        
        # Add the entire split directory with images
        split_artifact.add_dir(split_dir)
        split_run.log_artifact(split_artifact, aliases=aliases, tags=tags)
        
        print(f"Logged: aqua-{split_name} ({len(indices)} samples)")
        
    wandb.finish()
    
    print("\n")
    print("LINEAGE CREATED: Registry Artifact → Train/Val/Test Splits")

    # Step 6: Create EDA Exploration Table for Workshop
    print("\nStep 6: Creating EDA exploration table...")
    
    eda_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="dataset-eda-exploration",
        job_type="data-analysis",
        tags=["aqua", "eda", "exploration", "workshop"],
        notes="Stratified sample with image statistics for interactive data exploration",
        config={
            "samples_per_class": EDA_SAMPLES_PER_CLASS,
            "total_classes": num_classes,
            "seed": SEED
        }
    )
    
    # Reference the raw artifact for lineage
    eda_run.use_artifact(registry_artifact_path)
    
    # Get stratified sample indices
    sampled_indices = create_stratified_sample(
        image_files, class_names, EDA_SAMPLES_PER_CLASS, seed=SEED
    )
        
    # Create W&B Table with rich columns for exploration
    columns = [
        "image",           # The actual image for visual inspection
        "class",           # Class name for grouping/filtering
        "class_id",        # Numeric ID for sorting
        "width",           # Image dimensions
        "height",
        "aspect_ratio",    # Width/height ratio
        "brightness",      # Mean pixel value (0-255) - lighting indicator
        "contrast",        # Std of pixels - dynamic range
        "r_mean",          # Red channel mean
        "g_mean",          # Green channel mean  
        "b_mean",          # Blue channel mean
        "blue_ratio",      # Blue dominance (underwater indicator)
        "green_ratio",     # Green dominance (underwater indicator)
        "saturation",      # Color intensity
    ]
    
    eda_table = wandb.Table(columns=columns)
    
    # Populate the table
    for i, idx in enumerate(sampled_indices):
        img_info = image_files[idx]
        img_path = img_info["path"]
        class_name = img_info["class_name"]
        class_id = class_names.index(class_name)
        
        # Load image
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        width, height = img.size
        aspect_ratio = round(width / height, 2)
        
        # Compute statistics
        stats = compute_image_stats(img)
        
        # Add row to table
        eda_table.add_data(
            wandb.Image(img, caption=f"{class_name}"),
            class_name,
            class_id,
            width,
            height,
            aspect_ratio,
            stats["brightness"],
            stats["contrast"],
            stats["r_mean"],
            stats["g_mean"],
            stats["b_mean"],
            stats["blue_ratio"],
            stats["green_ratio"],
            stats["saturation"],
        )
    
    # Log the exploration table
    wandb.log({"dataset_exploration": eda_table})
        
    wandb.finish()
    
    # Step 7: Download and log pretrained model weights
    # In air-gapped environments, participants can't reach HuggingFace.
    # This step downloads the pretrained weights and logs them as W&B artifacts
    # so participants can pull them from the internal W&B instance.
    print("\nStep 7: Logging pretrained model weights as artifacts...")

    weights_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="pretrained-weights-upload",
        job_type="model-upload",
        tags=["pretrained", "setup", "workshop"],
        notes="Upload pretrained ImageNet weights for workshop models",
    )

    for model_name in WORKSHOP_MODELS:
        print(f"  Downloading {model_name} from timm (HuggingFace)...")
        model = timm.create_model(model_name, pretrained=True)

        weights_path = f"{model_name}_imagenet.pth"
        torch.save(model.state_dict(), weights_path)

        artifact = wandb.Artifact(
            name=f"pretrained-{model_name}",
            type="pretrained-weights",
            description=f"ImageNet pretrained weights for {model_name} (via timm)",
            metadata={
                "model_name": model_name,
                "source": "timm/huggingface",
                "pretrained_dataset": "imagenet",
                "num_params": sum(p.numel() for p in model.parameters()),
            }
        )
        artifact.add_file(weights_path)
        weights_run.log_artifact(artifact, aliases=["latest"])

        print(f"  Logged: pretrained-{model_name}")

    wandb.finish()
    print("Pretrained weights uploaded to W&B.")

    # Step 8: Prepare participant local data directory
    # Copies dataset splits and pretrained weights into workshop_material/
    # so participants have everything pre-loaded locally.
    print("\nStep 8: Preparing participant local data directory...")
    PARTICIPANT_DIR = os.path.join("..", "workshop_material")
    PARTICIPANT_DATA_DIR = os.path.join(PARTICIPANT_DIR, "data")
    PARTICIPANT_WEIGHTS_DIR = os.path.join(PARTICIPANT_DIR, "pretrained_weights")

    # Copy dataset splits (train/val/test with class subfolders)
    if os.path.exists(PARTICIPANT_DATA_DIR):
        shutil.rmtree(PARTICIPANT_DATA_DIR)
    shutil.copytree(SPLITS_DIR, PARTICIPANT_DATA_DIR)
    for split in ["train", "val", "test"]:
        split_path = os.path.join(PARTICIPANT_DATA_DIR, split)
        if os.path.exists(split_path):
            n_files = sum(len(files) for _, _, files in os.walk(split_path))
            print(f"  Copied {split} split ({n_files} files) to {split_path}")

    # Copy pretrained weights
    if os.path.exists(PARTICIPANT_WEIGHTS_DIR):
        shutil.rmtree(PARTICIPANT_WEIGHTS_DIR)
    os.makedirs(PARTICIPANT_WEIGHTS_DIR, exist_ok=True)
    for model_name in WORKSHOP_MODELS:
        src = f"{model_name}_imagenet.pth"
        dst = os.path.join(PARTICIPANT_WEIGHTS_DIR, src)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src} to {PARTICIPANT_WEIGHTS_DIR}/")

    print(f"\nParticipant data ready at: {os.path.abspath(PARTICIPANT_DIR)}")

    # Step 9: Set up Workshop Registry and CI/CD automation
    # This creates a dedicated "workshop" registry, links a placeholder
    # artifact to create the "aqua-classifier" collection, and attaches
    # an automation that fires a webhook to GitHub Actions whenever the
    # "production" alias is added to any artifact version.
    #
    # PREREQUISITE: A webhook integration must already be configured in
    # W&B Team Settings (Settings > Webhooks) with the name matching
    # AUTOMATION_WEBHOOK_NAME. The webhook should point to your GitHub
    # repo's repository_dispatch endpoint.
    print("\nStep 9: Setting up Workshop Registry and CI/CD automation...")

    api = wandb.Api()

    # 9a: Create the "workshop" registry and the "aqua-classifier" collection.
    WORKSHOP_REGISTRY_NAME = "sie-workshop-uk-2026"
    print(f"  Creating '{WORKSHOP_REGISTRY_NAME}' registry...")
    try:
        registry = api.create_registry(
            name=WORKSHOP_REGISTRY_NAME,
            visibility="organization",
            description="Workshop registry for aquatic species classification. Safe to delete after the workshop.",
        )
        print(f"  Registry created: {registry.name}")
    except ValueError:
        # Registry already exists (e.g., re-running setup)
        registry = api.registry(name=WORKSHOP_REGISTRY_NAME)
        print(f"  Registry already exists: {registry.name}")

    print("  Creating collection in registry...")
    setup_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="registry-collection-setup",
        job_type="admin-setup",
        tags=["admin", "setup", "registry"],
    )

    placeholder_metadata = {
        "purpose": "Initialize registry collection for workshop",
        "created_by": "setup.py",
        "created_at": datetime.now().isoformat(),
    }
    placeholder_path = "_placeholder_init.json"
    with open(placeholder_path, "w") as f:
        json.dump(placeholder_metadata, f, indent=2)

    placeholder_artifact = wandb.Artifact(
        name="aqua-classifier-init",
        type="model",
        description="Placeholder to initialize the aqua-classifier registry collection",
        metadata=placeholder_metadata,
    )
    placeholder_artifact.add_file(placeholder_path)
    setup_run.log_artifact(placeholder_artifact)

    setup_run.link_artifact(
        artifact=placeholder_artifact,
        target_path=f"wandb-registry-sie-workshop-uk-2026/{MODEL_REGISTRY_COLLECTION}",
    )
    wandb.finish()
    os.remove(placeholder_path)
    print(f"  Registry collection created: {MODEL_REGISTRY_COLLECTION}")

    # 9b: Fetch the webhook integration and create the automation
    resolved_webhook_name = webhook_name or AUTOMATION_WEBHOOK_NAME
    print(f"  Looking for webhook integration: '{resolved_webhook_name}'...")

    try:
        available_webhooks = list(api.webhook_integrations(entity=WANDB_ENTITY))
        webhook = next(
            hook for hook in available_webhooks
            if resolved_webhook_name.lower() in getattr(hook, "name", "").lower()
            or resolved_webhook_name.lower() in getattr(hook, "url_endpoint", "").lower()
        )
    except StopIteration:
        print(f"  ERROR: No webhook integration matching '{resolved_webhook_name}' found.")
        if available_webhooks:
            print(f"  Available webhooks:")
            for hook in available_webhooks:
                name = getattr(hook, "name", "unnamed")
                url = getattr(hook, "url_endpoint", "unknown")
                print(f"    - {name}: {url}")
        else:
            print(f"  No webhook integrations configured for entity '{WANDB_ENTITY}'.")
        print(f"\n  Create the webhook in W&B Team Settings > Webhooks, then re-run setup.")
        print(f"  Or pass --webhook-name to match a different webhook.")
        print("\nDone! Setup complete (automation skipped).")
        return

    # 9c: Define the automation event and action
    collection = api.artifact_collection(
        "model",
        f"wandb-registry-sie-workshop-uk-2026/{MODEL_REGISTRY_COLLECTION}",
    )

    # Event: trigger when alias matching "^production$" is added
    event = OnAddArtifactAlias(
        scope=collection,
        filter=ArtifactEvent.alias.matches_regex(r"^production$"),
    )

    # Action: fire the webhook with the GitHub Actions payload
    action = SendWebhook.from_integration(
        webhook,
        payload={
            "event_type": "webhook",
            "client_payload": {
                "event_author": "${event_author}",
                "artifact_version": "${artifact_version}",
                "artifact_version_string": "${artifact_version_string}",
                "artifact_collection_name": "${artifact_collection_name}",
                "project_name": "${project_name}",
                "entity_name": "${entity_name}",
            },
        },
    )

    # 9d: Create the automation
    automation = api.create_automation(
        event >> action,
        name=AUTOMATION_NAME,
        description=(
            "Triggers a GitHub Actions workflow when a model is promoted to "
            "production in the Registry. Creates a review issue with the "
            "artifact metadata, author, and lineage link for audit and approval."
        ),
    )

    print(f"  Automation created: {automation.name}")
    print(f"  Trigger: alias '^production$' added to '{MODEL_REGISTRY_COLLECTION}'")
    print(f"  Action: webhook -> GitHub Actions")

    # Cleanup temporary working files
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    shutil.rmtree(SPLITS_DIR, ignore_errors=True)
    shutil.rmtree("wandb", ignore_errors=True)
    for model_name in WORKSHOP_MODELS:
        weights_file = f"{model_name}_imagenet.pth"
        if os.path.exists(weights_file):
            os.remove(weights_file)
    
    print("\nDone! Setup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup W&B workshop: upload dataset, splits, EDA table, pretrained model weights, and registry automation")
    parser.add_argument("--entity", type=str, required=True, help="W&B entity/team (required)")
    parser.add_argument("--host", type=str, default=None, help="W&B host URL (e.g. https://your-instance.wandb.io)")
    parser.add_argument("--webhook-name", type=str, default=None,
                        help=f"Name of the webhook integration for CI/CD automation (default: '{AUTOMATION_WEBHOOK_NAME}')")
    args = parser.parse_args()
    main(entity=args.entity, host=args.host, webhook_name=args.webhook_name)
