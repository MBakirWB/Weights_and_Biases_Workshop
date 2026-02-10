# Workshop Admin Setup

Run this **once** before the workshop from a machine with internet access and connectivity to the W&B instance.

This script downloads the AQUA dataset from HuggingFace, creates train/val/test splits, builds an EDA exploration table, and uploads pretrained model weights (resnet50, efficientnet_b0) -- all as W&B artifacts. It also copies the splits and weights into `../workshop_material/` so participants have everything pre-loaded locally.

## Prerequisites

- Python 3.11+
- Access to the target W&B instance
- Internet access (HuggingFace downloads)
- A W&B team created on the instance for workshop participants (this is the `--entity` value)
- **For the CI/CD automation (Step 9):** A webhook integration must be configured in the W&B team settings **before** running `setup.py`. The script looks for a webhook named `github-model-cicd` by default (override with `--webhook-name`). To set this up:
  1. Get a GitHub token with `repo` scope from the workshop owners
  2. In W&B: **Team Settings > Team secrets > New secret**
     - Name: `GITHUB_TOKEN`
     - Secret: the GitHub access token
  3. In W&B: **Team Settings > Webhooks > New webhook**
     - Name: `github-model-cicd`
     - URL: `https://api.github.com/repos/MBakirWB/Automation_Jobs/dispatches`
     - Access token: select `GITHUB_TOKEN` from the secrets dropdown
  4. If the webhook doesn't exist when `setup.py` runs, Step 9 will be skipped and the rest of the setup completes normally

## Setup

```bash
git clone <repo-url>
cd 2026-Workshop/admin_setup_only
python -m venv workshop-setup
source workshop-setup/bin/activate
pip install -r ../requirements.txt
```

## Run

```bash
python setup.py --entity <your-wandb-team> --host <wandb-instance-url>
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--entity` | Yes | — | W&B team or username |
| `--host` | No | `https://api.wandb.ai` | W&B instance URL (for self-hosted) |
| `--webhook-name` | No | `github-model-cicd` | Name of the webhook integration for the CI/CD automation |

The project is fixed to `SIE-Workshop-2026`.

## What it creates in W&B

| Artifact | Type | Description |
|----------|------|-------------|
| `aqua-raw-dataset` | dataset | Full AQUA dataset (8,171 images, 20 classes) |
| `aqua-train` | dataset | Training split (80%) |
| `aqua-val` | dataset | Validation split (10%) |
| `aqua-test` | dataset | Test split (10%) |
| `pretrained-resnet50` | pretrained-weights | ImageNet weights for resnet50 |
| `pretrained-efficientnet_b0` | pretrained-weights | ImageNet weights for efficientnet_b0 |

It also:
- Logs a `dataset-eda-exploration` run with an interactive table for data exploration
- Creates a dedicated `sie-workshop-uk-2026` registry (via `api.create_registry()`) with an `aqua-classifier` collection inside it
- Sets up a `github-model-cicd` automation that fires a webhook when the `production` alias is added to any artifact in that collection
- **Cleanup:** After the workshop, delete the entire `sie-workshop-uk-2026` registry to remove all workshop artifacts in one step

## What it creates locally

After running, `../workshop_material/` will contain pre-loaded data for participants:

```
workshop_material/
  data/
    train/       # ~6,500 images in class subfolders
    val/         # ~800 images in class subfolders
    test/        # ~800 images in class subfolders
  pretrained_weights/
    resnet50_imagenet.pth
    efficientnet_b0_imagenet.pth
```

Participants use this local data during the workshop. The notebook calls `use_artifact()` to build lineage in W&B without re-downloading the data.

## Fallback: data-only setup (no W&B required)

If the workshop environment doesn't have the pre-loaded data (e.g., the admin forgot to run `setup.py`, or you're running on a fresh machine), use the fallback script to prepare just the local data -- no W&B account or instance needed:

```bash
cd admin_setup_only
pip install datasets torch timm Pillow numpy scikit-learn
python prepare_local_data.py
```

This downloads the dataset from HuggingFace, creates identical splits, downloads pretrained weights, and places everything into `../workshop_material/`. It does **not** upload anything to W&B -- you'll need to run `setup.py` separately for that.

## Participant environment

Participants need these packages pre-installed (see `../requirements.txt`):

```
wandb torch torchvision timm scikit-learn numpy Pillow tqdm
```

They do **not** need `datasets` (HuggingFace) -- all data is pre-loaded locally and tracked via W&B artifacts.
