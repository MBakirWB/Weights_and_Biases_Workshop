# W&B Workshop 2026: Aquatic Species Classification

Build an image classifier for marine species and learn the full MLOps lifecycle with Weights & Biases -- from first experiment to production model.

## What you'll do

Using the AQUA underwater dataset (20 marine species, 8K+ images), you'll train a baseline model, run a hyperparameter sweep, and promote the best model to production -- all tracked end-to-end in W&B.

**Topics covered:**
- Experiment tracking (runs, config, metrics, alerts, commit=False, define_metric)
- Visual logging (images, tables, ROC curves, per-class metrics)
- Artifacts (versioning, lineage, TTL, reference artifacts)
- Resuming runs (resume by ID, continue training seamlessly)
- Offline mode (train without connectivity, sync later)
- Model Registry (staging, promotion)
- Sweeps (hyperparameter optimization, CLI sweeps, parallel agents)
- Automations (CI/CD triggers from registry events)
- Programmatic API (optional -- query runs, filters, metadata)
- Programmatic Reports (optional -- Reports API, PanelGrid)
- SDK Settings Reference (optional -- network, git, distributed training)

## Getting started

### 1. Set up your Python environment

If you already have a pre-provisioned environment (e.g., a JupyterHub kernel), skip to step 2.

Otherwise, create a virtual environment and install dependencies:

```bash
cd 2026-Workshop
python -m venv workshop
source workshop/bin/activate
pip install -r requirements.txt
```

> **Note:** If you plan to run the notebook in Jupyter, make sure `jupyter` and `ipykernel` are installed in the same environment, then register the kernel:
> ```bash
> pip install jupyter ipykernel
> python -m ipykernel install --user --name wandb_workshop --display-name "W&B Workshop"
> ```

### 2. Configure your `.env` file

Open `workshop_material/.env` and fill in your W&B credentials:

```
WANDB_ENTITY=your-team-name
WANDB_PROJECT=SIE-Workshop-2026
WANDB_BASE_URL=https://your-wandb-instance.example.com
WANDB_API_KEY=your-api-key-here
```

All workshop files (notebook, sweep scripts, shared worker) read from this single file.

Find your API key at: **W&B UI > Profile > Settings > API Keys**

### 3. Set your name in the notebook

Open `workshop_material/aqua_with_wandb.ipynb` and set `YOUR_NAME` near the top of the Setup section (Cell 6):

```python
YOUR_NAME = "alice"  # Use your first name, lowercase, no spaces
```

This namespaces your artifacts, registry entries, and run groups so multiple participants can work in the same W&B project without conflicts. If you also plan to use the CLI sweep script (`sweep_train.py`), set `YOUR_NAME` there too.

### 4. Verify local data

The datasets and pretrained model weights should already be pre-loaded in `workshop_material/`:

```
workshop_material/
  data/
    train/               # ~6,500 training images (20 class subfolders)
    val/                 # ~800 validation images
    test/                # ~800 test images
  pretrained_weights/
    resnet50_imagenet.pth
    efficientnet_b0_imagenet.pth
```

The notebook reads data from these local directories. It also calls `use_artifact()` to track lineage in W&B -- so your training runs record exactly which dataset versions were used, without needing to download anything.

If the `data/` or `pretrained_weights/` directories are missing, ask your workshop facilitator. In a pinch, you can run the fallback script yourself (requires internet access):

```bash
cd admin_setup_only
pip install datasets torch timm Pillow numpy scikit-learn
python prepare_local_data.py
```

### 5. Open the workshop notebook

Open `workshop_material/aqua_with_wandb.ipynb` and follow along. The notebook handles W&B authentication automatically using your `.env` credentials.

## Supplementary material

New to W&B? Check `supplementary_101_notebooks/` for standalone introductions to experiment tracking and artifacts/registry.

## Repo structure

```
2026-Workshop/
  workshop_material/
    aqua_with_wandb.ipynb       # Main workshop notebook (start here)
    workshop_utils.py           # ML boilerplate (data loading, training loops)
    .env                        # Your W&B credentials (fill in once)
    sweep_train.py              # Standalone sweep training script (CLI sweeps)
    sweep_config.yaml           # Sweep search space config
    shared_worker.py            # Shared-mode demo (multi-process logging)
    data/                       # Pre-loaded dataset splits (train/val/test)
    pretrained_weights/         # Pre-loaded model weights
  supplementary_101_notebooks/  # Optional deep-dives on W&B concepts
  admin_setup_only/             # Admin-only: dataset + model upload scripts
  requirements.txt              # Python dependencies
```
