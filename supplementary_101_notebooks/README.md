# Supplementary 101 Notebooks

These notebooks are standalone introductions to W&B Experiment Tracking and Artifacts. They serve as the blueprint for the main workshop. If you are new to W&B, it is highly recommended you step through these prior to the workshop.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your environment:**
   Copy `.env` and fill in your values:
   ```
   YOUR_NAME=<your-first-name>
   WANDB_ENTITY=<your-team-name>
   WANDB_PROJECT=SIE-Workshop-Supplementary-Material
   WANDB_BASE_URL=<your-wandb-instance-url>
   WANDB_API_KEY=<your-api-key>
   ```

3. **Run the notebooks in order.**

## Notebooks

| Notebook | Covers |
|----------|--------|
| W&B_101_Intro_to_Experiment_Tracking | Runs, config, history, summary, scalar/media logging, sweeps, alerts |
| W&B_101_Intro_to_W&B_Artifacts | Artifact creation, versioning, aliases, TTL, lineage, reference artifacts, Registry overview |

## Cleanup

Each notebook has a cleanup cell at the end that removes all generated files (`notebook_generated_material/`). Uncomment and run it when you're done.
