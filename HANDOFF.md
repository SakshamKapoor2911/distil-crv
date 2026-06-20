# Distil-CRV Handoff Guide

## Environment Setup (Run Once on Lambda)
1. **Navigate to the workspace:**
   ```bash
   cd /home/skapoor/distil-crv
   ```
2. **Create Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Login to HuggingFace (required for gated Llama-3.1 models):**
   ```bash
   huggingface-cli login
   ```
4. **Login to Weights & Biases:**
   ```bash
   wandb login
   ```

## Phase 1 Execution

### 1. Download Datasets
You must cache the datasets locally on the server before running the profiling script to avoid repeated network overhead.
```bash
python experiments/scripts/download_data.py
```

### 2. Run Baseline VRAM Profiling
Execute the baseline script on a single 3090 GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/scripts/phase1_baseline.py --subset_size 200
```
*Results will be saved to `experiments/results/phase1/baseline_crv_profiling.json`.*

### 3. Build & Deploy Dashboard
The repository is equipped with a static HTML dashboard that parses the `experiments/results/` folder.
To view it locally:
```bash
python scripts/build_dashboard.py
# Open dashboard/index.html in your browser
```
Once you push these changes to the `main` branch, the GitHub Action (`.github/workflows/dashboard.yml`) will automatically build and host the dashboard on GitHub Pages!

## Next Steps
After completing the baseline profiling, update the Task Tracker and we will move onto **Phase 2: Distil-CRV Core Implementation** where we will build the actual LoRA/Transformer auxiliary models.
