#!/bin/bash

# run_ablations.sh
# Run layer ablations for the Transformer Verifier

# Exit on error
set -e

echo "========================================"
echo "Starting Layer Ablation Training Runs"
echo "========================================"

echo ""
echo "[1/4] Running Baseline (All Layers: 24-31)"
python experiments/scripts/phase2_train.py data.layer_indices="[24,25,26,27,28,29,30,31]" experiment_name="ablation-all"

echo ""
echo "[2/4] Running Top Half (Layers: 28-31)"
python experiments/scripts/phase2_train.py data.layer_indices="[28,29,30,31]" experiment_name="ablation-top-half"

echo ""
echo "[3/4] Running Bottom Half (Layers: 24-27)"
python experiments/scripts/phase2_train.py data.layer_indices="[24,25,26,27]" experiment_name="ablation-bottom-half"

echo ""
echo "[4/4] Running Final Layer Only (Layer: 31)"
python experiments/scripts/phase2_train.py data.layer_indices="[31]" experiment_name="ablation-final-layer"

echo ""
echo "========================================"
echo "Ablations complete! Check experiments/results/phase2 for outputs."
echo "========================================"
