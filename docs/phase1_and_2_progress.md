# Distil-CRV: Progress Report

## Phase 1: Foundation & Baseline Profiling
**Status:** Completed
- Verified and downloaded the GSM8K dataset using the HuggingFace `datasets` library.
- Fixed the offline dashboard generation issue (using `replace()` instead of `.format()` to prevent CSS brace conflicts).
- Wrote and tested the `phase1_baseline.py` script to simulate the profiling of VRAM usage and generation latency for the heavy baseline (CRV).
- Successfully cached real hidden states from the 500-example GSM8K subset using an adapter around Llama-3.1-8B. The extraction saves [batch, seq_len, 4096] hidden states and mock CRV soft targets to `data/phase1_cache` while consuming minimal memory footprint. 

## Phase 2: Distil-CRV Core Implementation
**Status:** In Progress
- Finalized the dual-path implementation strategy targeting a Transformer Verifier (primary) and a LoRA Adapter (secondary).
- Implemented the `BaseVerifier`, `TransformerVerifier` (small 50M parameter encoder model), and `LoRAVerifier` using PyTorch and `peft`.
- Wrote the knowledge distillation training script (`VerifierTrainer`), employing a combined KL Divergence (soft labels) and Cross-Entropy (hard labels) loss strategy.
- Created `CachedHiddenStatesDataset` and Phase 2 DataLoader configurations to load Phase 1 hidden states efficiently for offline training.
- Defined Hydra configurations for both Transformer and LoRA experiments.

## Next Steps
- Verify the 5-batch training dry-run of the `TransformerVerifier` on the extracted Phase 1 dataset.
- Evaluate Checkpoints, perform Layer Ablations, and assess Cross-Domain Generalization metrics (Phase 3).
