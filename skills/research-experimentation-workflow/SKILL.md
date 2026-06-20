---
name: research-experimentation-workflow
description: End-to-end framework for accelerating ML research and navigating constraints on a Dual RTX 3090 (48GB VRAM) server.
---

# ML Research Experimentation Workflow

This skill documents the rapid prototyping workflow successfully used to execute the **Distil-CRV** research project. By following these standardized steps, future research projects can bypass common pitfalls, hardware bottlenecks, and accelerate from inception to ICLR paper draft in record time (significantly faster than projected timelines).

## 1. Hardware Context: Dual RTX 3090 (48GB VRAM)
When dealing with large language models (like Llama-3.1-8B), you must respect the physical constraints of our specific Lambda server setup. 
- **Inference only:** 8B models in `bfloat16` take ~16GB of VRAM and fit comfortably on a single GPU (`cuda:0`).
- **Full Fine-Tuning / LoRA:** Backpropagating through an 8B model requires storing massive computational graphs. 
  - *Failure Mode:* `CUDA Out of Memory` is almost guaranteed if not careful.
  - *Resolution:* You MUST explicitly use `torch.bfloat16`, enable `gradient_checkpointing_enable()` on the base LLM, and drop `batch_size` to 1. 

## 2. Decoupled Caching for Rapid Ablation
**The Distil-CRV Trick:** If a research experiment requires training an auxiliary model (like a Verifier or Reward Model) on top of a frozen LLM, *do not run the LLM during the training loop!*
1. Run a one-time data extraction pipeline that passes the dataset through the frozen LLM.
2. Save the required outputs (e.g., hidden states, logits) directly to disk using PyTorch (`torch.save`).
3. Train the auxiliary model exclusively on the disk cache. 

*Result:* Training iterations drop from minutes/hours to seconds. This allowed us to run deep ablation studies across all 32 layers of Llama in under 5 minutes.

## 3. Explaining & Debugging Tokens (The BPE Trap)
When visualizing token-level importance (like Gradient-Based Saliency) or debugging generation, modern LLM tokenizers (using Byte-Pair Encoding) will output special characters.
- **Spaces** become `Ġ` or ` ` (U+2581).
- **Newlines** become `Ċ`.
*Resolution:* Always intercept and clean these tokens before rendering them into HTML or UI:
```python
import html
t = html.escape(token.replace("<|begin_of_text|>", ""))
t = t.replace("Ġ", " ").replace("Ċ", "<br>").replace(" ", " ")
```

## 4. Accelerated Workflow Pipeline
1. **Baseline Profiling:** Prove the problem exists (e.g., measuring the heavy 38.5GB VRAM and 1.5s latency of the baseline CRV).
2. **Caching:** Extract required states to disk.
3. **Core Training (Distillation):** Train hyper-efficient, standalone architectures on the cache.
4. **Explainability (XAI):** Extract mathematical proofs of the model's behavior (e.g., Saliency).
5. **Efficiency Baselining:** Train the traditional approach (LoRA) to mathematically prove the architectural superiority of the new method.
6. **Paper Drafting:** Translate all logged JSON metrics and visualizations into an ICLR formatted `main.tex`.
