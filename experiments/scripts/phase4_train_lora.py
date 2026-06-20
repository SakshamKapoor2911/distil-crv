import torch
import torch.optim as optim
import argparse
import sys
import os
import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.lora_verifier import LoRAVerifier
from src.training.distiller import DistillationLoss
from src.data.cache import HiddenStateCache

def load_data(cache_dir, raw_data_path, num_examples=500):
    cache = HiddenStateCache(cache_dir=cache_dir)
    cached_ids = set(cache.list_cached_examples()[:num_examples])
    
    from datasets import load_from_disk
    dataset = load_from_disk(raw_data_path)['train']
    
    examples = []
    for i in range(len(dataset)):
        ex_id = f"gsm8k_train_{i}"
        if ex_id in cached_ids:
            # Load soft targets
            cached = cache.load_example(ex_id)
            examples.append({
                'id': ex_id,
                'reasoning_trace': f"{dataset[i]['question']}\\n{dataset[i]['answer']}",
                'crv_logits': cached['verification_logits'],
                'label': cached['label']
            })
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data/phase1_cache")
    parser.add_argument("--raw_data", default="data/raw/gsm8k")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    # 1. Load tokenizer & base model
    print("Loading tokenizer and Llama-3.1-8B in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    # Enable gradient checkpointing for memory
    base_model.gradient_checkpointing_enable()
    
    # 2. Wrap with LoRA
    print("Wrapping with LoRAVerifier...")
    verifier = LoRAVerifier(
        base_model=base_model,
        lora_r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"]
    )
    # Ensure verification heads are also in bfloat16 and on the correct device
    verifier = verifier.to(dtype=torch.bfloat16, device=args.device)
    
    print(f"Trainable Parameters: {verifier.get_param_count():,}")
    
    # 3. Load Data
    print("Loading data...")
    examples = load_data(args.cache_dir, args.raw_data, num_examples=500)
    print(f"Loaded {len(examples)} examples.")
    
    # 4. Training setup
    optimizer = optim.AdamW(verifier.parameters(), lr=1e-4)
    loss_fn = DistillationLoss()
    
    torch.cuda.reset_peak_memory_stats()
    
    print("Starting LoRA Training...")
    verifier.train()
    
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_ce = 0
        epoch_kl = 0
        
        pbar = tqdm(examples, desc=f"Epoch {epoch+1}/{args.epochs}")
        for ex in pbar:
            # Tokenize
            inputs = tokenizer(
                ex['reasoning_trace'], 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            input_ids = inputs['input_ids'].to(args.device)
            attention_mask = inputs['attention_mask'].to(args.device)
            
            # Since attention_mask=0 is False, PyTorch mask convention: 1 to keep, 0 to ignore
            # but LoRAVerifier converts 0 to True for ignoring
            # Actually, HuggingFace mask is 1 for attend, 0 for ignore
            
            # Prepare targets
            crv_logits = ex['crv_logits'].unsqueeze(0).to(args.device, dtype=torch.bfloat16)
            label = torch.tensor([ex['label']]).to(args.device)
            
            # Forward
            optimizer.zero_grad()
            outputs = verifier(input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss_dict = loss_fn(
                outputs['verification_logits'], 
                crv_logits, 
                label
            )
            loss = loss_dict['loss']
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ce += loss_dict['ce_loss'].item()
            epoch_kl += loss_dict['kl_loss'].item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            # Clean cache to prevent OOM
            del input_ids, attention_mask, outputs, loss
            torch.cuda.empty_cache()
            
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(examples):.4f}")
        
    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    
    print("Measuring forward pass latency...")
    latencies = []
    verifier.eval()
    with torch.no_grad():
        for i in range(10): # 10 samples
            ex = examples[i]
            inputs = tokenizer(ex['reasoning_trace'], return_tensors="pt", truncation=True, max_length=2048).to(args.device)
            start_infer = time.time()
            _ = verifier(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            torch.cuda.synchronize()
            latencies.append(time.time() - start_infer)
    
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    
    # Save results
    results_dir = "experiments/results/phase4"
    os.makedirs(results_dir, exist_ok=True)
    
    result_data = {
        "model": "LoRA",
        "parameters": verifier.get_param_count(),
        "final_train_loss": round(epoch_loss / len(examples), 4),
        "final_ce_loss": round(epoch_ce / len(examples), 4),
        "final_kl_loss": round(epoch_kl / len(examples), 4),
        "peak_vram_gb": round(peak_vram, 2),
        "training_time_s": round(elapsed, 2),
        "forward_pass_latency_ms": round(avg_latency_ms, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    res_path = os.path.join(results_dir, "lora_baseline.json")
    with open(res_path, "w") as f:
        json.dump(result_data, f, indent=2)
        
    # Save checkpoint
    torch.save(verifier.state_dict(), os.path.join(results_dir, "lora_verifier.pt"))
    print(f"\\nSaved results to {res_path} and model to lora_verifier.pt")

if __name__ == "__main__":
    main()
