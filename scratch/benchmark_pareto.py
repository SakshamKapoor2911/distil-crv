import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def count_macs(model, seq_len=1):
    """
    Analytically calculate Multiply-Accumulate operations (MACs) for a single forward pass.
    Since this is a Transformer architecture, the computational cost is entirely 
    dominated by the Linear projections (Q, K, V, O, and MLP matrices).
    """
    macs = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # MACs = input_dim * output_dim * sequence_length
            macs += module.in_features * module.out_features * seq_len
    return macs

def benchmark_tps(model, tokenizer, device, num_prompts=50, input_len=256, output_len=128):
    """
    Benchmark End-to-End Tokens-Per-Second (TPS) using a raw PyTorch generation loop.
    This bypasses complex serving engines to measure native mathematical throughput.
    """
    # Create a dummy token sequence to simulate prompt context
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, input_len), device=device)
    
    # Warmup phase to compile kernels and initialize CUDA contexts
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(5):
            _ = model.generate(input_ids, max_new_tokens=10, do_sample=False, use_cache=True)
            
    torch.cuda.synchronize(device)
    
    # Execution phase
    print(f"Running TPS benchmark over {num_prompts} iterations...")
    start_time = time.time()
    
    total_generated_tokens = 0
    for _ in tqdm(range(num_prompts), desc="Benchmarking Generation"):
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=output_len,
                min_new_tokens=output_len,
                do_sample=False,
                use_cache=True
            )
            total_generated_tokens += output_len
            
    torch.cuda.synchronize(device)
    end_time = time.time()
    
    elapsed = end_time - start_time
    tps = total_generated_tokens / elapsed
    return tps, elapsed

def main(args):
    print(f"Loading Tokenizer from {args.baseline_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.baseline_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("\n--- 1. Dense Baseline Model ---")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.baseline_model, torch_dtype=torch.float16, device_map=args.device
    )
    baseline_model.eval()
    
    baseline_macs = count_macs(baseline_model, seq_len=1)
    print(f"Baseline Theoretical MACs/token: {baseline_macs / 1e9:.2f} G-MACs")
    
    baseline_tps, baseline_time = benchmark_tps(
        baseline_model, tokenizer, args.device, 
        num_prompts=args.num_prompts, input_len=args.input_len, output_len=args.output_len
    )
    print(f"Baseline Physical TPS: {baseline_tps:.2f} tokens/sec")
    
    # Clear VRAM before loading the sparse model
    del baseline_model
    torch.cuda.empty_cache()
    
    print("\n--- 2. Structurally Pruned Model ---")
    try:
        pruned_model = AutoModelForCausalLM.from_pretrained(
            args.pruned_model, torch_dtype=torch.float16, device_map=args.device
        )
        pruned_model.eval()
        
        pruned_macs = count_macs(pruned_model, seq_len=1)
        print(f"Pruned Theoretical MACs/token: {pruned_macs / 1e9:.2f} G-MACs")
        
        pruned_tps, pruned_time = benchmark_tps(
            pruned_model, tokenizer, args.device,
            num_prompts=args.num_prompts, input_len=args.input_len, output_len=args.output_len
        )
        print(f"Pruned Physical TPS: {pruned_tps:.2f} tokens/sec")
        
        # Calculate strict Pareto statistics
        mac_reduction = (1 - (pruned_macs / baseline_macs)) * 100
        tps_improvement = ((pruned_tps / baseline_tps) - 1) * 100
        
        print("\n==================================================")
        print("           PARETO EFFICIENCY RESULTS")
        print("==================================================")
        print(f" Theoretical MACs Reduction:  {mac_reduction:.2f}%")
        print(f" Empirical TPS Acceleration:  +{tps_improvement:.2f}%")
        print("==================================================")
        
    except Exception as e:
        print(f"Error loading pruned model (It may not be generated yet): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pareto efficiency of structural pruning.")
    parser.add_argument("--baseline_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--pruned_model", type=str, default="outputs/sparse_lora_heal/healed_adapters")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_prompts", type=int, default=50, help="Number of inference calls.")
    parser.add_argument("--input_len", type=int, default=256, help="Input prompt sequence length.")
    parser.add_argument("--output_len", type=int, default=128, help="Output generation length.")
    
    args = parser.parse_args()
    main(args)
