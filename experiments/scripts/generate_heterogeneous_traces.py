import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_local_dataset(path, split="train"):
    try:
        return load_from_disk(path)[split]
    except Exception as e:
        print(f"[ERROR] Could not load {path}: {e}")
        return []

def extract_answer(text):
    """Attempt to extract the final answer from generated text."""
    # Common format: '#### <answer>'
    match = re.search(r'####\s*(.*)', text)
    if match:
        return match.group(1).strip()
    
    # Fallback: boxed answer (often in MATH)
    match = re.search(r'\\boxed\{(.*?)\}', text)
    if match:
        return match.group(1).strip()
        
    # Final fallback: last number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return ""

def generate_traces(args):
    """Generate reasoning traces via sampling to induce natural errors."""
    print(f"Loading {args.model_name} for generation on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    
    # Load raw datasets
    datasets = {
        "math": load_local_dataset("data/raw/math", split="train"),
        "svamp": load_local_dataset("data/raw/svamp", split="train")
    }
    
    correct_traces = []
    incorrect_traces = []
    
    target_per_class = args.target_per_class  # default 250
    
    print(f"\n[START] Generating traces (Target: {target_per_class} correct, {target_per_class} incorrect)")
    print(f"Sampling configuration: Temperature = {args.temperature}, N_samples = {args.n_samples}")
    
    # We iterate over a mixture of MATH and SVAMP questions
    # To interleave, we zip them up to min length
    math_examples = list(datasets["math"])
    svamp_examples = list(datasets["svamp"])
    
    if not math_examples and not svamp_examples:
        print("[ERROR] No data found. Please run download_data.py first.")
        return
        
    mixed_examples = []
    for i in range(max(len(math_examples), len(svamp_examples))):
        if i < len(math_examples):
            # MATH uses 'problem' and 'solution'
            mixed_examples.append({
                "dataset": "math",
                "id": f"math_{i}",
                "question": math_examples[i].get("problem", ""),
                "ground_truth": extract_answer(math_examples[i].get("solution", ""))
            })
        if i < len(svamp_examples):
            # SVAMP uses 'Body', 'Question', and 'Answer'
            body = svamp_examples[i].get("Body", "")
            q = svamp_examples[i].get("Question", "")
            ans = str(svamp_examples[i].get("Answer", ""))
            mixed_examples.append({
                "dataset": "svamp",
                "id": f"svamp_{i}",
                "question": f"{body} {q}",
                "ground_truth": ans
            })

    for example in tqdm(mixed_examples, desc="Generating from prompts"):
        # Stop early if we have enough of both
        if len(correct_traces) >= target_per_class and len(incorrect_traces) >= target_per_class:
            break
            
        prompt = f"Question: {example['question']}\nLet's think step by step.\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        # We sample multiple times per question to get a distribution of correct and incorrect answers
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=args.n_samples,
                pad_token_id=tokenizer.pad_token_id
            )
            
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for text in generated_texts:
            # We strip the prompt to get just the reasoning trace
            trace = text[len(prompt):].strip()
            pred_ans = extract_answer(trace)
            
            # Simple equivalence check (in reality, may need more robust sympy parsing)
            is_correct = (pred_ans == example["ground_truth"]) and (pred_ans != "")
            
            trace_data = {
                "id": f"{example['id']}_{len(correct_traces) + len(incorrect_traces)}",
                "dataset": example['dataset'],
                "question": example['question'],
                "reasoning_trace": trace,
                "ground_truth_answer": example['ground_truth'],
                "predicted_answer": pred_ans,
                "label": 1 if is_correct else 0
            }
            
            if is_correct and len(correct_traces) < target_per_class:
                correct_traces.append(trace_data)
            elif not is_correct and len(incorrect_traces) < target_per_class:
                incorrect_traces.append(trace_data)

    print(f"\n[SUMMARY] Generation Complete")
    print(f"Correct traces collected: {len(correct_traces)} / {target_per_class}")
    print(f"Incorrect traces collected: {len(incorrect_traces)} / {target_per_class}")
    
    # Save the blended generation
    os.makedirs("data/raw/blended_heterogeneous", exist_ok=True)
    blended_data = correct_traces + incorrect_traces
    
    with open("data/raw/blended_heterogeneous/generated_traces.json", "w") as f:
        json.dump(blended_data, f, indent=2)
        
    print("Saved generated traces to data/raw/blended_heterogeneous/generated_traces.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="T>0 induces natural logical flaws in reasoning.")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of sampled traces per question.")
    parser.add_argument("--target_per_class", type=int, default=250,
                        help="How many Correct and Incorrect traces to collect.")
    
    args = parser.parse_args()
    main(args)
