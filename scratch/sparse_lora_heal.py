import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk, load_dataset

def get_healing_lora_config():
    """Strict PEFT Configuration for Sparse Healing (Layers 25-30)"""
    target_modules = [
        f"model.layers.{i}.self_attn.o_proj" for i in range(25, 31)
    ] + [
        f"model.layers.{i}.mlp.down_proj" for i in range(25, 31)
    ]
    
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        # True is the default PEFT initialization: Kaiming uniform/normal for A, Zero for B
        init_lora_weights=True,
    )

def main(args):
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
        torch_dtype=torch.float16,
    )
    
    # Apply strict PEFT configuration
    peft_config = get_healing_lora_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # We load the expanded dataset instead of the 120-sample calibration set
    print(f"Loading expanded dataset from {args.dataset_path}")
    
    # Try loading from disk first, fallback to loading JSON traces if it's the blended set
    if os.path.exists(os.path.join(args.dataset_path, "generated_traces.json")):
        dataset = load_dataset("json", data_files=os.path.join(args.dataset_path, "generated_traces.json"))
        # Format into causal LM text
        def format_text(example):
            return {"text": f"Question: {example['question']}\n{example['reasoning_trace']}"}
        dataset = dataset.map(format_text)
        train_dataset = dataset["train"]
    else:
        dataset = load_from_disk(args.dataset_path)
        train_dataset = dataset["train"] if "train" in dataset else dataset
        
    # Healing Training Arguments
    # We use a conservative learning rate (e.g. 5e-5) with cosine decay to gently smooth the stream
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        fp16=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
    )
    
    print("Beginning Sparse LoRA Healing Phase...")
    trainer.train()
    
    # Save the healed adapters
    model.save_pretrained(os.path.join(args.output_dir, "healed_adapters"))
    print("Healing complete. Adapters saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", type=str, default="cuda:0")
    # Default to 5e-5 for gentle healing instead of standard 2e-4
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    # Default to the expanded blended dataset rather than the 120-calib set
    parser.add_argument("--dataset_path", type=str, default="data/raw/blended_heterogeneous")
    parser.add_argument("--output_dir", type=str, default="outputs/sparse_lora_heal")
    parser.add_argument("--epochs", type=int, default=3)
    
    args = parser.parse_args()
    main(args)
