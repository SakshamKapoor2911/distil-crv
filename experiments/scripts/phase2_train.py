import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.factory import build_verifier
from src.models.factory import build_verifier
from src.training.distiller import DistillationLoss, VerifierTrainer
from src.data.loader import build_dataloader
import wandb

@hydra.main(version_base=None, config_path="../configs/phase2", config_name="transformer_verifier")
def main(cfg: DictConfig):
    # Initialize logging
    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.experiment_name,
        config=dict(cfg),
        tags=list(cfg.logging.get("wandb_tags", [])),
    )
    
    device = "cuda:0"
    
    base_model = None
    if cfg.verifier.type == "lora":
        print("Loading base LLM for LoRA...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            torch_dtype=torch.float16,
            device_map="auto",  
        )
    
    # Build verifier
    print(f"Building {cfg.verifier.type} verifier...")
    verifier = build_verifier(cfg.verifier, base_model=base_model if cfg.verifier.type == "lora" else None)
    verifier = verifier.to(device)
    
    print(f"Verifier parameters: {verifier.get_param_count():,}")
    
    # Build optimizer
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=cfg.training.learning_rate)
    
    # Build loss
    loss_fn = DistillationLoss(
        temperature=cfg.training.loss.temperature,
        alpha_soft=cfg.training.loss.alpha_soft,
        alpha_hard=cfg.training.loss.alpha_hard,
    )
    
    # Build trainer
    trainer = VerifierTrainer(verifier, optimizer, loss_fn, device=device)
    
    print("Loading cached hidden states...")
    train_loader = build_dataloader(
        cache_dir="data/phase1_cache",
        batch_size=cfg.training.batch_size,
        shuffle=True
    )
    
    print(f"\n[STARTING] Training {cfg.verifier.type} verifier for {cfg.training.epochs} epochs...")
    
    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.training.epochs}")
        for step, batch in enumerate(train_loader):
            metrics = trainer.train_step(batch)
            
            if step % 10 == 0:
                print(f"Step {step} - Loss: {metrics['train_loss']:.4f} (KL: {metrics['kl_loss']:.4f}, CE: {metrics['ce_loss']:.4f})")
            
            wandb.log(metrics)
            
    print("\n[SUCCESS] Training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()
