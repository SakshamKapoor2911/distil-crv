import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.factory import build_verifier
from src.training.distiller import DistillationLoss, VerifierTrainer
# from src.data.loader import build_dataloader # Temporarily mocked
import wandb

@hydra.main(version_base=None, config_path="configs/phase2", config_name="transformer_verifier")
def main(cfg: DictConfig):
    # Initialize logging
    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.experiment_name,
        config=dict(cfg),
        tags=list(cfg.logging.get("wandb_tags", [])),
    )
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load base LLM (for extracting hidden states)
    print("Loading base LLM...")
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
    
    print("Implementation done! Exiting before data loop...")
    wandb.finish()

if __name__ == "__main__":
    main()
