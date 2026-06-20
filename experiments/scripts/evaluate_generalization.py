import torch
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.transformer_verifier import TransformerVerifier
from src.data.loader import build_dataloader
from src.training.distiller import DistillationLoss

def evaluate(model_path: str, cache_dir: str, device: str = "cuda:0"):
    print(f"Loading weights from {model_path}...")
    
    # Initialize the model using the same architecture as Phase 2
    verifier = TransformerVerifier(
        hidden_dim=4096,
        n_layers=4,
        n_heads=8,
        n_error_types=5
    ).to(device)
    
    verifier.load_state_dict(torch.load(model_path, map_location=device))
    verifier.eval()
    
    print(f"Loading evaluation dataset from {cache_dir}...")
    loader = build_dataloader(
        cache_dir=cache_dir,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    loss_fn = DistillationLoss()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            # Move to device
            hidden_states = batch['hidden_states'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            crv_logits = batch['crv_logits'].to(device)
            error_type_logits = batch['error_type_logits'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = verifier(hidden_states, attention_mask=attention_mask)
            
            # Compute loss
            loss_dict = loss_fn(outputs, {
                'crv_logits': crv_logits,
                'error_type_logits': error_type_logits,
                'labels': labels
            })
            
            total_loss += loss_dict['loss'].item()
            total_ce += loss_dict['ce_loss'].item()
            total_kl += loss_dict['kl_loss'].item()
            
            # Accuracy on the hard label
            preds = torch.argmax(outputs['verification_logits'], dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
            
    num_batches = len(loader)
    avg_loss = total_loss / num_batches
    avg_ce = total_ce / num_batches
    avg_kl = total_kl / num_batches
    accuracy = correct_predictions / total_predictions
    
    print("\n=== Generalization Evaluation Results ===")
    print(f"Model: {model_path}")
    print(f"Dataset: {cache_dir} ({total_predictions} examples)")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"  Cross-Entropy: {avg_ce:.4f}")
    print(f"  KL-Divergence: {avg_kl:.4f}")
    print(f"Verification Accuracy: {accuracy * 100:.2f}%")
    print("=========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generalization on new datasets")
    parser.add_argument("--model_path", required=True, help="Path to .pt weights file")
    parser.add_argument("--cache_dir", required=True, help="Path to phase1_cache directory for evaluation")
    args = parser.parse_args()
    
    evaluate(args.model_path, args.cache_dir)
