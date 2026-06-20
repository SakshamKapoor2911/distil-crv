import torch
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.transformer_verifier import TransformerVerifier
from src.data.loader import build_dataloader
from src.training.distiller import DistillationLoss

def evaluate(model_path: str, cache_dir: str, layer_indices: list, model_type: str = "transformer", device: str = "cuda:0"):
    print(f"Loading weights from {model_path}...")
    
    if model_type == "transformer":
        verifier = TransformerVerifier(hidden_dim=4096, n_layers=4, n_heads=8, n_error_types=5).to(device)
    elif model_type == "linear":
        from src.models.baseline_verifiers import LinearVerifier
        verifier = LinearVerifier(hidden_dim=4096, n_error_types=5).to(device)
    elif model_type == "mlp":
        from src.models.baseline_verifiers import MLPVerifier
        verifier = MLPVerifier(hidden_dim=4096, mlp_hidden_dim=3660, n_error_types=5).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    verifier.load_state_dict(torch.load(model_path, map_location=device))
    verifier.eval()
    
    print(f"Loading evaluation dataset from {cache_dir}...")
    loader = build_dataloader(
        cache_dir=cache_dir,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        layer_indices=layer_indices
    )
    
    loss_fn = DistillationLoss()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            # Move to device
            hidden_states = batch['hidden_states'].to(device, dtype=torch.float32)
            attention_mask = batch['attention_mask'].to(device)
            crv_logits = batch['crv_logits'].to(device, dtype=torch.float32)
            error_type_logits = batch['error_type_logits'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = verifier(hidden_states, attention_mask=attention_mask)
            
            # Compute loss
            loss_dict = loss_fn(
                outputs['verification_logits'],
                crv_logits,
                labels
            )
            
            total_loss += loss_dict['loss'].item()
            total_ce += loss_dict['ce_loss'].item()
            total_kl += loss_dict['kl_loss'].item()
            
            # Accuracy on the hard label
            preds = torch.argmax(outputs['verification_logits'], dim=-1)
            probs = torch.softmax(outputs['verification_logits'], dim=-1)[:, 1] # Prob of class 1
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
            
    num_batches = len(loader)
    avg_loss = total_loss / num_batches
    avg_ce = total_ce / num_batches
    avg_kl = total_kl / num_batches
    accuracy = correct_predictions / total_predictions
    
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    correct_count = np.sum(y_true == 1)
    incorrect_count = np.sum(y_true == 0)
    
    print("\n=== Generalization Evaluation Results ===")
    print(f"Model: {model_path}")
    print(f"Dataset: {cache_dir} ({total_predictions} examples)")
    print(f"Class Distribution: Tested on {correct_count} correct traces and {incorrect_count} incorrect traces")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"  Cross-Entropy: {avg_ce:.4f}")
    print(f"  KL-Divergence: {avg_kl:.4f}")
    print(f"Verification Accuracy: {accuracy * 100:.2f}%")
    print(f"AUROC: {auroc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("=========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generalization on new datasets")
    parser.add_argument("--model_path", required=True, help="Path to .pt weights file")
    parser.add_argument("--cache_dir", required=True, help="Path to phase1_cache directory for evaluation")
    parser.add_argument("--layer_indices", type=str, default="[24,25,26,27,28,29,30,31]", help="JSON string of layer indices")
    parser.add_argument("--model_type", type=str, default="transformer", help="Type of model (transformer, linear, mlp)")
    args = parser.parse_args()
    
    import json
    layer_indices = json.loads(args.layer_indices)
    evaluate(args.model_path, args.cache_dir, layer_indices, args.model_type)
