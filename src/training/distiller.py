import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """Combined loss: KL divergence on soft targets + cross-entropy on hard labels."""
    
    def __init__(
        self,
        temperature: float = 3.0,
        alpha_soft: float = 0.7,  # Weight for soft target loss
        alpha_hard: float = 0.3,  # Weight for hard label loss
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_soft = alpha_soft
        self.alpha_hard = alpha_hard
    
    def forward(
        self,
        verifier_logits,  # Distil-CRV output
        crv_logits,       # CRV baseline (soft targets)
        true_labels,      # Hard labels (correct/incorrect)
    ):
        """
        Args:
            verifier_logits: [batch, 2]
            crv_logits: [batch, 2] (soft targets from CRV)
            true_labels: [batch] (hard binary labels)
        """
        
        # Soft target loss (KL divergence at temperature)
        soft_target_probs = F.softmax(crv_logits / self.temperature, dim=-1)
        verifier_log_probs = F.log_softmax(verifier_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(verifier_log_probs, soft_target_probs, reduction='batchmean')
        
        # Hard label loss (cross-entropy)
        ce_loss = F.cross_entropy(verifier_logits, true_labels)
        
        # Combined
        total_loss = self.alpha_soft * kl_loss + self.alpha_hard * ce_loss
        
        return {
            'loss': total_loss,
            'kl_loss': kl_loss,
            'ce_loss': ce_loss,
        }

class VerifierTrainer:
    """Trainer for distillation."""
    
    def __init__(self, verifier, optimizer, loss_fn, device='cuda'):
        self.verifier = verifier
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def train_step(self, batch):
        """Single training step."""
        hidden_states = batch['hidden_states'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        crv_logits = batch['crv_logits'].to(self.device)  # Soft targets
        labels = batch['labels'].to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass through verifier
        output = self.verifier(hidden_states, attention_mask)
        verifier_logits = output['verification_logits']
        
        # Compute loss
        loss_dict = self.loss_fn(verifier_logits, crv_logits, labels)
        loss = loss_dict['loss']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'train_loss': loss.item(),
            'kl_loss': loss_dict['kl_loss'].item(),
            'ce_loss': loss_dict['ce_loss'].item(),
        }
    
    @torch.no_grad()
    def eval_step(self, batch):
        """Validation step."""
        hidden_states = batch['hidden_states'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        output = self.verifier(hidden_states, attention_mask)
        verifier_logits = output['verification_logits']
        
        # Simple accuracy
        preds = verifier_logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean()
        
        return {
            'val_accuracy': accuracy.item(),
            'confidence_mean': output['confidence'].mean().item(),
        }
