import torch
import torch.nn as nn
from typing import Dict

class GradientSaliencyExplainer:
    """
    Computes token-level importance (saliency) by calculating the gradient 
    of the verification logits with respect to the input hidden states.
    """
    def __init__(self, verifier: nn.Module, device: str = "cuda:0"):
        self.verifier = verifier
        self.device = device
        self.verifier.to(device)
        self.verifier.eval()
        
    def compute_saliency(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor = None,
        target_class: int = 1, # 0 = incorrect, 1 = correct
    ) -> torch.Tensor:
        """
        Computes the gradient-based saliency map for the hidden states.
        
        Args:
            hidden_states: Float tensor of shape [batch, num_layers, seq_len, 4096] or [batch, seq_len, 4096]
            attention_mask: Boolean tensor of shape [batch, seq_len]
            target_class: Which class to compute gradients for.
            
        Returns:
            saliency_scores: Tensor of shape [batch, seq_len] with L2 normalized gradients.
        """
        # Move to device and ensure float32
        hidden_states = hidden_states.to(self.device, dtype=torch.float32)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        # We need gradients with respect to the input hidden states
        hidden_states.requires_grad_(True)
        
        # Forward pass
        outputs = self.verifier(hidden_states, attention_mask=attention_mask)
        verification_logits = outputs['verification_logits'] # [batch, 2]
        
        # Select the logits for the target class
        target_logits = verification_logits[:, target_class]
        
        # Compute gradients of target_logits with respect to hidden_states
        self.verifier.zero_grad()
        target_logits.sum().backward()
        
        gradients = hidden_states.grad # [batch, seq_len, 4096] or [batch, num_layers, seq_len, 4096]
        
        # If input was 4D, pool gradients across layers using mean (same as forward pass)
        if gradients.dim() == 4:
            gradients = gradients.mean(dim=1) # [batch, seq_len, 4096]
            
        # Compute L2 norm across the hidden dimension to get token-level importance
        # Shape: [batch, seq_len]
        saliency_scores = torch.norm(gradients, p=2, dim=-1)
        
        # Clean up
        hidden_states.requires_grad_(False)
        self.verifier.zero_grad()
        
        return saliency_scores
