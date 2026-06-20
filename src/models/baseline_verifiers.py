import torch
import torch.nn as nn
from src.models.base_verifier import BaseVerifier

class LinearVerifier(BaseVerifier):
    def __init__(self, hidden_dim: int = 4096, n_error_types: int = 5):
        super().__init__()
        self.verification_head = nn.Linear(hidden_dim, 2)
        self.error_type_head = nn.Linear(hidden_dim, n_error_types)

    def forward(self, hidden_states, attention_mask=None, token_ids=None):
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.mean(dim=1)
            
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            x_pool = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            x_pool = hidden_states.mean(dim=1)
            
        verification_logits = self.verification_head(x_pool)
        error_type_logits = self.error_type_head(x_pool)
        
        confidence = torch.softmax(verification_logits, dim=-1)[:, 1]
        
        return {
            'verification_logits': verification_logits,
            'error_type_logits': error_type_logits,
            'confidence': confidence,
            'reasoning_scores': torch.zeros(hidden_states.size(0), hidden_states.size(1), device=hidden_states.device)
        }

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLPVerifier(BaseVerifier):
    def __init__(self, hidden_dim: int = 4096, mlp_hidden_dim: int = 3660, n_error_types: int = 5):
        super().__init__()
        
        self.verification_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 2)
        )
        
        self.error_type_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_error_types)
        )

    def forward(self, hidden_states, attention_mask=None, token_ids=None):
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.mean(dim=1)
            
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            x_pool = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            x_pool = hidden_states.mean(dim=1)
            
        verification_logits = self.verification_head(x_pool)
        error_type_logits = self.error_type_head(x_pool)
        
        confidence = torch.softmax(verification_logits, dim=-1)[:, 1]
        
        return {
            'verification_logits': verification_logits,
            'error_type_logits': error_type_logits,
            'confidence': confidence,
            'reasoning_scores': torch.zeros(hidden_states.size(0), hidden_states.size(1), device=hidden_states.device)
        }

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
