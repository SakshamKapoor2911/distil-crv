import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.models.base_verifier import BaseVerifier

class TransformerVerifier(BaseVerifier):
    """
    Standalone small transformer that learns verification features 
    from hidden states. Model-agnostic and generalizable.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,  # Input hidden dim from Llama
        verifier_dim: int = 512,  # Internal verifier dimension
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        n_error_types: int = 5,  # ASA, AME, SBE, RSE, etc.
    ):
        super().__init__()
        
        # Project LLM hidden states to verifier dimension
        self.input_proj = nn.Linear(hidden_dim, verifier_dim)
        
        # Standalone transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=verifier_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification heads
        self.verification_head = nn.Sequential(
            nn.Linear(verifier_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # correct / incorrect
        )
        
        self.error_type_head = nn.Sequential(
            nn.Linear(verifier_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_error_types),
        )
        
        self.token_importance_head = nn.Linear(verifier_dim, 1)
        
        self.hidden_dim = hidden_dim
        self.verifier_dim = verifier_dim
    
    def forward(self, hidden_states, attention_mask=None, token_ids=None):
        # hidden_states: [batch, seq_len, hidden_dim] or [batch, num_layers, seq_len, hidden_dim]
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.mean(dim=1)
            
        # Project to verifier space
        x = self.input_proj(hidden_states)  # [batch, seq_len, verifier_dim]
        
        # Apply transformer
        # PyTorch transformer expects True for tokens to IGNORE (padding)
        padding_mask = ~attention_mask if attention_mask is not None else None
        x_pre_pool = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Pool over sequence dimension (mean of non-padded tokens)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            x_pool = (x_pre_pool * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x_pool = x_pre_pool.mean(dim=1)  # [batch, verifier_dim]
        
        # Classification outputs
        verification_logits = self.verification_head(x_pool)  # [batch, 2]
        error_type_logits = self.error_type_head(x_pool)  # [batch, n_error_types]
        
        # Token-level reasoning importance
        token_importance = self.token_importance_head(x_pre_pool).squeeze(-1)  # [batch, seq_len]
        
        # Calibrated confidence from softmax
        verification_probs = torch.softmax(verification_logits, dim=-1)
        confidence = verification_probs[:, 1]  # Probability of "correct"
        
        return {
            'verification_logits': verification_logits,
            'error_type_logits': error_type_logits,
            'confidence': confidence,
            'reasoning_scores': token_importance,
        }
    
    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
