from abc import ABC, abstractmethod
import torch.nn as nn

class BaseVerifier(nn.Module, ABC):
    """Abstract base for all verification approaches."""
    
    @abstractmethod
    def forward(self, hidden_states, attention_mask=None, token_ids=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim] from LLM layers
            attention_mask: [batch, seq_len] (optional)
            token_ids: [batch, seq_len] (optional, for error type classification)
        
        Returns:
            {
                'verification_logits': [batch, 2],  # correct/incorrect binary
                'error_type_logits': [batch, n_error_types],  # ASA, AME, etc.
                'confidence': [batch],  # [0, 1] calibrated probability
                'reasoning_scores': [batch, seq_len],  # token-level importance
            }
        """
        pass
    
    @abstractmethod
    def get_param_count(self) -> int:
        """Total trainable parameters."""
        pass
