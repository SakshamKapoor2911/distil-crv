import torch.nn as nn
from peft import get_peft_model, LoraConfig
from src.models.base_verifier import BaseVerifier

class LoRAVerifier(BaseVerifier):
    """
    LoRA adapter on top of base LLM. Lightweight alternative 
    for inference-time verification with minimal overhead.
    """
    
    def __init__(
        self,
        base_model,  # Llama-3.1-8B or other LLM
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: list = ["q_proj", "v_proj"],  # Llama-specific
        n_error_types: int = 5,
    ):
        super().__init__()
        
        # Attach LoRA to base model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.base_model = get_peft_model(base_model, lora_config)
        
        # Verification heads that operate on LLM hidden states
        hidden_dim = base_model.config.hidden_size
        
        self.verification_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(lora_dropout),
            nn.Linear(256, 2),
        )
        
        self.error_type_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(lora_dropout),
            nn.Linear(256, n_error_types),
        )
        
        self.n_error_types = n_error_types
    
    def forward(self, input_ids, attention_mask=None):
        # Forward pass through the LLM with LoRA
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract final hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Pool over sequence
        if attention_mask is not None:
            mask = ~attention_mask.unsqueeze(-1)
            x = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = hidden_states.mean(dim=1)
        
        verification_logits = self.verification_head(x)
        error_type_logits = self.error_type_head(x)
        
        import torch
        verification_probs = torch.softmax(verification_logits, dim=-1)
        confidence = verification_probs[:, 1]
        
        return {
            'verification_logits': verification_logits,
            'error_type_logits': error_type_logits,
            'confidence': confidence,
            'reasoning_scores': torch.zeros(hidden_states.shape[:-1]),  # Not computed for LoRA
        }
    
    def get_param_count(self) -> int:
        # Only count LoRA params
        lora_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        return lora_params
