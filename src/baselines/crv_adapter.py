"""
Adapter to load Llama-3.1-8B and extract hidden states for distillation.
Minimal dependency on facebookresearch/CRV internals.
"""

import torch
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaFeatureExtractor:
    """
    Wrapper around Llama-3.1-8B for extracting hidden states.
    If CRV weights are unavailable, we extract real hidden states 
    and mock CRV logits to unblock the pipeline.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        device: str = "cuda:1",  # GPU 1 for LLM
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize Llama model for feature extraction.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            dtype: Data type (float16 for memory efficiency)
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="eager",  # Avoid attention weirdness
        )
        self.model.eval()  # Inference mode
        
    def extract_hidden_states(
        self,
        text: str,
        layer_indices: List[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract hidden states from specified layers for a given input text.
        
        Args:
            text: Input text (e.g., CoT reasoning trace)
            layer_indices: Which layers to extract (default: [24, 25, ..., 31])
        
        Returns:
            {
                'input_ids': [seq_len],
                'hidden_states_dict': {
                    24: [1, seq_len, 4096],
                    ...
                },
                'seq_len': int,
            }
        """
        if layer_indices is None:
            layer_indices = list(range(24, 32))  # Top 8 layers
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        seq_len = inputs['input_ids'].shape[1]
        
        # Register hooks to capture hidden states
        hidden_states_dict = {}
        hooks = []
        
        def hook_fn(layer_idx):
            def _hook(module, input, output):
                # output is a tuple: (last_hidden_state, ...)
                hidden_states_dict[layer_idx] = output[0].detach().cpu()
            return _hook
        
        # Attach hooks to transformer layers
        for layer_idx in layer_indices:
            layer_module = self.model.model.layers[layer_idx]
            hook = layer_module.register_forward_hook(hook_fn(layer_idx))
            hooks.append(hook)
        
        # Forward pass (no grad)
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Detach hooks
        for hook in hooks:
            hook.remove()
        
        return {
            'input_ids': inputs['input_ids'].cpu().squeeze(0),
            'hidden_states_dict': hidden_states_dict,
            'seq_len': seq_len,
        }

def load_llama_extractor(
    device: str = "cuda:1",
    dtype: torch.dtype = torch.float16,
) -> LlamaFeatureExtractor:
    """Factory function to load Llama extractor."""
    return LlamaFeatureExtractor(device=device, dtype=dtype)


def mock_crv_logits(
    num_examples: int,
    num_error_types: int = 5,
) -> Dict[str, torch.Tensor]:
    """
    Mock CRV logits for unblocking the pipeline.
    """
    return {
        'verification_logits': torch.randn(num_examples, 2),
        'error_type_logits': torch.randn(num_examples, num_error_types),
    }
