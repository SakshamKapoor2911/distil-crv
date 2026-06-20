"""
Caching utilities for extracted hidden states and labels.
Enables efficient data loading without re-running the extractor.
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, Optional

class HiddenStateCache:
    """Manages on-disk cache for hidden states and CRV logits."""
    
    def __init__(self, cache_dir: str = "data/phase1_cache"):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cached tensors
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def save_example(
        self,
        example_id: str,
        hidden_states_dict: Dict[int, torch.Tensor],
        verification_logits: torch.Tensor,
        error_type_logits: torch.Tensor,
        label: int,
        seq_len: int,
    ) -> None:
        """
        Save extracted features for a single example to disk.
        """
        example_cache_dir = self.cache_dir / example_id
        example_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save hidden states
        for layer_idx, hs in hidden_states_dict.items():
            layer_path = example_cache_dir / f"layer_{layer_idx}.pt"
            torch.save(hs, layer_path)
        
        # Save logits and label
        torch.save(verification_logits, example_cache_dir / "verification_logits.pt")
        torch.save(error_type_logits, example_cache_dir / "error_type_logits.pt")
        torch.save(torch.tensor(label), example_cache_dir / "label.pt")
        
        # Save metadata
        metadata = {
            'example_id': example_id,
            'seq_len': seq_len,
            'label': label,
        }
        with open(example_cache_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def load_example(self, example_id: str) -> Dict:
        """
        Load cached example from disk.
        """
        example_cache_dir = self.cache_dir / example_id
        
        if not example_cache_dir.exists():
            raise FileNotFoundError(f"Cache not found for {example_id}")
        
        # Load metadata
        with open(example_cache_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        # Load hidden states for all layers
        hidden_states = {}
        for layer_file in sorted(example_cache_dir.glob("layer_*.pt")):
            layer_idx = int(layer_file.stem.split('_')[1])
            hidden_states[layer_idx] = torch.load(layer_file, map_location='cpu')
        
        # Load logits and label
        verification_logits = torch.load(
            example_cache_dir / "verification_logits.pt",
            map_location='cpu'
        )
        error_type_logits = torch.load(
            example_cache_dir / "error_type_logits.pt",
            map_location='cpu'
        )
        label = torch.load(
            example_cache_dir / "label.pt",
            map_location='cpu'
        ).item()
        
        return {
            'hidden_states': hidden_states,
            'verification_logits': verification_logits,
            'error_type_logits': error_type_logits,
            'label': label,
            'seq_len': metadata['seq_len'],
        }
    
    def list_cached_examples(self) -> list:
        """List all cached example IDs."""
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]
    
    def get_cache_size_gb(self) -> float:
        """Estimate total cache size in GB."""
        total_bytes = sum(
            f.stat().st_size
            for f in self.cache_dir.rglob('*')
            if f.is_file()
        )
        return total_bytes / (1024 ** 3)
