import torch
from torch.utils.data import Dataset, DataLoader
from src.data.cache import HiddenStateCache

class CachedHiddenStatesDataset(Dataset):
    """PyTorch Dataset that loads cached hidden states and targets."""
    
    def __init__(self, cache_dir: str = "data/phase1_cache", layer_indices: list = None):
        self.cache = HiddenStateCache(cache_dir)
        self.example_ids = sorted(self.cache.list_cached_examples())
        self.layer_indices = layer_indices
    
    def __len__(self):
        return len(self.example_ids)
    
    def __getitem__(self, idx):
        example_id = self.example_ids[idx]
        cached = self.cache.load_example(example_id)
        
        # Stack hidden states from selected layers into a single tensor
        # Shape: [num_layers, seq_len, hidden_dim]
        available_layers = sorted(cached['hidden_states'].keys())
        layer_indices = self.layer_indices if self.layer_indices is not None else available_layers
        
        hidden_states = torch.stack([
            cached['hidden_states'][layer_idx].squeeze(0)
            for layer_idx in layer_indices
        ])
        
        return {
            'hidden_states': hidden_states,  # [8, seq_len, 4096]
            'verification_logits': cached['verification_logits'],  # [2]
            'error_type_logits': cached['error_type_logits'],  # [5]
            'label': torch.tensor(cached['label'], dtype=torch.long),  # scalar
            'seq_len': cached['seq_len'],
        }


def collate_fn(batch):
    """Custom collate function to pad variable-length sequences."""
    # Find max sequence length in batch
    max_seq_len = max(b['seq_len'] for b in batch)
    
    batch_hidden_states = []
    batch_verification_logits = []
    batch_error_type_logits = []
    batch_labels = []
    batch_attention_masks = []
    
    for b in batch:
        hs = b['hidden_states']  # [8, seq_len, 4096]
        seq_len = hs.shape[1]
        
        # Pad sequence dimension to max_seq_len
        if seq_len < max_seq_len:
            pad_amount = max_seq_len - seq_len
            hs = torch.nn.functional.pad(hs, (0, 0, 0, pad_amount))  # Pad seq_len dim
        
        batch_hidden_states.append(hs)
        batch_verification_logits.append(b['verification_logits'])
        batch_error_type_logits.append(b['error_type_logits'])
        batch_labels.append(b['label'])
        
        # Attention mask: 1 for real tokens, 0 for padding
        mask = torch.ones(max_seq_len, dtype=torch.bool)
        if seq_len < max_seq_len:
            mask[seq_len:] = False
        batch_attention_masks.append(mask)
    
    return {
        'hidden_states': torch.stack(batch_hidden_states),  # [batch, 8, seq_len, 4096]
        'crv_logits': torch.stack(batch_verification_logits),  # [batch, 2] # NOTE: Changed key to match trainer expectation 'crv_logits'
        'error_type_logits': torch.stack(batch_error_type_logits),  # [batch, 5]
        'labels': torch.stack(batch_labels),  # [batch]
        'attention_mask': torch.stack(batch_attention_masks),  # [batch, seq_len]
    }


def build_dataloader(
    cache_dir: str = "data/phase1_cache",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    layer_indices: list = None,
):
    """Factory function to build a DataLoader from cached hidden states."""
    dataset = CachedHiddenStatesDataset(cache_dir, layer_indices=layer_indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
