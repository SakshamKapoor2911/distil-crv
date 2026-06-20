"""
Lightweight adapter to load CRV fingerprints and labels.
Minimal coupling to facebookresearch/CRV.
"""

def load_crv_fingerprints(dataset_name: str, split: str):
    """Load structural fingerprints from CRV baseline.
    
    Args:
        dataset_name: Name of dataset (e.g., 'gsm8k', 'synthetic_math')
        split: Data split (e.g., 'train', 'val', 'test')
    
    Returns:
        Dictionary mapping example_id -> fingerprint tensor
    """
    # TODO: Week 1 implementation
    pass

def extract_crv_labels(crv_output):
    """Extract error labels from CRV attribution graphs."""
    # TODO: Week 1 implementation
    pass
