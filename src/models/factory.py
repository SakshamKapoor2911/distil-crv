def build_verifier(config: dict, base_model=None):
    """Factory function to instantiate verifier."""
    verifier_type = config.get("type", "transformer")
    
    if verifier_type == "transformer":
        from src.models.transformer_verifier import TransformerVerifier
        return TransformerVerifier(
            hidden_dim=config.get("hidden_dim", 4096),  # Llama-3.1 hidden dim
            verifier_dim=config.get("verifier_dim", 512),
            n_layers=config.get("n_layers", 4),
            n_heads=config.get("n_heads", 8),
            ff_dim=config.get("ff_dim", 2048),
            dropout=config.get("dropout", 0.1),
            n_error_types=config.get("n_error_types", 5),
        )
    
    elif verifier_type == "lora":
        from src.models.lora_verifier import LoRAVerifier
        assert base_model is not None, "LoRA verifier requires base_model"
        return LoRAVerifier(
            base_model=base_model,
            lora_r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.1),
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
            n_error_types=config.get("n_error_types", 5),
        )
        
    elif verifier_type == "linear":
        from src.models.baseline_verifiers import LinearVerifier
        return LinearVerifier(
            hidden_dim=config.get("hidden_dim", 4096),
            n_error_types=config.get("n_error_types", 5),
        )
        
    elif verifier_type == "mlp":
        from src.models.baseline_verifiers import MLPVerifier
        return MLPVerifier(
            hidden_dim=config.get("hidden_dim", 4096),
            mlp_hidden_dim=config.get("mlp_hidden_dim", 3660),
            n_error_types=config.get("n_error_types", 5),
        )
    
    else:
        raise ValueError(f"Unknown verifier type: {verifier_type}")
