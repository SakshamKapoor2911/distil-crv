import torch
import argparse
import sys
import os
import json
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.transformer_verifier import TransformerVerifier
from src.explainability.saliency import GradientSaliencyExplainer
from src.data.loader import build_dataloader

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: monospace; background-color: #1e1e1e; color: #d4d4d4; padding: 20px; }}
.example {{ border: 1px solid #333; margin-bottom: 20px; padding: 15px; border-radius: 5px; }}
h2 {{ color: #569cd6; }}
.token {{ padding: 2px 0px; border-radius: 2px; }}
.score {{ color: #888; font-size: 0.8em; margin-left: 10px; }}
</style>
</head>
<body>
<h1>Distil-CRV Error Detection & Token Highlighting</h1>
<p>The tokens highlighted in red represent the specific reasoning steps the verifier identified as flawed (or most critical for its decision).</p>
{content}
</body>
</html>
"""

def generate_html_highlight(tokens, scores):
    """
    Renders tokens with a background color intensity proportional to their saliency score.
    """
    # Normalize scores to [0, 1]
    if scores.max() > 0:
        norm_scores = scores / scores.max()
    else:
        norm_scores = scores
        
    html = "<div>"
    for token, score in zip(tokens, norm_scores):
        # Clean token
        t = token.replace("<|begin_of_text|>", "").replace(" ", " ")
        if not t: continue
        
        # Calculate color intensity (red)
        # alpha goes from 0 (transparent) to 0.8 (bright red)
        alpha = score.item() * 0.8
        html += f'<span class="token" style="background-color: rgba(255, 0, 0, {alpha});">{t}</span>'
    html += "</div>"
    return html

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="experiments/results/phase2/ablation-final-layer.pt")
    parser.add_argument("--cache_dir", default="data/math_cache")
    parser.add_argument("--raw_data", default="data/raw/synthetic_math/data.json")
    parser.add_argument("--output_file", default="experiments/results/phase3/highlights.html")
    args = parser.parse_args()
    
    device = "cuda:0"
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    
    print(f"Loading verifier from {args.model_path}...")
    verifier = TransformerVerifier(hidden_dim=4096, n_layers=4, n_heads=8, n_error_types=5)
    verifier.load_state_dict(torch.load(args.model_path, map_location=device))
    
    explainer = GradientSaliencyExplainer(verifier, device=device)
    
    print(f"Loading raw data...")
    with open(args.raw_data, "r") as f:
        raw_examples = json.load(f)
        
    print(f"Loading cache...")
    loader = build_dataloader(
        cache_dir=args.cache_dir,
        batch_size=1, # Process one by one for explainability
        shuffle=False,
        layer_indices=[31]
    )
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    html_content = ""
    
    for i, batch in enumerate(loader):
        if i >= 5: break # Highlight top 5 examples
        
        hidden_states = batch['hidden_states']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Get raw text
        example = raw_examples[i]
        reasoning_trace = f"{example['problem']}\\n{example['solution']}"
        
        # Tokenize to get string tokens
        # Note: Llama tokenizer tokens mapping
        input_ids = tokenizer(reasoning_trace, return_tensors="pt")['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Compute saliency for the "Incorrect" class (0) to see what it blames
        saliency_scores = explainer.compute_saliency(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            target_class=0 # Saliency for 'Incorrect' logit
        )
        
        saliency_scores = saliency_scores[0] # batch size 1
        
        # Truncate to min(len(tokens), len(scores))
        min_len = min(len(tokens), saliency_scores.shape[0])
        
        # Render HTML
        html_content += f'<div class="example">'
        html_content += f'<h2>Example {i+1}</h2>'
        html_content += f'<p><strong>Label:</strong> {"Correct" if labels[0].item() == 1 else "Incorrect"}</p>'
        html_content += generate_html_highlight(tokens[:min_len], saliency_scores[:min_len])
        html_content += f'</div>'
        
    final_html = HTML_TEMPLATE.format(content=html_content)
    with open(args.output_file, "w") as f:
        f.write(final_html)
        
    print(f"\\n[SUCCESS] Generated visualizations at {args.output_file}")

if __name__ == "__main__":
    main()
