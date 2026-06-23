import torch
import argparse
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.transformer_verifier import TransformerVerifier
from src.explainability.saliency import GradientSaliencyExplainer

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
<h1>Distil-CRV Adversarial Saliency Probing</h1>
<p>The tokens highlighted in red represent the specific reasoning steps the verifier identified as flawed (or most critical for its decision).</p>
{content}
</body>
</html>
"""

def generate_html_highlight(tokens, scores):
    if scores.max() > 0:
        norm_scores = scores / scores.max()
    else:
        norm_scores = scores
        
    html_str = "<div>"
    import html
    for token, score in zip(tokens, norm_scores):
        t = token.replace("<|begin_of_text|>", "")
        t = html.escape(t)
        t = t.replace("Ġ", " ").replace("Ċ", "<br>").replace(" ", " ")
        if not t: continue
        
        alpha = score.item() * 0.8
        html_str += f'<span class="token" style="background-color: rgba(255, 0, 0, {alpha});">{t}</span>'
    html_str += "</div>"
    return html_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="experiments/results/phase2/ablation-final-layer.pt")
    parser.add_argument("--output_file", default="experiments/results/phase5/adversarial_saliency.html")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    device = args.device
    
    print(f"Loading tokenizer and base Llama-3.1-8B model...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    base_model.eval()
    
    print(f"Loading 15M Transformer Verifier from {args.model_path}...")
    verifier = TransformerVerifier(hidden_dim=4096, n_layers=4, n_heads=8, n_error_types=5)
    verifier.load_state_dict(torch.load(args.model_path, map_location=device))
    
    explainer = GradientSaliencyExplainer(verifier, device=device)
    
    # 1. Define Adversarial Traces
    traces = [
        {
            "name": "Original Correct Math",
            "text": "John has 5 apples, eats 2. 5 - 2 = 3"
        },
        {
            "name": "Adversarial Incorrect Math",
            "text": "John has 5 apples, eats 2. 5 - 2 = 4"
        }
    ]
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    html_content = ""
    
    print("\\nRunning Adversarial Saliency Evaluation...")
    for i, trace_info in enumerate(traces):
        name = trace_info["name"]
        text = trace_info["text"]
        print(f"Processing: {name}")
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        with torch.no_grad():
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # We want Layer 31 (the final layer before lm_head, index -1)
            # output_hidden_states returns tuple of length 33 (embedding + 32 layers)
            hidden_states = outputs.hidden_states[-1]
            
        # Compute saliency for the "Incorrect" class (0)
        saliency_scores = explainer.compute_saliency(
            hidden_states=hidden_states.to(torch.float32), # explainer might expect float32
            attention_mask=attention_mask,
            target_class=0 
        )
        
        # Verify if the model thought it was correct or incorrect
        verifier.eval()
        with torch.no_grad():
            ver_out = verifier(hidden_states.to(torch.float32), attention_mask=attention_mask)
            pred_class = torch.argmax(ver_out["verification_logits"], dim=-1)[0].item()
            pred_label = "Correct" if pred_class == 1 else "Incorrect"
            
        saliency_scores = saliency_scores[0]
        min_len = min(len(tokens), saliency_scores.shape[0])
        
        html_content += f'<div class="example">'
        html_content += f'<h2>{name}</h2>'
        html_content += f'<p><strong>Text:</strong> {text}</p>'
        html_content += f'<p><strong>Predicted Label:</strong> {pred_label}</p>'
        html_content += generate_html_highlight(tokens[:min_len], saliency_scores[:min_len])
        html_content += f'</div>'
        
    final_html = HTML_TEMPLATE.format(content=html_content)
    with open(args.output_file, "w") as f:
        f.write(final_html)
        
    print(f"\\n[SUCCESS] Generated adversarial visualizations at {args.output_file}")

if __name__ == "__main__":
    main()
