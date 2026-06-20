import glob
import json
import os

def summarize_ablations():
    results_dir = "experiments/results/phase2"
    files = glob.glob(os.path.join(results_dir, "ablation-*.json"))
    
    if not files:
        print("No ablation results found.")
        return
        
    print("| Ablation Run | Parameters | Train Loss | CE Loss | KL Loss |")
    print("|---|---|---|---|---|")
    
    for f in sorted(files):
        with open(f, "r") as file:
            data = json.load(file)
            name = os.path.basename(f).replace(".json", "")
            params = data.get("parameters", "N/A")
            train_loss = data.get("final_train_loss", "N/A")
            ce_loss = data.get("final_ce_loss", "N/A")
            kl_loss = data.get("final_kl_loss", "N/A")
            
            print(f"| {name} | {params:,} | {train_loss} | {ce_loss} | {kl_loss} |")

if __name__ == "__main__":
    summarize_ablations()
