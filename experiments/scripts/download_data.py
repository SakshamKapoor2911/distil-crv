import os
from datasets import load_dataset

def main():
    os.makedirs("data/raw", exist_ok=True)
    print("Downloading GSM8K...")
    try:
        gsm8k = load_dataset("openai/gsm8k", "main")
        gsm8k.save_to_disk("data/raw/gsm8k")
        print("GSM8K downloaded successfully.")
    except Exception as e:
        print(f"Failed to download GSM8K: {e}")
    
    print("Downloading Synthetic Math (facebook/crv)...")
    try:
        crv_data = load_dataset("facebook/crv", "default")
        crv_data.save_to_disk("data/raw/crv_synthetic_math")
        print("CRV Synthetic Math downloaded successfully.")
    except Exception as e:
        print(f"Failed to download CRV Synthetic Math (Make sure HF_TOKEN is set if gated): {e}")

if __name__ == "__main__":
    main()
