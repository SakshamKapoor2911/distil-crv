import os
from datasets import load_dataset

def main():
    os.makedirs("data/raw", exist_ok=True)
    print("Downloading GSM8K...")
    try:
        gsm8k = load_dataset("gsm8k", "main")
        gsm8k.save_to_disk("data/raw/gsm8k")
        print("GSM8K downloaded successfully.")
    except Exception as e:
        print(f"Failed to download GSM8K: {e}")
    
    print("Downloading MATH benchmark...")
    try:
        math_data = load_dataset("hendrycks/competition_math")
        math_data.save_to_disk("data/raw/math")
        print("MATH benchmark downloaded successfully.")
    except Exception as e:
        print(f"Failed to download MATH: {e}")

    print("Downloading SVAMP benchmark...")
    try:
        svamp_data = load_dataset("ChilleD/SVAMP")
        svamp_data.save_to_disk("data/raw/svamp")
        print("SVAMP benchmark downloaded successfully.")
    except Exception as e:
        print(f"Failed to download SVAMP: {e}")

    print("Downloading ASDiv benchmark (Zero-Shot Holdout)...")
    try:
        asdiv_data = load_dataset("EleutherAI/asdiv")
        asdiv_data.save_to_disk("data/raw/asdiv")
        print("ASDiv benchmark downloaded successfully.")
    except Exception as e:
        print(f"Failed to download ASDiv: {e}")

if __name__ == "__main__":
    main()
