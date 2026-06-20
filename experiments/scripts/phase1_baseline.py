"""Phase 1: Reproduce CRV baseline on GSM8K subset."""
import argparse

def main(args):
    # TODO: Week 1-2 implementation
    # 1. Load CRV repo
    # 2. Run on 200 GSM8K examples
    # 3. Profile VRAM + latency
    # 4. Save results to experiments/results/phase1/
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, default=200)
    parser.add_argument("--output_dir", default="experiments/results/phase1")
    args = parser.parse_args()
    main(args)
