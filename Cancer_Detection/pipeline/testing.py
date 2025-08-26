#!/usr/bin/env python3

import runpy
import argparse
from pathlib import Path
import sys

class Runner:
    def __init__(self, mode: str, dataset: str = None):
        self.mode = mode
        self.dataset = dataset
        # Base folder = this script's parent directory
        self.base_dir = Path(__file__).resolve().parent / "Cancer_Detection" / "main_model"

    def run(self):
        if self.mode == "train":
            script = self.base_dir / "training_utils_uncertainty.py"
            print(f"▶️  Training via `{script}`")
            runpy.run_path(str(script), run_name="__main__")
            print("✅ Model trained")

        elif self.mode == "test":
            if self.dataset != "cancer":
                sys.exit(f"❌ Unknown dataset '{self.dataset}' for testing")
            script = self.base_dir / "testing.py"
            print(f"▶️  Testing via `{script}`")
            runpy.run_path(str(script), run_name="__main__")
            print("✅ Model tested")

        else:
            sys.exit(f"❌ Unknown mode '{self.mode}'")

def main():
    p = argparse.ArgumentParser(
        description="Run training or testing scripts for Cancer_Detection."
    )
    p.add_argument(
        "--mode", 
        choices=["train", "test"], 
        required=True, 
        help="Whether to train or test the model"
    )
    p.add_argument(
        "--dataset",
        choices=["cancer"],
        help="For testing: which dataset to run on"
    )
    args = p.parse_args()

    runner = Runner(mode=args.mode, dataset=args.dataset)
    runner.run()

if __name__ == "__main__":
    main()
