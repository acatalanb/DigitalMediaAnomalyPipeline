#!/usr/bin/env python3
"""
Run VideoAnomalyPipeline experiments for multiple models and datasets.

# Default: runs all 3 models × 1 epoch value on crime-ucf (3 total runs)
python run_video_detection.py

# Specific models only
python run_video_detection.py --models "CNN-LSTM" "3D CNN"

# Specific models + epochs
python run_video_detection.py --models "Video Transformer" --epochs 5 20

# Custom dataset
python run_video_detection.py --dataset crime-ucf --epochs 10

# All models with custom epochs and dataset
python run_video_detection.py --dataset crime-ucf --epochs 5 20
"""

import argparse
import subprocess
import sys

def run_command(cmd: list[str]) -> int:
    """Run a command and stream its output. Returns the exit code."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    result = subprocess.run(cmd, check=False)
    return result.returncode

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VideoAnomalyPipeline with specified models, epochs, and dataset."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["CNN-LSTM", "3D CNN", "Video Transformer"],
        help='One or more model names to run (default: "CNN-LSTM" "3D CNN" "Video Transformer")',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[5],
        help="One or more epoch values to run (default: 5)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="crime-ucf",
        help="Dataset name to use (default: crime-ucf)",
    )
    args = parser.parse_args()

    failed = False
    for model in args.models:
        for epochs in args.epochs:
            cmd = [
                "python", "VideoAnomalyPipeline.py",
                "--mode", "all",
                "--model", model,
                "--epochs", str(epochs),
                "--dataset", args.dataset,
            ]
            code = run_command(cmd)
            if code != 0:
                print(f"\n[ERROR] Command failed with exit code {code}")
                failed = True
                # Continue to the next command even if one fails
                # (change to `sys.exit(code)` if you want to stop on first failure)

    if failed:
        sys.exit(1)
    print("\nAll runs completed successfully.")

if __name__ == "__main__":
    main()
