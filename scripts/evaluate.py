"""Evaluation script for ORBIT."""

import argparse
import torch
import numpy as np
from pathlib import Path

from orbit.model import ORBIT
from orbit.utils import get_device


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ORBIT model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--test-dir", type=str, required=True, help="Test data directory")

    args = parser.parse_args()

    device = get_device()
    model = ORBIT.from_pretrained(args.checkpoint).to(device)

    print(f"Loaded model from {args.checkpoint}")
    print(f"Testing on {args.test_dir}")
    print("\n(Placeholder evaluation)")


if __name__ == "__main__":
    main()
