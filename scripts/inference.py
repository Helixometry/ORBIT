"""Inference script for ORBIT."""

import argparse
import torch
from pathlib import Path

from orbit.model import ORBIT
from orbit.encoders import SpeechTextEncoders
from orbit.utils import get_device


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Inference with ORBIT")
    parser.add_argument("--audio", type=str, required=True, help="Audio file path")
    parser.add_argument("--text", type=str, required=True, help="Text transcription")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")

    args = parser.parse_args()

    device = get_device()
    model = ORBIT.from_pretrained(args.checkpoint).to(device)

    print(f"\nORBIT Inference Results")
    print("=" * 60)
    print(f"Audio: {Path(args.audio).name}")
    print(f"Text: {args.text[:50]}...")
    print(f"\n(Placeholder inference)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
