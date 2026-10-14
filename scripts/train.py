"""Training script for ORBIT."""

import argparse
import logging
import torch
import torch.nn as nn
import yaml
from pathlib import Path

from orbit.model import ORBIT
from orbit.utils import seed_everything, get_device

logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, device, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        speech = batch['speech'].to(device)
        text = batch['text'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(speech, text, compute_adversarial=True)

        # Classification loss
        loss = nn.CrossEntropyLoss()(outputs['logits'], labels)

        # Adversarial loss
        if model.use_adversary and 'adversarial_predictions' in outputs:
            # Gradient reversal and adversarial loss
            for adv_pred in outputs['adversarial_predictions']:
                adv_loss = nn.CrossEntropyLoss()(adv_pred, batch['language'].to(device))
                loss -= args['training']['loss_weights']['adversary'] * adv_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['training']['gradient_clip'])
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

    return total_loss / (batch_idx + 1)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ORBIT model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--epochs", type=int, help="Override epochs")

    args = parser.parse_args()

    # Setup
    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config['training']['epochs'] = args.epochs

    seed_everything(config.get('seed', 42))
    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize model
    model = ORBIT(**config['model']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training for {config['training']['epochs']} epochs")

    for epoch in range(config['training']['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        logger.info("  (Placeholder training)")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            model.save(str(checkpoint_path), config['model'])


if __name__ == "__main__":
    main()
