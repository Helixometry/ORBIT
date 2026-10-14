"""Language adversary module for ORBIT."""

import torch
import torch.nn as nn


class LanguageAdversary(nn.Module):
    """
    Language discriminator for suppressing language-specific information.

    Multi-tap approach: operates at different representation levels.

    Args:
        hidden_dim (int): Feature dimension
        num_languages (int): Number of language classes
        num_layers (int): Number of hidden layers
    """

    def __init__(self, hidden_dim: int, num_languages: int = 10, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        input_dim = hidden_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
            ])
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dim, num_languages))

        self.discriminator = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict language from features.

        Args:
            features: (batch, hidden_dim)

        Returns:
            Language logits: (batch, num_languages)
        """
        return self.discriminator(features)


class MultiTapAdversary(nn.Module):
    """
    Multi-tap language adversaries at different representation levels.

    Args:
        hidden_dim (int): Feature dimension
        num_taps (int): Number of adversary taps
        num_languages (int): Number of language classes
    """

    def __init__(self, hidden_dim: int, num_taps: int = 3, num_languages: int = 10):
        super().__init__()
        self.adversaries = nn.ModuleList([
            LanguageAdversary(hidden_dim, num_languages, num_layers=2)
            for _ in range(num_taps)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate predictions from multiple adversaries.

        Args:
            features: (batch, hidden_dim)

        Returns:
            Aggregated language logits: (batch, num_languages)
        """
        predictions = []
        for adv in self.adversaries:
            pred = adv(features)
            predictions.append(pred)

        # Average predictions from all taps
        return sum(predictions) / len(predictions)
