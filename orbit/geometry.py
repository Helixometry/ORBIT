"""Bi-geometric projections for ORBIT."""

import torch
import torch.nn as nn


class BiGeometricProjection(nn.Module):
    """
    Bi-geometric projections combining spherical and hyperbolic spaces.

    Args:
        hidden_dim (int): Feature dimension
        use_spherical (bool): Enable spherical projection
        use_hyperbolic (bool): Enable hyperbolic projection
    """

    def __init__(
        self,
        hidden_dim: int,
        use_spherical: bool = True,
        use_hyperbolic: bool = True,
    ):
        super().__init__()
        self.use_spherical = use_spherical
        self.use_hyperbolic = use_hyperbolic

        if use_spherical:
            self.spherical_proj = nn.Linear(hidden_dim, hidden_dim)

        if use_hyperbolic:
            self.hyperbolic_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable weights for combining projections
        self.register_parameter(
            "alpha",
            nn.Parameter(torch.tensor([0.5]))
        )
        self.register_parameter(
            "beta",
            nn.Parameter(torch.tensor([0.5]))
        )

        # Consensus clustering
        self.consensus = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply bi-geometric projections.

        Args:
            x: Input features (batch, hidden_dim)

        Returns:
            Projected features (batch, hidden_dim)
        """
        projections = []

        # Spherical projection (normalized)
        if self.use_spherical:
            s = self.spherical_proj(x)
            s = torch.nn.functional.normalize(s, dim=-1)
            projections.append(self.alpha * s)

        # Hyperbolic projection (Poincaré ball)
        if self.use_hyperbolic:
            h = self.hyperbolic_proj(x)
            h = torch.tanh(h)  # Poincaré ball normalization
            projections.append(self.beta * h)

        # Combine projections
        if projections:
            combined = sum(projections)
        else:
            combined = x

        # Consensus clustering
        output = self.consensus(combined)

        return output
