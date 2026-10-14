"""Core ORBIT model for zero-shot cross-lingual Alzheimer's detection."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ORBIT(nn.Module):
    """
    ORBIT: Language-Invariant Multimodal Bi-Geometric Adversarial Learning.

    Detects Alzheimer's disease in zero-shot cross-lingual settings by synergizing:
    - Multimodal fusion of speech and text
    - Adversarial learning to suppress language-specific patterns
    - Bi-geometric projections (spherical + hyperbolic)

    Args:
        speech_dim (int): Speech encoder output dimension (e.g., 1024 for WavLM)
        text_dim (int): Text encoder output dimension (e.g., 768 for XLM-R)
        hidden_dim (int): Fusion hidden dimension
        num_heads (int): Number of cross-attention heads
        use_adversary (bool): Enable language adversaries
        num_adversaries (int): Number of adversary layers
        use_geometry (bool): Enable bi-geometric projections
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        speech_dim: int = 1024,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        use_adversary: bool = True,
        num_adversaries: int = 3,
        use_geometry: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.use_adversary = use_adversary
        self.use_geometry = use_geometry

        # Projection layers for alignment
        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Multimodal fusion
        from .fusion import MultimodalFusion
        self.fusion = MultimodalFusion(hidden_dim, num_heads, dropout)

        # Language adversaries
        if use_adversary:
            from .adversary import LanguageAdversary
            self.adversaries = nn.ModuleList([
                LanguageAdversary(hidden_dim, num_layers=2)
                for _ in range(num_adversaries)
            ])

        # Bi-geometric projections
        if use_geometry:
            from .geometry import BiGeometricProjection
            self.geometry = BiGeometricProjection(hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # Binary: Healthy vs Alzheimer
        )

    def forward(
        self,
        speech_features: torch.Tensor,
        text_features: torch.Tensor,
        compute_adversarial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ORBIT.

        Args:
            speech_features: Speech encoder outputs (batch, seq_len, speech_dim)
            text_features: Text encoder outputs (batch, seq_len, text_dim)
            compute_adversarial: Compute adversarial loss during training

        Returns:
            Dictionary with logits, probabilities, and optional adversarial info
        """
        # Project to common space
        speech_proj = self.speech_proj(speech_features)  # (batch, seq_len, hidden_dim)
        text_proj = self.text_proj(text_features)        # (batch, seq_len, hidden_dim)

        # Multimodal fusion
        fused = self.fusion(speech_proj, text_proj)  # (batch, hidden_dim)

        # Apply bi-geometric projections
        if self.use_geometry:
            fused = self.geometry(fused)

        # Classification
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)

        result = {
            "logits": logits,
            "healthy_prob": probs[:, 0],
            "alzheimer_prob": probs[:, 1],
            "features": fused,
        }

        # Adversarial info if requested
        if compute_adversarial and self.use_adversary:
            adv_predictions = []
            for adv in self.adversaries:
                adv_pred = adv(fused.detach())
                adv_predictions.append(adv_pred)
            result["adversarial_predictions"] = adv_predictions

        return result

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "ORBIT":
        """Load pretrained ORBIT model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def save(self, checkpoint_path: str, config: Optional[Dict] = None) -> None:
        """Save ORBIT model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {
                "speech_dim": self.speech_dim,
                "text_dim": self.text_dim,
                "hidden_dim": self.hidden_dim,
            },
        }
        torch.save(checkpoint, checkpoint_path)
