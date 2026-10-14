"""Multimodal fusion module for ORBIT."""

import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """
    Cross-attention based multimodal fusion of speech and text.

    Args:
        hidden_dim (int): Feature dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Cross-attention layers
        self.speech_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_speech_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Fusion gates
        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        speech: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse speech and text through cross-attention.

        Args:
            speech: (batch, seq_len, hidden_dim)
            text: (batch, seq_len, hidden_dim)

        Returns:
            Fused representation: (batch, hidden_dim)
        """
        batch_size = speech.shape[0]

        # Cross-attention: speech attends to text
        speech_attended, _ = self.speech_to_text_attn(speech, text, text)
        speech_fused = self.norm1(speech + self.dropout(speech_attended))

        # Cross-attention: text attends to speech
        text_attended, _ = self.text_to_speech_attn(text, speech, speech)
        text_fused = self.norm2(text + self.dropout(text_attended))

        # Average pooling to get utterance-level representations
        speech_pool = speech_fused.mean(dim=1)  # (batch, hidden_dim)
        text_pool = text_fused.mean(dim=1)      # (batch, hidden_dim)

        # Learned fusion gate
        combined = torch.cat([speech_pool, text_pool], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))

        # Gated combination
        fused = gate * speech_pool + (1 - gate) * text_pool
        fused = self.norm3(fused)

        return fused
