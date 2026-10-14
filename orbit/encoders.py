"""Speech and text encoders for ORBIT."""

import torch
import torch.nn as nn
from typing import Optional


class SpeechTextEncoders(nn.Module):
    """
    Load and manage speech and text encoders.

    Args:
        speech_model (str): Speech encoder model name ('wavlm', 'unispeech', 'whisper')
        text_model (str): Text encoder model name ('xlm-r', 'mbert', 'mt5')
        freeze_speech (bool): Freeze speech encoder weights
        freeze_text (bool): Freeze text encoder weights
    """

    def __init__(
        self,
        speech_model: str = "wavlm",
        text_model: str = "xlm-r",
        freeze_speech: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()
        self.speech_model_name = speech_model
        self.text_model_name = text_model

        # Load models (placeholder implementations)
        self.speech_encoder = self._load_speech_encoder(speech_model)
        self.text_encoder = self._load_text_encoder(text_model)

        # Freeze if requested
        if freeze_speech:
            for param in self.speech_encoder.parameters():
                param.requires_grad = False

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def _load_speech_encoder(self, model_name: str):
        """Load speech encoder."""
        if model_name == "wavlm":
            try:
                from transformers import AutoModel
                return AutoModel.from_pretrained("microsoft/wavlm-large")
            except:
                return nn.Linear(1, 1024)  # Placeholder
        elif model_name == "unispeech":
            try:
                from transformers import AutoModel
                return AutoModel.from_pretrained("microsoft/unispeech-sat-large")
            except:
                return nn.Linear(1, 1024)  # Placeholder
        else:
            return nn.Linear(1, 1024)  # Placeholder

    def _load_text_encoder(self, model_name: str):
        """Load text encoder."""
        if model_name == "xlm-r":
            try:
                from transformers import AutoModel
                return AutoModel.from_pretrained("xlm-roberta-large")
            except:
                return nn.Linear(1, 768)  # Placeholder
        elif model_name == "mbert":
            try:
                from transformers import AutoModel
                return AutoModel.from_pretrained("bert-base-multilingual-cased")
            except:
                return nn.Linear(1, 768)  # Placeholder
        else:
            return nn.Linear(1, 768)  # Placeholder

    def encode_speech(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode speech.

        Args:
            audio: Audio tensor (batch, sample_length)

        Returns:
            Speech embeddings (batch, seq_len, 1024)
        """
        with torch.no_grad():
            outputs = self.speech_encoder(audio, output_hidden_states=True)
            return outputs.last_hidden_state

    def encode_text(self, text_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode text.

        Args:
            text_ids: Text token IDs (batch, seq_len)
            attention_mask: Attention mask

        Returns:
            Text embeddings (batch, seq_len, 768)
        """
        with torch.no_grad():
            outputs = self.text_encoder(text_ids, attention_mask=attention_mask, output_hidden_states=True)
            return outputs.last_hidden_state
