"""
Wav2Vec2 feature extraction 
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from transformers import Wav2Vec2Model
except Exception as e:
    raise ImportError("transformers required") from e


class Wav2VecExtractor(nn.Module):
    """Extracts Wav2Vec2 embeddings from raw 16 kHz waveforms."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        freeze_base: bool = False,
        trainable_layers: int = 12,
    ) -> None:
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        self.output_dim: int = self.wav2vec.config.hidden_size

        if freeze_base:
            for p in self.wav2vec.parameters():
                p.requires_grad = False

            if trainable_layers > 0 and hasattr(self.wav2vec, "encoder"):
                for layer in self.wav2vec.encoder.layers[-trainable_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T] waveform at 16 kHz

        Returns:
            embeddings: [B, T', 768]
        """
        if audio.dim() != 2:
            raise ValueError(f"Expected [B, T], got {tuple(audio.shape)}")

        # Validate audio before processing
        if not torch.isfinite(audio).all():
            audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # REMOVED: Don't return zeros for quiet audio - let it process
        # Clamp to valid range
        audio = audio.clamp(-1.0, 1.0)
        
        try:
            # Forward pass
            outputs = self.wav2vec(audio)
            embeddings = outputs.last_hidden_state
            
            # Validate output
            if not torch.isfinite(embeddings).all():
                return torch.zeros_like(embeddings)
            
            return embeddings
            
        except Exception as e:
            print(f"Wav2Vec2 error: {e}")
            B = audio.shape[0]
            T_out = audio.shape[1] // 320
            return torch.zeros(B, max(1, T_out), 768, device=audio.device)

    def get_output_dim(self) -> int:
        return self.output_dim


class AttentionPooling(nn.Module):
    """Simple learnable attention pooling over the time axis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}")

        scores = self.score(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return pooled


class Wav2VecWithPooling(nn.Module):
    """Wav2Vec2 feature extractor with optional temporal pooling."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        freeze_base: bool = False,
        pooling: str = "mean",
        trainable_layers: int = 12,
    ) -> None:
        super().__init__()
        pooling = pooling.lower()
        if pooling not in {"mean", "max", "attention", "none"}:
            raise ValueError("pooling must be: 'mean', 'max', 'attention', 'none'")

        self.extractor = Wav2VecExtractor(
            model_name=model_name, 
            freeze_base=freeze_base, 
            trainable_layers=trainable_layers
        )
        self.pooling = pooling

        if self.pooling == "attention":
            self.attn_pool = AttentionPooling(self.extractor.get_output_dim())
        else:
            self.attn_pool = None

    def forward(self, audio: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.extractor(audio)

        if self.pooling == "none":
            return x

        if self.pooling == "mean":
            return x.mean(dim=1)

        if self.pooling == "max":
            return x.max(dim=1).values

        return self.attn_pool(x, mask)

    def get_output_dim(self) -> int:
        return self.extractor.get_output_dim()


__all__ = ["Wav2VecExtractor", "AttentionPooling", "Wav2VecWithPooling"]