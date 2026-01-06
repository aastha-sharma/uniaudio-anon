# src/models/experts_w2v.py - WITH DROPOUT=0.0 FOR INITIAL TRAINING """ Wav2Vec2-based expert models """ 
import torch 
import torch.nn as nn 
from src.features.wav2vec import Wav2VecExtractor, AttentionPooling

class Wav2VecTransformerExpert(nn.Module):
    """Wav2Vec2 + Attention Pooling + Classifier"""
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        freeze_base: bool = False,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.wav2vec = Wav2VecExtractor(
            model_name=model_name,
            freeze_base=False,
            trainable_layers=0
        )

        # Unfreeze only top-2 encoder blocks
        for layer in self.wav2vec.wav2vec.encoder.layers:
            for p in layer.parameters():
                p.requires_grad = False
        for layer in self.wav2vec.wav2vec.encoder.layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.pool = AttentionPooling(768)
        self.norm = nn.LayerNorm(768)

        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, audio):
        H = self.wav2vec(audio)           # [B, T, 768]
        z = self.pool(H)                  # [B, 768]
        z = self.norm(z)
        return self.classifier(z).squeeze(-1)

class Wav2VecResNetExpert(nn.Module):
    """Wav2Vec2 + CNN - MODIFIED FOR BETTER LEARNING"""
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        freeze_base: bool = False,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()

        self.wav2vec = Wav2VecExtractor(
            model_name=model_name,
            freeze_base=False,
            trainable_layers=0
        )

        # Unfreeze only top-2 encoder blocks
        for layer in self.wav2vec.wav2vec.encoder.layers:
            for p in layer.parameters():
                p.requires_grad = False
        for layer in self.wav2vec.wav2vec.encoder.layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        # CNN with NO DROPOUT for initial training
        self.cnn = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.0),  # ← Changed to 0.0

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.0),  # ← Changed to 0.0

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.0),  # ← Changed to 0.0
            nn.Linear(64, 1)
        )

    def forward(self, audio):
        embeddings = self.wav2vec(audio)  # [B, T, 768]
        x = embeddings.transpose(1, 2)    # [B, 768, T]
        x = self.cnn(x).squeeze(-1)       # [B, 128]
        logits = self.classifier(x).squeeze(-1)  # [B]
        return logits