# src/features/extractors.py - Production-ready version
import torch
import torch.nn as nn
import torchaudio

class LogMel(nn.Module):
    """
    Log-Mel Spectrogram (power) with CMVN and channel dimension
    
    Returns: [B, 1, n_mels, T] ready for CNN input
    """
    def __init__(self, sr=16000, n_mels=64, top_db=80, f_min=20.0, f_max=None, train_specaug=False):
        super().__init__()
        if f_max is None:
            f_max = sr / 2.0
        
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=400,
            win_length=400,
            hop_length=160,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=2.0,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            norm="slaney",
            mel_scale="htk"
        )
        
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
        
        # Optional SpecAugment for training
        self.train_specaug = train_specaug
        if train_specaug:
            self.specaug = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=max(1, n_mels // 8)),
                torchaudio.transforms.TimeMasking(time_mask_param=50),
            )

    def forward(self, x):  # x: [B, T]
        S = self.mel(x)              # [B, n_mels, T]
        S = self.db(S)               # Log power dB
        S = S.unsqueeze(1)           # [B, 1, n_mels, T]
        
        # Per-utterance CMVN (Cepstral Mean and Variance Normalization)
        mean = S.mean(dim=(-1, -2), keepdim=True)
        std = S.std(dim=(-1, -2), keepdim=True).clamp_min(1e-5)
        S = (S - mean) / std
        
        # Apply SpecAugment only during training
        if self.training and self.train_specaug:
            S = self.specaug(S)
        
        return S  # [B, 1, n_mels, T]


class MFCC(nn.Module):
    """
    MFCC with optional delta/delta-delta features and CMVN
    
    Returns: [B, 1, n_mfcc * (1 or 3), T] ready for CNN input
    """
    def __init__(self, sr=16000, n_mfcc=40, add_deltas=False, f_min=20.0, f_max=None):
        super().__init__()
        if f_max is None:
            f_max = sr / 2.0
        
        self.add_deltas = add_deltas
        
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,
                "win_length": 400,
                "hop_length": 160,
                "window_fn": torch.hann_window,
                "center": True,
                "pad_mode": "reflect",
                "power": 2.0,
                "f_min": f_min,
                "f_max": f_max,
                "norm": "slaney",
                "mel_scale": "htk",
            }
        )

    def forward(self, x):  # x: [B, T]
        M = self.mfcc(x)  # [B, n_mfcc, T]
        
        # Optional: Add delta and delta-delta features
        if self.add_deltas:
            d1 = torchaudio.functional.compute_deltas(M)
            d2 = torchaudio.functional.compute_deltas(d1)
            M = torch.cat([M, d1, d2], dim=1)  # [B, 3*n_mfcc, T]
        
        M = M.unsqueeze(1)  # [B, 1, n_mfcc (or 3*n_mfcc), T]
        
        # Per-utterance CMVN
        mean = M.mean(dim=(-1, -2), keepdim=True)
        std = M.std(dim=(-1, -2), keepdim=True).clamp_min(1e-5)
        M = (M - mean) / std
        
        return M  # [B, 1, F, T] where F = n_mfcc or 3*n_mfcc
