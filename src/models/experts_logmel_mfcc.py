"""
LOGMEL and MFCC Expert Models
Based on DeepfakeBranch architecture from train_expert.py
"""
import torch
import torch.nn as nn
import torchvision
from src.features.extractors import LogMel, MFCC


class ResNet18Embed(nn.Module):
    """ResNet-18 encoder that outputs 512-d embeddings"""
    def __init__(self, in_ch=1, out_dim=512):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            in_ch, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(512, out_dim)
    
    def forward(self, S):
        h = self.backbone(S)
        return self.proj(h)


class LogmelExpert(nn.Module):
    """LogMel + ResNet18 expert"""
    def __init__(self):
        super().__init__()
        self.feat = LogMel()
        self.resnet = ResNet18Embed(in_ch=1, out_dim=512)
        self.classifier = nn.Linear(512, 1)
    
    def extract_features(self, audio):
        """Extract features before classifier"""
        S = self.feat(audio)
        z = self.resnet(S)
        return z.unsqueeze(-1).unsqueeze(-1)  # [B, 512, 1, 1] for compatibility
    
    def forward(self, audio):
        S = self.feat(audio)
        z = self.resnet(S)
        logit = self.classifier(z)
        return logit.squeeze(1)


class MFCCExpert(nn.Module):
    """MFCC + ResNet18 expert"""
    def __init__(self):
        super().__init__()
        self.feat = MFCC()
        self.resnet = ResNet18Embed(in_ch=1, out_dim=512)
        self.classifier = nn.Linear(512, 1)
    
    def extract_features(self, audio):
        """Extract features before classifier"""
        S = self.feat(audio)
        z = self.resnet(S)
        return z.unsqueeze(-1).unsqueeze(-1)  # [B, 512, 1, 1] for compatibility
    
    def forward(self, audio):
        S = self.feat(audio)
        z = self.resnet(S)
        logit = self.classifier(z)
        return logit.squeeze(1)