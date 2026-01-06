# unified_training/unified_model.py
"""
Unified Deepfake Detector (4 Experts) + Soft Gate
WITH IMPROVEMENTS:
- Per-expert logit calibration (scale + bias)
- Better checkpoint loading
- Diagnostic methods
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating_network import SoftGate, fuse_logits

# External experts
from src.models.experts_w2v import Wav2VecTransformerExpert, Wav2VecResNetExpert
from src.models.experts_logmel_mfcc import LogmelExpert, MFCCExpert


class LogitCalibration(nn.Module):
    """Per-expert affine calibration: s' = a*s + b"""
    def __init__(self, n_experts=4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_experts))
        self.bias = nn.Parameter(torch.zeros(n_experts))
    
    def forward(self, expert_logits):
        """
        expert_logits: list of [B, 1] tensors
        returns: list of calibrated [B, 1] tensors
        """
        calibrated = []
        for i, logit in enumerate(expert_logits):
            cal_logit = self.scale[i] * logit + self.bias[i]
            calibrated.append(cal_logit)
        return calibrated


class UnifiedDeepfakeDetector(nn.Module):
    def __init__(self, checkpoint_dir: str | Path = "../checkpoints", use_signal_stats: bool = True):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)

        # Instantiate experts
        self.logmel_expert = LogmelExpert()
        self.mfcc_expert = MFCCExpert()
        self.w2v_transformer_expert = Wav2VecTransformerExpert()
        self.w2v_resnet_expert = Wav2VecResNetExpert()

        # Load expert checkpoints
        self._load_pretrained_weights()

        # Expert embedding dims
        expert_dims = [512, 512, 128, 768]

        # Logit calibration layer
        self.logit_calibration = LogitCalibration(n_experts=4)

        # Gate
        self.gate = SoftGate(
            expert_dims=expert_dims,
            hidden_dim=256,
            n_experts=4,
            use_signal_stats=use_signal_stats,
            initial_temp=2.0,
            sample_rate=16000,
        )

    # ---------------- Checkpoint Loading ---------------- #
    def _remap_state_dict_keys(self, sd: dict) -> dict:
        """Remap older naming (enc.* -> resnet.*, head -> classifier)"""
        remapped = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("enc."):
                nk = nk.replace("enc.", "resnet.")
            if nk.startswith("head."):
                nk = nk.replace("head.", "classifier.")
            remapped[nk] = v
        return remapped

    def _safe_load(self, module: nn.Module, ckpt_path: Path, strict_default: bool = True, name: str = ""):
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid checkpoint format for {ckpt_path}")
        try:
            module.load_state_dict(state, strict=strict_default)
            print(f"  ✓ Loaded {name} from {ckpt_path.name} (strict={strict_default})")
            return
        except Exception:
            pass
        remap = self._remap_state_dict_keys(state)
        missing, unexpected = module.load_state_dict(remap, strict=False)
        print(f"  ✓ Loaded {name} from {ckpt_path.name} with key remap (strict=False)")
        if missing:
            print(f"    • missing: {len(missing)} keys")
        if unexpected:
            print(f"    • unexpected: {len(unexpected)} keys")

    def _load_pretrained_weights(self):
        ckpt_dir = self.checkpoint_dir
        self._safe_load(self.logmel_expert, ckpt_dir / "logmel_pretrained.pth", strict_default=True, name="logmel")
        self._safe_load(self.mfcc_expert, ckpt_dir / "mfcc_pretrained.pth", strict_default=True, name="mfcc")
        self._safe_load(
            self.w2v_transformer_expert,
            ckpt_dir / "w2v_transformer_pretrained.pth",
            strict_default=True,
            name="w2v_transformer",
        )
        self._safe_load(self.w2v_resnet_expert, ckpt_dir / "w2v_rn_pretrained.pth", strict_default=True, name="w2v_rn")
        print("✓ All expert checkpoints loaded")

    # ---------------- Forward ---------------- #
    def forward(self, audio: torch.Tensor, return_expert_outputs: bool = False):
        """
        audio: [B, T] @ 16kHz
        returns:
            fused_logit: [B,1]
            gate_weights: [B,4]
            (optional) expert_outputs: dict
        """
        # LOGMEL
        logmel_z = self.logmel_expert.extract_features(audio).view(audio.size(0), -1)
        logmel_logit = self.logmel_expert.classifier(logmel_z).view(-1, 1)

        # MFCC
        mfcc_z = self.mfcc_expert.extract_features(audio).view(audio.size(0), -1)
        mfcc_logit = self.mfcc_expert.classifier(mfcc_z).view(-1, 1)

        # W2V Transformer
        w2v_tx_H = self.w2v_transformer_expert.wav2vec(audio)
        w2v_tx_z = self.w2v_transformer_expert.pool(w2v_tx_H)
        w2v_tx_z = self.w2v_transformer_expert.norm(w2v_tx_z)
        w2v_tx_logit = self.w2v_transformer_expert.classifier(w2v_tx_z).view(-1, 1)

        # W2V ResNet
        w2v_rn_H = self.w2v_resnet_expert.wav2vec(audio)
        x = w2v_rn_H.transpose(1, 2)
        w2v_rn_z = self.w2v_resnet_expert.cnn(x).squeeze(-1)
        w2v_rn_logit = self.w2v_resnet_expert.classifier(w2v_rn_z).view(-1, 1)

        # Gather expert outputs
        expert_embeddings = [logmel_z, mfcc_z, w2v_rn_z, w2v_tx_z]
        expert_logits = [logmel_logit, mfcc_logit, w2v_rn_logit, w2v_tx_logit]
        
        # CALIBRATE LOGITS before gating
        expert_logits_cal = self.logit_calibration(expert_logits)

        # Gate + fuse (use calibrated logits)
        gate_weights, gate_alpha = self.gate(expert_embeddings, audio)
        fused_logit = fuse_logits(expert_logits_cal, gate_weights)

        if return_expert_outputs:
            return fused_logit, gate_weights, {
                "embeddings": {
                    "logmel": logmel_z,
                    "mfcc": mfcc_z,
                    "w2v_rn": w2v_rn_z,
                    "w2v_tx": w2v_tx_z,
                },
                "logits": {
                    "logmel": logmel_logit,
                    "mfcc": mfcc_logit,
                    "w2v_rn": w2v_rn_logit,
                    "w2v_tx": w2v_tx_logit,
                },
                "logits_calibrated": {
                    "logmel": expert_logits_cal[0],
                    "mfcc": expert_logits_cal[1],
                    "w2v_rn": expert_logits_cal[2],
                    "w2v_tx": expert_logits_cal[3],
                },
                "gate_alpha": gate_alpha,
            }

        return fused_logit, gate_weights

    # ---------------- Freeze / Unfreeze helpers ---------------- #
    def freeze_experts(self, unfreeze_heads: bool = False):
        for expert in [self.logmel_expert, self.mfcc_expert, self.w2v_transformer_expert, self.w2v_resnet_expert]:
            for p in expert.parameters():
                p.requires_grad = False
            if unfreeze_heads and hasattr(expert, "classifier"):
                for p in expert.classifier.parameters():
                    p.requires_grad = True
        print("✓ Experts frozen" + (" (heads trainable)" if unfreeze_heads else ""))

    def unfreeze_partial(self):
        # LOGMEL/MFCC: layer3, layer4, proj, classifier
        for e in [self.logmel_expert, self.mfcc_expert]:
            for path in ["resnet.backbone.layer3", "resnet.backbone.layer4", "resnet.proj", "classifier"]:
                mod = e
                for p in path.split("."):
                    if not hasattr(mod, p):
                        mod = None
                        break
                    mod = getattr(mod, p)
                if mod is not None:
                    for param in mod.parameters(): param.requires_grad = True

        # W2V TX: classifier, pool, norm, last 2 blocks
        tx = self.w2v_transformer_expert
        for sub in ["classifier", "pool", "norm"]:
            if hasattr(tx, sub):
                for p in getattr(tx, sub).parameters(): p.requires_grad = True
        enc = getattr(getattr(tx, "wav2vec", None), "wav2vec", getattr(tx, "wav2vec", None))
        if enc and hasattr(enc, "encoder") and hasattr(enc.encoder, "layers"):
            for layer in enc.encoder.layers[-2:]:
                for p in layer.parameters(): p.requires_grad = True

        # W2V RN: cnn + classifier + last 2 blocks
        rn = self.w2v_resnet_expert
        if hasattr(rn, "cnn"):
            for p in rn.cnn.parameters(): p.requires_grad = True
        if hasattr(rn, "classifier"):
            for p in rn.classifier.parameters(): p.requires_grad = True
        enc_rn = getattr(getattr(rn, "wav2vec", None), "wav2vec", getattr(rn, "wav2vec", None))
        if enc_rn and hasattr(enc_rn, "encoder") and hasattr(enc_rn.encoder, "layers"):
            for layer in enc_rn.encoder.layers[-2:]:
                for p in layer.parameters(): p.requires_grad = True

        print("✓ Partial unfreeze done")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        print("✓ All params trainable")

    def get_trainable_params_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\nParameter Statistics:")
        print(f"  Total:     {total:,}")
        print(f"  Trainable: {trainable:,}")
        print(f"  Frozen:    {total - trainable:,}")
        print(f"  Trainable %: {100.0 * trainable / total:.2f}%")