import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

# ---  IMPORTS ---
from src.models.experts_w2v import Wav2VecResNetExpert
from src.models.experts_logmel_mfcc import LogmelExpert, MFCCExpert

class GatingNetwork3Branch(nn.Module):
    """3-way gating network with temperature-scaled softmax."""
    def __init__(self, input_dim=512, hidden_dim=256, num_experts=3, temperature=1.5):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(input_dim * num_experts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_experts),
        )

    def set_temperature(self, temp: float):
        self.temperature = max(float(temp), 0.1)

    def get_temperature(self) -> float:
        return self.temperature

    def forward(self, expert_embeddings):
        # expert_embeddings: list of 3 tensors [B, D]
        x = torch.cat(expert_embeddings, dim=1)
        logits = self.net(x)
        return logits  # Return logits, we apply softmax in the main model

class UnifiedDeepfakeDetector3Branch(nn.Module):
    def __init__(self, checkpoint_dir="./checkpoints", embedding_dim=512, min_gate_weight=0.12):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.embedding_dim = embedding_dim
        self.min_gate_weight = min_gate_weight  # CRITICAL FIX

        print("Initializing 3-Branch Unified Model...")

        # 1. Initialize Experts
        self.logmel = LogmelExpert()
        self.mfcc = MFCCExpert()
        self.w2v_rn = Wav2VecResNetExpert()
        
        # W2V embedding projection (128 -> 512 to match others)
        self.w2v_proj = nn.Linear(128, 512)
        
        # Load Weights
        self._load_expert_weights()

        # 2. Gate
        self.gate = GatingNetwork3Branch(
            input_dim=embedding_dim,
            hidden_dim=256,
            num_experts=3,
            temperature=1.5,
        )

        # 3. Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        print("3-Branch Unified Model initialized.")

    def _load_expert_weights(self):
        experts = [
            (self.logmel, "logmel_pretrained.pth", "logmel"),
            (self.mfcc, "mfcc_pretrained.pth", "mfcc"),
            (self.w2v_rn, "w2v_rn_pretrained.pth", "w2v_rn"),
        ]
        for expert, ckpt_name, name in experts:
            ckpt = self.checkpoint_dir / ckpt_name
            if ckpt.exists():
                try:
                    state = torch.load(ckpt, map_location="cpu")
                    # Handle module. prefix
                    if any(k.startswith("module.") for k in state.keys()):
                        state = {k.replace("module.", ""): v for k, v in state.items()}
                    expert.load_state_dict(state, strict=False)
                    print(f"Loaded {name} from {ckpt_name}")
                except Exception as e:
                    print(f"Error loading {name}: {e}")
            else:
                print(f"Warning: {name} checkpoint not found at {ckpt}")

    def forward(self, audio, return_expert_outputs=False):
        # 1. Get Logits (for Aux Loss)
        logmel_logit = self.logmel(audio).unsqueeze(1)
        mfcc_logit = self.mfcc(audio).unsqueeze(1)
        w2v_rn_logit = self.w2v_rn(audio).unsqueeze(1)
        
        # 2. Get Embeddings (for Gate/Fusion)
        # LogMel/MFCC
        logmel_emb = self.logmel.extract_features(audio).squeeze(-1).squeeze(-1)
        mfcc_emb = self.mfcc.extract_features(audio).squeeze(-1).squeeze(-1)
        
        # Wav2Vec (Requires projection 128->512)
        w2v_seq = self.w2v_rn.wav2vec(audio) 
        w2v_cnn_out = self.w2v_rn.cnn(w2v_seq.transpose(1, 2)).squeeze(-1)
        w2v_emb = self.w2v_proj(w2v_cnn_out)

        # 3. Gating
        expert_embeddings = [logmel_emb, mfcc_emb, w2v_emb]
        gate_logits = self.gate(expert_embeddings)
        
        # Softmax with temperature
        gate_weights = F.softmax(gate_logits / self.gate.temperature, dim=1)

        # --- CRITICAL FIX: HARD MINIMUM CONSTRAINT ---
        if self.training and self.min_gate_weight > 0:
            gate_weights = torch.clamp(gate_weights, min=self.min_gate_weight)
            gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True)
        # ---------------------------------------------

        # 4. Fusion
        fused_embedding = (
            gate_weights[:, 0:1] * logmel_emb +
            gate_weights[:, 1:2] * mfcc_emb +
            gate_weights[:, 2:3] * w2v_emb
        )
        fused_logit = self.fusion_head(fused_embedding)

        if return_expert_outputs:
            expert_outputs = {
                "logits": {
                    "logmel": logmel_logit,
                    "mfcc": mfcc_logit,
                    "w2v_rn": w2v_rn_logit,
                },
                "embeddings": {
                    "logmel": logmel_emb,
                    "mfcc": mfcc_emb,
                    "w2v_rn": w2v_emb,
                },
            }
            return fused_logit, gate_weights, expert_outputs

        return fused_logit, gate_weights

    # --- Freezing Helpers ---
    def freeze_experts(self, unfreeze_heads=True):
        print("Freezing expert backbones...")
        for expert in [self.logmel, self.mfcc, self.w2v_rn]:
            for p in expert.parameters(): p.requires_grad = False
        
        # Always unfreeze projections and heads
        for p in self.w2v_proj.parameters(): p.requires_grad = True
        for p in self.gate.parameters(): p.requires_grad = True
        for p in self.fusion_head.parameters(): p.requires_grad = True

        if unfreeze_heads:
            # Assumes experts have a .classifier or .fc head
            if hasattr(self.logmel, 'fc'):
                 for p in self.logmel.fc.parameters(): p.requires_grad = True
            if hasattr(self.mfcc, 'fc'):
                 for p in self.mfcc.fc.parameters(): p.requires_grad = True
            if hasattr(self.w2v_rn, 'fc'):
                 for p in self.w2v_rn.fc.parameters(): p.requires_grad = True

    def unfreeze_partial(self):
        print("Unfreezing partial backbones...")
        self.freeze_experts(unfreeze_heads=True)
        # Unfreeze Layer 4 of ResNets
        for expert in [self.logmel, self.mfcc]:
            if hasattr(expert, 'resnet') and hasattr(expert.resnet, 'layer4'):
                for p in expert.resnet.layer4.parameters(): p.requires_grad = True
        # Unfreeze W2V CNN
        if hasattr(self.w2v_rn, 'cnn'):
            for p in self.w2v_rn.cnn.parameters(): p.requires_grad = True

    def unfreeze_all(self):
        print("Unfreezing ALL...")
        for p in self.parameters(): p.requires_grad = True