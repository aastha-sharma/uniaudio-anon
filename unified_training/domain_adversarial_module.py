"""
Domain Adversarial Neural Networks (DANN) for Cross-Domain Deepfake Detection

Implements gradient reversal layer and domain-adversarial training wrapper
for the 3-branch unified deepfake detector.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer
    
    Forward: identity (pass through)
    Backward: multiply gradient by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer wrapper
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class DomainClassifier(nn.Module):
    """
    Domain classifier for DANN
    
    Takes feature embeddings and predicts domain (0=speech, 1=singing)
    """
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # x: [batch, input_dim]
        return self.classifier(x)


class UnifiedDeepfakeDetector3Branch_DANN(nn.Module):
    """
    DANN wrapper for 3-branch unified deepfake detector
    
    Adds domain classifiers to expert branches (focusing on Wav2Vec2)
    """
    def __init__(self, base_model, use_all_experts=False):
        """
        Args:
            base_model: Pre-trained UnifiedDeepfakeDetector3Branch
            use_all_experts: If True, add domain classifiers to all experts.
                           If False, only add to Wav2Vec2 (most transferable)
        """
        super().__init__()
        
        self.base_model = base_model
        self.use_all_experts = use_all_experts
        
        # Gradient reversal layers
        self.grl_logmel = GradientReversalLayer()
        self.grl_mfcc = GradientReversalLayer()
        self.grl_w2v = GradientReversalLayer()
        
        # Domain classifiers (512 = embedding dim from ResNet)
        if use_all_experts:
            self.domain_classifier_logmel = DomainClassifier(input_dim=512)
            self.domain_classifier_mfcc = DomainClassifier(input_dim=512)
        
        self.domain_classifier_w2v = DomainClassifier(input_dim=512)
    
    def forward(self, x, return_expert_outputs=False, return_domain_logits=False):
        """
        Forward pass with optional domain classification
        
        Args:
            x: Audio input [batch, time]
            return_expert_outputs: If True, return expert outputs
            return_domain_logits: If True, compute domain predictions
        
        Returns:
            fused_logit: Final deepfake prediction
            gate_weights: Attention weights
            expert_outputs: (optional) Expert predictions and embeddings
            domain_logits: (optional) Domain predictions for each expert
        """
        # Get base model predictions
        if return_expert_outputs:
            fused_logit, gate_weights, expert_outputs = self.base_model(
                x, return_expert_outputs=True
            )
        else:
            fused_logit, gate_weights = self.base_model(x)
            expert_outputs = None
        
        # Domain classification (only if requested, i.e., during training)
        domain_logits = None
        if return_domain_logits and expert_outputs is not None:
            domain_logits = {}
            
            # Apply gradient reversal and domain classification
            if self.use_all_experts:
                # LogMel
                reversed_logmel = self.grl_logmel(expert_outputs['embeddings']['logmel'])
                domain_logits['logmel'] = self.domain_classifier_logmel(reversed_logmel)
                
                # MFCC
                reversed_mfcc = self.grl_mfcc(expert_outputs['embeddings']['mfcc'])
                domain_logits['mfcc'] = self.domain_classifier_mfcc(reversed_mfcc)
            
            # Wav2Vec2 (always)
            reversed_w2v = self.grl_w2v(expert_outputs['embeddings']['w2v_rn'])
            domain_logits['w2v_rn'] = self.domain_classifier_w2v(reversed_w2v)
        
        if return_domain_logits:
            return fused_logit, gate_weights, expert_outputs, domain_logits
        elif return_expert_outputs:
            return fused_logit, gate_weights, expert_outputs
        else:
            return fused_logit, gate_weights
    
    def set_lambda(self, lambda_value):
        """Set gradient reversal strength for all GRLs"""
        self.grl_logmel.set_lambda(lambda_value)
        self.grl_mfcc.set_lambda(lambda_value)
        self.grl_w2v.set_lambda(lambda_value)
    
    def freeze_base_model(self):
        """Freeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = True


class DANNLoss(nn.Module):
    """
    Combined loss for DANN training
    
    L_total = L_fake + lambda_aux * L_aux + lambda_domain * L_domain
    """
    def __init__(self, lambda_aux=0.1, lambda_domain=0.1, use_all_experts=False):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.lambda_domain = lambda_domain
        self.use_all_experts = use_all_experts
    
    def forward(self, fused_logit, expert_outputs, gate_weights, 
                domain_logits, labels, domain_labels):
        """
        Compute combined loss
        
        Args:
            fused_logit: Main deepfake prediction [batch, 1]
            expert_outputs: Dict with 'logits' for each expert
            gate_weights: Attention weights [batch, n_experts]
            domain_logits: Dict with domain predictions for each expert
            labels: Deepfake labels (0=bonafide, 1=spoof) [batch]
            domain_labels: Domain labels (0=speech, 1=singing) [batch]
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        labels = labels.float().view(-1, 1)
        domain_labels = domain_labels.float().view(-1, 1)
        
        # Main deepfake detection loss
        loss_main = F.binary_cross_entropy_with_logits(fused_logit, labels)
        
        # Auxiliary expert losses
        expert_names = ['logmel', 'mfcc', 'w2v_rn']
        aux_losses = []
        for name in expert_names:
            aux_loss = F.binary_cross_entropy_with_logits(
                expert_outputs['logits'][name], labels
            )
            aux_losses.append(aux_loss)
        loss_aux = sum(aux_losses) / len(aux_losses)
        
        # Domain classification losses
        domain_losses = []
        for name, logit in domain_logits.items():
            domain_loss = F.binary_cross_entropy_with_logits(logit, domain_labels)
            domain_losses.append(domain_loss)
        loss_domain = sum(domain_losses) / len(domain_losses)
        
        # Combined loss
        total_loss = (
            loss_main + 
            self.lambda_aux * loss_aux + 
            self.lambda_domain * loss_domain
        )
        
        loss_dict = {
            'total': float(total_loss.item()),
            'main': float(loss_main.item()),
            'aux': float(loss_aux.item()),
            'domain': float(loss_domain.item()),
        }
        
        return total_loss, loss_dict


def compute_lambda_schedule(epoch, max_epochs, schedule='exponential'):
    """
    Compute gradient reversal strength lambda
    
    Args:
        epoch: Current epoch
        max_epochs: Total epochs
        schedule: 'exponential', 'linear', or 'constant'
    
    Returns:
        lambda value in [0, 1]
    """
    if schedule == 'constant':
        return 1.0
    
    p = epoch / max(1, max_epochs)
    
    if schedule == 'exponential':
        # From DANN paper: lambda_p = 2/(1 + exp(-10*p)) - 1
        return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    
    elif schedule == 'linear':
        return p
    
    return 1.0