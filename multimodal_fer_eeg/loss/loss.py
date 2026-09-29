"""
Loss functions for Multimodal FER + EEG Deception Detection.

Implements multi-task loss:
L_total = α L_face + β L_eeg + γ L_fusion + λ L_congruence

Where:
- L_face: Cross-entropy for facial emotion classification
- L_eeg: Cross-entropy for EEG emotion + MSE for valence/arousal regression
- L_fusion: Cross-entropy for fused emotion classification
- L_congruence: Binary cross-entropy for deception detection or contrastive loss
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Loss weight configuration."""
    alpha: float = 1.0    # L_face
    beta: float = 1.0     # L_eeg
    gamma: float = 1.0    # L_fusion
    lambda_: float = 0.5  # L_congruence
    
    # EEG sub-loss weights
    eeg_cls_weight: float = 1.0      # EEG emotion classification
    eeg_valence_weight: float = 0.5  # Valence regression
    eeg_arousal_weight: float = 0.5  # Arousal regression


class FERLoss(nn.Module):
    """Facial Emotion Recognition loss (Cross-Entropy)."""
    
    def __init__(self, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


class EEGLoss(nn.Module):
    """
    EEG loss combining discrete emotion classification and continuous valence/arousal regression.
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        valence_weight: float = 0.5,
        arousal_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.valence_weight = valence_weight
        self.arousal_weight = arousal_weight
        
        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.reg_criterion = nn.MSELoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        valence_pred: torch.Tensor,
        arousal_pred: torch.Tensor,
        emotion_targets: torch.Tensor,
        valence_targets: torch.Tensor,
        arousal_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits: (B, num_classes) EEG emotion logits
            valence_pred: (B, 1) predicted valence
            arousal_pred: (B, 1) predicted arousal
            emotion_targets: (B,) discrete emotion labels
            valence_targets: (B, 1) continuous valence targets [-1, 1]
            arousal_targets: (B, 1) continuous arousal targets [0, 1]
            
        Returns:
            total_loss, loss_components_dict
        """
        # Classification loss
        cls_loss = self.cls_criterion(logits, emotion_targets)
        
        # Regression losses
        valence_loss = self.reg_criterion(valence_pred, valence_targets)
        arousal_loss = self.reg_criterion(arousal_pred, arousal_targets)
        
        total = (
            self.cls_weight * cls_loss +
            self.valence_weight * valence_loss +
            self.arousal_weight * arousal_loss
        )
        
        return total, {
            "eeg_cls": cls_loss.item(),
            "eeg_valence": valence_loss.item(),
            "eeg_arousal": arousal_loss.item(),
        }


class FusionLoss(nn.Module):
    """Fused emotion classification loss."""
    
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


class CongruenceLoss(nn.Module):
    """
    Deception/Incongruence detection loss.
    
    Supports multiple modes:
    - BCE: Binary cross-entropy for genuine vs deceptive
    - Contrastive: Contrastive loss pulling congruent pairs together, pushing incongruent apart
    - Triplet: Triplet loss with anchor (face), positive (congruent EEG), negative (incongruent EEG)
    - Similarity: Direct similarity regression (cosine similarity target 1 for congruent, 0 for incongruent)
    """
    
    def __init__(
        self,
        mode: str = "bce",           # bce, contrastive, triplet, similarity
        margin: float = 1.0,         # for contrastive/triplet
        temperature: float = 0.1,    # for similarity/contrastive
    ):
        super().__init__()
        self.mode = mode
        self.margin = margin
        self.temperature = temperature
        
        if mode == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif mode == "contrastive":
            self.criterion = nn.CosineEmbeddingLoss(margin=margin)
        elif mode == "similarity":
            self.criterion = nn.MSELoss()
        elif mode == "triplet":
            self.criterion = nn.TripletMarginLoss(margin=margin, p=2)
        else:
            raise ValueError(f"Unknown congruence loss mode: {mode}")
    
    def forward(
        self,
        z_face: torch.Tensor,
        z_eeg: torch.Tensor,
        congruence_labels: torch.Tensor,  # 0 = genuine, 1 = deceptive
        deception_logits: Optional[torch.Tensor] = None,
        similarity: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            z_face: (B, D) face latent features
            z_eeg: (B, D) EEG latent features
            congruence_labels: (B,) 0=genuine, 1=deceptive
            deception_logits: (B, 2) or (B, 1) deception classifier logits (for BCE mode)
            similarity: (B, 1) precomputed similarity scores (for similarity mode)
            
        Returns:
            loss, loss_components_dict
        """
        if self.mode == "bce":
            # Binary classification: genuine (0) vs deceptive (1)
            if deception_logits is None:
                raise ValueError("deception_logits required for BCE mode")
            # Expect logits for 2 classes, get prob of class 1 (deceptive)
            if deception_logits.size(1) == 2:
                deception_prob = deception_logits[:, 1]
            else:
                deception_prob = deception_logits.squeeze(1)
            target = congruence_labels.float()
            loss = self.criterion(deception_prob, target)
            return loss, {"congruence_bce": loss.item()}
        
        elif self.mode == "contrastive":
            # Cosine embedding loss: target=1 for congruent (pull together), target=-1 for incongruent (push apart)
            target = 1.0 - 2.0 * congruence_labels.float()  # 0->1, 1->-1
            loss = self.criterion(z_face, z_eeg, target)
            return loss, {"congruence_contrastive": loss.item()}
        
        elif self.mode == "similarity":
            # Direct similarity regression
            if similarity is None:
                similarity = F.cosine_similarity(z_face, z_eeg, dim=1).unsqueeze(1)
            # Target: 1 for congruent, 0 for incongruent
            target = (1.0 - congruence_labels.float()).unsqueeze(1)
            loss = self.criterion(similarity, target)
            return loss, {"congruence_similarity": loss.item()}
        
        elif self.mode == "triplet":
            # Triplet loss: anchor=face, positive=congruent EEG, negative=incongruent EEG
            # This requires sampling triplets - simplified version uses all pairs
            # In practice, you'd use a triplet sampler
            congruent_mask = (congruence_labels == 0)
            incongruent_mask = (congruence_labels == 1)
            
            if congruent_mask.sum() == 0 or incongruent_mask.sum() == 0:
                return torch.tensor(0.0, device=z_face.device), {"congruence_triplet": 0.0}
            
            # Sample anchor from congruent
            anchor_idx = torch.where(congruent_mask)[0][0]
            anchor = z_face[anchor_idx:anchor_idx+1]
            positive = z_eeg[anchor_idx:anchor_idx+1]
            
            # Sample negative from incongruent
            neg_idx = torch.where(incongruent_mask)[0][0]
            negative = z_eeg[neg_idx:neg_idx+1]
            
            loss = self.criterion(anchor, positive, negative)
            return loss, {"congruence_triplet": loss.item()}


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for the full multimodal model.
    
    L_total = α L_face + β L_eeg + γ L_fusion + λ L_congruence
    """
    
    def __init__(self, weights: LossWeights, congruence_mode: str = "bce"):
        super().__init__()
        self.weights = weights
        
        self.face_loss = FERLoss(label_smoothing=0.1)
        self.eeg_loss = EEGLoss(
            cls_weight=weights.eeg_cls_weight,
            valence_weight=weights.eeg_valence_weight,
            arousal_weight=weights.eeg_arousal_weight,
        )
        self.fusion_loss = FusionLoss(label_smoothing=0.1)
        self.congruence_loss = CongruenceLoss(mode=congruence_mode)
    
    def forward(
        self,
        outputs: dict,
        targets: dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: Model outputs dict with keys:
                - face_logits: (B, C)
                - eeg_logits: (B, C)
                - fused_logits: (B, C)
                - valence: (B, 1)
                - arousal: (B, 1)
                - deception_logits: (B, 2) or (B, 1)
                - similarity: (B, 1)
                - z_face: (B, D)
                - z_eeg: (B, D)
            targets: Target dict with keys:
                - face_emotion_idx: (B,)
                - eeg_emotion_idx: (B,)
                - fused_emotion_idx: (B,) - typically same as face_emotion_idx
                - valence: (B, 1)
                - arousal: (B, 1)
                - congruence: (B,) 0=genuine, 1=deceptive
                
        Returns:
            total_loss, loss_components_dict
        """
        # Face emotion loss
        l_face = self.face_loss(outputs["face_logits"], targets["face_emotion_idx"])
        
        # EEG loss (classification + regression)
        l_eeg, eeg_components = self.eeg_loss(
            outputs["eeg_logits"],
            outputs["valence"],
            outputs["arousal"],
            targets["eeg_emotion_idx"],
            targets["valence"],
            targets["arousal"],
        )
        
        # Fusion loss
        l_fusion = self.fusion_loss(outputs["fused_logits"], targets["fused_emotion_idx"])
        
        # Congruence loss
        l_congruence, cong_components = self.congruence_loss(
            outputs["z_face"],
            outputs["z_eeg"],
            targets["congruence"],
            deception_logits=outputs.get("deception_logits"),
            similarity=outputs.get("similarity"),
        )
        
        # Weighted total
        total = (
            self.weights.alpha * l_face +
            self.weights.beta * l_eeg +
            self.weights.gamma * l_fusion +
            self.weights.lambda_ * l_congruence
        )
        
        components = {
            "total": total.item(),
            "face": l_face.item(),
            "eeg": l_eeg.item(),
            "fusion": l_fusion.item(),
            "congruence": l_congruence.item(),
            **{f"eeg_{k}": v for k, v in eeg_components.items()},
            **{f"congruence_{k}": v for k, v in cong_components.items()},
        }
        
        return total, components


def build_loss(config: dict) -> MultiTaskLoss:
    """Factory function to build loss from config."""
    loss_cfg = config["loss"]
    weights = LossWeights(
        alpha=loss_cfg["alpha"],
        beta=loss_cfg["beta"],
        gamma=loss_cfg["gamma"],
        lambda_=loss_cfg["lambda_"],
        eeg_cls_weight=1.0,
        eeg_valence_weight=0.5,
        eeg_arousal_weight=0.5,
    )
    
    # Get congruence mode from model config
    congruence_mode = config["model"]["deception"].get("loss_mode", "bce")
    
    return MultiTaskLoss(weights, congruence_mode=congruence_mode)


if __name__ == "__main__":
    # Quick test
    import yaml
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    loss_fn = build_loss(config)
    print(loss_fn)
    
    # Dummy outputs and targets
    B, C, D = 4, 8, 256
    outputs = {
        "face_logits": torch.randn(B, C),
        "eeg_logits": torch.randn(B, C),
        "fused_logits": torch.randn(B, C),
        "valence": torch.randn(B, 1),
        "arousal": torch.randn(B, 1),
        "deception_logits": torch.randn(B, 2),
        "similarity": torch.rand(B, 1),
        "z_face": torch.randn(B, D),
        "z_eeg": torch.randn(B, D),
    }
    targets = {
        "face_emotion_idx": torch.randint(0, C, (B,)),
        "eeg_emotion_idx": torch.randint(0, C, (B,)),
        "fused_emotion_idx": torch.randint(0, C, (B,)),
        "valence": torch.rand(B, 1) * 2 - 1,
        "arousal": torch.rand(B, 1),
        "congruence": torch.randint(0, 2, (B,)),
    }
    
    total, components = loss_fn(outputs, targets)
    print(f"\nTotal loss: {total.item():.4f}")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")