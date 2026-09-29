"""
Model architectures for Multimodal FER + EEG Deception Detection.

Components:
- Visual Stream: ResNet/CNN backbone with attention for facial emotion recognition
- EEG Stream: 1D-CNN + BiLSTM for temporal/spectral EEG processing
- Fusion Module: Cross-attention or concatenation-based late fusion
- Deception/Incongruence Head: Binary classifier for genuineness detection
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


# ──────────────────────────────────────────────────────────────────────
# Attention Modules
# ──────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """Channel Attention Module (from CBAM)."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (from CBAM)."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


# ──────────────────────────────────────────────────────────────────────
# Visual Stream (Facial Emotion Recognition)
# ──────────────────────────────────────────────────────────────────────
class VisualStream(nn.Module):
    """
    Visual stream for Facial Emotion Recognition.
    
    Uses a ResNet backbone (default: ResNet18) with optional CBAM attention.
    Outputs latent features and emotion class logits.
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        latent_dim: int = 256,
        num_classes: int = 8,
        freeze_backbone: bool = False,
        attention: str = "cbam",  # none, cbam, spatial
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Load backbone
        if backbone_name == "resnet18":
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif backbone_name == "resnet34":
            backbone = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif backbone_name == "resnet50":
            backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feat_dim = 2048
        elif backbone_name == "mobilenet_v3_small":
            backbone = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Remove final classification layer
        if "resnet" in backbone_name:
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Keep avgpool separate
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif "mobilenet" in backbone_name:
            self.backbone = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Attention module
        self.attention_type = attention
        if attention == "cbam":
            self.attention = CBAM(feat_dim)
        elif attention == "spatial":
            self.attention = SpatialAttention()
        else:
            self.attention = nn.Identity()
        
        # Feature projection to latent space
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # Emotion classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) face images
            
        Returns:
            z_face: (B, latent_dim) latent features
            logits: (B, num_classes) emotion logits
        """
        # Backbone features
        feat = self.backbone(x)  # (B, C, H', W')
        
        # Apply attention
        feat = self.attention(feat)
        
        # Global pooling
        feat = self.pool(feat)  # (B, C, 1, 1)
        feat = feat.flatten(1)  # (B, C)
        
        # Project to latent space
        z_face = self.projector(feat)  # (B, latent_dim)
        
        # Classification
        logits = self.classifier(z_face)
        
        return z_face, logits


# ──────────────────────────────────────────────────────────────────────
# EEG Stream (Brainwave Analysis)
# ──────────────────────────────────────────────────────────────────────
class EEGStream(nn.Module):
    """
    EEG stream for emotional valence/arousal classification.
    
    Architecture: 1D-CNN feature extractor + BiLSTM temporal modeling.
    Handles both temporal (C, T) and spectral (C, n_bands) inputs.
    """
    
    def __init__(
        self,
        n_channels: int = 32,
        n_bands: int = 4,
        cnn_channels: list[int] = None,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        latent_dim: int = 256,
        num_classes: int = 8,  # valence/arousal states or discrete emotions
        dropout: float = 0.3,
        use_spectral: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.lstm_hidden = lstm_hidden
        self.latent_dim = latent_dim
        self.use_spectral = use_spectral
        
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]
        
        # 1D-CNN for temporal processing
        # Input: (B, C, T) -> treat channels as input channels
        temporal_cnn_layers = []
        in_ch = n_channels
        for out_ch in cnn_channels:
            temporal_cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.temporal_cnn = nn.Sequential(*temporal_cnn_layers)
        
        # Calculate CNN output length
        # With 4 maxpool2 layers: T -> T/16
        # For 4s @ 256Hz = 1024 samples -> 1024/16 = 64
        self.temporal_feat_dim = cnn_channels[-1]
        
        # BiLSTM for temporal modeling
        self.bilstm = nn.LSTM(
            input_size=self.temporal_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Spectral branch (if using PSD features)
        if use_spectral:
            self.spectral_net = nn.Sequential(
                nn.Linear(n_channels * n_bands, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            spectral_feat_dim = 128
        else:
            spectral_feat_dim = 0
        
        # Combined feature dimension
        combined_dim = lstm_hidden * 2 + spectral_feat_dim  # BiLSTM is bidirectional
        
        # Projector to latent space
        self.projector = nn.Sequential(
            nn.Linear(combined_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Emotion/valence-arousal classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_classes),
        )
        
        # Valence/Arousal regression heads (continuous)
        self.valence_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),  # [-1, 1]
        )
        self.arousal_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # [0, 1]
        )
    
    def forward(
        self,
        temporal: torch.Tensor,
        spectral: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            temporal: (B, C, T) EEG temporal signal
            spectral: (B, C, n_bands) EEG spectral features (optional)
            
        Returns:
            z_eeg: (B, latent_dim) latent features
            logits: (B, num_classes) discrete emotion logits
            valence: (B, 1) continuous valence [-1, 1]
            arousal: (B, 1) continuous arousal [0, 1]
        """
        B = temporal.size(0)
        
        # Temporal CNN
        temp_feat = self.temporal_cnn(temporal)  # (B, C_out, T')
        temp_feat = temp_feat.transpose(1, 2)    # (B, T', C_out)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(temp_feat)     # (B, T', 2*hidden)
        # Global average pooling over time
        temp_pooled = lstm_out.mean(dim=1)       # (B, 2*hidden)
        
        # Spectral branch
        if self.use_spectral and spectral is not None:
            spec_flat = spectral.flatten(1)      # (B, C * n_bands)
            spec_feat = self.spectral_net(spec_flat)  # (B, 128)
            combined = torch.cat([temp_pooled, spec_feat], dim=1)
        else:
            combined = temp_pooled
        
        # Project to latent space
        z_eeg = self.projector(combined)
        
        # Classification
        logits = self.classifier(z_eeg)
        
        # Continuous valence/arousal
        valence = self.valence_head(z_eeg)
        arousal = self.arousal_head(z_eeg)
        
        return z_eeg, logits, valence, arousal


# ──────────────────────────────────────────────────────────────────────
# Fusion Module
# ──────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between visual and EEG features."""
    
    def __init__(
        self,
        latent_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        # Cross-attention: face attends to EEG, EEG attends to face
        self.face_to_eeg = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.eeg_to_face = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)
        self.norm4 = nn.LayerNorm(latent_dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim * 2, latent_dim)
    
    def forward(
        self,
        z_face: torch.Tensor,
        z_eeg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_face: (B, D) visual latent
            z_eeg: (B, D) EEG latent
            
        Returns:
            z_fused: (B, D) fused latent
        """
        # Add sequence dimension for attention
        z_face_seq = z_face.unsqueeze(1)  # (B, 1, D)
        z_eeg_seq = z_eeg.unsqueeze(1)    # (B, 1, D)
        
        # Face attends to EEG
        face_attn, _ = self.face_to_eeg(z_face_seq, z_eeg_seq, z_eeg_seq)
        face_attn = self.norm1(z_face_seq + face_attn)
        
        # EEG attends to Face
        eeg_attn, _ = self.eeg_to_face(z_eeg_seq, z_face_seq, z_face_seq)
        eeg_attn = self.norm2(z_eeg_seq + eeg_attn)
        
        # Feed-forward on each
        face_ffn = self.norm3(face_attn + self.ffn(face_attn))
        eeg_ffn = self.norm4(eeg_attn + self.ffn(eeg_attn))
        
        # Concatenate and project
        fused = torch.cat([face_ffn, eeg_ffn], dim=1).flatten(1)  # (B, 2D)
        z_fused = self.output_proj(fused)  # (B, D)
        
        return z_fused


class ConcatFusion(nn.Module):
    """Simple concatenation-based fusion."""
    
    def __init__(self, latent_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, z_face: torch.Tensor, z_eeg: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z_face, z_eeg], dim=1)
        return self.fusion(combined)


class BilinearFusion(nn.Module):
    """Bilinear pooling fusion."""
    
    def __init__(self, latent_dim: int = 256, output_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Low-rank bilinear projection
        self.face_proj = nn.Linear(latent_dim, output_dim)
        self.eeg_proj = nn.Linear(latent_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z_face: torch.Tensor, z_eeg: torch.Tensor) -> torch.Tensor:
        face_p = self.face_proj(z_face)
        eeg_p = self.eeg_proj(z_eeg)
        # Hadamard product (element-wise multiplication)
        fused = face_p * eeg_p
        return self.dropout(fused)


def get_fusion_module(fusion_type: str, latent_dim: int, **kwargs) -> nn.Module:
    """Factory for fusion modules."""
    if fusion_type == "cross_attention":
        return CrossAttentionFusion(latent_dim, **kwargs)
    elif fusion_type == "concat":
        return ConcatFusion(latent_dim, **kwargs)
    elif fusion_type == "bilinear":
        return BilinearFusion(latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


# ──────────────────────────────────────────────────────────────────────
# Deception / Incongruence Detection Head
# ──────────────────────────────────────────────────────────────────────
class DeceptionHead(nn.Module):
    """
    Deception/Incongruence detection head.
    
    Computes divergence between visual and EEG latent representations
    to detect when facial expression doesn't match brain activity.
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        head_type: str = "similarity_binary",  # similarity_binary, distance_threshold, mlp_classifier
        similarity_metric: str = "cosine",     # cosine, euclidean
        threshold: float = 0.7,
    ):
        super().__init__()
        self.head_type = head_type
        self.similarity_metric = similarity_metric
        self.threshold = threshold
        
        if head_type == "similarity_binary":
            # Simple similarity + learnable threshold
            self.similarity_scorer = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        elif head_type == "mlp_classifier":
            # MLP on concatenated features
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(latent_dim // 2, 2),  # binary: genuine vs deceptive
            )
        elif head_type == "distance_threshold":
            # Fixed threshold on distance (no learnable params)
            pass
        else:
            raise ValueError(f"Unknown deception head type: {head_type}")
    
    def compute_similarity(self, z_face: torch.Tensor, z_eeg: torch.Tensor) -> torch.Tensor:
        """Compute similarity between face and EEG latents."""
        if self.similarity_metric == "cosine":
            sim = F.cosine_similarity(z_face, z_eeg, dim=1)  # (B,)
        elif self.similarity_metric == "euclidean":
            dist = torch.norm(z_face - z_eeg, p=2, dim=1)  # (B,)
            sim = 1.0 / (1.0 + dist)  # Convert to similarity [0, 1]
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        return sim.unsqueeze(1)  # (B, 1)
    
    def forward(
        self,
        z_face: torch.Tensor,
        z_eeg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_face: (B, D) visual latent
            z_eeg: (B, D) EEG latent
            
        Returns:
            deception_logits: (B, 2) or (B, 1) depending on head type
            similarity: (B, 1) similarity score
            is_deceptive: (B,) boolean predictions
        """
        similarity = self.compute_similarity(z_face, z_eeg)
        
        if self.head_type == "similarity_binary":
            # Learnable mapping from similarity to deception probability
            deception_prob = self.similarity_scorer(similarity)
            deception_logits = torch.cat([1 - deception_prob, deception_prob], dim=1)
            is_deceptive = (deception_prob.squeeze(1) > self.threshold).long()
            
        elif head_type == "mlp_classifier":
            combined = torch.cat([z_face, z_eeg], dim=1)
            deception_logits = self.classifier(combined)
            deception_prob = F.softmax(deception_logits, dim=1)[:, 1]
            is_deceptive = (deception_prob > self.threshold).long()
            
        elif head_type == "distance_threshold":
            # Fixed threshold
            is_deceptive = (similarity.squeeze(1) < self.threshold).long()
            deception_logits = torch.cat([similarity, 1 - similarity], dim=1)
            deception_prob = 1 - similarity.squeeze(1)
        
        return deception_logits, similarity, is_deceptive


# ──────────────────────────────────────────────────────────────────────
# Complete Multimodal Model
# ──────────────────────────────────────────────────────────────────────
class MultimodalDeceptionModel(nn.Module):
    """
    Complete multimodal model for FER + EEG deception detection.
    
    Combines:
    - Visual stream (FER)
    - EEG stream (valence/arousal)
    - Fusion module (joint emotion classification)
    - Deception head (incongruence detection)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        latent_dim = model_cfg["latent_dim"]
        num_emotion_classes = model_cfg["num_emotion_classes"]
        
        # Visual stream
        self.visual_stream = VisualStream(
            backbone_name=model_cfg["visual"]["backbone"],
            pretrained=model_cfg["visual"]["pretrained"],
            latent_dim=latent_dim,
            num_classes=num_emotion_classes,
            freeze_backbone=model_cfg["visual"]["freeze_backbone"],
            attention=model_cfg["visual"]["attention"],
        )
        
        # EEG stream
        self.eeg_stream = EEGStream(
            n_channels=config["eeg"].get("num_channels", 32),
            n_bands=len(config["eeg"]["bands"]),
            cnn_channels=model_cfg["eeg"]["cnn_channels"],
            lstm_hidden=model_cfg["eeg"]["lstm_hidden"],
            lstm_layers=model_cfg["eeg"]["lstm_layers"],
            latent_dim=latent_dim,
            num_classes=num_emotion_classes,
            dropout=model_cfg["eeg"]["dropout"],
            use_spectral=True,
        )
        
        # Fusion module
        fusion_cfg = model_cfg["fusion"]
        self.fusion = get_fusion_module(
            fusion_type=fusion_cfg["type"],
            latent_dim=latent_dim,
            num_heads=fusion_cfg.get("num_heads", 4),
            dropout=fusion_cfg.get("dropout", 0.2),
        )
        
        # Fused emotion classifier
        self.fused_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_emotion_classes),
        )
        
        # Deception head
        deception_cfg = model_cfg["deception"]
        self.deception_head = DeceptionHead(
            latent_dim=latent_dim,
            head_type=deception_cfg["type"],
            similarity_metric=deception_cfg["similarity_metric"],
            threshold=deception_cfg["threshold"],
        )
    
    def forward(
        self,
        face: torch.Tensor,
        eeg_temporal: torch.Tensor,
        eeg_spectral: torch.Tensor,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            face: (B, 3, H, W) face images
            eeg_temporal: (B, C, T) EEG temporal
            eeg_spectral: (B, C, n_bands) EEG spectral
            
        Returns:
            Dictionary with all outputs
        """
        # Visual stream
        z_face, face_logits = self.visual_stream(face)
        
        # EEG stream
        z_eeg, eeg_logits, valence, arousal = self.eeg_stream(eeg_temporal, eeg_spectral)
        
        # Fusion
        z_fused = self.fusion(z_face, z_eeg)
        
        # Fused emotion classification
        fused_logits = self.fused_classifier(z_fused)
        
        # Deception detection
        deception_logits, similarity, is_deceptive = self.deception_head(z_face, z_eeg)
        
        return {
            "z_face": z_face,
            "z_eeg": z_eeg,
            "z_fused": z_fused,
            "face_logits": face_logits,
            "eeg_logits": eeg_logits,
            "fused_logits": fused_logits,
            "valence": valence,
            "arousal": arousal,
            "deception_logits": deception_logits,
            "similarity": similarity,
            "is_deceptive": is_deceptive,
        }
    
    @torch.no_grad()
    def verify_authenticity(
        self,
        face_sample: torch.Tensor,
        eeg_temporal: torch.Tensor,
        eeg_spectral: torch.Tensor,
        threshold: float = 0.7,
        emotion_labels: list[str] = None,
    ) -> dict:
        """
        Inference utility for authenticity verification.
        
        Args:
            face_sample: (1, 3, H, W) or (3, H, W)
            eeg_temporal: (1, C, T) or (C, T)
            eeg_spectral: (1, C, n_bands) or (C, n_bands)
            threshold: Deception threshold
            emotion_labels: List of emotion label strings
            
        Returns:
            Dictionary with verification results
        """
        self.eval()
        
        # Add batch dim if needed
        if face_sample.dim() == 3:
            face_sample = face_sample.unsqueeze(0)
        if eeg_temporal.dim() == 2:
            eeg_temporal = eeg_temporal.unsqueeze(0)
        if eeg_spectral.dim() == 2:
            eeg_spectral = eeg_spectral.unsqueeze(0)
        
        outputs = self.forward(face_sample, eeg_temporal, eeg_spectral)
        
        # Get predictions
        face_pred = outputs["face_logits"].argmax(dim=1).item()
        eeg_pred = outputs["eeg_logits"].argmax(dim=1).item()
        fused_pred = outputs["fused_logits"].argmax(dim=1).item()
        
        similarity = outputs["similarity"].item()
        deception_prob = F.softmax(outputs["deception_logits"], dim=1)[0, 1].item()
        is_deceptive = deception_prob > threshold
        
        # Valence/Arousal
        valence = outputs["valence"].item()
        arousal = outputs["arousal"].item()
        
        # Labels
        if emotion_labels is None:
            emotion_labels = self.config["model"]["emotion_labels"]
        
        # Diagnostic breakdown
        diagnostics = {
            "similarity_score": similarity,
            "deception_probability": deception_prob,
            "valence": valence,
            "arousal": arousal,
            "face_emotion": emotion_labels[face_pred],
            "eeg_emotion": emotion_labels[eeg_pred],
            "fused_emotion": emotion_labels[fused_pred],
            "valence_interpretation": "positive" if valence > 0 else "negative",
            "arousal_interpretation": "high" if arousal > 0.5 else "low",
        }
        
        # Add incongruence explanation
        if is_deceptive:
            diagnostics["incongruence_explanation"] = (
                f"Face shows '{emotion_labels[face_pred]}' but EEG indicates "
                f"'{emotion_labels[eeg_pred]}' (valence={valence:.2f}, arousal={arousal:.2f}). "
                f"Similarity={similarity:.3f} < threshold={threshold}"
            )
        else:
            diagnostics["incongruence_explanation"] = (
                f"Face and EEG are congruent. "
                f"Face: '{emotion_labels[face_pred]}', EEG: '{emotion_labels[eeg_pred]}'. "
                f"Similarity={similarity:.3f} >= threshold={threshold}"
            )
        
        return {
            "predicted_face_emotion": emotion_labels[face_pred],
            "predicted_eeg_emotion": emotion_labels[eeg_pred],
            "fused_emotion": emotion_labels[fused_pred],
            "authenticity_score": 1.0 - deception_prob,
            "is_deceptive": bool(is_deceptive),
            "deception_probability": deception_prob,
            "similarity_score": similarity,
            "valence": valence,
            "arousal": arousal,
            "diagnostics": diagnostics,
        }


def build_model(config: dict) -> MultimodalDeceptionModel:
    """Factory function to build model from config."""
    return MultimodalDeceptionModel(config)


if __name__ == "__main__":
    # Quick test
    import yaml
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model = build_model(config)
    print(model)
    
    # Test forward
    B = 4
    face = torch.randn(B, 3, 224, 224)
    eeg_temp = torch.randn(B, 32, 1024)
    eeg_spec = torch.randn(B, 32, 4)
    
    out = model(face, eeg_temp, eeg_spec)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {v}")
    
    # Test verify_authenticity
    result = model.verify_authenticity(face[0], eeg_temp[0], eeg_spec[0])
    print("\nVerification result:")
    for k, v in result.items():
        print(f"  {k}: {v}")