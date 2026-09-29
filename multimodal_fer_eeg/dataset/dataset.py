"""
Dataset module for Multimodal FER + EEG Deception Detection.

Supports:
- Facial Emotion Recognition (FER) from image frames
- EEG signal processing (synthetic generator for demo, real data loader when available)
- Cross-modal pairing for incongruence/deception detection
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────
# Emotion label mapping
# ──────────────────────────────────────────────────────────────────────
EMOTION_LABELS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprised"
]
EMOTION_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
IDX_TO_EMOTION = {idx: label for idx, label in enumerate(EMOTION_LABELS)}


# ──────────────────────────────────────────────────────────────────────
# Vision Transforms
# ──────────────────────────────────────────────────────────────────────
def get_vision_transforms(config: dict, is_train: bool = True) -> transforms.Compose:
    """Build torchvision transform pipeline for facial images."""
    vision_cfg = config["vision"]
    img_size = tuple(vision_cfg["image_size"])
    mean = vision_cfg["mean"]
    std = vision_cfg["std"]

    if is_train:
        aug = vision_cfg["augment"]
        transform_list = [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=aug.get("horizontal_flip", 0.5)),
            transforms.RandomAffine(
                degrees=aug.get("random_affine_degrees", 15),
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(
                brightness=aug.get("color_jitter", 0.2),
                contrast=aug.get("color_jitter", 0.2),
                saturation=aug.get("color_jitter", 0.2),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        transform_list = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)


# ──────────────────────────────────────────────────────────────────────
# EEG Signal Processing
# ──────────────────────────────────────────────────────────────────────
def compute_psd_bands(
    eeg_signal: np.ndarray,
    fs: int,
    bands: list[dict],
) -> np.ndarray:
    """
    Compute Power Spectral Density for specified frequency bands.
    
    Args:
        eeg_signal: (C, T) array - C channels, T time samples
        fs: Sampling frequency in Hz
        bands: List of dicts with 'low', 'high' keys
        
    Returns:
        (C, len(bands)) array of band power per channel
    """
    from scipy.integrate import trapezoid
    n_channels, n_samples = eeg_signal.shape
    band_powers = np.zeros((n_channels, len(bands)), dtype=np.float32)
    
    for ch in range(n_channels):
        freqs, psd = signal.welch(eeg_signal[ch], fs=fs, nperseg=min(256, n_samples // 4))
        for b_idx, band in enumerate(bands):
            mask = (freqs >= band["low"]) & (freqs <= band["high"])
            if mask.any():
                band_powers[ch, b_idx] = trapezoid(psd[mask], freqs[mask])
            else:
                band_powers[ch, b_idx] = 0.0
    
    return band_powers  # (C, n_bands)


def baseline_correction(eeg_signal: np.ndarray, baseline_duration: float = 0.5, fs: int = 256) -> np.ndarray:
    """Remove baseline drift using initial segment mean."""
    n_baseline = int(baseline_duration * fs)
    baseline_mean = eeg_signal[:, :n_baseline].mean(axis=1, keepdims=True)
    return eeg_signal - baseline_mean


def zscore_per_channel(eeg_signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize each channel independently."""
    mean = eeg_signal.mean(axis=1, keepdims=True)
    std = eeg_signal.std(axis=1, keepdims=True) + eps
    return (eeg_signal - mean) / std


def preprocess_eeg(
    raw_eeg: np.ndarray,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full EEG preprocessing pipeline.
    
    Returns:
        temporal: (C, T) preprocessed temporal signal
        spectral: (C, n_bands) PSD band features
    """
    eeg_cfg = config["eeg"]
    fs = eeg_cfg["sampling_rate"]
    bands = eeg_cfg["bands"]
    
    # Baseline correction
    temporal = baseline_correction(raw_eeg, fs=fs)
    # Z-score per channel
    temporal = zscore_per_channel(temporal)
    # Compute spectral features
    spectral = compute_psd_bands(temporal, fs, bands)
    # Normalize spectral features (min-max per band across channels)
    spec_min = spectral.min(axis=0, keepdims=True)
    spec_max = spectral.max(axis=0, keepdims=True)
    spectral = (spectral - spec_min) / (spec_max - spec_min + 1e-8)
    
    return temporal.astype(np.float32), spectral.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Synthetic EEG Generator (for demo when real EEG unavailable)
# ──────────────────────────────────────────────────────────────────────
class SyntheticEEGGenerator:
    """
    Generates realistic synthetic EEG data conditioned on emotion labels.
    Used when real EEG data is not available.
    
    Based on known EEG-emotion correlates:
    - Positive valence (Happy): ↑ Alpha asymmetry (left frontal), ↓ Beta
    - Negative valence (Fear, Anger, Sad): ↑ Beta/Gamma (right frontal), ↑ Theta
    - High arousal (Fear, Anger, Surprised): ↑ Beta, ↑ Gamma
    - Low arousal (Sad, Neutral): ↑ Alpha, ↑ Theta
    """
    
    EMOTION_EEG_PROFILES = {
        "Anger":       {"valence": -0.8, "arousal": 0.9, "theta": 1.2, "alpha": 0.7, "beta": 1.3, "gamma": 1.2},
        "Contempt":    {"valence": -0.4, "arousal": 0.5, "theta": 1.0, "alpha": 0.9, "beta": 1.1, "gamma": 1.0},
        "Disgust":     {"valence": -0.9, "arousal": 0.6, "theta": 1.1, "alpha": 0.8, "beta": 1.2, "gamma": 1.1},
        "Fear":        {"valence": -0.9, "arousal": 0.95, "theta": 1.3, "alpha": 0.6, "beta": 1.4, "gamma": 1.3},
        "Happy":       {"valence": 0.9,  "arousal": 0.7, "theta": 0.8, "alpha": 1.3, "beta": 0.9, "gamma": 0.9},
        "Neutral":     {"valence": 0.0,  "arousal": 0.2, "theta": 1.0, "alpha": 1.0, "beta": 1.0, "gamma": 1.0},
        "Sad":         {"valence": -0.7, "arousal": 0.3, "theta": 1.2, "alpha": 1.2, "beta": 0.8, "gamma": 0.8},
        "Surprised":   {"valence": 0.1,  "arousal": 0.85, "theta": 1.0, "alpha": 0.7, "beta": 1.3, "gamma": 1.2},
    }
    
    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.eeg_cfg = config["eeg"]
        self.n_channels = self.eeg_cfg["num_channels"]
        self.fs = self.eeg_cfg["sampling_rate"]
        self.window_sec = self.eeg_cfg["window_seconds"]
        self.n_samples = self.fs * self.window_sec
        self.bands = self.eeg_cfg["bands"]
        self.rng = np.random.RandomState(seed)
        
        # 10-20 channel names (standard 32-channel subset)
        self.channel_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
            "O1", "Oz", "O2", "Fz", "Cz", "Pz", "AFz", "POz"
        ][:self.n_channels]
        
        # Frontal asymmetry channels (for valence)
        self.left_frontal = [i for i, ch in enumerate(self.channel_names) if ch in ["Fp1", "F3", "F7", "FC5", "FC1"]]
        self.right_frontal = [i for i, ch in enumerate(self.channel_names) if ch in ["Fp2", "F4", "F8", "FC6", "FC2"]]
    
    def generate(self, emotion_label: str, noise_level: float = 0.3) -> np.ndarray:
        """
        Generate synthetic EEG for a given emotion.
        
        Returns:
            (C, T) temporal signal
        """
        profile = self.EMOTION_EEG_PROFILES.get(emotion_label, self.EMOTION_EEG_PROFILES["Neutral"])
        
        # Generate base oscillatory activity per band
        t = np.arange(self.n_samples) / self.fs
        signal = np.zeros((self.n_channels, self.n_samples), dtype=np.float32)
        
        for ch in range(self.n_channels):
            ch_signal = np.zeros(self.n_samples)
            
            # Add band-specific oscillations with emotion-modulated amplitude
            for band in self.bands:
                band_name = band["name"]
                center_freq = (band["low"] + band["high"]) / 2
                amplitude = profile.get(band_name, 1.0)
                
                # Add some channel-specific variation
                ch_amplitude = amplitude * (1 + 0.2 * self.rng.randn())
                
                # Generate oscillation with phase noise
                phase_noise = 0.1 * self.rng.randn(self.n_samples).cumsum()
                ch_signal += ch_amplitude * np.sin(2 * np.pi * center_freq * t + phase_noise)
            
            # Add 1/f background noise
            ch_signal += noise_level * self.rng.randn(self.n_samples) / np.sqrt(np.arange(1, self.n_samples + 1))
            
            # Frontal asymmetry for valence
            if ch in self.left_frontal:
                ch_signal += profile["valence"] * 0.5 * np.sin(2 * np.pi * 10 * t)  # Alpha asymmetry
            elif ch in self.right_frontal:
                ch_signal -= profile["valence"] * 0.5 * np.sin(2 * np.pi * 10 * t)
            
            signal[ch] = ch_signal
        
        return signal
    
    def generate_congruent_pair(self, emotion_label: str) -> Tuple[np.ndarray, int]:
        """Generate EEG that matches the facial emotion (congruent/genuine)."""
        eeg = self.generate(emotion_label)
        temporal, spectral = preprocess_eeg(eeg, self.config)
        return temporal, spectral, 0  # 0 = genuine/congruent
    
    def generate_incongruent_pair(self, face_emotion: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate EEG that mismatches the facial emotion (incongruent/deceptive).
        Picks an emotion with opposite valence/arousal.
        """
        face_profile = self.EMOTION_EEG_PROFILES[face_emotion]
        face_valence = face_profile["valence"]
        face_arousal = face_profile["arousal"]
        
        # Find emotion with most different valence/arousal
        best_diff = -1
        best_emotion = "Neutral"
        for emo, profile in self.EMOTION_EEG_PROFILES.items():
            if emo == face_emotion:
                continue
            diff = abs(profile["valence"] - face_valence) + abs(profile["arousal"] - face_arousal)
            if diff > best_diff:
                best_diff = diff
                best_emotion = emo
        
        eeg = self.generate(best_emotion)
        temporal, spectral = preprocess_eeg(eeg, self.config)
        return temporal, spectral, 1  # 1 = deceptive/incongruent


# ──────────────────────────────────────────────────────────────────────
# PyTorch Dataset Classes
# ──────────────────────────────────────────────────────────────────────
class FERDataset(Dataset):
    """
    Facial Emotion Recognition Dataset.
    
    Expected directory structure:
    root/
        images/
            {subject_id}/
                {Emotion}.jpg
        emotions.csv  (columns: set_id, gender, age, country)
    
    Each subject has 8 emotion images. We treat each (subject, emotion) as a sample.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        config: dict,
        transform: Optional[Callable] = None,
        is_train: bool = True,
        subjects: Optional[list[int]] = None,
    ):
        self.root = Path(root)
        self.config = config
        self.transform = transform or get_vision_transforms(config, is_train)
        self.is_train = is_train
        
        # Load metadata
        csv_path = self.root.parent / "emotions.csv"
        if not csv_path.exists():
            csv_path = self.root / ".." / "emotions.csv"
        self.metadata = pd.read_csv(csv_path)
        
        # Build sample index: (subject_id, emotion_label)
        self.samples = []
        self.subjects = subjects if subjects is not None else sorted(self.metadata["set_id"].unique())
        
        for subject_id in self.subjects:
            subject_dir = self.root / str(subject_id)
            if not subject_dir.exists():
                continue
            for emotion in EMOTION_LABELS:
                img_path = subject_dir / f"{emotion}.jpg"
                if img_path.exists():
                    self.samples.append({
                        "subject_id": subject_id,
                        "emotion": emotion,
                        "emotion_idx": EMOTION_TO_IDX[emotion],
                        "image_path": img_path,
                    })
        
        print(f"[FERDataset] Loaded {len(self.samples)} samples from {len(self.subjects)} subjects")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "face": image,
            "face_emotion_idx": sample["emotion_idx"],
            "face_emotion_label": sample["emotion"],
            "subject_id": sample["subject_id"],
        }


class MultimodalDeceptionDataset(Dataset):
    """
    Multimodal dataset pairing FER images with EEG signals for deception detection.
    
    Each sample contains:
    - Face image
    - EEG temporal signal (C, T)
    - EEG spectral features (C, n_bands)
    - Face emotion label (0-7)
    - EEG emotion label (0-7) - same for congruent, different for incongruent
    - Congruence label (0 = genuine, 1 = deceptive)
    
    If real EEG data is provided via eeg_root, it loads from .npy/.mat/.pkl files.
    Otherwise, uses SyntheticEEGGenerator.
    """
    
    def __init__(
        self,
        fer_root: Union[str, Path],
        config: dict,
        eeg_root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        is_train: bool = True,
        subjects: Optional[list[int]] = None,
        deception_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.fer_root = Path(fer_root)
        self.config = config
        self.eeg_root = Path(eeg_root) if eeg_root else None
        self.transform = transform or get_vision_transforms(config, is_train)
        self.is_train = is_train
        self.deception_ratio = deception_ratio
        self.seed = seed
        
        # Load FER metadata
        csv_path = self.fer_root.parent / "emotions.csv"
        if not csv_path.exists():
            csv_path = self.fer_root / ".." / "emotions.csv"
        self.metadata = pd.read_csv(csv_path)
        
        self.subjects = subjects if subjects is not None else sorted(self.metadata["set_id"].unique())
        
        # Initialize EEG generator or loader
        self.eeg_generator = SyntheticEEGGenerator(config, seed=seed)
        self.use_synthetic = self.eeg_root is None or not self.eeg_root.exists()
        
        if self.use_synthetic:
            print("[MultimodalDeceptionDataset] Using SYNTHETIC EEG generator")
        else:
            print(f"[MultimodalDeceptionDataset] Loading REAL EEG from {self.eeg_root}")
            self._index_real_eeg()
        
        # Build paired samples
        self.samples = self._build_paired_samples()
        print(f"[MultimodalDeceptionDataset] Built {len(self.samples)} paired samples "
              f"({sum(1 for s in self.samples if s['congruence']==0)} congruent, "
              f"{sum(1 for s in self.samples if s['congruence']==1)} incongruent)")
    
    def _index_real_eeg(self):
        """Index real EEG files. Expected naming: {subject_id}_{emotion}.npy"""
        self.eeg_index = {}
        for ext in [".npy", ".mat", ".pkl"]:
            for f in self.eeg_root.rglob(f"*{ext}"):
                # Parse subject and emotion from filename
                # Expected: subject_0_emotion_Happy.npy or similar
                stem = f.stem
                # Try to extract subject_id and emotion
                # This is dataset-specific; adjust as needed
                pass  # Implement based on actual naming convention
    
    def _load_real_eeg(self, subject_id: int, emotion: str) -> Optional[np.ndarray]:
        """Load real EEG file. Returns (C, T) or None if not found."""
        # Implement based on actual file structure
        return None
    
    def _build_paired_samples(self) -> list[dict]:
        """Build list of (face, eeg, congruence) samples."""
        samples = []
        rng = random.Random(self.seed)
        
        for subject_id in self.subjects:
            subject_dir = self.fer_root / str(subject_id)
            if not subject_dir.exists():
                continue
            
            for face_emotion in EMOTION_LABELS:
                face_path = subject_dir / f"{face_emotion}.jpg"
                if not face_path.exists():
                    continue
                
                # Congruent sample (genuine)
                samples.append({
                    "subject_id": subject_id,
                    "face_path": face_path,
                    "face_emotion": face_emotion,
                    "face_emotion_idx": EMOTION_TO_IDX[face_emotion],
                    "eeg_emotion": face_emotion,
                    "eeg_emotion_idx": EMOTION_TO_IDX[face_emotion],
                    "congruence": 0,  # genuine
                })
                
                # Incongruent sample (deceptive) - with probability deception_ratio
                if rng.random() < self.deception_ratio:
                    # Pick a different emotion for EEG
                    other_emotions = [e for e in EMOTION_LABELS if e != face_emotion]
                    eeg_emotion = rng.choice(other_emotions)
                    samples.append({
                        "subject_id": subject_id,
                        "face_path": face_path,
                        "face_emotion": face_emotion,
                        "face_emotion_idx": EMOTION_TO_IDX[face_emotion],
                        "eeg_emotion": eeg_emotion,
                        "eeg_emotion_idx": EMOTION_TO_IDX[eeg_emotion],
                        "congruence": 1,  # deceptive
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Load face image
        face_img = Image.open(sample["face_path"]).convert("RGB")
        if self.transform:
            face_img = self.transform(face_img)
        
        # Load/generate EEG
        if self.use_synthetic:
            if sample["congruence"] == 0:
                temporal, spectral, _ = self.eeg_generator.generate_congruent_pair(sample["eeg_emotion"])
            else:
                temporal, spectral, _ = self.eeg_generator.generate_incongruent_pair(sample["face_emotion"])
        else:
            # Load real EEG
            eeg_raw = self._load_real_eeg(sample["subject_id"], sample["eeg_emotion"])
            if eeg_raw is not None:
                temporal, spectral = preprocess_eeg(eeg_raw, self.config)
            else:
                # Fallback to synthetic
                temporal, spectral, _ = self.eeg_generator.generate_congruent_pair(sample["eeg_emotion"])
        
        return {
            "face": face_img,
            "eeg_temporal": torch.from_numpy(temporal),      # (C, T)
            "eeg_spectral": torch.from_numpy(spectral),      # (C, n_bands)
            "face_emotion_idx": torch.tensor(sample["face_emotion_idx"], dtype=torch.long),
            "eeg_emotion_idx": torch.tensor(sample["eeg_emotion_idx"], dtype=torch.long),
            "congruence": torch.tensor(sample["congruence"], dtype=torch.long),
            "subject_id": sample["subject_id"],
            "face_emotion_label": sample["face_emotion"],
            "eeg_emotion_label": sample["eeg_emotion"],
        }


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with subject-wise splits."""
    fer_root = config["data"]["image_root"]
    
    # Full dataset
    full_dataset = MultimodalDeceptionDataset(
        fer_root=fer_root,
        config=config,
        eeg_root=config["data"].get("eeg_root"),
        is_train=True,
        deception_ratio=0.5,
        seed=config["training"]["seed"],
    )
    
    # Subject-wise split to prevent data leakage
    subjects = full_dataset.subjects
    n_subjects = len(subjects)
    n_test = max(1, int(n_subjects * 0.15))
    n_val = max(1, int(n_subjects * 0.15))
    n_train = n_subjects - n_test - n_val
    
    rng = np.random.RandomState(config["training"]["seed"])
    shuffled = rng.permutation(subjects)
    train_subjects = shuffled[:n_train].tolist()
    val_subjects = shuffled[n_train:n_train + n_val].tolist()
    test_subjects = shuffled[n_train + n_val:].tolist()
    
    print(f"[DataLoader] Subject split: Train={len(train_subjects)}, Val={len(val_subjects)}, Test={len(test_subjects)}")
    
    train_dataset = MultimodalDeceptionDataset(
        fer_root=fer_root,
        config=config,
        eeg_root=config["data"].get("eeg_root"),
        transform=get_vision_transforms(config, is_train=True),
        is_train=True,
        subjects=train_subjects,
        deception_ratio=0.5,
        seed=config["training"]["seed"],
    )
    
    val_dataset = MultimodalDeceptionDataset(
        fer_root=fer_root,
        config=config,
        eeg_root=config["data"].get("eeg_root"),
        transform=get_vision_transforms(config, is_train=False),
        is_train=False,
        subjects=val_subjects,
        deception_ratio=0.5,
        seed=config["training"]["seed"],
    )
    
    test_dataset = MultimodalDeceptionDataset(
        fer_root=fer_root,
        config=config,
        eeg_root=config["data"].get("eeg_root"),
        transform=get_vision_transforms(config, is_train=False),
        is_train=False,
        subjects=test_subjects,
        deception_ratio=0.5,
        seed=config["training"]["seed"],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Quick test
    cfg = load_config()
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    batch = next(iter(train_loader))
    print("\nBatch keys:", batch.keys())
    print("Face shape:", batch["face"].shape)
    print("EEG temporal shape:", batch["eeg_temporal"].shape)
    print("EEG spectral shape:", batch["eeg_spectral"].shape)
    print("Face emotion:", batch["face_emotion_idx"])
    print("EEG emotion:", batch["eeg_emotion_idx"])
    print("Congruence:", batch["congruence"])