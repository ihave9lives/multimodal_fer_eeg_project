#!/usr/bin/env python
"""
Main entry point for Multimodal FER + EEG Deception Detection.

Usage:
    python main.py --config configs/default.yaml --mode train
    python main.py --config configs/default.yaml --mode verify --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

from models.models import MultimodalDeceptionModel, build_model
from dataset.dataset import create_dataloaders, MultimodalDeceptionDataset, get_vision_transforms
from engine.engine import run_training, load_checkpoint, get_device, set_seed, TrainingConfig, build_optimizer, build_scheduler, Trainer
from loss.loss import build_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal FER + EEG Deception Detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["train", "verify", "test"], default="train", help="Mode to run")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint for verify/test mode")
    parser.add_argument("--face-image", type=str, default=None, help="Path to face image for verification")
    parser.add_argument("--eeg-file", type=str, default=None, help="Path to EEG file (.npy) for verification")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--output", type=str, default=None, help="Output path for results")
    return parser.parse_args()


def verify_single_sample(
    model: MultimodalDeceptionModel,
    face_path: str,
    eeg_path: str = None,
    config: dict = None,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.7,
):
    """
    Verify authenticity of a single face-EEG pair.
    
    If EEG file not provided, generates synthetic EEG matching the face emotion (for testing).
    """
    from PIL import Image
    import numpy as np
    from dataset.dataset import preprocess_eeg, SyntheticEEGGenerator, EMOTION_LABELS, EMOTION_TO_IDX
    
    model.eval()
    model.to(device)
    
    # Load and transform face image
    transform = get_vision_transforms(config, is_train=False)
    face_img = Image.open(face_path).convert("RGB")
    face_tensor = transform(face_img).unsqueeze(0).to(device)
    
    # Load or generate EEG
    if eeg_path and Path(eeg_path).exists():
        eeg_raw = np.load(eeg_path)  # Expected: (C, T)
        temporal, spectral = preprocess_eeg(eeg_raw, config)
    else:
        # Generate synthetic EEG - try to infer emotion from filename
        face_emotion = "Neutral"
        for emo in EMOTION_LABELS:
            if emo.lower() in face_path.lower():
                face_emotion = emo
                break
        
        generator = SyntheticEEGGenerator(config)
        temporal, spectral, _ = generator.generate_congruent_pair(face_emotion)
    
    eeg_temporal = torch.from_numpy(temporal).unsqueeze(0).to(device)
    eeg_spectral = torch.from_numpy(spectral).unsqueeze(0).to(device)
    
    # Run verification
    with torch.no_grad():
        result = model.verify_authenticity(
            face_tensor,
            eeg_temporal,
            eeg_spectral,
            threshold=threshold,
            emotion_labels=config["model"]["emotion_labels"] if config else EMOTION_LABELS,
        )
    
    return result


def test_mode(args):
    """Run test evaluation on test set."""
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    if args.device != "auto":
        config["training"]["device"] = args.device
    
    device = get_device(config["training"]["device"])
    set_seed(config["training"]["seed"])
    
    # Data
    _, _, test_loader = create_dataloaders(config)
    
    # Model
    model = build_model(config)
    
    # Load checkpoint
    if args.checkpoint:
        load_checkpoint(model, path=args.checkpoint, device=device)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No checkpoint provided, using random weights")
    
    # Loss
    loss_fn = build_loss(config)
    
    # Training config for Trainer
    training_config = TrainingConfig(**config["training"])
    
    # Trainer (just for validation)
    trainer = Trainer(
        model=model,
        train_loader=test_loader,  # Not used
        val_loader=test_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=build_optimizer(model, training_config),
        scheduler=None,
        config=training_config,
        device=device,
    )
    
    # Run validation
    test_metrics = trainer.validate(test_loader, "test")
    
    print("\n=== Test Results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return test_metrics


def verify_mode(args):
    """Verify a single sample."""
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    if args.device != "auto":
        config["training"]["device"] = args.device
    
    device = get_device(config["training"]["device"])
    
    # Model
    model = build_model(config)
    
    # Load checkpoint
    if args.checkpoint:
        load_checkpoint(model, path=args.checkpoint, device=device)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("WARNING: No checkpoint provided, using random weights")
    
    if not args.face_image:
        print("ERROR: --face-image required for verify mode")
        return
    
    result = verify_single_sample(
        model=model,
        face_path=args.face_image,
        eeg_path=args.eeg_file,
        config=config,
        device=device,
        threshold=config["model"]["deception"]["threshold"],
    )
    
    print("\n=== Verification Result ===")
    for k, v in result.items():
        if k != "diagnostics":
            print(f"  {k}: {v}")
    
    print("\n=== Diagnostics ===")
    for k, v in result["diagnostics"].items():
        print(f"  {k}: {v}")
    
    if args.output:
        import json
        with open(args.output, "w") as f:
            # Convert non-serializable
            serializable = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in result.items()}
            serializable["diagnostics"] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in result["diagnostics"].items()}
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return result


def main():
    args = parse_args()
    
    if args.mode == "train":
        run_training(args.config)
    elif args.mode == "test":
        test_mode(args)
    elif args.mode == "verify":
        verify_mode(args)


if __name__ == "__main__":
    main()