"""
Training engine for Multimodal FER + EEG Deception Detection.

Provides training loop, validation, checkpointing, logging, and inference utilities.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.models import MultimodalDeceptionModel, build_model
from loss.loss import MultiTaskLoss, build_loss
from dataset.dataset import create_dataloaders, EMOTION_LABELS


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine_annealing"
    warmup_epochs: int = 5
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_best_only: bool = True
    early_stopping_patience: int = 15
    eval_every: int = 1


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 15, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig, num_training_steps: int):
    """Build learning rate scheduler from config."""
    lr = float(config.lr)
    if config.scheduler == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs - config.warmup_epochs,
            eta_min=lr * 0.01,
        )
    elif config.scheduler == "cosine_warmup":
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step: int):
            if step < config.warmup_epochs:
                return float(step) / float(max(1, config.warmup_epochs))
            progress = float(step - config.warmup_epochs) / float(max(1, config.epochs - config.warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        return LambdaLR(optimizer, lr_lambda)
    elif config.scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
    else:
        return None


class Trainer:
    """
    Main training class for the multimodal deception detection model.
    """
    
    def __init__(
        self,
        model: MultimodalDeceptionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: MultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision and device.type == "cuda" else None
        if config.mixed_precision and device.type == "cuda":
            self.autocast_ctx = autocast(device_type="cuda", dtype=torch.float16)
        else:
            self.autocast_ctx = nullcontext()
        
        # Logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode="min",
        )
        
        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {"train": [], "val": []}
    
    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = {"total": 0.0}
        component_sums = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Prepare targets
            targets = {
                "face_emotion_idx": batch["face_emotion_idx"],
                "eeg_emotion_idx": batch["eeg_emotion_idx"],
                "fused_emotion_idx": batch["face_emotion_idx"],  # Same as face for fused
                "valence": self._emotion_to_valence(batch["eeg_emotion_idx"]),
                "arousal": self._emotion_to_arousal(batch["eeg_emotion_idx"]),
                "congruence": batch["congruence"],
            }
            
            # Forward pass with mixed precision
            with self.autocast_ctx:
                outputs = self.model(
                    batch["face"],
                    batch["eeg_temporal"],
                    batch["eeg_spectral"],
                )
                loss, components = self.loss_fn(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.gradient_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.gradient_clip))
                self.optimizer.step()
            
            # Accumulate losses
            epoch_losses["total"] += loss.item()
            for k, v in components.items():
                component_sums[k] = component_sums.get(k, 0.0) + v
            num_batches += 1
            self.global_step += 1
            
            # Log batch
            if batch_idx % 50 == 0:
                self.writer.add_scalar("train/batch_loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in component_sums.items()}
        avg_losses["total"] = epoch_losses["total"] / num_batches
        
        return avg_losses
    
    def validate(self, loader: DataLoader, prefix: str = "val") -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        epoch_losses = {"total": 0.0}
        component_sums = {}
        num_batches = 0
        
        # Metrics accumulation
        all_face_preds = []
        all_face_targets = []
        all_eeg_preds = []
        all_eeg_targets = []
        all_fused_preds = []
        all_fused_targets = []
        all_congruence_preds = []
        all_congruence_targets = []
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                targets = {
                    "face_emotion_idx": batch["face_emotion_idx"],
                    "eeg_emotion_idx": batch["eeg_emotion_idx"],
                    "fused_emotion_idx": batch["face_emotion_idx"],
                    "valence": self._emotion_to_valence(batch["eeg_emotion_idx"]),
                    "arousal": self._emotion_to_arousal(batch["eeg_emotion_idx"]),
                    "congruence": batch["congruence"],
                }
                
                with self.autocast_ctx:
                    outputs = self.model(
                        batch["face"],
                        batch["eeg_temporal"],
                        batch["eeg_spectral"],
                    )
                    loss, components = self.loss_fn(outputs, targets)
                
                epoch_losses["total"] += loss.item()
                for k, v in components.items():
                    component_sums[k] = component_sums.get(k, 0.0) + v
                num_batches += 1
                
                # Collect predictions for metrics
                all_face_preds.append(outputs["face_logits"].argmax(dim=1).cpu())
                all_face_targets.append(targets["face_emotion_idx"].cpu())
                all_eeg_preds.append(outputs["eeg_logits"].argmax(dim=1).cpu())
                all_eeg_targets.append(targets["eeg_emotion_idx"].cpu())
                all_fused_preds.append(outputs["fused_logits"].argmax(dim=1).cpu())
                all_fused_targets.append(targets["fused_emotion_idx"].cpu())
                
                # Deception predictions
                if outputs["deception_logits"].size(1) == 2:
                    dec_pred = outputs["deception_logits"].argmax(dim=1)
                else:
                    dec_pred = (outputs["deception_logits"] > 0).long()
                all_congruence_preds.append(dec_pred.cpu())
                all_congruence_targets.append(targets["congruence"].cpu())
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in component_sums.items()}
        avg_losses["total"] = epoch_losses["total"] / num_batches
        
        # Compute metrics
        metrics = self._compute_metrics(
            all_face_preds, all_face_targets,
            all_eeg_preds, all_eeg_targets,
            all_fused_preds, all_fused_targets,
            all_congruence_preds, all_congruence_targets,
            prefix,
        )
        avg_losses.update(metrics)
        
        return avg_losses
    
    def _compute_metrics(
        self,
        face_preds, face_targets,
        eeg_preds, eeg_targets,
        fused_preds, fused_targets,
        cong_preds, cong_targets,
        prefix: str,
    ) -> dict[str, float]:
        """Compute classification metrics."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
        
        # Concatenate
        face_preds = torch.cat(face_preds).numpy()
        face_targets = torch.cat(face_targets).numpy()
        eeg_preds = torch.cat(eeg_preds).numpy()
        eeg_targets = torch.cat(eeg_targets).numpy()
        fused_preds = torch.cat(fused_preds).numpy()
        fused_targets = torch.cat(fused_targets).numpy()
        cong_preds = torch.cat(cong_preds).numpy()
        cong_targets = torch.cat(cong_targets).numpy()
        
        metrics = {}
        
        # Face emotion metrics
        metrics[f"{prefix}_face_acc"] = accuracy_score(face_targets, face_preds)
        metrics[f"{prefix}_face_f1_macro"] = f1_score(face_targets, face_preds, average="macro")
        
        # EEG emotion metrics
        metrics[f"{prefix}_eeg_acc"] = accuracy_score(eeg_targets, eeg_preds)
        metrics[f"{prefix}_eeg_f1_macro"] = f1_score(eeg_targets, eeg_preds, average="macro")
        
        # Fused emotion metrics
        metrics[f"{prefix}_fused_acc"] = accuracy_score(fused_targets, fused_preds)
        metrics[f"{prefix}_fused_f1_macro"] = f1_score(fused_targets, fused_preds, average="macro")
        
        # Deception detection metrics
        metrics[f"{prefix}_deception_acc"] = accuracy_score(cong_targets, cong_preds)
        metrics[f"{prefix}_deception_f1"] = f1_score(cong_targets, cong_preds, average="binary")
        
        # Log confusion matrices
        if prefix == "val":
            self._log_confusion_matrix(face_targets, face_preds, "face", prefix)
            self._log_confusion_matrix(cong_targets, cong_preds, "deception", prefix)
        
        return metrics
    
    def _log_confusion_matrix(self, targets, preds, name: str, prefix: str):
        """Log confusion matrix to TensorBoard."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import io
        
        cm = confusion_matrix(targets, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        if name == "face":
            labels = EMOTION_LABELS
        else:
            labels = ["Genuine", "Deceptive"]
        
        ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
               xticklabels=labels, yticklabels=labels,
               title=f"{prefix}_{name}_confusion_matrix",
               ylabel="True label", xlabel="Predicted label")
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        # Save to buffer and log
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        from PIL import Image
        img = Image.open(buf)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        self.writer.add_image(f"{prefix}/{name}_confusion_matrix", img_tensor, self.epoch)
        plt.close(fig)
    
    def _emotion_to_valence(self, emotion_idx: torch.Tensor) -> torch.Tensor:
        """Map emotion index to valence target [-1, 1]."""
        # Valence mapping based on emotion
        valence_map = torch.tensor([
            -0.8,  # Anger
            -0.4,  # Contempt
            -0.9,  # Disgust
            -0.9,  # Fear
            0.9,   # Happy
            0.0,   # Neutral
            -0.7,  # Sad
            0.1,   # Surprised
        ], device=self.device)
        return valence_map[emotion_idx].unsqueeze(1).float()
    
    def _emotion_to_arousal(self, emotion_idx: torch.Tensor) -> torch.Tensor:
        """Map emotion index to arousal target [0, 1]."""
        arousal_map = torch.tensor([
            0.9,   # Anger
            0.5,   # Contempt
            0.6,   # Disgust
            0.95,  # Fear
            0.7,   # Happy
            0.2,   # Neutral
            0.3,   # Sad
            0.85,  # Surprised
        ], device=self.device)
        return arousal_map[emotion_idx].unsqueeze(1).float()
    
    def train(self) -> dict:
        """Run full training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.config.eval_every == 0:
                val_metrics = self.validate(self.val_loader, "val")
            else:
                val_metrics = {}
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("val_total", train_metrics["total"]))
                else:
                    self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - epoch_start
            self._log_epoch(train_metrics, val_metrics, epoch_time)
            
            # Save checkpoint
            is_best = val_metrics.get("val_total", float("inf")) < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_total"]
            self._save_checkpoint(is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics.get("val_total", train_metrics["total"])):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s")
        
        # Final test evaluation
        test_metrics = self.validate(self.test_loader, "test")
        self._log_epoch({}, test_metrics, 0, is_test=True)
        
        self.writer.close()
        return self.history
    
    def _log_epoch(self, train_metrics: dict, val_metrics: dict, epoch_time: float, is_test: bool = False):
        """Log epoch metrics."""
        prefix = "test" if is_test else "epoch"
        
        # Console log
        if not is_test:
            msg = f"Epoch {self.epoch:3d}/{self.config.epochs} | "
            msg += f"Train Loss: {train_metrics['total']:.4f} | "
            if val_metrics:
                msg += f"Val Loss: {val_metrics.get('val_total', 0):.4f} | "
                msg += f"Face Acc: {val_metrics.get('val_face_acc', 0):.4f} | "
                msg += f"EEG Acc: {val_metrics.get('val_eeg_acc', 0):.4f} | "
                msg += f"Fused Acc: {val_metrics.get('val_fused_acc', 0):.4f} | "
                msg += f"Dec Acc: {val_metrics.get('val_deception_acc', 0):.4f} | "
            msg += f"Time: {epoch_time:.1f}s"
            print(msg)
        
        # TensorBoard
        for k, v in train_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, self.epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, self.epoch)
        
        # History
        if not is_test:
            self.history["train"].append({"epoch": self.epoch, **train_metrics})
            if val_metrics:
                self.history["val"].append({"epoch": self.epoch, **val_metrics})
    
    def _save_checkpoint(self, is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "history": self.history,
        }
        
        # Save last
        if not self.config.save_best_only:
            torch.save(checkpoint, self.save_dir / f"checkpoint_epoch_{self.epoch}.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pt")
            print(f"  → Saved best model (val_loss={self.best_val_loss:.4f})")
        
        # Always save latest
        torch.save(checkpoint, self.save_dir / "latest_model.pt")


def load_checkpoint(
    model: MultimodalDeceptionModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    path: str = "./checkpoints/best_model.pt",
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


def run_training(config_path: str = "configs/default.yaml"):
    """Main training entry point."""
    import yaml
    
    # Load config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Set seed
    set_seed(config_dict["training"]["seed"])
    
    # Device
    device = get_device(config_dict["training"]["device"])
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, test_loader = create_dataloaders(config_dict)
    
    # Model
    model = build_model(config_dict)
    
    # Loss
    loss_fn = build_loss(config_dict)
    
    # Optimizer & Scheduler
    training_config = TrainingConfig(**config_dict["training"])
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config, len(train_loader) * training_config.epochs)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
        device=device,
    )
    
    # Train
    history = trainer.train()
    
    return model, history


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    model, history = run_training(config_path)