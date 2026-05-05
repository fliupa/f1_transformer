"""Training loop V2: SWA, OneCycleLR, augmentation, class weights.

Improvements over V1:
- Stochastic Weight Averaging (SWA) for better generalization
- OneCycleLR scheduler instead of CosineAnnealingWarmRestarts
- Class weights to reduce dominant-driver bias
- Data augmentation: noise injection, context dropout
- Linear warmup period
- Gradient accumulation for larger effective batch size
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
import time
import copy

from .metrics import compute_metrics, print_metrics
from .losses import F1PredictionLoss


class AugmentedDataset(TensorDataset):
    """TensorDataset with optional noise augmentation and context dropout."""

    def __init__(self, context, candidates, winners, time_gaps,
                 noise_std=0.02, context_dropout_prob=0.1):
        super().__init__(context, candidates, winners, time_gaps)
        self.noise_std = noise_std
        self.context_dropout_prob = context_dropout_prob

    def __getitem__(self, idx):
        context, candidates, winners, time_gaps = super().__getitem__(idx)

        # Augment: add noise to numeric features (skip embedding indices in candidates)
        if self.noise_std > 0:
            # candidates: columns 3+ are numeric
            noise = torch.randn_like(candidates[:, 3:]) * self.noise_std
            candidates_aug = candidates.clone()
            candidates_aug[:, 3:] += noise
            candidates = candidates_aug

            # context: all columns are numeric
            noise_ctx = torch.randn_like(context) * self.noise_std * 0.5
            context = context + noise_ctx

        # Augment: randomly zero out 1-2 context races
        if self.context_dropout_prob > 0 and torch.rand(1).item() < self.context_dropout_prob:
            n_drop = np.random.randint(1, 3)
            drop_indices = torch.randperm(len(context))[:n_drop]
            context[drop_indices] = 0.0
            time_gaps[drop_indices] = 365.0

        return context, candidates, winners, time_gaps


class SWA:
    """Stochastic Weight Averaging: averages model weights over last N epochs."""

    def __init__(self, model: nn.Module, start_epoch: int = 10):
        self.model = model
        self.start_epoch = start_epoch
        self.swa_model = None
        self.n_averaged = 0

    def update(self, model: nn.Module):
        """Accumulate exponential moving average of weights."""
        if self.swa_model is None:
            self.swa_model = copy.deepcopy(model)
            self.n_averaged = 1
        else:
            decay = 1.0 / (self.n_averaged + 1)
            for swa_p, p in zip(self.swa_model.parameters(), model.parameters()):
                swa_p.data = decay * p.data + (1 - decay) * swa_p.data
            self.n_averaged += 1

    def set_weights(self, model: nn.Module):
        """Set model weights to SWA average."""
        if self.swa_model is not None:
            model.load_state_dict(self.swa_model.state_dict())


class TrainerV2:
    """Trainer with SWA, OneCycleLR, augmentation, and class weights."""

    def __init__(
        self,
        model: nn.Module,
        train_data: Dict[str, torch.Tensor],
        val_data: Dict[str, torch.Tensor],
        device: torch.device,
        batch_size: int = 32,
        learning_rate: float = 1e-3,  # Higher for OneCycle
        weight_decay: float = 1e-4,
        epochs: int = 80,
        grad_clip: float = 1.0,
        patience: int = 15,
        use_amp: bool = True,
        checkpoint_dir: Optional[Path] = None,
        log_interval: int = 10,
        num_workers: int = 2,
        warmup_epochs: int = 5,
        swa_start: int = 20,
        label_smoothing: float = 0.10,
        noise_std: float = 0.02,
        context_dropout_prob: float = 0.15,
        grad_accum_steps: int = 2,
        use_class_weights: bool = True,
        num_classes: int = 20,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.patience = patience
        self.use_amp = use_amp and "cuda" in str(device)
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.warmup_epochs = warmup_epochs
        self.swa_start = swa_start
        self.grad_accum_steps = grad_accum_steps

        # Create loaders with augmentation for training
        self.train_loader = self._create_loader(
            train_data, batch_size, shuffle=True, num_workers=num_workers,
            augment=True, noise_std=noise_std, context_dropout_prob=context_dropout_prob,
        )
        self.val_loader = self._create_loader(
            val_data, batch_size, shuffle=False, num_workers=num_workers,
            augment=False,
        )

        # Class weights (reduce bias toward dominant drivers)
        class_weights = None
        if use_class_weights:
            class_weights = self._compute_class_weights(train_data["winners"])
            class_weights = class_weights.to(device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # OneCycleLR scheduler (configured after seeing data)
        steps_per_epoch = len(self.train_loader) // grad_accum_steps
        total_steps = epochs * steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,  # 10% of steps for warmup
            anneal_strategy='cos',
            final_div_factor=100,
        )

        # Loss
        self.criterion = F1PredictionLoss(
            num_classes=self.num_classes,
            class_weights=class_weights,
            use_focal=False,
            label_smoothing=label_smoothing,
        )

        # AMP
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # SWA
        self.swa = SWA(model, start_epoch=swa_start)

        # Tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.checkpoint_filename: str = "best_v2.pt"
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accs: List[float] = []

    def _compute_class_weights(self, winners: torch.Tensor) -> torch.Tensor:
        """Compute inverse-frequency class weights.

        Dominant drivers (e.g. VER with 8 wins) get lower weight.
        Rare winners get higher weight.
        """
        winner_counts = torch.bincount(winners, minlength=self.num_classes).float()
        # Add smoothing
        winner_counts = winner_counts + 1
        weights = 1.0 / winner_counts
        # Normalize so mean = 1
        weights = weights / weights.mean()
        return weights

    def _create_loader(self, data, batch_size, shuffle, num_workers,
                       augment=False, noise_std=0.02, context_dropout_prob=0.15):
        """Create DataLoader, optionally with augmentation."""
        if augment:
            dataset = AugmentedDataset(
                data["context"], data["candidates"], data["winners"],
                data.get("time_gaps", torch.zeros(data["context"].shape[0], data["context"].shape[1])),
                noise_std=noise_std,
                context_dropout_prob=context_dropout_prob,
            )
        else:
            dataset = TensorDataset(
                data["context"], data["candidates"], data["winners"],
                data.get("time_gaps", torch.zeros(data["context"].shape[0], data["context"].shape[1])),
            )

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory="cuda" in str(self.device),
            drop_last=False,
        )

    def train_epoch(self) -> float:
        """Train for one epoch with gradient accumulation and augmentation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, (context, candidates, winners, time_gaps) in enumerate(pbar):
            context = context.to(self.device)
            candidates = candidates.to(self.device)
            winners = winners.to(self.device)
            time_gaps = time_gaps.to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                logits = self.model(context, candidates, time_gaps)
                loss = self.criterion(logits, winners)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            if batch_idx % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{loss.item()*self.grad_accum_steps:.4f}", "lr": f"{lr:.2e}"})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, loader=None) -> Dict[str, float]:
        """Run validation."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_winners = []

        for context, candidates, winners, time_gaps in loader:
            context = context.to(self.device)
            candidates = candidates.to(self.device)
            winners = winners.to(self.device)
            time_gaps = time_gaps.to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                logits = self.model(context, candidates, time_gaps)
                loss = self.criterion(logits, winners)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_winners.append(winners.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_winners = torch.cat(all_winners, dim=0)
        metrics = compute_metrics(all_logits, all_winners)
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    def train(self, early_stopping: bool = True) -> Dict[str, List[float]]:
        """Run full training loop with SWA."""
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics["loss"])
            self.val_accs.append(val_metrics["accuracy"])

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.3f} ({val_metrics['accuracy']*100:.1f}%) | "
                f"Top3: {val_metrics['top3_accuracy']:.3f} | "
                f"MRR: {val_metrics['mrr']:.3f} | "
                f"LR: {lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # SWA update (after swa_start epoch)
            if epoch >= self.swa_start:
                self.swa.update(self.model)

            # Save best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(self.checkpoint_filename, epoch, val_metrics)
                print(f"  >>> New best model! (Acc: {val_metrics['accuracy']:.3f})")
            else:
                self.patience_counter += 1

            if early_stopping and self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={self.patience})")
                break

        # Apply SWA weights for final eval
        if self.swa.n_averaged > 0:
            print(f"\nApplying SWA ({self.swa.n_averaged} averaged models)")
            self.swa.set_weights(self.model)

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.3f} at epoch {self.best_epoch+1}")

        self._load_checkpoint(self.checkpoint_filename)
        print_metrics(self.validate(), prefix="Best Model (V2)")

        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "val_acc": self.val_accs,
        }

    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "best_val_acc": self.best_val_acc,
        }, path)

    def _load_checkpoint(self, filename: str) -> bool:
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        if not path.exists():
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return True
