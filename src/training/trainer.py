"""Training loop for F1 winner prediction transformer.

Supports:
- Mixed precision (AMP) for Colab GPU
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Checkpoint saving
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Callable
from tqdm import tqdm
import time
import os

from .metrics import compute_metrics, print_metrics
from .losses import F1PredictionLoss


class Trainer:
    """Trainer for F1WinnerTransformer with AMP support."""

    def __init__(
        self,
        model: nn.Module,
        train_data: Dict[str, torch.Tensor],
        val_data: Dict[str, torch.Tensor],
        device: torch.device,
        batch_size: int = 32,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-5,
        epochs: int = 80,
        grad_clip: float = 1.0,
        patience: int = 12,
        use_amp: bool = True,
        checkpoint_dir: Optional[Path] = None,
        log_interval: int = 10,
        num_workers: int = 2,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.patience = patience
        self.use_amp = use_amp and "cuda" in str(device)
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        # Create data loaders
        self.train_loader = self._create_loader(train_data, batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = self._create_loader(val_data, batch_size, shuffle=False, num_workers=num_workers)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6,
        )

        # Loss
        self.criterion = F1PredictionLoss(
            num_classes=20,
            use_focal=False,
            label_smoothing=0.05,
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accs: List[float] = []

    def _create_loader(self, data: Dict[str, torch.Tensor], batch_size: int, shuffle: bool, num_workers: int = 2):
        """Create a DataLoader from data dict."""
        dataset = TensorDataset(
            data["context"],
            data["candidates"],
            data["winners"],
            data.get("time_gaps", torch.zeros(data["context"].shape[0], data["context"].shape[1])),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory="cuda" in str(self.device),
            drop_last=False,
        )

    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (context, candidates, winners, time_gaps) in enumerate(pbar):
            context = context.to(self.device)
            candidates = candidates.to(self.device)
            winners = winners.to(self.device)
            time_gaps = time_gaps.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                logits = self.model(context, candidates, time_gaps)
                loss = self.criterion(logits, winners)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Run validation. Returns metrics dict."""
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

            with autocast(enabled=self.use_amp):
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
        """Run full training loop.

        Returns:
            history dict with train_loss, val_loss, val_acc
        """
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

            # Logging
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

            # Save best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint("best.pt", epoch, val_metrics)
                print(f"  >>> New best model! (Acc: {val_metrics['accuracy']:.3f})")
            else:
                self.patience_counter += 1

            # Early stopping
            if early_stopping and self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={self.patience})")
                break

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.3f} at epoch {self.best_epoch+1}")

        # Load best model
        self._load_checkpoint("best.pt")

        # Final validation metrics
        print_metrics(self.validate(), prefix="Best Model")

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
        """Load model checkpoint. Returns True if successful."""
        path = self.checkpoint_dir / filename
        if not path.exists():
            return False
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return True

    def load_best_model(self):
        """Load the best saved model."""
        success = self._load_checkpoint("best.pt")
        if success:
            self.model.to(self.device)
            print(f"Loaded best model (val_acc={self.best_val_acc:.3f})")
        return success
