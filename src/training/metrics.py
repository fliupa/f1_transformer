"""Evaluation metrics for F1 winner prediction."""
import torch
import numpy as np
from typing import Dict


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        logits: (batch, num_drivers) raw logits
        targets: (batch,) integer indices of true winners

    Returns:
        dict of metric name -> value
    """
    probs = torch.softmax(logits, dim=-1)
    batch_size = logits.shape[0]

    # Accuracy
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).sum().item()
    accuracy = correct / batch_size if batch_size > 0 else 0

    # Top-3 accuracy
    _, top3_indices = torch.topk(probs, k=min(3, logits.shape[1]), dim=-1)
    top3_correct = sum(1 for i in range(batch_size) if targets[i].item() in top3_indices[i].tolist())
    top3_accuracy = top3_correct / batch_size if batch_size > 0 else 0

    # Top-5 accuracy
    k = min(5, logits.shape[1])
    _, top5_indices = torch.topk(probs, k=k, dim=-1)
    top5_correct = sum(1 for i in range(batch_size) if targets[i].item() in top5_indices[i].tolist())
    top5_accuracy = top5_correct / batch_size if batch_size > 0 else 0

    # Mean Reciprocal Rank
    # Rank of the true winner (1-indexed)
    sorted_indices = torch.argsort(probs, dim=-1, descending=True)
    ranks = []
    for i in range(batch_size):
        rank = (sorted_indices[i] == targets[i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    mrr = np.mean([1.0 / r for r in ranks])

    # Mean position error (in terms of probability ranking)
    mean_rank = np.mean(ranks)

    # Log loss
    log_probs = torch.log(probs + 1e-10)
    log_loss = -log_probs[torch.arange(batch_size), targets].mean().item()

    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "log_loss": log_loss,
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty-print evaluation metrics."""
    title = f"{prefix} Metrics" if prefix else "Metrics"
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Accuracy:       {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.3f} ({metrics['top3_accuracy']*100:.1f}%)")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.3f} ({metrics['top5_accuracy']*100:.1f}%)")
    print(f"  MRR:            {metrics['mrr']:.4f}")
    print(f"  Mean Rank:      {metrics['mean_rank']:.2f}")
    print(f"  Log Loss:       {metrics['log_loss']:.4f}")
    print(f"{'='*50}")
