"""Loss functions for F1 winner prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class F1PredictionLoss(nn.Module):
    """Combined loss for F1 winner prediction.

    - Cross-entropy for winner classification
    - Optional class weights to handle dominant drivers
    - Optional focal loss component for hard examples
    """

    def __init__(
        self,
        num_classes: int = 20,
        class_weights: torch.Tensor = None,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.ones(num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_drivers) raw logits
            targets: (batch,) integer indices of winners

        Returns:
            scalar loss
        """
        device = logits.device
        weights = self.class_weights.to(device)

        if self.use_focal:
            return self._focal_loss(logits, targets, weights)
        else:
            return F.cross_entropy(
                logits, targets,
                weight=weights,
                label_smoothing=self.label_smoothing,
            )

    def _focal_loss(self, logits, targets, weights):
        """Focal loss: focuses on hard-to-classify examples."""
        ce_loss = F.cross_entropy(logits, targets, weight=weights, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
        return focal_loss


def compute_class_weights(winner_counts: torch.Tensor, num_classes: int = 20) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle class imbalance.

    Args:
        winner_counts: (num_classes,) tensor of win counts per driver position
        num_classes: total number of driver slots

    Returns:
        (num_classes,) normalized weights
    """
    total = winner_counts.sum()
    weights = total / (num_classes * (winner_counts + 1))  # +1 to avoid division by zero
    weights = weights / weights.sum() * num_classes  # Normalize
    return weights.float()
