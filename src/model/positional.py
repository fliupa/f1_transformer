"""Time-aware positional encoding for F1 race sequences.

Standard positional encoding assumes equally-spaced positions.
In F1, races are weeks or months apart. This module encodes the actual
time gap (in days) between each context race and the target race.
"""
import torch
import torch.nn as nn
import math


class TimeAwarePositionalEncoding(nn.Module):
    """Positional encoding based on real time gaps (days) between races.

    Args:
        d_model: Embedding dimension
        max_time_gap: Maximum expected time gap in normalized units (default ~365 days)
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, max_time_gap: float = 365.0, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_time_gap = max_time_gap
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute frequency bands
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, time_gaps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_gaps: (batch, seq_len) tensor of time gaps in days
                        or (seq_len,) that will be broadcast

        Returns:
            (batch, seq_len, d_model) positional encodings
        """
        if time_gaps.dim() == 1:
            time_gaps = time_gaps.unsqueeze(0)  # (1, seq_len)

        batch, seq_len = time_gaps.shape
        device = time_gaps.device

        # Normalize time gaps to [0, 1]
        normalized = time_gaps / self.max_time_gap  # (batch, seq_len)

        # Compute sinusoidal encodings
        pe = torch.zeros(batch, seq_len, self.d_model, device=device)
        # normalized: (batch, seq_len) -> (batch, seq_len, 1)
        arg = normalized.unsqueeze(-1) * self.div_term.to(device)  # (batch, seq_len, d_model//2)

        pe[:, :, 0::2] = torch.sin(arg)
        pe[:, :, 1::2] = torch.cos(arg)

        # Also add a sequential position component (order in the context)
        # This helps the model distinguish between race t-1, t-2, etc.
        positions = torch.arange(seq_len, device=device).float().unsqueeze(0)  # (1, seq_len)
        seq_pe = torch.zeros(batch, seq_len, self.d_model, device=device)
        seq_arg = positions.unsqueeze(-1) * self.div_term.to(device)  # (1, seq_len, d_model//2)
        seq_pe[:, :, 0::2] = torch.sin(seq_arg)
        seq_pe[:, :, 1::2] = torch.cos(seq_arg)

        # Combine: 70% time-aware + 30% sequential
        combined = 0.7 * pe + 0.3 * seq_pe

        return self.dropout(combined)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding as an alternative.

    Simple learned embeddings for each position in the context window.
    """

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) input embeddings

        Returns:
            (batch, seq_len, d_model) with added positional encoding
        """
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        return self.dropout(x + self.pe(positions))
