"""F1 Winner Prediction Transformer Model.

Dual-stream architecture with cross-attention:
1. Race History Encoder: Transformer encoder processes a sequence of N previous races
2. Driver Candidate Encoder: MLP encodes per-driver features for the target race
3. Cross-Attention: Driver queries attend to race history keys/values
4. Prediction Head: Produces win probabilities over 20 drivers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .positional import TimeAwarePositionalEncoding


class F1WinnerTransformer(nn.Module):
    """Transformer model for predicting F1 race winners.

    Architecture:
    - Embeds categorical features (driver, constructor, circuit)
    - Encodes race history with transformer encoder + time-aware PE
    - Cross-attention between driver candidates and race history
    - Linear head produces win probability per driver
    """

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 6,
        n_encoder_layers: int = 3,
        n_cross_attn_layers: int = 2,
        d_ff: int = 768,
        dropout: float = 0.15,
        context_window: int = 10,
        num_drivers: int = 80,
        num_constructors: int = 20,
        num_circuits: int = 45,
        driver_embed_dim: int = 16,
        constructor_embed_dim: int = 8,
        circuit_embed_dim: int = 8,
        d_candidate_raw: int = 32,  # Per-driver features after embedding indices
        d_context_raw: int = 45,    # Race context features
        use_time_aware_pe: bool = True,
        max_drivers_per_race: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_window = context_window
        self.num_drivers = num_drivers
        self.num_constructors = num_constructors
        self.num_circuits = num_circuits
        self.max_drivers_per_race = max_drivers_per_race

        # ─── Embedding Layers ───────────────────────────────────────
        self.driver_embed = nn.Embedding(num_drivers + 1, driver_embed_dim, padding_idx=0)
        self.constructor_embed = nn.Embedding(num_constructors + 1, constructor_embed_dim, padding_idx=0)
        self.circuit_embed = nn.Embedding(num_circuits + 1, circuit_embed_dim, padding_idx=0)

        # Calculate actual feature dimensions
        # d_candidate_raw includes: 3 embedding indices + numeric features
        # After embedding lookup + concat with numerics:
        # embedding_dims: driver(16) + constructor(8) + circuit(8) = 32
        # rest of d_candidate_raw are numeric features after index 3
        self._embed_total = driver_embed_dim + constructor_embed_dim + circuit_embed_dim
        self._candidate_numeric_feats = d_candidate_raw - 3  # Subtract 3 embedding indices

        # ─── Input Projections ──────────────────────────────────────
        # Candidate features: embeddings + numeric features -> d_model
        self.candidate_proj = nn.Sequential(
            nn.Linear(self._embed_total + self._candidate_numeric_feats, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Context features projection (all numeric) -> d_model
        self.context_proj = nn.Sequential(
            nn.Linear(d_context_raw, d_model),  # All features are numeric
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # ─── Positional Encoding ─────────────────────────────────────
        if use_time_aware_pe:
            self.pos_encoding = TimeAwarePositionalEncoding(d_model, dropout=dropout)
        else:
            self.pos_encoding = None

        # ─── Race History Transformer Encoder ─────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better stability
        )
        self.race_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # ─── Cross-Attention Layers ──────────────────────────────────
        # Driver candidates query the encoded race history
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_cross_attn_layers)
        ])

        # ─── Prediction Head ─────────────────────────────────────────
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Embeddings: normal distribution
        nn.init.normal_(self.driver_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.constructor_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.circuit_embed.weight, mean=0, std=0.02)
        # Zero out padding index
        with torch.no_grad():
            self.driver_embed.weight[0] = 0
            self.constructor_embed.weight[0] = 0
            self.circuit_embed.weight[0] = 0

    def _embed_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Process candidate driver features.

        Args:
            candidates: (batch, 20, d_candidate_raw)
                Columns 0: driver_idx, 1: constructor_idx, 2: circuit_idx, 3+: numeric features

        Returns:
            (batch, 20, d_model)
        """
        batch, num_drivers, _ = candidates.shape

        # Extract embedding indices (integer)
        driver_idx = candidates[:, :, 0].long()
        constructor_idx = candidates[:, :, 1].long()
        circuit_idx = candidates[:, :, 2].long()

        # Lookup embeddings
        d_emb = self.driver_embed(driver_idx)     # (batch, 20, driver_embed_dim)
        c_emb = self.constructor_embed(constructor_idx)  # (batch, 20, constructor_embed_dim)
        circ_emb = self.circuit_embed(circuit_idx)  # (batch, 20, circuit_embed_dim)

        # Numeric features
        numeric = candidates[:, :, 3:]  # (batch, 20, rest)

        # Concatenate all
        combined = torch.cat([d_emb, c_emb, circ_emb, numeric], dim=-1)

        # Project to d_model
        return self.candidate_proj(combined)

    def _embed_context(self, context: torch.Tensor) -> torch.Tensor:
        """Process context race summary features.

        Context has all numeric features including placeholder driver indices
        for top-3 finishers and winner.

        Args:
            context: (batch, N, d_context_raw)

        Returns:
            (batch, N, d_model)
        """
        # All features are numeric/indices, project directly
        return self.context_proj(context)

    def forward(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        time_gaps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: (batch, context_window, d_context_raw)
                     Race history summaries for previous N races
            candidates: (batch, max_drivers, d_candidate_raw)
                        Features for each driver in the target race
            time_gaps: (batch, context_window) optional
                       Days between each context race and target race

        Returns:
            logits: (batch, max_drivers) win logits (before softmax)
        """
        # 1. Embed candidate drivers
        driver_feats = self._embed_candidates(candidates)  # (B, 20, d_model)

        # 2. Embed context races
        context_feats = self._embed_context(context)  # (B, N, d_model)

        # 3. Add time-aware positional encoding
        if self.pos_encoding is not None and time_gaps is not None:
            pe = self.pos_encoding(time_gaps)  # (B, N, d_model)
            context_feats = context_feats + pe

        # 4. Encode race history with transformer encoder
        # Create padding mask (no padding needed, all sequences are same length)
        context_encoded = self.race_encoder(context_feats)  # (B, N, d_model)

        # 5. Cross-attention: drivers attend to race history
        # Drivers are queries, race history are keys/values
        driver_repr = driver_feats
        for cross_attn in self.cross_attn_layers:
            driver_repr = cross_attn(driver_repr, context_encoded)

        # 6. Prediction head: produce scalar logit per driver
        logits = self.pred_head(driver_repr).squeeze(-1)  # (B, 20)

        return logits

    def predict_proba(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        time_gaps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict win probabilities for each driver.

        Returns:
            probs: (batch, max_drivers) win probabilities (softmax)
        """
        logits = self.forward(context, candidates, time_gaps)
        return F.softmax(logits, dim=-1)

    def predict_winner(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        time_gaps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the winner and confidence.

        Returns:
            winner_idx: (batch,) index of predicted winner
            confidence: (batch,) probability for predicted winner
        """
        probs = self.predict_proba(context, candidates, time_gaps)
        winner_idx = torch.argmax(probs, dim=-1)
        confidence = probs.gather(1, winner_idx.unsqueeze(-1)).squeeze(-1)
        return winner_idx, confidence


class CrossAttentionBlock(nn.Module):
    """Cross-attention block where queries attend to key-value pairs.

    Q comes from driver candidates, K and V come from race history.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (batch, seq_q, d_model) - driver representations
            context: (batch, seq_kv, d_model) - encoded race history

        Returns:
            (batch, seq_q, d_model) - updated driver representations
        """
        # Cross-attention: queries attend to context
        attn_out, _ = self.cross_attn(queries, context, context)
        x = self.norm1(queries + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


def create_model_from_config(config: dict) -> F1WinnerTransformer:
    """Create model instance from configuration dictionary."""
    model_cfg = config["model"]
    return F1WinnerTransformer(
        d_model=model_cfg.get("d_model", 192),
        n_heads=model_cfg.get("n_heads", 6),
        n_encoder_layers=model_cfg.get("n_encoder_layers", 3),
        n_cross_attn_layers=model_cfg.get("n_cross_attn_layers", 2),
        d_ff=model_cfg.get("d_ff", 768),
        dropout=model_cfg.get("dropout", 0.15),
        context_window=model_cfg.get("context_window", 10),
        num_drivers=model_cfg.get("max_drivers", 80),
        num_constructors=model_cfg.get("max_constructors", 20),
        num_circuits=model_cfg.get("max_circuits", 45),
        driver_embed_dim=model_cfg.get("driver_embed_dim", 16),
        constructor_embed_dim=model_cfg.get("constructor_embed_dim", 8),
        circuit_embed_dim=model_cfg.get("circuit_embed_dim", 8),
        use_time_aware_pe=model_cfg.get("use_time_aware_pe", True),
    )
