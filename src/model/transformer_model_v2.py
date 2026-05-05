"""F1 Winner Prediction Transformer V2.

Improved architecture over V1:
- Smaller (d_model=128) to reduce overfitting on ~192 training seqs
- Candidate self-attention: drivers "compete" before attending to context
- Gated cross-attention: learnable gate controls context vs self info
- Context residual: mean-pooled context encoding augments driver representations
- Higher dropout (0.25) and label smoothing (0.10)
- Pre-LN throughout for training stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .positional import TimeAwarePositionalEncoding


class GatedCrossAttentionBlock(nn.Module):
    """Cross-attention with a learned gate controlling context vs self information.

    gate = sigmoid(W * [self_repr, context_attended])
    output = gate * context_attended + (1 - gate) * self_repr
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Gate network: learns how much to trust context vs self
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Stochastic depth probability
        self.drop_path_prob = dropout * 0.4

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (batch, seq_q, d_model) - driver representations
            context: (batch, seq_kv, d_model) - encoded race history

        Returns:
            (batch, seq_q, d_model) - updated driver representations
        """
        # Cross-attention: drivers attend to race history
        attn_out, _ = self.cross_attn(queries, context, context)

        # Learned gate: how much context vs self?
        gate_input = torch.cat([queries, attn_out], dim=-1)
        gate = self.gate(gate_input)  # (batch, seq_q, 1)
        gated = gate * attn_out + (1 - gate) * queries

        x = self.norm1(queries + gated)

        # Feed-forward with stochastic depth
        ff_out = self.ff(x)
        if self.training:
            # Stochastic depth during training
            keep_prob = 1.0 - self.drop_path_prob
            if torch.rand(1).item() > keep_prob:
                ff_out = torch.zeros_like(ff_out)
            else:
                ff_out = ff_out / keep_prob

        x = self.norm2(x + ff_out)
        return x


class F1WinnerTransformerV2(nn.Module):
    """Transformer V2: compact, regularized architecture for F1 winner prediction.

    Key improvements over V1:
    1. Candidate self-attention before cross-attention
    2. Gated cross-attention with stochastic depth
    3. Context residual connection to prediction head
    4. Smaller d_model (128) with higher dropout (0.25)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        n_cross_attn_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.25,
        context_window: int = 10,
        num_drivers: int = 80,
        num_constructors: int = 20,
        num_circuits: int = 45,
        driver_embed_dim: int = 16,
        constructor_embed_dim: int = 8,
        circuit_embed_dim: int = 8,
        d_candidate_raw: int = 36,
        d_context_raw: int = 110,
        use_time_aware_pe: bool = True,
        max_drivers_per_race: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_window = context_window
        self.max_drivers_per_race = max_drivers_per_race

        # ─── Embedding Layers ───────────────────────────────────────
        self.driver_embed = nn.Embedding(num_drivers + 1, driver_embed_dim, padding_idx=0)
        self.constructor_embed = nn.Embedding(num_constructors + 1, constructor_embed_dim, padding_idx=0)
        self.circuit_embed = nn.Embedding(num_circuits + 1, circuit_embed_dim, padding_idx=0)

        self._embed_total = driver_embed_dim + constructor_embed_dim + circuit_embed_dim
        self._candidate_numeric_feats = d_candidate_raw - 3

        # ─── Input Projections ──────────────────────────────────────
        self.candidate_proj = nn.Sequential(
            nn.Linear(self._embed_total + self._candidate_numeric_feats, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.context_proj = nn.Sequential(
            nn.Linear(d_context_raw, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ─── Positional Encoding ─────────────────────────────────────
        if use_time_aware_pe:
            self.pos_encoding = TimeAwarePositionalEncoding(d_model, dropout=dropout)
        else:
            self.pos_encoding = None

        # ─── Candidate Self-Attention ────────────────────────────────
        # Drivers "compete" with each other before looking at context
        self.candidate_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.candidate_self_norm = nn.LayerNorm(d_model)

        # ─── Race History Transformer Encoder ─────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm
        )
        self.race_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # ─── Gated Cross-Attention Layers ─────────────────────────────
        self.cross_attn_layers = nn.ModuleList([
            GatedCrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_cross_attn_layers)
        ])

        # ─── Context Residual Projection ─────────────────────────────
        # Mean-pool context encoding -> concat with driver repr before prediction
        self.context_pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
        )

        # ─── Prediction Head ─────────────────────────────────────────
        # Input: driver_repr (d_model) + context_pooled (d_model//2)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.driver_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.constructor_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.circuit_embed.weight, mean=0, std=0.02)
        with torch.no_grad():
            self.driver_embed.weight[0] = 0
            self.constructor_embed.weight[0] = 0
            self.circuit_embed.weight[0] = 0

    def _embed_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Process candidate driver features."""
        driver_idx = candidates[:, :, 0].long()
        constructor_idx = candidates[:, :, 1].long()
        circuit_idx = candidates[:, :, 2].long()

        d_emb = self.driver_embed(driver_idx)
        c_emb = self.constructor_embed(constructor_idx)
        circ_emb = self.circuit_embed(circuit_idx)
        numeric = candidates[:, :, 3:]

        combined = torch.cat([d_emb, c_emb, circ_emb, numeric], dim=-1)
        return self.candidate_proj(combined)

    def _embed_context(self, context: torch.Tensor) -> torch.Tensor:
        """Process context race summary features."""
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
            candidates: (batch, max_drivers, d_candidate_raw)
            time_gaps: (batch, context_window)

        Returns:
            logits: (batch, max_drivers)
        """
        batch_size = candidates.shape[0]

        # 1. Embed candidates and context
        driver_feats = self._embed_candidates(candidates)  # (B, 20, d_model)
        context_feats = self._embed_context(context)       # (B, N, d_model)

        # 2. Time-aware positional encoding on context
        if self.pos_encoding is not None and time_gaps is not None:
            context_feats = context_feats + self.pos_encoding(time_gaps)

        # 3. Candidate self-attention: drivers compete with each other
        driver_feats_attn, _ = self.candidate_self_attn(driver_feats, driver_feats, driver_feats)
        driver_feats = self.candidate_self_norm(driver_feats + driver_feats_attn)

        # 4. Encode race history
        context_encoded = self.race_encoder(context_feats)  # (B, N, d_model)

        # 5. Gated cross-attention: drivers attend to race history
        driver_repr = driver_feats
        for cross_attn in self.cross_attn_layers:
            driver_repr = cross_attn(driver_repr, context_encoded)

        # 6. Context residual: mean-pool context and concat with driver repr
        context_pooled = context_encoded.mean(dim=1)                      # (B, d_model)
        context_pooled = self.context_pool_proj(context_pooled)            # (B, d_model//2)
        context_pooled = context_pooled.unsqueeze(1).expand(-1, self.max_drivers_per_race, -1)  # (B, 20, d_model//2)

        driver_with_context = torch.cat([driver_repr, context_pooled], dim=-1)  # (B, 20, d_model + d_model//2)

        # 7. Prediction head
        logits = self.pred_head(driver_with_context).squeeze(-1)  # (B, 20)

        return logits

    def predict_proba(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        time_gaps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict win probabilities for each driver."""
        logits = self.forward(context, candidates, time_gaps)
        return F.softmax(logits, dim=-1)

    def predict_winner(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        time_gaps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the winner and confidence."""
        probs = self.predict_proba(context, candidates, time_gaps)
        winner_idx = torch.argmax(probs, dim=-1)
        confidence = probs.gather(1, winner_idx.unsqueeze(-1)).squeeze(-1)
        return winner_idx, confidence
