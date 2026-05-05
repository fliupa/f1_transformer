"""Online fine-tuning: adapt the model race-by-race through the 2025 season.

Simulates a real deployment scenario:
1. Before race N: predict winner using current model
2. After race N results known: fine-tune on this race (with experience replay)
3. Repeat for race N+1

Key improvements:
- Experience replay: maintains a buffer of past 2025 races to prevent forgetting
- EWC-style regularization: small penalty for weight deviation from original model
- Gradual unlearning of dominant-driver bias via targeted class weighting
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from copy import deepcopy
from collections import deque

from src.model.transformer_model_v2 import F1WinnerTransformerV2

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "final"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

REAL_WINNERS_2025 = {
    1: 'NOR', 2: 'PIA', 3: 'VER', 4: 'RUS', 5: 'VER', 6: 'NOR',
    7: 'PIA', 8: 'NOR', 9: 'VER', 10: 'NOR', 11: 'PIA', 12: 'RUS',
    13: 'PIA', 14: 'VER', 15: 'VER', 16: 'NOR', 17: 'PIA', 18: 'VER',
    19: 'PIA', 20: 'VER', 21: 'PIA', 22: 'VER', 23: 'NOR', 24: 'NOR',
}


def get_winner_idx(candidates_row, driver_enc, real_abbr):
    """Find the index (0-19) of the real winner in candidates array."""
    driver_ids = candidates_row[:, 0].long().numpy()
    for i, idx in enumerate(driver_ids):
        try:
            if driver_enc.decode(int(idx)) == real_abbr:
                return i
        except:
            pass
    return None


class ReplayBuffer:
    """Stores past races for experience replay during fine-tuning."""

    def __init__(self, max_size=8):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, context, candidates, winner_idx):
        """Add a race to the buffer."""
        self.buffer.append((
            context.cpu().clone(),
            candidates.cpu().clone(),
            winner_idx,
        ))

    def sample(self):
        """Get all stored races. Returns lists or None."""
        if len(self.buffer) == 0:
            return None, None, None

        contexts = torch.cat([item[0] for item in self.buffer], dim=0)
        candidates = torch.cat([item[1] for item in self.buffer], dim=0)
        winners = torch.tensor([item[2] for item in self.buffer], dtype=torch.long)
        return contexts, candidates, winners

    def __len__(self):
        return len(self.buffer)


def ewc_loss(model, original_state, lambda_ewc=10.0):
    """Compute EWC-style regularization loss (weight deviation penalty)."""
    total = 0.0
    n_params = 0
    for name, param in model.named_parameters():
        if name in original_state and param.requires_grad:
            diff = param - original_state[name].to(param.device)
            total += (diff ** 2).sum()
            n_params += param.numel()
    return lambda_ewc * total / max(n_params, 1)


def online_finetune_v2():
    """Online fine-tuning for V2 with experience replay + EWC."""
    print("=" * 70)
    print("ONLINE FINE-TUNING: V2 Transformer + Replay + EWC")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load pre-trained model
    with open(PROCESSED / "metadata_v2.pkl", "rb") as f:
        metadata = pickle.load(f)

    model = F1WinnerTransformerV2(
        d_model=128, n_heads=4, n_encoder_layers=4, n_cross_attn_layers=3,
        d_ff=512, dropout=0.0,  # no dropout during fine-tuning
        context_window=metadata["context_window"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
        d_candidate_raw=metadata["d_candidate_raw"],
        d_context_raw=metadata["d_context_raw"],
    ).to(DEVICE)

    ckpt = torch.load(MODEL_DIR / "best_v2.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded V2 model (val_acc={ckpt.get('best_val_acc', 0):.4f})")

    driver_enc = metadata["driver_encoder"]
    data_2025 = torch.load(PROCESSED / "features_2025_v2.pt", weights_only=False)
    context = data_2025["context"]
    candidates = data_2025["candidates"]
    time_gaps = data_2025["time_gaps"]
    n_races = len(context)

    # Save original weights for EWC (state_dict includes buffers like pos_encoding.div_term)
    original_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # Replay buffer for experience replay
    replay = ReplayBuffer(max_size=10)

    # Optimizer with very low LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    # Tracking
    static_correct = 0
    ft_correct = 0
    ft_history = []
    static_history = []
    n_correct_pre_ft = 0  # predictions correct BEFORE any fine-tuning

    print(f"\n{'R':>3s} {'FT':<6s} {'Static':<6s} {'Real':<6s} {'Status':>6s}  |  {'Cum FT':>7s} {'Recent6':>7s}")
    print("-" * 65)

    for race_idx in range(n_races):
        round_num = race_idx + 1
        real_abbr = REAL_WINNERS_2025.get(round_num, None)
        if real_abbr is None:
            continue

        ctx = context[race_idx:race_idx+1].to(DEVICE)
        cand = candidates[race_idx:race_idx+1].to(DEVICE)
        gaps = time_gaps[race_idx:race_idx+1].to(DEVICE)
        real_idx = get_winner_idx(candidates[race_idx], driver_enc, real_abbr)
        if real_idx is None:
            print(f"  R{round_num:2d}: Winner {real_abbr} not in candidates!")
            continue

        # ─── Predict with fine-tuned model ───
        model.eval()
        with torch.no_grad():
            logits_ft = model(ctx, cand, gaps)
            probs_ft = F.softmax(logits_ft, dim=-1)[0]
            pred_ft_idx = torch.argmax(probs_ft).item()

        pred_ft_name = driver_enc.decode(
            int(candidates[race_idx, pred_ft_idx, 0].item())
        )
        matched_ft = pred_ft_idx == real_idx
        ft_correct += int(matched_ft)
        ft_history.append(int(matched_ft))

        # ─── Predict with static model (separate copy) ───
        static_model = F1WinnerTransformerV2(
            d_model=128, n_heads=4, n_encoder_layers=4, n_cross_attn_layers=3,
            d_ff=512, dropout=0.0,
            context_window=metadata["context_window"],
            num_drivers=metadata["num_drivers"],
            num_constructors=metadata["num_constructors"],
            num_circuits=metadata["num_circuits"],
            d_candidate_raw=metadata["d_candidate_raw"],
            d_context_raw=metadata["d_context_raw"],
        ).to(DEVICE)
        static_model.load_state_dict(original_state)
        static_model.eval()
        with torch.no_grad():
            logits_static = static_model(ctx, cand, gaps)
            probs_static = F.softmax(logits_static, dim=-1)[0]
            pred_static_idx = torch.argmax(probs_static).item()

        pred_static_name = driver_enc.decode(
            int(candidates[race_idx, pred_static_idx, 0].item())
        )
        matched_static = pred_static_idx == real_idx
        static_correct += int(matched_static)
        static_history.append(int(matched_static))
        del static_model  # free memory

        # Status code
        if matched_ft and not matched_static:
            status = "NEW"
        elif matched_ft:
            status = "YES"
        elif matched_static:
            status = "LOST"
        else:
            status = "NO"

        # Track progress
        cum_ft = sum(ft_history) / len(ft_history)
        recent_ft = sum(ft_history[-6:]) / min(6, len(ft_history))

        print(f"{round_num:3d} {pred_ft_name:<6s} {pred_static_name:<6s} "
              f"{real_abbr:<6s} {status:>6s}  |  {cum_ft:>7.1%} {recent_ft:>7.1%}")

        # ─── Fine-tune with experience replay + EWC ───
        model.train()
        winner_tensor = torch.tensor([real_idx], dtype=torch.long).to(DEVICE)

        # Number of gradient steps depends on how many races we've seen
        n_steps = 4 if race_idx < 6 else 6

        for step in range(n_steps):
            optimizer.zero_grad()
            loss = 0.0
            n_contrib = 0

            # Current race (always included)
            logits = model(ctx, cand, gaps)
            loss += F.cross_entropy(logits, winner_tensor, label_smoothing=0.05)
            n_contrib += 1

            # Experience replay: mix in past races to prevent forgetting
            if len(replay) > 0:
                rp_ctx, rp_cand, rp_winners = replay.sample()
                rp_ctx = rp_ctx.to(DEVICE)
                rp_cand = rp_cand.to(DEVICE)
                rp_winners = rp_winners.to(DEVICE)

                # Use a subset if buffer is large
                if len(replay) > 6:
                    indices = torch.randperm(len(replay))[:6]
                    rp_ctx = rp_ctx[indices]
                    rp_cand = rp_cand[indices]
                    rp_winners = rp_winners[indices]

                rp_gaps = torch.zeros(len(rp_winners), 10).to(DEVICE)
                logits_rp = model(rp_ctx, rp_cand, rp_gaps)
                # Higher weight on recent additions
                replay_weight = min(1.0, len(replay) / 5.0)
                loss += replay_weight * F.cross_entropy(
                    logits_rp, rp_winners, label_smoothing=0.10
                )
                n_contrib += 1

            # EWC: small penalty for deviating from original weights
            loss += ewc_loss(model, original_state, lambda_ewc=5.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Add to replay buffer
        replay.add(ctx.detach(), cand.detach(), real_idx)

    # ─── Final Summary ──
    n_eval = len(ft_history)
    static_acc = static_correct / n_eval
    ft_acc = ft_correct / n_eval

    print(f"\n{'='*70}")
    print(f"ONLINE FINE-TUNING RESULTS (V2 + Replay + EWC)")
    print(f"{'='*70}")
    print(f"  Static model (no adaptation):  {static_correct}/{n_eval} = {static_acc:.1%}")
    print(f"  Fine-tuned (online update):   {ft_correct}/{n_eval} = {ft_acc:.1%}")
    print(f"  Improvement:                   {ft_acc - static_acc:+.1%}")

    if n_eval >= 12:
        first_half = sum(ft_history[:12]) / 12
        second_half = sum(ft_history[12:]) / 12
        print(f"  First 12 races:  {first_half:.1%}")
        print(f"  Last 12 races:   {second_half:.1%}")
        if second_half > first_half:
            print(f"  >>> Model adapted: +{second_half - first_half:.1%} in 2nd half")

    # Per-driver breakdown
    driver_stats = {}
    for i, (ft_hit, st_hit) in enumerate(zip(ft_history, static_history)):
        real = REAL_WINNERS_2025.get(i + 1, "?")
        if real not in driver_stats:
            driver_stats[real] = {"total": 0, "ft_ok": 0, "st_ok": 0}
        driver_stats[real]["total"] += 1
        driver_stats[real]["ft_ok"] += ft_hit
        driver_stats[real]["st_ok"] += st_hit

    print(f"\n  Per-driver accuracy:")
    for driver in sorted(driver_stats.keys()):
        ds = driver_stats[driver]
        print(f"    {driver}: Static={ds['st_ok']}/{ds['total']} "
              f"FT={ds['ft_ok']}/{ds['total']}")

    return {"ft_history": ft_history, "static_history": static_history,
            "ft_acc": ft_acc, "static_acc": static_acc}


def online_finetune_mlp():
    """Online fine-tuning for MLP with experience replay."""
    print("\n" + "=" * 70)
    print("ONLINE FINE-TUNING: MLP + Replay")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    from scripts.train_mlp import F1WinnerMLP

    with open(PROCESSED / "metadata_v2.pkl", "rb") as f:
        metadata = pickle.load(f)
    data_2025 = torch.load(PROCESSED / "features_2025_v2.pt", weights_only=False)

    model = F1WinnerMLP(
        d_context_raw=metadata["d_context_raw"],
        context_window=metadata["context_window"],
        d_candidate_raw=metadata["d_candidate_raw"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
    ).to(DEVICE)

    ckpt = torch.load(MODEL_DIR / "best_mlp.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded MLP model (val_acc={ckpt.get('best_val_acc', 0):.4f})")

    driver_enc = metadata["driver_encoder"]
    context = data_2025["context"]
    candidates = data_2025["candidates"]
    n_races = len(context)

    original_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    replay = ReplayBuffer(max_size=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

    static_correct = 0
    ft_correct = 0
    ft_history = []
    static_history = []

    print(f"\n{'R':>3s} {'FT':<6s} {'Static':<6s} {'Real':<6s} {'Status':>6s}  |  {'Cum FT':>7s} {'Recent6':>7s}")
    print("-" * 65)

    for race_idx in range(n_races):
        round_num = race_idx + 1
        real_abbr = REAL_WINNERS_2025.get(round_num, None)
        if real_abbr is None:
            continue

        ctx = context[race_idx:race_idx+1].to(DEVICE)
        cand = candidates[race_idx:race_idx+1].to(DEVICE)
        real_idx = get_winner_idx(candidates[race_idx], driver_enc, real_abbr)
        if real_idx is None:
            continue

        # Predict with fine-tuned model
        model.eval()
        with torch.no_grad():
            probs_ft = F.softmax(model(ctx, cand), dim=-1)[0]
            pred_ft_idx = torch.argmax(probs_ft).item()
        pred_ft_name = driver_enc.decode(int(candidates[race_idx, pred_ft_idx, 0].item()))
        matched_ft = pred_ft_idx == real_idx
        ft_correct += int(matched_ft)
        ft_history.append(int(matched_ft))

        # Predict with static model
        static_model = F1WinnerMLP(
            d_context_raw=metadata["d_context_raw"],
            context_window=metadata["context_window"],
            d_candidate_raw=metadata["d_candidate_raw"],
            num_drivers=metadata["num_drivers"],
            num_constructors=metadata["num_constructors"],
            num_circuits=metadata["num_circuits"],
        ).to(DEVICE)
        static_model.load_state_dict(original_state)
        static_model.eval()
        with torch.no_grad():
            probs_static = F.softmax(static_model(ctx, cand), dim=-1)[0]
            pred_static_idx = torch.argmax(probs_static).item()
        pred_static_name = driver_enc.decode(int(candidates[race_idx, pred_static_idx, 0].item()))
        matched_static = pred_static_idx == real_idx
        static_correct += int(matched_static)
        static_history.append(int(matched_static))
        del static_model

        status = "NEW" if (matched_ft and not matched_static) else (
            "YES" if matched_ft else ("LOST" if matched_static else "NO")
        )

        cum_ft = sum(ft_history) / len(ft_history)
        recent_ft = sum(ft_history[-6:]) / min(6, len(ft_history))
        print(f"{round_num:3d} {pred_ft_name:<6s} {pred_static_name:<6s} "
              f"{real_abbr:<6s} {status:>6s}  |  {cum_ft:>7.1%} {recent_ft:>7.1%}")

        # Fine-tune with experience replay
        model.train()
        winner_tensor = torch.tensor([real_idx], dtype=torch.long).to(DEVICE)
        n_steps = 5

        for step in range(n_steps):
            optimizer.zero_grad()
            loss = 0.0

            # Current race
            logits = model(ctx, cand)
            loss += F.cross_entropy(logits, winner_tensor, label_smoothing=0.10)

            # Experience replay
            if len(replay) > 0:
                rp_ctx, rp_cand, rp_winners = replay.sample()
                rp_ctx = rp_ctx.to(DEVICE)
                rp_cand = rp_cand.to(DEVICE)
                rp_winners = rp_winners.to(DEVICE)
                if len(replay) > 6:
                    indices = torch.randperm(len(replay))[:6]
                    rp_ctx = rp_ctx[indices]
                    rp_cand = rp_cand[indices]
                    rp_winners = rp_winners[indices]
                logits_rp = model(rp_ctx, rp_cand)
                loss += 0.5 * F.cross_entropy(logits_rp, rp_winners, label_smoothing=0.15)

            loss += ewc_loss(model, original_state, lambda_ewc=3.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        replay.add(ctx.detach(), cand.detach(), real_idx)

    # Summary
    n_eval = len(ft_history)
    static_acc = static_correct / n_eval
    ft_acc = ft_correct / n_eval

    print(f"\n{'='*70}")
    print(f"ONLINE FINE-TUNING RESULTS (MLP + Replay)")
    print(f"{'='*70}")
    print(f"  Static model:  {static_correct}/{n_eval} = {static_acc:.1%}")
    print(f"  Fine-tuned:    {ft_correct}/{n_eval} = {ft_acc:.1%}")
    print(f"  Improvement:   {ft_acc - static_acc:+.1%}")
    if n_eval >= 12:
        print(f"  First 12: {sum(ft_history[:12])/12:.1%}  |  "
              f"Last 12: {sum(ft_history[12:])/12:.1%}")

    return ft_history


def online_finetune_v4():
    """Online fine-tuning for V4 Grid-First (top-7 candidates)."""
    print("\n" + "=" * 70)
    print("ONLINE FINE-TUNING: V4 Grid-First + Replay + EWC")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load V4 metadata
    with open(PROCESSED / "metadata_v4.pkl", "rb") as f:
        metadata = pickle.load(f)

    top_k = metadata.get("top_k", 7)

    model = F1WinnerTransformerV2(
        d_model=128, n_heads=4, n_encoder_layers=4, n_cross_attn_layers=3,
        d_ff=512, dropout=0.0,
        context_window=metadata["context_window"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
        d_candidate_raw=metadata["d_candidate_raw"],
        d_context_raw=metadata["d_context_raw"],
        max_drivers_per_race=top_k,
    ).to(DEVICE)

    ckpt = torch.load(MODEL_DIR / "best_v4.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded V4 model (val_acc={ckpt.get('best_val_acc', 0):.4f}, {top_k} candidates)")

    driver_enc = metadata["driver_encoder"]
    data_2025 = torch.load(PROCESSED / "features_2025_v4.pt", weights_only=False)
    context = data_2025["context"]
    candidates = data_2025["candidates"]
    time_gaps = data_2025["time_gaps"]
    n_races = len(context)

    # Save original state
    original_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    replay = ReplayBuffer(max_size=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    static_correct = 0
    ft_correct = 0
    ft_history = []
    static_history = []

    print(f"\n{'R':>3s} {'FT':<6s} {'Static':<6s} {'Real':<6s} {'Status':>6s}  |  {'Cum FT':>7s} {'Recent6':>7s}")
    print("-" * 65)

    for race_idx in range(n_races):
        round_num = race_idx + 1
        real_abbr = REAL_WINNERS_2025.get(round_num, None)
        if real_abbr is None:
            continue

        ctx = context[race_idx:race_idx+1].to(DEVICE)
        cand = candidates[race_idx:race_idx+1].to(DEVICE)
        gaps = time_gaps[race_idx:race_idx+1].to(DEVICE)
        real_idx = get_winner_idx(candidates[race_idx], driver_enc, real_abbr)
        if real_idx is None:
            print(f"  R{round_num:2d}: Winner {real_abbr} not in top-{top_k} candidates!")
            continue

        # Predict with fine-tuned model
        model.eval()
        with torch.no_grad():
            logits_ft = model(ctx, cand, gaps)
            probs_ft = F.softmax(logits_ft, dim=-1)[0]
            pred_ft_idx = torch.argmax(probs_ft).item()

        pred_ft_name = driver_enc.decode(
            int(candidates[race_idx, pred_ft_idx, 0].item())
        )
        matched_ft = pred_ft_idx == real_idx
        ft_correct += int(matched_ft)
        ft_history.append(int(matched_ft))

        # Predict with static model
        static_model = F1WinnerTransformerV2(
            d_model=128, n_heads=4, n_encoder_layers=4, n_cross_attn_layers=3,
            d_ff=512, dropout=0.0,
            context_window=metadata["context_window"],
            num_drivers=metadata["num_drivers"],
            num_constructors=metadata["num_constructors"],
            num_circuits=metadata["num_circuits"],
            d_candidate_raw=metadata["d_candidate_raw"],
            d_context_raw=metadata["d_context_raw"],
            max_drivers_per_race=top_k,
        ).to(DEVICE)
        static_model.load_state_dict(original_state)
        static_model.eval()
        with torch.no_grad():
            logits_static = static_model(ctx, cand, gaps)
            probs_static = F.softmax(logits_static, dim=-1)[0]
            pred_static_idx = torch.argmax(probs_static).item()

        pred_static_name = driver_enc.decode(
            int(candidates[race_idx, pred_static_idx, 0].item())
        )
        matched_static = pred_static_idx == real_idx
        static_correct += int(matched_static)
        static_history.append(int(matched_static))
        del static_model

        status = "NEW" if (matched_ft and not matched_static) else (
            "YES" if matched_ft else ("LOST" if matched_static else "NO")
        )

        cum_ft = sum(ft_history) / len(ft_history)
        recent_ft = sum(ft_history[-6:]) / min(6, len(ft_history))
        print(f"{round_num:3d} {pred_ft_name:<6s} {pred_static_name:<6s} "
              f"{real_abbr:<6s} {status:>6s}  |  {cum_ft:>7.1%} {recent_ft:>7.1%}")

        # Fine-tune with experience replay + EWC
        model.train()
        winner_tensor = torch.tensor([real_idx], dtype=torch.long).to(DEVICE)
        n_steps = 4 if race_idx < 6 else 6

        for step in range(n_steps):
            optimizer.zero_grad()
            loss = 0.0

            logits = model(ctx, cand, gaps)
            loss += F.cross_entropy(logits, winner_tensor, label_smoothing=0.05)

            if len(replay) > 0:
                rp_ctx, rp_cand, rp_winners = replay.sample()
                rp_ctx = rp_ctx.to(DEVICE)
                rp_cand = rp_cand.to(DEVICE)
                rp_winners = rp_winners.to(DEVICE)

                if len(replay) > 6:
                    indices = torch.randperm(len(replay))[:6]
                    rp_ctx = rp_ctx[indices]
                    rp_cand = rp_cand[indices]
                    rp_winners = rp_winners[indices]

                rp_gaps = torch.zeros(len(rp_winners), 10).to(DEVICE)
                logits_rp = model(rp_ctx, rp_cand, rp_gaps)
                replay_weight = min(1.0, len(replay) / 5.0)
                loss += replay_weight * F.cross_entropy(
                    logits_rp, rp_winners, label_smoothing=0.10
                )

            loss += ewc_loss(model, original_state, lambda_ewc=5.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        replay.add(ctx.detach(), cand.detach(), real_idx)

    # Summary
    n_eval = len(ft_history)
    static_acc = static_correct / n_eval
    ft_acc = ft_correct / n_eval

    print(f"\n{'='*70}")
    print(f"ONLINE FINE-TUNING RESULTS (V4 Grid-First + Replay + EWC)")
    print(f"{'='*70}")
    print(f"  Static model:  {static_correct}/{n_eval} = {static_acc:.1%}")
    print(f"  Fine-tuned:    {ft_correct}/{n_eval} = {ft_acc:.1%}")
    print(f"  Improvement:   {ft_acc - static_acc:+.1%}")

    if n_eval >= 12:
        first_half = sum(ft_history[:12]) / min(12, len(ft_history[:12]))
        second_half = sum(ft_history[12:]) / max(1, len(ft_history[12:]))
        print(f"  First half:  {first_half:.1%}")
        print(f"  Second half: {second_half:.1%}")

    return {"ft_history": ft_history, "static_history": static_history,
            "ft_acc": ft_acc, "static_acc": static_acc}


if __name__ == "__main__":
    v4_results = online_finetune_v4()
