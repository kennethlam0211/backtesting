"""
TradingNetworkLite — Two-Actor Multi-Timeframe Trading Network
================================================================
See architecture.txt for full design documentation.

Two actors: Direction (3-way: no-go/long/short), Portfolio Manager (3-way: clear/long/short)
FiLM-conditioned shared encoders, N-BEATS movement decomposition,
CrossAttentionHyperConnection, curriculum loss weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict
import numpy as np


def scale_gradient(x, scale=0.3):
    """Scale gradient in backward pass without affecting forward pass.
    Forward: returns x unchanged. Backward: gradient *= scale.
    Used to reduce value head gradient interference on shared backbone."""
    return x * scale + x.detach() * (1 - scale)


# =============================================================================
# STREAMLINED TRADING NETWORK v4 - DISCRETE ACTIONS (3: clear/buy/sell)
# =============================================================================
# Key features:
# 1. DISCRETE action space: 0=clear, 1=buy, 2=sell
# 2. Reduced hidden dims for faster training
# 3. PReLU for negative momentum signals
# 4. New encoders: pfolio_info, nts, n_px_1
# =============================================================================


class LiteTCNBlock(nn.Module):
    """Lightweight TCN block with PReLU"""
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.ln = nn.LayerNorm(channels)
        self.act = nn.PReLU(num_parameters=channels, init=0.25)

    def forward(self, x):  # x: (B, C, T)
        residual = x
        x = self.conv(x)
        x = self.ln(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        return x + residual


class NBEATSBlock(nn.Module):
    """Single N-BEATS block: FC stack → basis coefficients → backcast + forecast."""
    def __init__(self, input_len: int, hidden_dim: int, n_basis: int, basis_type: str = 'generic'):
        super().__init__()
        self.input_len = input_len
        self.n_basis = n_basis
        self.basis_type = basis_type

        # FC stack (4 layers as in original N-BEATS)
        self.fc = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.PReLU(num_parameters=hidden_dim, init=0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(num_parameters=hidden_dim, init=0.25),
        )

        # Basis coefficient projections
        self.theta_b = nn.Linear(hidden_dim, n_basis)  # backcast coefficients
        self.theta_f = nn.Linear(hidden_dim, n_basis)  # forecast/feature coefficients

        if basis_type == 'trend':
            # Polynomial basis (degree = n_basis)
            t = torch.linspace(0, 1, input_len).unsqueeze(0)  # (1, T)
            powers = torch.arange(n_basis).unsqueeze(1).float()  # (n_basis, 1)
            self.register_buffer('basis', t ** powers)  # (n_basis, T)
        elif basis_type == 'seasonal':
            # Fourier basis
            t = torch.linspace(0, 2 * 3.14159, input_len).unsqueeze(0)  # (1, T)
            freqs = torch.arange(1, n_basis // 2 + 1).unsqueeze(1).float()
            sin_basis = torch.sin(freqs * t)  # (n_basis//2, T)
            cos_basis = torch.cos(freqs * t)  # (n_basis//2, T)
            self.register_buffer('basis', torch.cat([sin_basis, cos_basis], dim=0)[:n_basis])
        else:
            # Generic: learned basis
            self.basis_matrix = nn.Linear(n_basis, input_len, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T) — single channel input
        Returns:
            backcast: (B, T) — reconstruction to subtract (residual learning)
            features: (B, n_basis) — basis coefficients as features
        """
        h = self.fc(x)  # (B, hidden_dim)

        theta_b = self.theta_b(h)  # (B, n_basis)
        theta_f = self.theta_f(h)  # (B, n_basis)

        if self.basis_type in ('trend', 'seasonal'):
            backcast = torch.matmul(theta_b, self.basis)  # (B, T)
        else:
            backcast = self.basis_matrix(theta_b)  # (B, T)

        return backcast, theta_f  # theta_f = extracted features


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: γ·x + β, conditioned on frequency embedding.
    One FiLM per feature type, shared γ/β generators across frequencies."""
    def __init__(self, feature_dim, embed_dim=8):
        super().__init__()
        self.gamma = nn.Linear(embed_dim, feature_dim)
        self.beta = nn.Linear(embed_dim, feature_dim)

    def forward(self, x, freq_embed):
        # x: (B, T, F) or (B, F), freq_embed: (embed_dim,)
        gamma = self.gamma(freq_embed)  # (F,)
        beta = self.beta(freq_embed)    # (F,)
        return gamma * x + beta


class BasisFiLM(nn.Module):
    """FiLM on N-BEATS basis coefficients — adapts trend/seasonal interpretation per freq.
    42-day polynomial trend ≠ 84-min polynomial trend. ~224 extra params total."""
    def __init__(self, embed_dim=8, n_trend=4, n_seasonal=8):
        super().__init__()
        self.gamma_t = nn.Linear(embed_dim, n_trend)
        self.beta_t = nn.Linear(embed_dim, n_trend)
        self.gamma_s = nn.Linear(embed_dim, n_seasonal)
        self.beta_s = nn.Linear(embed_dim, n_seasonal)

    def forward(self, trend_coefs, seasonal_coefs, freq_embed):
        # trend_coefs: (B, 4), seasonal_coefs: (B, 8), freq_embed: (embed_dim,)
        gamma_t = self.gamma_t(freq_embed)
        beta_t = self.beta_t(freq_embed)
        gamma_s = self.gamma_s(freq_embed)
        beta_s = self.beta_s(freq_embed)
        return gamma_t * trend_coefs + beta_t, gamma_s * seasonal_coefs + beta_s


class MultiSourceHyperConnection(nn.Module):
    """HyperConnection with multiple gated sources (e.g. 720→1m direct skip + 15→1m cascade).
    output = LayerNorm(lower_tf + Σ(gate_i * source_i))"""
    def __init__(self, dim, num_sources):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
            for _ in range(num_sources)
        ])
        self.ln = nn.LayerNorm(dim)

    def forward(self, lower_tf, sources):
        # lower_tf: (B, dim), sources: list of (B, dim)
        out = lower_tf
        for gate, src in zip(self.gates, sources):
            out = out + gate(src) * src
        return self.ln(out)


class CrossAttentionHyperConnection(nn.Module):
    """Content-dependent cross-attention: lower TF attends to higher TF sources.
    Unlike static gated fusion, attention weights are dynamically computed based
    on current content — 'what matters right now' instead of fixed modulation.

    lower_tf (query) attends to stacked sources (keys/values) via multi-head attention.
    Residual connection + LayerNorm for stable training."""
    def __init__(self, dim, num_sources, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, lower_tf, sources):
        # lower_tf: (B, dim), sources: list of (B, dim)
        B = lower_tf.size(0)

        # Query from lower TF
        q = self.q_proj(lower_tf).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Stack sources for batched K/V projection
        src = torch.stack(sources, dim=1)  # (B, num_sources, dim)
        k = self.k_proj(src).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Multi-head attention: (B, heads, 1, head_dim) @ (B, heads, head_dim, num_sources)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1)
        out = self.out_proj(out)

        return self.ln(lower_tf + out)


class SessionSummaryEncoder(nn.Module):
    """Encodes 14 session metrics → 8 dims.
    [ttl_pnl, mdd, winrate, mu_norm, sigma_norm, sharpe, subscription_days,
     pass_exam, activation_progress, is_50k, is_100k, is_150k, is_exam_plan, is_no_activation_plan]
    PM only — answers 'how is the whole session going?'"""
    def __init__(self, input_dim=14, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.PReLU(num_parameters=16, init=0.25),
            nn.Linear(16, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):  # x: (B, 15)
        return self.net(x)


class MovementEncoder(nn.Module):
    """Hybrid N-BEATS + LSTM movement encoder with internal BasisFiLM.

    Two parallel paths:
    1. N-BEATS: trend + seasonal decomposition per channel (structured features)
       └─ BasisFiLM adapts basis coefficients per frequency
    2. LSTM: Conv1d → LSTM for temporal memory across the sequence

    Concatenates both outputs for the best of both worlds:
    structured decomposition + sequential context.
    """
    def __init__(self, input_dim=2, hidden_dim=24, output_dim=24, embed_dim=8):
        super().__init__()
        self.input_dim = input_dim

        # ── N-BEATS path ─────────────────────────────────────────────
        self.input_proj = nn.Conv1d(input_dim, input_dim, kernel_size=5, padding=2)
        self.act = nn.PReLU(num_parameters=input_dim, init=0.25)

        n_trend_basis = 4   # polynomial degree 0-3
        n_seasonal_basis = 8  # 4 sin + 4 cos harmonics

        self.trend_blocks = nn.ModuleList([
            NBEATSBlock(84, hidden_dim, n_trend_basis, basis_type='trend')
            for _ in range(input_dim)
        ])
        self.seasonal_blocks = nn.ModuleList([
            NBEATSBlock(84, hidden_dim, n_seasonal_basis, basis_type='seasonal')
            for _ in range(input_dim)
        ])

        nbeats_features = (n_trend_basis + n_seasonal_basis) * input_dim  # 24

        # ── Internal BasisFiLM (adapts coefficients per frequency) ───
        self.basis_film = BasisFiLM(embed_dim=embed_dim, n_trend=n_trend_basis, n_seasonal=n_seasonal_basis)

        # ── LSTM path ────────────────────────────────────────────────
        lstm_conv_dim = 8
        lstm_hidden_dim = 16
        self.lstm_conv = nn.Conv1d(input_dim, lstm_conv_dim, kernel_size=3, padding=1)
        self.lstm_act = nn.PReLU(num_parameters=lstm_conv_dim, init=0.25)
        self.lstm = nn.LSTM(lstm_conv_dim, lstm_hidden_dim, num_layers=1, batch_first=True)
        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim)

        # ── Output: N-BEATS(24) + LSTM(16) = 40 → output_dim ────────
        self.output = nn.Linear(nbeats_features + lstm_hidden_dim, output_dim)

    def forward(self, x, freq_embed=None):  # x: (B, 84, 2), freq_embed: (embed_dim,) or None
        # ── N-BEATS path ─────────────────────────────────────────────
        x_nbeats = self.act(self.input_proj(x.transpose(1, 2))).transpose(1, 2)  # (B, 84, 2)

        all_features = []
        for ch in range(self.input_dim):
            signal = x_nbeats[:, :, ch]  # (B, 84)

            # Trend block
            trend_backcast, trend_coefs = self.trend_blocks[ch](signal)
            residual = signal - trend_backcast  # remove trend

            # Seasonal block on residual
            _, seasonal_coefs = self.seasonal_blocks[ch](residual)

            # BasisFiLM: adapt coefficient interpretation per frequency
            if freq_embed is not None:
                trend_coefs, seasonal_coefs = self.basis_film(trend_coefs, seasonal_coefs, freq_embed)

            all_features.append(trend_coefs)      # (B, 4)
            all_features.append(seasonal_coefs)    # (B, 8)

        nbeats_out = torch.cat(all_features, dim=-1)  # (B, 24)

        # ── LSTM path ────────────────────────────────────────────────
        x_lstm = self.lstm_act(self.lstm_conv(x.transpose(1, 2)))  # (B, 16, 84)
        x_lstm = x_lstm.transpose(1, 2)  # (B, 84, 16)
        _, (h_n, _) = self.lstm(x_lstm)  # h_n: (1, B, 16)
        lstm_out = self.lstm_norm(h_n[-1])  # (B, 16)

        # ── Combine ──────────────────────────────────────────────────
        combined = torch.cat([nbeats_out, lstm_out], dim=-1)  # (B, 40)
        return self.output(combined)


class UDEncoder(nn.Module):
    """Simple 2-layer MLP for UD signals (21 normalized scalars → 16).
    Conv1d on (1,21) is overkill — MLP is simpler and sufficient."""
    def __init__(self, input_dim=21, hidden_dim=16, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(num_parameters=hidden_dim, init=0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):  # x: (B, 21)
        return self.net(x)


class TargetEncoder(nn.Module):
    """Group-aware target encoder: splits 108 features into price targets [0:28]
    and UD targets [28:108], encodes each with dedicated Conv+Pool, then concatenates.

    Price targets (28 cols, std 0.01-0.9): intrady_hl, pre_ohlc, sma, bbands + neg/pos
    UD targets (80 cols, std 0.1-5.0): 4 freqs × 20 sorted UD price levels

    Separate encoders prevent high-variance UD targets from drowning out
    small price-level signals. Total output = 16 + 32 = 48."""
    PRICE_SPLIT = 28  # boundary between price targets and UD targets

    def __init__(self, input_dim=88, output_dim=48):
        super().__init__()
        price_dim = self.PRICE_SPLIT   # 28
        ud_dim = input_dim - price_dim  # 60

        # Price targets: 28 → 16 → 12 → pool → 16
        self.price_conv1 = nn.Conv1d(price_dim, 16, kernel_size=3, padding=1)
        self.price_act1 = nn.PReLU(num_parameters=16, init=0.25)
        self.price_conv2 = nn.Conv1d(16, 12, kernel_size=3, padding=1)
        self.price_act2 = nn.PReLU(num_parameters=12, init=0.25)
        self.price_pool = nn.AdaptiveAvgPool1d(1)
        self.price_max_pool = nn.AdaptiveMaxPool1d(1)
        self.price_out = nn.Linear(12 * 3, 16)

        # UD targets: 60 → 24 → 16 → pool → 32
        self.ud_conv1 = nn.Conv1d(ud_dim, 24, kernel_size=3, padding=1)
        self.ud_act1 = nn.PReLU(num_parameters=24, init=0.25)
        self.ud_conv2 = nn.Conv1d(24, 16, kernel_size=3, padding=1)
        self.ud_act2 = nn.PReLU(num_parameters=16, init=0.25)
        self.ud_pool = nn.AdaptiveAvgPool1d(1)
        self.ud_max_pool = nn.AdaptiveMaxPool1d(1)
        self.ud_out = nn.Linear(16 * 3, 32)

    def forward(self, x):  # x: (B, 20, 88)
        x_price = x[:, :, :self.PRICE_SPLIT].transpose(1, 2)  # (B, 28, 20)
        x_ud = x[:, :, self.PRICE_SPLIT:].transpose(1, 2)     # (B, 60, 20)

        # Price path
        p = self.price_act1(self.price_conv1(x_price))
        p = self.price_act2(self.price_conv2(p))                               # (B, 12, 20)
        p_avg = self.price_pool(p).squeeze(-1)
        p_max = self.price_max_pool(p).squeeze(-1)
        p_min = (-self.price_max_pool(-p)).squeeze(-1)
        price_out = self.price_out(torch.cat([p_avg, p_max, p_min], dim=-1))  # (B, 16)

        # UD path
        u = self.ud_act1(self.ud_conv1(x_ud))
        u = self.ud_act2(self.ud_conv2(u))                                     # (B, 16, 20)
        u_avg = self.ud_pool(u).squeeze(-1)
        u_max = self.ud_max_pool(u).squeeze(-1)
        u_min = (-self.ud_max_pool(-u)).squeeze(-1)
        ud_out = self.ud_out(torch.cat([u_avg, u_max, u_min], dim=-1))  # (B, 32)

        return torch.cat([price_out, ud_out], dim=-1)  # (B, 48)


class VolatilityContinuousEncoder(nn.Module):
    """Conv + Conv + LSTM for continuous volatility features (20_std, 20_atr, volume, vix).
    LSTM preserves temporal ordering — volatility clustering has meaningful sequence
    structure that triple pooling (avg/max/min) destroys. 4 input channels → 16 output dims."""
    def __init__(self, input_dim=4, hidden_dim=12, output_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.act1 = nn.PReLU(num_parameters=hidden_dim, init=0.5)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act2 = nn.PReLU(num_parameters=hidden_dim, init=0.5)
        self.lstm = nn.LSTM(hidden_dim, output_dim, batch_first=True)
        self.ln = nn.LayerNorm(output_dim)
        self.magnitude_head = nn.Linear(output_dim, 1)

    def forward(self, x, return_magnitude=False):  # x: (B, 20, 4)
        x = x.transpose(1, 2)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = x.transpose(1, 2)  # (B, T, hidden_dim) for LSTM
        _, (h_n, _) = self.lstm(x)
        hidden = self.ln(h_n[-1])
        if return_magnitude:
            magnitude = F.hardsigmoid(self.magnitude_head(hidden)).squeeze(-1)  # (B,) [0, 1]
            return hidden, magnitude
        return hidden


class VolatilityEventEncoder(nn.Module):
    """Simple MLP for binary event flags (fomc, nfp, cpi, ppi, gdp).
    No temporal processing needed — flags are binary indicators.
    5 input → 8 output dims."""
    def __init__(self, input_dim=5, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.PReLU(num_parameters=16, init=0.25),
            nn.Linear(16, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):  # x: (B, 5) — last timestep of events
        return self.net(x)


class SessionEncoder(nn.Module):
    """Conv + Conv + Pool"""
    def __init__(self, input_dim=4, hidden_dim=12, output_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.act1 = nn.PReLU(num_parameters=hidden_dim, init=0.25)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act2 = nn.PReLU(num_parameters=hidden_dim, init=0.25)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(hidden_dim * 3, output_dim)  # avg + max + min

    def forward(self, x):  # x: (B, 20, 4)
        x = x.transpose(1, 2)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x_avg = self.avg_pool(x).squeeze(-1)
        x_max = self.max_pool(x).squeeze(-1)
        x_min = (-self.max_pool(-x)).squeeze(-1)
        x = torch.cat([x_avg, x_max, x_min], dim=-1)
        return self.output(x)


class DirectionalEncoder(nn.Module):
    """Encodes directional bias signals (UD_flag + FVG).
    Linear projection → LayerNorm → LSTM for temporal pattern detection in binary/ternary signals."""
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.PReLU(num_parameters=1, init=0.5)
        self.lstm = nn.LSTM(hidden_dim, output_dim, batch_first=True)

    def forward(self, x):  # (B, 20, 2)
        x = self.act(self.ln(self.input_proj(x)))  # (B, 20, 8)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # (B, 8)


class TAEncoder(nn.Module):
    """TCN + LSTM kept (TA needs temporal memory)"""
    def __init__(self, input_dim=6, hidden_dim=12, output_dim=16):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.PReLU(num_parameters=1, init=0.5)  # Use scalar learnable param for (B, T, F) input
        self.tcn = LiteTCNBlock(hidden_dim, dilation=2)
        self.lstm = nn.LSTM(hidden_dim, output_dim, batch_first=True)
        
    def forward(self, x):  # x: (B, 20, 7) or (B, 20, 12)
        x = self.act(self.input_proj(x))  # (B, 20, hidden_dim)
        x = x.transpose(1, 2)  # (B, hidden_dim, 20) for TCN
        x = self.tcn(x)
        x = x.transpose(1, 2)  # (B, 20, hidden_dim) for LSTM
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class GatedFusion(nn.Module):
    """Lightweight gated fusion"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.PReLU(num_parameters=out_dim, init=0.25)
        )
        
    def forward(self, x):
        return self.proj(self.gate(x) * x)


class FreqStem(nn.Module):
    """Per-frequency CNN stem that normalizes raw features before shared encoder.

    Preserves input shape (B, T, F) so downstream shared encoder is unchanged.
    Each frequency gets its own learned temporal preprocessing.
    """
    def __init__(self, input_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size, padding=kernel_size // 2)
        self.act = nn.PReLU(num_parameters=input_dim, init=0.25)
        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size, padding=kernel_size // 2)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):  # x: (B, T, F)
        residual = x
        x = x.transpose(1, 2)          # (B, F, T)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = x.transpose(1, 2)          # (B, T, F)
        x = self.ln(x + residual)       # residual + layernorm
        return x


class InputGate(nn.Module):
    """Learnable per-feature sigmoid gate applied BEFORE FreqStems/encoders.
    Each raw input feature is multiplied by a learned importance weight in [0,1].
    Initialized at 0.5 (neutral) so all features start equally weighted."""
    def __init__(self, num_features):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(num_features))

    @property
    def gate_values(self):
        return torch.sigmoid(self.gate_logits)

    def forward(self, x):
        return x * torch.sigmoid(self.gate_logits)


class HyperConnection(nn.Module):
    """Simplified hyperconnection without manifold basis"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, lower_tf, higher_tf):
        """Fuse higher TF context into lower TF"""
        gated = self.gate(higher_tf) * higher_tf
        return self.ln(lower_tf + gated)


class PortfolioEncoder(nn.Module):
    """Encodes portfolio state (11,) - flat current snapshot

    Input shape: (B, 11) where 11 features are:
        floating_pnl/10, per_contract_pnl/10, step_pnl/10,
        position_usage, rth, nts, n_px_1,
        kelly_f, mdd_buffer, confidence, win_rate
    """
    def __init__(self, input_dim=11, output_dim=16, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.PReLU(num_parameters=16, init=0.25),
            nn.Linear(16, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, pfolio_info):
        # pfolio_info: (B, 11)
        return self.net(pfolio_info)

class DirectionActorHead(nn.Module):
    """3-way direction actor: no-go / long / short."""
    def __init__(self, input_dim=64, hidden_dim=32, n_actions=3, dropout=0.15):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(hidden_dim, init=0.25),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions),  # 3 actions: no-go / long / short
        )
        self.value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(hidden_dim, init=0.25),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, shared_features, value_grad_scale=0.3):
        logits = self.policy(shared_features)       # (B, 3)
        # Clamp logits to prevent softmax collapse → NaN log_prob
        logits = logits.clamp(-10, 10)
        probs = F.softmax(logits, dim=-1)
        value = self.value(scale_gradient(shared_features, value_grad_scale))  # (B, 1)
        return logits, probs, value


class PortfolioTimeSeriesEncoder(nn.Module):
    """Encodes 60-step portfolio history for the portfolio manager."""
    def __init__(self, input_dim=11, hidden_dim=16, output_dim=16):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.act = nn.PReLU(num_parameters=hidden_dim, init=0.25)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, pfolio_ts):  # (B, 60, 11)
        x = pfolio_ts.transpose(1, 2)   # (B, 11, 60)
        x = self.act(self.conv(x))       # (B, 16, 60)
        x = x.transpose(1, 2)           # (B, 60, 16)
        _, (h_n, _) = self.lstm(x)
        return self.output(h_n[-1])      # (B, 16)


class PortfolioManagerHead(nn.Module):
    """Portfolio manager: outputs Beta-distributed confidence [0,1] for position sizing.
    Uses shared features + direction conviction + portfolio context.
    Stop ratio head gets extra direct inputs: magnitude(1) + ta_1m(16) = 17
    for market-structure-aware stop placement."""
    def __init__(self, shared_dim=96, conviction_dim=3, pfolio_snap_dim=16, pfolio_ts_dim=16, session_dim=8,
                 stop_extra_dim=17, hidden_dim=64, dropout=0.15):
        super().__init__()
        total_input = shared_dim + conviction_dim + pfolio_snap_dim + pfolio_ts_dim + session_dim
        stop_input = total_input + stop_extra_dim  # 171 + 57 = 228

        self.policy = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.PReLU(hidden_dim, init=0.25),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.PReLU(32, init=0.25),
            nn.Dropout(dropout),
            nn.Linear(32, 2),  # alpha, beta params for Beta distribution
        )
        self.stop_policy = nn.Sequential(
            nn.Linear(stop_input, hidden_dim),
            nn.PReLU(hidden_dim, init=0.25),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.PReLU(32, init=0.25),
            nn.Dropout(dropout),
            nn.Linear(32, 2),  # stop_alpha, stop_beta for Beta distribution
        )
        self.value = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.PReLU(hidden_dim, init=0.25),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, shared_features, dir_probs, pfolio_snap_ctx, pfolio_ts_ctx, session_ctx,
                magnitude=None, ta_1m=None, value_grad_scale=0.3):
        conviction = dir_probs.detach()  # (B, 3)

        combined = torch.cat([shared_features, conviction, pfolio_snap_ctx, pfolio_ts_ctx, session_ctx], dim=-1)

        raw = self.policy(combined)                # (B, 2)
        alpha = F.softplus(raw[:, 0].clamp(-5, 5)) + 1.01        # > 1 for unimodal
        beta_param = F.softplus(raw[:, 1].clamp(-5, 5)) + 1.01   # > 1 for unimodal
        alpha = torch.where(torch.isnan(alpha), torch.full_like(alpha, 2.0), alpha)
        beta_param = torch.where(torch.isnan(beta_param), torch.full_like(beta_param, 2.0), beta_param)

        # Stop ratio: PM context + direct market structure for stop placement
        stop_combined = torch.cat([combined, magnitude.unsqueeze(-1), ta_1m], dim=-1)
        stop_raw = self.stop_policy(stop_combined)
        stop_alpha = F.softplus(stop_raw[:, 0].clamp(-5, 5)) + 1.01
        stop_beta = F.softplus(stop_raw[:, 1].clamp(-5, 5)) + 1.01
        stop_alpha = torch.where(torch.isnan(stop_alpha), torch.full_like(stop_alpha, 2.0), stop_alpha)
        stop_beta = torch.where(torch.isnan(stop_beta), torch.full_like(stop_beta, 2.0), stop_beta)

        s = value_grad_scale
        combined_scaled = torch.cat([scale_gradient(shared_features, s), conviction, scale_gradient(pfolio_snap_ctx, s), scale_gradient(pfolio_ts_ctx, s), scale_gradient(session_ctx, s)], dim=-1)
        value = self.value(combined_scaled)      # (B, 1)

        return alpha, beta_param, stop_alpha, stop_beta, value


class TradingNetworkLite(nn.Module):
    """
    Two-actor trading network with DISCRETE actions.

    Architecture (see architecture.txt for full details):
        FreqStem+FiLM → Shared Encoders → GatedFusion → HyperConnections → Final Fusion → shared_features (128)
        ├── Direction Actor (128→64→3): 3-way (no-go / long / short)
        └── PM (171→64→32→3): shared(128) + conviction(3) + pfolio_snap(16) + pfolio_ts(16) + session_sum(8)

    Key features:
    - Dual FiLM conditioning (input stems + N-BEATS basis coefficients)
    - Split movement encoder: lo-freq (720/60) vs hi-freq (15/1)
    - DirectionalEncoder: UD_flag + FVG (2→8) shared across all freqs
    - Split volatility encoder: continuous (4→16) + binary events (5→8)
    - Group-aware TargetEncoder: price[0:28]→16 + UD[28:88]→32 = 48
    - MultiSourceHyperConnection (direct 720→1m skip)
    - Targets enter at 1m GatedFusion (not final fusion)
    - PM dual portfolio view (60-step time series + session summary)
    """
    FREQ_IDS = {'720': 0, '60': 1, '15': 2, '1': 3}

    def __init__(self, num_actions=2, dropout=0.15,
                 l1_coef=1e-5, use_input_gates=False):
        super().__init__()
        self.num_actions = num_actions
        self.dropout = nn.Dropout(dropout)
        self.l1_coef = l1_coef
        # Adaptive value gradient scaling — auto-tuned by PPO trainer
        self.value_grad_scale = 0.3
        self.use_input_gates = use_input_gates

        # ═══════════════════════════════════════════════════════════════
        # INPUT GATES (learnable per-feature importance, applied before stems)
        # ═══════════════════════════════════════════════════════════════
        if use_input_gates:
            # mov_input_gates — movement encoder removed
            self.ta_input_gates = nn.ModuleDict({   # ta: merged tas+px_session+directional+day_targets
                '720': InputGate(12), '60': InputGate(12),  # tas(6)+px(4)+dir(2)
                '15': InputGate(12), '1': InputGate(22),    # 1m: tas(8)+px(2)+day(10)+dir(2)
            })
            # dir_input_gates — merged into ta
            self.vol_input_gates = nn.ModuleDict({  # (B,20,4): [20_std, 20_atr, volume, vix]
                '720': InputGate(4), '60': InputGate(4),
                '15': InputGate(4), '1': InputGate(4),
            })
            self.vol_event_input_gates = nn.ModuleDict({  # (B,5): [fomc, nfp, cpi, ppi, gdp] binary flags
                '720': InputGate(5), '60': InputGate(5),
                '15': InputGate(5), '1': InputGate(5),
            })
            # session_input_gates — merged into ta
            self.ud_input_gates = nn.ModuleDict({   # (B,21): up/down volume distribution across 21 bins
                '720': InputGate(21), '60': InputGate(21),
                '15': InputGate(21), '1': InputGate(21),
            })
            # target_input_gate — merged into ta_1
            self.pfolio_input_gate = InputGate(11)   # (B,60,11): [float_pnl, per_con_pnl, step_pnl, pos_usage, rth, nts, n_px_1, kelly_f, mdd_buffer, confidence, win_rate]
            self.session_summary_input_gate = InputGate(14)  # (B,14): [acc_pnl/target, mdd/threshold, win_rate, sharpe, calmar, sub_days, pass_exam, act_days, acct_50k/100k/150k, plan×3]

        # ═══════════════════════════════════════════════════════════════
        # FREQUENCY EMBEDDING (shared across all FiLM layers)
        # ═══════════════════════════════════════════════════════════════
        self.freq_embed = nn.Embedding(4, 8)  # 4 freqs → 8-dim embedding
        # Cache index tensors to avoid per-forward allocation
        for freq_key, idx in self.FREQ_IDS.items():
            self.register_buffer(f'_freq_idx_{freq_key}', torch.tensor(idx))

        # ═══════════════════════════════════════════════════════════════
        # PER-FREQUENCY CNN STEMS (preprocess before shared encoders)
        # ═══════════════════════════════════════════════════════════════
        # mov_stems — movement encoder removed
        self.ta_stems = nn.ModuleDict({
            '720': FreqStem(12), '60': FreqStem(12),  # tas(6)+px(4)+dir(2)
            '15': FreqStem(12), '1': FreqStem(22),    # 1m: tas(8)+px(2)+day(10)+dir(2)
        })
        # self.dir_stems — merged into ta
        # self.session_stems — merged into ta
        self.vol_stems = nn.ModuleDict({
            '720': FreqStem(4), '60': FreqStem(4),
            '15': FreqStem(4), '1': FreqStem(4),
        })  # continuous only (4 channels) — binary events don't need temporal stems

        # ═══════════════════════════════════════════════════════════════
        # FiLM LAYERS (one per feature type, applied after stems)
        # ═══════════════════════════════════════════════════════════════
        # No mov_film — BasisFiLM inside MovementEncoder handles per-freq adaptation
        # Outer FiLM would warp the signal before N-BEATS polynomial/Fourier decomposition
        self.vol_film = FiLMLayer(4, embed_dim=8)  # continuous only
        # self.session_film = FiLMLayer(4, embed_dim=8)  # merged into ta
        self.ta_film = FiLMLayer(12, embed_dim=8)       # for 720/60/15: tas(6)+px(4)+dir(2)
        self.ta_film_1m = FiLMLayer(22, embed_dim=8)    # 1m: tas(8)+px(2)+day(10)+dir(2)
        # self.dir_film = FiLMLayer(2, embed_dim=8)      # merged into ta

        # ═══════════════════════════════════════════════════════════════
        # ENCODERS (movement split lo/hi, others shared across freqs)
        # FiLM handles frequency adaptation
        # ═══════════════════════════════════════════════════════════════
        # Split movement: lo-freq (720/60) specialises in macro trend,
        # hi-freq (15/1) specialises in execution-level price action
        # movement_encoder_lo/hi — removed (zero correlation, zero sensitivity)
        self.ud_encoder = UDEncoder(output_dim=16)
        # self.target_encoder = TargetEncoder(input_dim=108, output_dim=48)  # merged into ta
        self.vol_cont_encoder = VolatilityContinuousEncoder(input_dim=4, output_dim=16)
        self.vol_event_encoder = VolatilityEventEncoder(input_dim=5, output_dim=8)
        # self.session_encoder = SessionEncoder(input_dim=4, output_dim=16)  # merged into ta
        # self.directional_encoder = DirectionalEncoder(input_dim=2, output_dim=8)  # merged into ta
        self.ta_encoder = TAEncoder(input_dim=12, output_dim=16)   # 720/60/15: tas(6)+px(4)+dir(2)
        self.ta_encoder_1m = TAEncoder(input_dim=22, output_dim=16)  # 1m: tas(8)+px(2)+day(10)+dir(2)

        # ═══════════════════════════════════════════════════════════════
        # PORTFOLIO/STATE ENCODERS
        # ═══════════════════════════════════════════════════════════════
        self.portfolio_encoder = PortfolioEncoder(input_dim=11, output_dim=16)
        self.pfolio_ts_encoder = PortfolioTimeSeriesEncoder(
            input_dim=11, hidden_dim=16, output_dim=16
        )
        self.session_summary_encoder = SessionSummaryEncoder(input_dim=14, output_dim=8)

        # ═══════════════════════════════════════════════════════════════
        # INTRA-TIMEFRAME FUSION (shared across all TFs)
        # all TFs: ud(16)+vol_cont(16)+vol_evt(8)+ta(16) = 56 → 48
        # ═══════════════════════════════════════════════════════════════
        self.fusion = GatedFusion(56, 48)

        # ═══════════════════════════════════════════════════════════════
        # CROSS-TIMEFRAME HYPERCONNECTIONS (top-down + direct 720→1m)
        # ═══════════════════════════════════════════════════════════════
        self.hyper_720_60 = HyperConnection(48)
        self.hyper_60_15 = HyperConnection(48)
        self.hyper_multi_1 = CrossAttentionHyperConnection(48, num_sources=2, num_heads=4)

        # ═══════════════════════════════════════════════════════════════
        # FINAL FUSION (market-only — no portfolio, direction actor stays pure)
        # 4 TFs × 48 = 192 → 160 → 128
        # ═══════════════════════════════════════════════════════════════
        self.final_fusion = nn.Sequential(
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.PReLU(num_parameters=128, init=0.25),
        )

        # ═══════════════════════════════════════════════════════════════
        # TWO-ACTOR HEADS
        # ═══════════════════════════════════════════════════════════════
        self.direction_actor = DirectionActorHead(input_dim=128, hidden_dim=64, n_actions=3)
        self.portfolio_manager = PortfolioManagerHead(
            shared_dim=128, conviction_dim=3, pfolio_snap_dim=16, pfolio_ts_dim=16, session_dim=8, hidden_dim=64
        )

        # Weight initialization
        self.apply(self._init_weights)
        # Spectral normalization on policy heads for stable policy updates
        self._apply_spectral_norm()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _apply_spectral_norm(self):
        """Apply spectral normalization to policy head linear layers.
        Constrains the Lipschitz constant of policy networks for more stable updates.
        Applied to policy heads only — value heads are left unconstrained."""
        for head in [self.direction_actor, self.portfolio_manager]:
            for module in head.policy.modules():
                if isinstance(module, nn.Linear):
                    nn.utils.spectral_norm(module)
        for module in self.portfolio_manager.stop_policy.modules():
            if isinstance(module, nn.Linear):
                nn.utils.spectral_norm(module)

    def _remove_spectral_norm(self):
        """Remove spectral norm hooks for faster inference (no SVD recomputation)."""
        for head in [self.direction_actor, self.portfolio_manager]:
            for module in head.policy.modules():
                if isinstance(module, nn.Linear):
                    try:
                        nn.utils.remove_spectral_norm(module)
                    except ValueError:
                        pass
        for module in self.portfolio_manager.stop_policy.modules():
            if isinstance(module, nn.Linear):
                try:
                    nn.utils.remove_spectral_norm(module)
                except ValueError:
                    pass

    def _get_freq_embed(self, freq_key):
        """Get frequency embedding vector for a given frequency key."""
        return self.freq_embed(getattr(self, f'_freq_idx_{freq_key}'))  # (8,)

    def _gate(self, x, gate_attr, freq_key=None):
        """Apply input gate if enabled, otherwise pass through.
        Uses string attr name to avoid accessing non-existent attributes when gates are off."""
        if not self.use_input_gates:
            return x
        gate = getattr(self, gate_attr)
        return gate[freq_key](x) if freq_key is not None else gate(x)

    def l1_first_layer_loss(self) -> torch.Tensor:
        """Sum of L1 norms of all first-contact layer weights.
        Drives useless input feature weights toward zero."""
        l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        # FreqStem conv1 weights (16 stems across 4 dicts)
        for stem_dict in [self.ta_stems, self.vol_stems]:
            for stem in stem_dict.values():
                l1 = l1 + stem.conv1.weight.abs().sum()
        # Non-stem encoder first layers
        l1 = l1 + self.ud_encoder.net[0].weight.abs().sum()
        l1 = l1 + self.vol_event_encoder.net[0].weight.abs().sum()
        return l1

    def get_gate_values(self) -> Dict[str, torch.Tensor]:
        """Return all input gate sigmoid values for TensorBoard logging."""
        if not self.use_input_gates:
            return {}
        gates = {}
        for name, attr in [('ta', 'ta_input_gates'),
                           ('vol', 'vol_input_gates'), ('vol_evt', 'vol_event_input_gates'),
                           ('ud', 'ud_input_gates')]:
            gate_dict = getattr(self, attr)
            for freq_key, gate_module in gate_dict.items():
                gates[f'{name}_{freq_key}'] = gate_module.gate_values.detach()
        # gates['targets'] — merged into ta
        gates['pfolio'] = self.pfolio_input_gate.gate_values.detach()
        gates['session_summary'] = self.session_summary_input_gate.gate_values.detach()
        return gates

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = obs['20_UD_720'].device

        # NaN check on all input observations (debug only — costs ~4% of forward time)
        if getattr(self, '_debug_nan', False):
            for key, val in obs.items():
                if torch.isnan(val).any():
                    nan_count = torch.isnan(val).sum().item()
                    print(f"  [NaN] INPUT obs['{key}']: {nan_count} NaN values out of {val.numel()}")

        # ═══════════════════════════════════════════════════════════════
        # FREQUENCY EMBEDDINGS
        # ═══════════════════════════════════════════════════════════════
        fe_720 = self._get_freq_embed('720')
        fe_60 = self._get_freq_embed('60')
        fe_15 = self._get_freq_embed('15')
        fe_1 = self._get_freq_embed('1')

        # ═══════════════════════════════════════════════════════════════
        # [GATE] → STEMS → FiLM → SHARED ENCODERS
        # ═══════════════════════════════════════════════════════════════
        # Movement: removed (zero correlation, zero sensitivity)

        # UD: [gate] → encoder (no stem, no FiLM)
        ud_720 = self.dropout(self.ud_encoder(self._gate(obs['20_UD_720'], 'ud_input_gates', '720')))
        ud_60 = self.dropout(self.ud_encoder(self._gate(obs['20_UD_60'], 'ud_input_gates', '60')))
        ud_15 = self.dropout(self.ud_encoder(self._gate(obs['20_UD_15'], 'ud_input_gates', '15')))
        ud_1 = self.dropout(self.ud_encoder(self._gate(obs['20_UD_1'], 'ud_input_gates', '1')))

        # Targets: merged into ta_1 — no separate encoder

        # Volatility: split continuous ([gate]→stem→FiLM→Conv encoder) vs events ([gate]→MLP on last step)
        def _encode_vol(vol_data, freq_key, fe):
            vol_cont = vol_data[:, :, :4]  # (B, 20, 4) continuous: 20_std, 20_atr, volume, vix
            vol_evt = vol_data[:, -1, 4:]   # (B, 5) binary events: fomc, nfp, cpi, ppi, gdp — last timestep
            if self.use_input_gates:
                vol_cont = self.vol_input_gates[freq_key](vol_cont)
                vol_evt = self.vol_event_input_gates[freq_key](vol_evt)
            cont_out = self.dropout(self.vol_cont_encoder(self.vol_film(self.vol_stems[freq_key](vol_cont), fe)))  # (B, 16)
            evt_out = self.dropout(self.vol_event_encoder(vol_evt))  # (B, 8)
            return cont_out, evt_out

        vol_cont_720, vol_evt_720 = _encode_vol(obs['vol_720'], '720', fe_720)
        vol_cont_60, vol_evt_60 = _encode_vol(obs['vol_60'], '60', fe_60)
        vol_cont_15, vol_evt_15 = _encode_vol(obs['vol_15'], '15', fe_15)

        # 1m vol: also extract magnitude for stop sizing
        vol_1_data = obs['vol_1']
        vol_cont_1_input = vol_1_data[:, :, :4]
        vol_evt_1_input = vol_1_data[:, -1, 4:]
        if self.use_input_gates:
            vol_cont_1_input = self.vol_input_gates['1'](vol_cont_1_input)
            vol_evt_1_input = self.vol_event_input_gates['1'](vol_evt_1_input)
        vol_cont_1, magnitude = self.vol_cont_encoder(self.vol_film(self.vol_stems['1'](vol_cont_1_input), fe_1), return_magnitude=True)
        vol_cont_1 = self.dropout(vol_cont_1)
        vol_evt_1 = self.dropout(self.vol_event_encoder(vol_evt_1_input))

        # Session/Directional: merged into ta — no separate encoders

        # TA: [gate] → stem → FiLM → encoder → dropout (1m has separate encoder + FiLM due to different input dim)
        ta_720 = self.dropout(self.ta_encoder(self.ta_film(self.ta_stems['720'](self._gate(obs['ta_720'], 'ta_input_gates', '720')), fe_720)))
        ta_60 = self.dropout(self.ta_encoder(self.ta_film(self.ta_stems['60'](self._gate(obs['ta_60'], 'ta_input_gates', '60')), fe_60)))
        ta_15 = self.dropout(self.ta_encoder(self.ta_film(self.ta_stems['15'](self._gate(obs['ta_15'], 'ta_input_gates', '15')), fe_15)))
        ta_1 = self.dropout(self.ta_encoder_1m(self.ta_film_1m(self.ta_stems['1'](self._gate(obs['ta_1'], 'ta_input_gates', '1')), fe_1)))

        # ═══════════════════════════════════════════════════════════════
        # PORTFOLIO STATE
        # ═══════════════════════════════════════════════════════════════
        pfolio_input = obs['pfolio_info']
        if self.use_input_gates:
            pfolio_input = self.pfolio_input_gate(pfolio_input)
        if pfolio_input.dim() == 3:
            pfolio_flat = pfolio_input[:, -1, :]       # (B, 11)
        else:
            pfolio_flat = pfolio_input
        portfolio_ctx = self.portfolio_encoder(pfolio_flat)  # (B, 16)

        # ═══════════════════════════════════════════════════════════════
        # INTRA-TIMEFRAME FUSION
        # 720/60/15: cat(mov24, ud16, vol_c16, vol_e8, ta16) = 80 → 48
        # 1m: cat(mov24, ud16, vol_c16, vol_e8, ta16) = 80 → 48
        # ═══════════════════════════════════════════════════════════════
        tf_720 = self.dropout(self.fusion(torch.cat([ud_720, vol_cont_720, vol_evt_720, ta_720], dim=-1)))
        tf_60 = self.dropout(self.fusion(torch.cat([ud_60, vol_cont_60, vol_evt_60, ta_60], dim=-1)))
        tf_15 = self.dropout(self.fusion(torch.cat([ud_15, vol_cont_15, vol_evt_15, ta_15], dim=-1)))
        tf_1 = self.dropout(self.fusion(torch.cat([ud_1, vol_cont_1, vol_evt_1, ta_1], dim=-1)))

        # ═══════════════════════════════════════════════════════════════
        # CROSS-TF HYPERCONNECTIONS (top-down + direct 720→1m skip)
        # ═══════════════════════════════════════════════════════════════
        tf_60_enh = self.hyper_720_60(tf_60, tf_720)
        tf_15_enh = self.hyper_60_15(tf_15, tf_60_enh)
        tf_1_enh = self.hyper_multi_1(tf_1, [tf_15_enh, tf_720])

        # ═══════════════════════════════════════════════════════════════
        # FINAL FUSION: cat(4×48) = 192 → 160 → 128  (market-only, no portfolio)
        # Direction actor sees ONLY market signals — portfolio goes exclusively to PM
        # ═══════════════════════════════════════════════════════════════
        all_contexts = torch.cat([tf_720, tf_60_enh, tf_15_enh, tf_1_enh], dim=-1)
        shared_features = self.final_fusion(all_contexts)  # (B, 128)

        # ═══════════════════════════════════════════════════════════════
        # TWO-ACTOR HEADS
        # ═══════════════════════════════════════════════════════════════
        # NaN check after encoder/fusion stages (debug only)
        if getattr(self, '_debug_nan', False):
            nan_checks = {
                'shared_features': shared_features, 'portfolio_ctx': portfolio_ctx,
                'magnitude': magnitude, 'ta_1': ta_1,
                'tf_720': tf_720, 'tf_1_enh': tf_1_enh,
            }
            for name, t in nan_checks.items():
                if t is not None and torch.isnan(t).any():
                    print(f"  [NaN] INTERMEDIATE {name}: {torch.isnan(t).sum().item()} NaN values")

        vgs = self.value_grad_scale
        dir_logits, dir_probs, dir_value = self.direction_actor(shared_features, value_grad_scale=vgs)

        # PM: shared(128) + conviction(3) + pfolio_snap(16) + pfolio_ts(16) + session_sum(8) = 171
        # Portfolio state goes ONLY to PM — direction actor stays pure market signal assessor
        if pfolio_input.dim() == 3:
            pfolio_ts_ctx = self.pfolio_ts_encoder(pfolio_input)   # (B, 16)
        else:
            pfolio_ts_ctx = torch.zeros(pfolio_input.size(0), 16, device=device)

        session_sum = obs.get('session_summary')
        if session_sum is not None:
            if self.use_input_gates:
                session_sum = self.session_summary_input_gate(session_sum)
            session_ctx = self.session_summary_encoder(session_sum)  # (B, 8)
        else:
            session_ctx = torch.zeros(pfolio_input.size(0), 8, device=device)

        # NaN check PM-specific inputs (debug only)
        if getattr(self, '_debug_nan', False):
            for name, t in [('pfolio_ts_ctx', pfolio_ts_ctx), ('session_ctx', session_ctx), ('dir_probs', dir_probs)]:
                if torch.isnan(t).any():
                    print(f"  [NaN] PM INPUT {name}: {torch.isnan(t).sum().item()} NaN values")

        pm_alpha, pm_beta, stop_alpha, stop_beta, pm_value = self.portfolio_manager(
            shared_features, dir_probs, portfolio_ctx, pfolio_ts_ctx, session_ctx,
            magnitude=magnitude, ta_1m=ta_1,
            value_grad_scale=vgs,
        )

        return {
            'direction_logits': dir_logits,       # (B, 3)
            'direction_probs': dir_probs,         # (B, 3)
            'direction_value': dir_value,         # (B, 1)
            'pm_alpha': pm_alpha,           # (B,)
            'pm_beta': pm_beta,             # (B,)
            'stop_alpha': stop_alpha,       # (B,)
            'stop_beta': stop_beta,         # (B,)
            'magnitude': magnitude,         # (B,)
            'pm_value': pm_value,           # (B, 1)
        }

    def get_action(
        self,
        obs: Dict[str, torch.Tensor],
        prev_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Get actions from both actors.

        Returns dict with keys:
            dir_action, dir_log_prob, dir_entropy, dir_value,
            pm_action, pm_log_prob, pm_entropy, pm_value, pm_max_prob
        """
        out = self.forward(obs)

        # --- Direction actor ---
        dir_probs = out['direction_probs']
        if torch.isnan(dir_probs).any():
            dir_probs = torch.ones(dir_probs.size(0), 3, device=dir_probs.device) / 3.0
        dir_dist = Categorical(probs=dir_probs)
        dir_action = dir_probs.argmax(dim=-1) if deterministic else dir_dist.sample()

        # --- Portfolio manager (Beta confidence) ---
        pm_alpha = out['pm_alpha']
        pm_beta = out['pm_beta']
        # NaN/Inf guard — prevents crash if weights explode
        bad = torch.isnan(pm_alpha) | torch.isinf(pm_alpha)
        pm_alpha = torch.where(bad, torch.full_like(pm_alpha, 2.0), pm_alpha)
        pm_beta = torch.where(bad | torch.isnan(pm_beta) | torch.isinf(pm_beta),
                              torch.full_like(pm_beta, 2.0), pm_beta)
        pm_dist = torch.distributions.Beta(pm_alpha, pm_beta)
        if deterministic:
            pm_confidence = pm_alpha / (pm_alpha + pm_beta)  # Beta mean
        else:
            pm_confidence = pm_dist.sample()
        pm_confidence = pm_confidence.clamp(1e-6, 1 - 1e-6)

        # --- Stop ratio (Beta) ---
        stop_alpha = out['stop_alpha']
        stop_beta = out['stop_beta']
        bad_s = torch.isnan(stop_alpha) | torch.isinf(stop_alpha)
        stop_alpha = torch.where(bad_s, torch.full_like(stop_alpha, 2.0), stop_alpha)
        stop_beta = torch.where(bad_s | torch.isnan(stop_beta) | torch.isinf(stop_beta),
                                torch.full_like(stop_beta, 2.0), stop_beta)
        stop_dist = torch.distributions.Beta(stop_alpha, stop_beta)
        if deterministic:
            stop_ratio = stop_alpha / (stop_alpha + stop_beta)
        else:
            stop_ratio = stop_dist.sample()
        stop_ratio = stop_ratio.clamp(1e-6, 1 - 1e-6)

        return {
            'direction_action': dir_action,
            'direction_log_prob': dir_dist.log_prob(dir_action),
            'direction_entropy': dir_dist.entropy(),
            'direction_value': out['direction_value'],

            'pm_confidence': pm_confidence,
            'pm_log_prob': pm_dist.log_prob(pm_confidence),
            'pm_entropy': pm_dist.entropy(),
            'pm_value': out['pm_value'],

            'stop_ratio': stop_ratio,
            'stop_log_prob': stop_dist.log_prob(stop_ratio),
            'stop_entropy': stop_dist.entropy(),
            'magnitude': out['magnitude'],
        }

    def evaluate_action(
        self,
        obs: Dict[str, torch.Tensor],
        actions_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate given actions for both actors (for PPO update).

        Args:
            obs: observation dict
            actions_dict: {'direction_action': (B,), 'pm_action': (B,)}

        Returns dict with log_prob, entropy, value for each actor.
        """
        out = self.forward(obs)

        # NaN guard — same fallback as get_action to prevent NaN loss
        dir_probs = out['direction_probs']
        if torch.isnan(dir_probs).any():
            dir_probs = torch.ones(dir_probs.size(0), 3, device=dir_probs.device) / 3.0

        dir_dist = Categorical(probs=dir_probs)

        pm_alpha = out['pm_alpha']
        pm_beta = out['pm_beta']
        # Belt-and-suspenders NaN/Inf guard — prevents crash if weights explode
        bad = torch.isnan(pm_alpha) | torch.isinf(pm_alpha)
        pm_alpha = torch.where(bad, torch.full_like(pm_alpha, 2.0), pm_alpha)
        pm_beta = torch.where(bad | torch.isnan(pm_beta) | torch.isinf(pm_beta),
                              torch.full_like(pm_beta, 2.0), pm_beta)
        pm_dist = torch.distributions.Beta(pm_alpha, pm_beta)

        stop_alpha = out['stop_alpha']
        stop_beta = out['stop_beta']
        bad_s = torch.isnan(stop_alpha) | torch.isinf(stop_alpha)
        stop_alpha = torch.where(bad_s, torch.full_like(stop_alpha, 2.0), stop_alpha)
        stop_beta = torch.where(bad_s | torch.isnan(stop_beta) | torch.isinf(stop_beta),
                                torch.full_like(stop_beta, 2.0), stop_beta)
        stop_dist = torch.distributions.Beta(stop_alpha, stop_beta)

        return {
            'direction_log_prob': dir_dist.log_prob(actions_dict['direction_action'].long()),
            'direction_entropy': dir_dist.entropy(),
            'direction_value': out['direction_value'],

            'pm_log_prob': pm_dist.log_prob(actions_dict['pm_confidence'].clamp(1e-6, 1 - 1e-6)),
            'pm_entropy': pm_dist.entropy(),
            'pm_value': out['pm_value'],

            'stop_log_prob': stop_dist.log_prob(actions_dict['stop_ratio'].clamp(1e-6, 1 - 1e-6)),
            'stop_entropy': stop_dist.entropy(),

            'magnitude': out['magnitude'],
        }


# Alias for backward compatibility
TradingNetwork = TradingNetworkLite


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Convert env observation tuple to network dict
# ═══════════════════════════════════════════════════════════════════════════════

def obs_to_dict(obs_tuple, feature_keys):
    """
    Convert environment observation tuple to dict format for network.
    
    Args:
        obs_tuple: tuple of numpy arrays from env._get_obs()
        feature_keys: list of feature names in same order as tuple
        
    Returns:
        dict with tensor values, batched
    """
    obs_dict = {}
    for key, arr in zip(feature_keys, obs_tuple):
        if isinstance(arr, np.ndarray):
            tensor = torch.from_numpy(arr).float()
        else:
            tensor = torch.tensor(arr).float()
        # Add batch dimension if needed
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # (N,) -> (1, N)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (T, F) -> (1, T, F)
        obs_dict[key] = tensor
    return obs_dict


def batch_obs_dicts(obs_list):
    """
    Batch multiple observation dicts into single dict with batched tensors.
    
    Args:
        obs_list: list of obs dicts
        
    Returns:
        single dict with batched tensors
    """
    batched = {}
    keys = obs_list[0].keys()
    for key in keys:
        batched[key] = torch.cat([obs[key] for obs in obs_list], dim=0)
    return batched


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    network = TradingNetworkLite(num_actions=2)

    total_params = sum(p.numel() for p in network.parameters())
    print(f"Total parameters: {total_params:,}")

    B = 32
    obs = {
        '20_UD_720': torch.randn(B, 21),
        '20_UD_60': torch.randn(B, 21),
        '20_UD_15': torch.randn(B, 21),
        '20_UD_1': torch.randn(B, 21),
        'vol_720': torch.randn(B, 20, 9),
        'vol_60': torch.randn(B, 20, 9),
        'vol_15': torch.randn(B, 20, 9),
        'vol_1': torch.randn(B, 20, 9),
        'ta_720': torch.randn(B, 20, 12),  # tas(6)+px(4)+dir(2)
        'ta_60': torch.randn(B, 20, 12),
        'ta_15': torch.randn(B, 20, 12),
        'ta_1': torch.randn(B, 20, 22),   # tas(8)+px(2)+day(10)+dir(2)
        'pfolio_info': torch.randn(B, 60, 11),
        'session_summary': torch.randn(B, 14),
    }

    # Forward pass
    output = network(obs)
    print(f"\n--- Forward Pass ---")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")

    # Get action
    out = network.get_action(obs)
    print(f"\n--- Get Action ---")
    print(f"  pm_action: {out['pm_action'].shape} | values: {out['pm_action'][:5].tolist()}")
    print(f"  direction_action: {out['direction_action'][:5].tolist()}")
    print(f"  pm_max_prob: {out['pm_max_prob'][:3].tolist()}")

    # Evaluate action (for PPO)
    actions_dict = {
        'direction_action': out['direction_action'],
        'pm_action': out['pm_action'],
    }
    eval_out = network.evaluate_action(obs, actions_dict)
    print(f"\n--- Evaluate Action ---")
    print(f"  PM log_prob match: {torch.allclose(out['pm_log_prob'], eval_out['pm_log_prob'])}")
    print(f"  Direction log_prob match: {torch.allclose(out['direction_log_prob'], eval_out['direction_log_prob'])}")

