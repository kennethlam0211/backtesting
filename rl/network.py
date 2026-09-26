"""Network v2: Two-actor design with gated fusion.

Actor 1 — Model Selector (γ=0.99):
    Input: variants(72,5)×3, session_summary(12), pfolio_info(60,18)
    Shared encoder: (72,5) → broadcast signal dist → (72,8) → attention top-5
    Output: 3 per-family confidences (Beta) + selector value

Actor 2 — K Predictor (γ=0.97):
    Input: vol_60(60, 6) — 60 × 1min vol bars before consensus step
    Output: categorical K from [20, 60, 100, 150, 200] + k value

Components: scale_gradient, InputGate, GatedFusion, LayerNorm+PReLU, spectral_norm,
            Dropout, L1 reg.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
from torch.nn.utils import spectral_norm


def scale_gradient(x, scale=0.3):
    """Scale gradient in backward pass without affecting forward pass."""
    return x * scale + x.detach() * (1 - scale)


def triple_pool(out):
    """Concat [last, max, mean] over time dim. out: (B, T, D) → (B, 3*D).
    Captures end-state, peak, and average — richer summary than just last hidden."""
    last = out[:, -1, :]
    mx = out.max(dim=1).values
    mn = out.mean(dim=1)
    return torch.cat([last, mx, mn], dim=-1)


class InputGate(nn.Module):
    """Per-feature learnable sigmoid gates."""
    def __init__(self, n_features):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(n_features) * 2.0)

    def forward(self, x):
        return x * torch.sigmoid(self.gate)


class GatedFusion(nn.Module):
    """Learnable gating for fusing multiple embeddings.
    Each input gets a sigmoid gate that learns its importance."""
    def __init__(self, n_inputs, embed_dim):
        super().__init__()
        self.gate = nn.Linear(n_inputs * embed_dim, n_inputs)

    def forward(self, *embeddings):
        """embeddings: list of (B, embed_dim) tensors."""
        concat = torch.cat(embeddings, dim=-1)  # (B, n_inputs * embed_dim)
        weights = torch.sigmoid(self.gate(concat))  # (B, n_inputs)
        weighted = []
        for i, emb in enumerate(embeddings):
            weighted.append(emb * weights[:, i:i+1])
        return torch.cat(weighted, dim=-1)


# ═══════════════════════════════════════════════════════════════════
# ACTOR 1: Model Selector (60min)
# ═══════════════════════════════════════════════════════════════════

class StrategyFamilyEncoder(nn.Module):
    """Shared encoder for one family (72 variants × 5) with attention.
    Input: (B, 72, 5) = [bar_pnl_norm, side, ewm_winrate, ewm_mean, ewm_sharpe]
    Internal: (B, 72, 8) = variant features + signal distribution broadcast (3 dims)

    Per-variant gate: learnable scalar per variant applied in logit space before
    softmax — `scores = scores + log(sigmoid(gate))`. Sigmoid ∈ [0, 1], so log ≤ 0.
    Suppresses low-gated variants in attention. Init at 0.0 (sigmoid(0)=0.5, max gradient).
    """
    def __init__(self, n_variants=72, feat_dim=5, embed_dim=8, top_k=5):
        super().__init__()
        self.top_k = top_k
        self.n_variants = n_variants
        fused_dim = feat_dim + 3   # variant feats (5) + signal dist (3) = 8
        self.input_gate = InputGate(fused_dim)
        # LayerNorm before MLP: equalizes mixed-scale features (bar_pnl_norm, side, winrate, mean, sharpe)
        # for stable gradient flow into the variant_mlp.
        self.input_norm = nn.LayerNorm(fused_dim)

        self.variant_mlp = nn.Sequential(
            nn.Linear(fused_dim, 12),
            nn.LayerNorm(12),
            nn.PReLU(init=0.25),
            nn.Dropout(0.2),
            nn.Linear(12, embed_dim),
            nn.PReLU(init=0.25),
        )
        self.attn_scorer = nn.Sequential(
            nn.Linear(embed_dim, 4),
            nn.PReLU(init=0.25),
            nn.Linear(4, 1),
        )
        # Per-variant learnable gate (added in logit space — see forward).
        # Init at 0.0 → sigmoid(0)=0.5 (max gradient zone), faster gate differentiation.
        self.variant_gate = nn.Parameter(torch.zeros(n_variants))

        # Soft ewm_mean gate: scores += ewm_mean_scale * ewm_mean (col 3 of input).
        # Network learns optimal scale; positive scale biases attention to positive-momentum variants.
        # Init at 0.0 → no initial gating, network discovers the right strength via gradient.
        self.ewm_mean_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, N, _ = x.shape

        # Signal distribution broadcast
        signals = x[:, :, 1]
        long_frac = (signals > 0).float().mean(dim=-1, keepdim=True)
        short_frac = (signals < 0).float().mean(dim=-1, keepdim=True)
        flat_frac = (signals == 0).float().mean(dim=-1, keepdim=True)
        dist = torch.cat([long_frac, short_frac, flat_frac], dim=-1)
        dist_broadcast = dist.unsqueeze(1).expand(B, N, 3)

        x_fused = torch.cat([x, dist_broadcast], dim=-1)
        x_gated = self.input_gate(x_fused)
        x_normed = self.input_norm(x_gated)            # LayerNorm equalizes feature scales
        embeddings = self.variant_mlp(x_normed)
        scores = self.attn_scorer(embeddings).squeeze(-1)   # (B, N)

        # Option B: soft ewm_mean gate — bias attention scores by recent momentum.
        # ewm_mean is at col 3 of input (5-col snapshot: bar_pnl, side, winrate, mean, sharpe).
        # Network learns ewm_mean_scale; positive scale → positive-momentum variants get higher scores.
        ewm_mean_per_variant = x[:, :, 3]                                     # (B, N)
        scores = scores + self.ewm_mean_scale * ewm_mean_per_variant

        # Per-variant gate: added in logit space before softmax.
        # log(sigmoid(g)) ≤ 0 → low-gated variants get suppressed in attention.
        gate_logit = torch.log(torch.sigmoid(self.variant_gate) + 1e-8)   # (N,)
        scores = scores + gate_logit.unsqueeze(0)                           # broadcast over batch
        # Note: env applies a hard mask on negative ewm_mean post-softmax (env._update_consensus_from_variants).

        # Full softmax over all N variants — gradient flows to every variant.
        attn_weights = F.softmax(scores / 0.5, dim=-1)  # temp=0.5 sharper softmax

        weighted_signal = (attn_weights * signals).sum(dim=-1)
        context = (attn_weights.unsqueeze(-1) * embeddings).sum(dim=1)

        return weighted_signal, context, attn_weights


class RegimeEncoder(nn.Module):
    """Shared encoder for regime features (n_fam*5 + 1 scalars).
    Embedding fed to both Selector and K predictor heads — single source of regime truth."""
    def __init__(self, input_dim, embed_dim=8):
        super().__init__()
        self.input_gate = InputGate(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.PReLU(num_parameters=16, init=0.25),
            nn.Dropout(0.2),
            nn.Linear(16, embed_dim),
            nn.PReLU(num_parameters=embed_dim, init=0.25),
        )

    def forward(self, x):
        return self.mlp(self.input_gate(x))


class SessionEncoder(nn.Module):
    """Encode session summary (12 features)."""
    def __init__(self, input_dim=12, embed_dim=4):
        super().__init__()
        self.input_gate = InputGate(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.LayerNorm(8),
            nn.PReLU(num_parameters=8, init=0.25),
            nn.Dropout(0.2),
            nn.Linear(8, embed_dim),
            nn.PReLU(num_parameters=embed_dim, init=0.25),
        )

    def forward(self, x):
        return self.mlp(self.input_gate(x))


class PortfolioTimeSeriesEncoder(nn.Module):
    """Encode rolling portfolio+vol (60, 18) → embedding. Conv1d + LSTM + triple pool.
    Input LayerNorm bounds the 18 portfolio features (some are unbounded — e.g.
    floating_pnl/NORM_FACTOR can hit ±10) before they reach the conv."""
    def __init__(self, feat_dim=18, embed_dim=8):
        super().__init__()
        self.input_norm = nn.LayerNorm(feat_dim)   # normalize per-feature per-timestep
        self.conv = nn.Conv1d(feat_dim, 12, kernel_size=5, padding=2)
        self.ln = nn.LayerNorm(12)
        self.act = nn.PReLU(init=0.25)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(12, embed_dim, batch_first=True)
        # Triple pool gives 3 × embed_dim; project back to embed_dim.
        self.pool_proj = nn.Linear(3 * embed_dim, embed_dim)

    def forward(self, x):
        x = self.input_norm(x)
        h = self.conv(x.transpose(1, 2))
        h = self.ln(h.transpose(1, 2))
        h = self.dropout(self.act(h))
        out, _ = self.lstm(h)
        return self.pool_proj(triple_pool(out))


class VariantTemporalEncoder(nn.Module):
    """Encode 3-channel variant time series (20, 72, 3) → embedding per family.
    Channels = [ewm_mean_norm, ewm_winrate, ewm_sharpe]. Per-channel Conv1d
    + learnable scalar sigmoid gate per channel (lets the model down-weight
    less-useful channels) → concat → LSTM for temporal pattern."""
    def __init__(self, n_variants=72, n_channels=3, embed_dim=8, ch_out=12):
        super().__init__()
        self.n_channels = n_channels
        self.conv_per_channel = nn.ModuleList([
            nn.Conv1d(n_variants, ch_out, kernel_size=5, padding=2) for _ in range(n_channels)
        ])
        # Per-channel learnable gate: sigmoid(2.0) ≈ 0.88 init (mostly open)
        self.channel_gate = nn.Parameter(torch.ones(n_channels) * 2.0)
        concat_dim = ch_out * n_channels
        self.ln = nn.LayerNorm(concat_dim)
        self.act = nn.PReLU(init=0.25)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(concat_dim, embed_dim, batch_first=True)
        # Triple pool gives 3 × embed_dim; project back to embed_dim.
        self.pool_proj = nn.Linear(3 * embed_dim, embed_dim)

    def forward(self, x):
        # x: (B, 20, 72, 3) — 20 timesteps, 72 variants, 3 channels
        gates = torch.sigmoid(self.channel_gate)             # (3,)
        per_ch = []
        for c in range(self.n_channels):
            xc = x[..., c]                                   # (B, 20, 72)
            hc = self.conv_per_channel[c](xc.transpose(1, 2))   # (B, ch_out, 20)
            hc = hc * gates[c]                               # learnable gate before concat
            per_ch.append(hc)
        h = torch.cat(per_ch, dim=1)                         # (B, ch_out*n_channels, 20)
        h = self.ln(h.transpose(1, 2))                       # (B, 20, concat_dim)
        h = self.dropout(self.act(h))
        out, _ = self.lstm(h)
        return self.pool_proj(triple_pool(out))              # (B, embed_dim)


class ModelSelectorHead(nn.Module):
    """Actor 1: per-family variant selection + per-family confidence.

    Loops over params.STRAT_FAMILIES. Uses ONE shared family_encoder (weights
    shared across families), a per-family sigmoid gate, and one Beta head per family.

    Output conventions (keys include `{fam.lower()}_*` for each family in STRAT_FAMILIES):
      - `{fam}_signal`  : weighted signal scalar per batch
      - `{fam}_attn`    : (B, 72) attention weights
      - `conf_{fam}_alpha` / `conf_{fam}_beta` : Beta parameters for the confidence distribution
    """
    def __init__(self, family_embed=8, session_embed=4, pfolio_embed=8, hidden_dim=24, selector_mode='deterministic', regime_embed=0):
        super().__init__()
        import params as _p
        self.selector_mode = selector_mode
        self.families = list(_p.STRAT_FAMILIES)
        n_fam = len(self.families)
        self.regime_embed = regime_embed

        # ONE shared encoder — all families reuse these weights (per-variant gate is INSIDE it).
        self.family_encoder = StrategyFamilyEncoder(72, 5, family_embed, top_k=5)
        self.session_encoder = SessionEncoder(13, session_embed)
        self.pfolio_encoder = PortfolioTimeSeriesEncoder(18, pfolio_embed)

        # Per-family learnable sigmoid gate on family context.
        # Init at 0.0 → sigmoid(0)=0.5 (max gradient zone), faster family differentiation.
        self.family_gate = nn.Parameter(torch.zeros(n_fam))

        # Variant temporal encoder (only when RL selector active) — also shared across families.
        # HyperConnection fusion: collapse ctx + temporal into single per-family stream
        # via residual gate `fused = ctx + sigmoid(hyper_gate) * temporal`. Init at 0.0 →
        # sigmoid=0.5, equal weight residual at start. Reduces fusion-input dim.
        if selector_mode == 'rl':
            self.variant_temporal_encoder = VariantTemporalEncoder(n_variants=72, n_channels=3, embed_dim=family_embed)
            self.hyper_gate = nn.Parameter(torch.zeros(n_fam))

        # Fusion dims scale with n_fam (no separate temporal_dim — fused per family)
        # +regime_embed (shared regime context, fed by TradingNetworkV2)
        sel_fusion_dim = n_fam * family_embed + session_embed + regime_embed
        self.sel_fusion = nn.Sequential(
            nn.Linear(sel_fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(num_parameters=hidden_dim, init=0.25),
            nn.Dropout(0.2),
        )
        conf_fusion_dim = n_fam * family_embed + session_embed + pfolio_embed + regime_embed
        self.conf_fusion = nn.Sequential(
            nn.Linear(conf_fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(num_parameters=hidden_dim, init=0.25),
            nn.Dropout(0.2),
        )

        # Per-family Beta heads — one trio (hidden, alpha, beta) per family, stored by family-lowercase key
        self.conf_hiddens = nn.ModuleDict({
            fam.lower(): nn.Sequential(spectral_norm(nn.Linear(hidden_dim, 8)), nn.PReLU(num_parameters=8, init=0.25))
            for fam in self.families
        })
        self.conf_alphas = nn.ModuleDict({fam.lower(): nn.Linear(8, 1) for fam in self.families})
        self.conf_betas  = nn.ModuleDict({fam.lower(): nn.Linear(8, 1) for fam in self.families})

        # Value heads
        self.selector_value_head = nn.Sequential(
            nn.Linear(hidden_dim, 8), nn.PReLU(num_parameters=8, init=0.25), nn.Linear(8, 1),
        )
        self.confidence_value_head = nn.Sequential(
            nn.Linear(hidden_dim, 8), nn.PReLU(num_parameters=8, init=0.25), nn.Linear(8, 1),
        )

    def _beta_params(self, fam, x):
        """Per-family Beta(α, β) params with softplus + 1.0 floor."""
        key = fam.lower()
        h = self.conf_hiddens[key](x)
        a = F.softplus(self.conf_alphas[key](h)).squeeze(-1) + 1.0
        b = F.softplus(self.conf_betas[key](h)).squeeze(-1) + 1.0
        return a, b

    def forward(self, obs, regime_emb=None):
        # 1. Encode each family's snapshot (shared encoder), apply per-family gate to context.
        # If RL: also fuse the temporal encoding into the same per-family stream via
        # HyperConnection residual: fused = ctx + sigmoid(hyper_gate) * temporal
        family_gates = torch.sigmoid(self.family_gate)   # (n_fam,)
        has_temporal = self.selector_mode == 'rl' and f'{self.families[0]}_metrics_ts' in obs
        if has_temporal:
            hyper_gates = torch.sigmoid(self.hyper_gate) # (n_fam,)
        fam_ctxs = []
        fam_results = {}     # temp collection to merge into return dict
        for i, fam in enumerate(self.families):
            sig, ctx, attn = self.family_encoder(obs[f'{fam}_variants'])
            ctx = ctx * family_gates[i]                  # per-family gate
            if has_temporal:
                ts = self.variant_temporal_encoder(obs[f'{fam}_metrics_ts'])
                ts = ts * family_gates[i]                # same family gate applied to temporal
                ctx = ctx + hyper_gates[i] * ts          # HyperConnection residual
            fam_ctxs.append(ctx)
            fam_results[f'{fam.lower()}_signal'] = sig
            fam_results[f'{fam.lower()}_attn']   = attn

        session_emb = self.session_encoder(obs['session_summary'])
        base_parts = list(fam_ctxs) + [session_emb]
        if regime_emb is not None and self.regime_embed > 0:
            base_parts.append(regime_emb)

        # 3. Selector branch (no pfolio)
        sel_shared = self.sel_fusion(torch.cat(base_parts, dim=-1))
        sel_value = self.selector_value_head(scale_gradient(sel_shared, 0.3))

        # 4. Confidence branch (with pfolio)
        pfolio_emb = self.pfolio_encoder(obs['pfolio_info'])
        conf_shared = self.conf_fusion(torch.cat(base_parts + [pfolio_emb], dim=-1))
        conf_value = self.confidence_value_head(scale_gradient(conf_shared, 0.3))

        # 5. Per-family Beta params
        for fam in self.families:
            a, b = self._beta_params(fam, conf_shared)
            fam_results[f'conf_{fam.lower()}_alpha'] = a
            fam_results[f'conf_{fam.lower()}_beta']  = b

        return {
            **fam_results,
            'selector_value': sel_value,
            'confidence_value': conf_value,
        }


# ═══════════════════════════════════════════════════════════════════
# ACTOR 2: K Predictor (vol only → stop distance)
# ═══════════════════════════════════════════════════════════════════

class VolTimeSeriesEncoder(nn.Module):
    """Encode rolling vol (60, 6) → embedding. Conv1d + LSTM + triple pool."""
    def __init__(self, feat_dim=6, embed_dim=8):
        super().__init__()
        self.conv = nn.Conv1d(feat_dim, 12, kernel_size=5, padding=2)
        self.ln = nn.LayerNorm(12)
        self.act = nn.PReLU(init=0.25)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(12, embed_dim, batch_first=True)
        self.pool_proj = nn.Linear(3 * embed_dim, embed_dim)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))
        h = self.ln(h.transpose(1, 2))
        h = self.dropout(self.act(h))
        out, _ = self.lstm(h)
        return self.pool_proj(triple_pool(out))


class KPredictorHead(nn.Module):
    """Actor 2: predicts K from vol time series (60, 6).
    Categorical over K_OPTIONS = [20, 60, 100, 150, 200].
    Has own value head for separate GAE."""
    N_K_OPTIONS = 5

    def __init__(self, vol_embed=8, hidden_dim=8, regime_embed=0):
        super().__init__()
        self.vol_encoder = VolTimeSeriesEncoder(6, vol_embed)
        self.regime_embed = regime_embed

        self.fusion = nn.Sequential(
            nn.Linear(vol_embed + regime_embed, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(num_parameters=hidden_dim, init=0.25),
            nn.Dropout(0.2),
        )

        # K head: categorical logits
        self.k_logits = nn.Linear(hidden_dim, self.N_K_OPTIONS)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.PReLU(num_parameters=4, init=0.25),
            nn.Linear(4, 1),
        )

    def forward(self, obs, regime_emb=None):
        vol_emb = self.vol_encoder(obs['vol_60'])
        if regime_emb is not None and self.regime_embed > 0:
            vol_emb = torch.cat([vol_emb, regime_emb], dim=-1)
        shared = self.fusion(vol_emb)

        k_logits = self.k_logits(shared)  # (B, 5)
        value = self.value_head(scale_gradient(shared, 0.3))

        return {
            'k_logits': k_logits,
            'value': value,
        }


# ═══════════════════════════════════════════════════════════════════
# COMBINED NETWORK
# ═══════════════════════════════════════════════════════════════════

class TradingNetworkV2(nn.Module):
    """Two-actor network.

    Actor 1 (ModelSelector): picks top-5 per family → per-family confidences
    Actor 2 (KPredictor): categorical K from [20, 60, 100, 150, 200]
    """
    def __init__(self, selector_mode='deterministic'):
        super().__init__()
        import params as _p
        n_fam = len(_p.STRAT_FAMILIES)
        regime_input_dim = n_fam * 5 + 1
        self.regime_embed = 8
        self.regime_encoder = RegimeEncoder(regime_input_dim, embed_dim=self.regime_embed)
        self.model_selector = ModelSelectorHead(selector_mode=selector_mode, regime_embed=self.regime_embed)
        self.k_predictor = KPredictorHead(regime_embed=self.regime_embed)

    def _encode_regime(self, obs):
        """Encode regime_features → (B, embed_dim). Returns None if key missing (safety)."""
        if 'regime_features' not in obs:
            return None
        return self.regime_encoder(obs['regime_features'])

    def forward_selector(self, obs):
        regime_emb = self._encode_regime(obs)
        return self.model_selector(obs, regime_emb=regime_emb)

    def forward_k(self, obs):
        regime_emb = self._encode_regime(obs)
        return self.k_predictor(obs, regime_emb=regime_emb)

    def _sample_beta(self, alpha, beta, deterministic=False):
        dist = Beta(alpha, beta)
        if deterministic:
            sample = alpha / (alpha + beta)
        else:
            sample = dist.sample()
        return sample, dist

    def get_action(self, selector_obs, k_obs, deterministic=False):
        """Full action: run both actors. Loops over params.STRAT_FAMILIES for confidences.

        Action layout: [k_index, conf_{fam} for fam in STRAT_FAMILIES]
        e.g. 6 families → 7 dims: [k, conf_s4, conf_cgf, conf_rofs, conf_rsi, conf_bb, conf_s3]
        """
        import params as _p
        sel_out = self.forward_selector(selector_obs)
        k_out = self.forward_k(k_obs)

        # Sample K (categorical)
        k_dist = Categorical(logits=k_out['k_logits'])
        if deterministic:
            k_index = k_out['k_logits'].argmax(dim=-1)
        else:
            k_index = k_dist.sample()

        # Sample per-family confidences (Beta) + accumulate log_prob / entropy
        clamp = lambda x: x.clamp(1e-6, 1 - 1e-6)
        conf_samples = {}
        conf_dists = {}
        conf_log_prob = None
        conf_entropy = None
        for fam in _p.STRAT_FAMILIES:
            a = sel_out[f'conf_{fam.lower()}_alpha']
            b = sel_out[f'conf_{fam.lower()}_beta']
            sample, dist = self._sample_beta(a, b, deterministic)
            conf_samples[fam] = sample
            conf_dists[fam] = dist
            lp = dist.log_prob(clamp(sample))
            ent = dist.entropy()
            conf_log_prob = lp if conf_log_prob is None else conf_log_prob + lp
            conf_entropy  = ent if conf_entropy  is None else conf_entropy  + ent

        k_log_prob = k_dist.log_prob(k_index)
        k_entropy = k_dist.entropy()
        joint_log_prob = k_log_prob + conf_log_prob
        joint_entropy  = k_entropy + conf_entropy

        out = {
            'k_index': k_index,
            'k_log_prob': k_log_prob,
            'conf_log_prob': conf_log_prob,
            'k_entropy': k_entropy,
            'conf_entropy': conf_entropy,
            'joint_log_prob': joint_log_prob,
            'joint_entropy': joint_entropy,
            'selector_value': sel_out['selector_value'],
            'confidence_value': sel_out['confidence_value'],
            'k_value': k_out['value'],
        }
        for fam in _p.STRAT_FAMILIES:
            key = fam.lower()
            out[f'conf_{key}'] = conf_samples[fam]
            out[f'{key}_signal'] = sel_out[f'{key}_signal'].detach()
            out[f'{key}_attn']   = sel_out[f'{key}_attn'].detach()
        return out

    def evaluate_action(self, selector_obs, k_obs, k_index, conf_by_family):
        """Re-evaluate stored actions for PPO update.

        Args:
            conf_by_family: dict {fam: tensor} of stored confidence samples per family.
                            Keys must match params.STRAT_FAMILIES.
        """
        import params as _p
        sel_out = self.forward_selector(selector_obs)
        k_out = self.forward_k(k_obs)

        k_dist = Categorical(logits=k_out['k_logits'])
        clamp = lambda x: x.clamp(1e-6, 1 - 1e-6)

        conf_log_prob = None
        conf_entropy  = None
        for fam in _p.STRAT_FAMILIES:
            a = sel_out[f'conf_{fam.lower()}_alpha']
            b = sel_out[f'conf_{fam.lower()}_beta']
            dist = Beta(a, b)
            lp  = dist.log_prob(clamp(conf_by_family[fam]))
            ent = dist.entropy()
            conf_log_prob = lp if conf_log_prob is None else conf_log_prob + lp
            conf_entropy  = ent if conf_entropy  is None else conf_entropy  + ent

        k_log_prob = k_dist.log_prob(k_index.long())
        k_entropy = k_dist.entropy()
        joint_log_prob = k_log_prob + conf_log_prob
        joint_entropy  = k_entropy + conf_entropy

        out = {
            'k_log_prob': k_log_prob,
            'conf_log_prob': conf_log_prob,
            'k_entropy': k_entropy,
            'conf_entropy': conf_entropy,
            'joint_log_prob': joint_log_prob,
            'joint_entropy': joint_entropy,
            'selector_value': sel_out['selector_value'],
            'confidence_value': sel_out['confidence_value'],
            'k_value': k_out['value'],
        }
        for fam in _p.STRAT_FAMILIES:
            out[f'{fam.lower()}_signal'] = sel_out[f'{fam.lower()}_signal']
        return out

    def l1_first_layer_loss(self):
        """L1 on input gate parameters for sparsity."""
        l1 = 0
        for module in self.modules():
            if isinstance(module, InputGate):
                l1 += module.gate.abs().sum()
        return l1

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def get_trainable_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    net = TradingNetworkV2()
    sel_params = sum(p.numel() for p in net.model_selector.parameters())
    k_params = sum(p.numel() for p in net.k_predictor.parameters())
    print(f'Total: {net.get_param_count():,} (selector: {sel_params:,}, k_predictor: {k_params:,})')

    B = 4
    sel_obs = {
        'S4_variants': torch.randn(B, 72, 5),
        'CGF_variants': torch.randn(B, 72, 5),
        'ROFS_variants': torch.randn(B, 72, 5),
        'session_summary': torch.randn(B, 12),
        'pfolio_info': torch.randn(B, 60, 18),
    }
    k_obs = {
        'vol_60': torch.randn(B, 60, 6),
    }

    out = net.get_action(sel_obs, k_obs)
    print(f'\nAction:')
    print(f'  k_index: {out["k_index"]}')
    print(f'  conf_s4: {out["conf_s4"]}')
    print(f'  conf_cgf: {out["conf_cgf"]}')
    print(f'  conf_rofs: {out["conf_rofs"]}')
    print(f'  s4_signal: {out["s4_signal"]}')
    print(f'  joint_log_prob: {out["joint_log_prob"]}')
