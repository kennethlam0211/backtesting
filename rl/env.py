"""Trading Environment v2: Strategy meta-learner + 1min U/D exit.

Architecture:
    60min: Network selects top-5 variants per family via attention.
           Weighted consensus → direction + size.
           magnitude_raw → K for 1min U/D.
    1min:  U/D state machine tracks trend with chosen K.
           Stop-loss based on last_u / last_d.
           Re-enter when trend resumes.

Action space: [k_index, conf_per_family × len(STRAT_FAMILIES)]
    K = K_OPTIONS[k_index] from [40, 70, 100, 150, 200]
    e.g. 6 families → 7 dims: [k_idx, conf_s4, conf_cgf, conf_rofs, conf_rsi, conf_bb, conf_s3]

Observation (keys iterate params.STRAT_FAMILIES = S4/CGF/ROFS/RSI/BB/S3):
    {fam}_variants:   (72, 3) — snapshot [bar_pnl_raw, side, ewm_winrate] sliced from pkl cols [0,1,2]
    pkl underlying:   (N, 72, 5) — [bar_pnl_raw, side, ewm_winrate, ewm_mean, ewm_sharpe]
    {fam}_metrics_ts: (20, 72, 3) — time series [ewm_mean, ewm_winrate, ewm_sharpe], RL mode only
    vol_60:           (60, 6) — 60 × 1min vol bars before consensus step
    pfolio_info:      (60, 18) — 12 portfolio + 6 vol
    session_summary:  (13,) — 9 base + target_mdd_ratio + target_size + nts
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
import numpy as np
from gym import spaces
from pm import PM
from utils.app_logger import get_logger
import params

logger = get_logger('env', level=params.LOG_LEVEL)


def symlog(x):
    """Signed log: handles positive and negative values."""
    return np.sign(x) * np.log1p(abs(x))


class TradingEnvV2(gym.Env):

    NORMALIZATION_FACTOR = params.NORM_FACTOR  # 320
    K_OPTIONS = [50, 100, 150, 200, 250]  # discrete K choices
    MDD_BUFFER_CAP = 0.2  # below this → forced 1 micro
    MAX_REENTER_AFTER_STOP = 0  # 0 = no re-entry, 1 = allow once, etc.
    MICRO_DIVISOR = 3   # max_micro = scaled_contracts * MICRO_PER_MINI / this
    CONFIDENCE_THRESHOLD = 0.1  # below this → "sitting out", reward = 0
    SIM_DAILY_MDD_THRESHOLD = -500  # 1-micro daily MDD ≤ this → trading locked for the day

    # Reward weights
    W_STEP = 1          # activation mode
    W_STATE = 1.5       # activation mode
    W_PNL = 5           # exam: r_pnl weight (immediate PnL)
    W_DELTA = 2         # exam: delta_state weight (trajectory)

    # Humble mdd_buffer wrap divisor — smaller = more aggressive wrapping (forces frequent reset)
    EXAM_CYCLE_DIV = 1
    ACTIVATION_CYCLE_DIV = 1

    def __init__(self, data, data_pool=None, env_id=0, num_envs=12, obs_noise_std=0.0, mode='eval', exam_success=0, exam_counts=0):
        super().__init__()
        if env_id == 0 and not hasattr(TradingEnvV2, '_config_logged'):
            from topstep import ExamSimulator as _ES
            logger.info(f'EnvV2 config: selector={params.SELECTOR_MODE} K_OPTIONS={self.K_OPTIONS} '
                       f'MDD_BUFFER_CAP={self.MDD_BUFFER_CAP} CONFIDENCE_THRESHOLD={self.CONFIDENCE_THRESHOLD} '
                       f'MAX_REENTER={self.MAX_REENTER_AFTER_STOP} MICRO_DIV={self.MICRO_DIVISOR} '
                       f'SIM_DAILY_MDD_THRESHOLD={self.SIM_DAILY_MDD_THRESHOLD} '
                       f'EXAM_CYCLE_DIV={self.EXAM_CYCLE_DIV} ACTIVATION_CYCLE_DIV={self.ACTIVATION_CYCLE_DIV} '
                       f'W_PNL={self.W_PNL} W_DELTA={self.W_DELTA} W_STEP={self.W_STEP} W_STATE={self.W_STATE}')
            logger.info(f'TopStep config: account={params.ACCOUNT_SIZE} MAX_DAILY_RESETS={_ES.MAX_DAILY_RESETS} '
                       f'BACKTOFUND_LIMIT={_ES.BACKTOFUND_LIMIT} CONSISTENCY={_ES.CONSISTENCY_THRESHOLD} '
                       f'SUBSCRIPTION_DAYS={_ES.SUBSCRIPTION_RENEWAL_DAYS} ACTIVATION_DAYS={_ES.ACTIVATION_SUCCESS_DAYS}')
            TradingEnvV2._config_logged = True
        self.data_pool = data_pool if data_pool is not None else [data]
        self.mode = mode
        self.pm = PM(mode=mode, exam_success=exam_success, exam_counts=exam_counts)
        self.env_id = env_id
        self.num_envs = num_envs
        self.obs_noise_std = obs_noise_std
        self.chunk_idx = env_id % len(self.data_pool)

        self._load_data(data)
        self.current_step = self._get_begin_step()

        # Trading state
        self.side = 0           # PM position: +1 long, -1 short, 0 flat
        self.last_u = 0.0       # U/D watermark high (always tracks)
        self.last_d = 0.0       # U/D watermark low (always tracks)
        self.ud_side = 0        # U/D current direction: +1 uptrend, -1 downtrend, 0 undecided
        self.current_k = 50.0   # current K from RL (locked per 60min)
        # Per-family state — dict keyed by params.STRAT_FAMILIES (S4, CGF, ROFS, RSI, BB, S3)
        self.family_confs = {fam: 0.5 for fam in params.STRAT_FAMILIES}
        self.family_attns = {fam: np.ones(72, dtype=np.float32) / 72.0 for fam in params.STRAT_FAMILIES}
        self._hour_start_acc_pnl = 0.0
        self._last_final_size = 0.0
        self.stop_loss_count = 0
        self.stale_consensus = False
        self._last_r_selector = 0.0
        self.size_capped = False
        self._consensus_step = 0
        self._prev_consensus_step = 0
        # Shadow 1-micro tracker (realized tracked by pm.sim_daily_pnl, updated in execute_trade)
        self.sim_daily_pnl = 0.0          # realized + floating (for MDD tracking)
        self.sim_daily_mdd = 0.0          # lowest sim_daily_pnl of the day
        self.sim_daily_locked = False
        self.ttl_days = 0                 # days elapsed since env.reset() — captured in snapshot, then reset

        # Kelly cache
        self._cached_kelly_ewm = {
            'kelly_f': 0.0, 'win_rate': 0.0, 'mu_norm': 0.0,
            'sigma_norm': 0.0, 'confidence': 0.3, 'n_trades': 0,
        }
        self._prev_pm_r = 0.0
        self._prev_r_state = 0.0
        self.training_progress = 0.0  # 0→1 over training, set by PPO

        # 60min state
        self._last_consensus = 0.0
        self._last_size = 0.0
        self._last_60min_step = -1

        # Spaces
        self.feature_shape = self._get_feature_shape()
        # Action: [magnitude_raw(0-1), conf_per_family × len(STRAT_FAMILIES)]
        # e.g. 6 families → 7 dims: [mag, conf_s4, conf_cgf, conf_rofs, conf_rsi, conf_bb, conf_s3]
        _action_dim = 1 + len(params.STRAT_FAMILIES)
        self.action_space = spaces.Box(
            low=np.zeros(_action_dim), high=np.ones(_action_dim), dtype=np.float32)
        self.pfolio_history = np.zeros(self.feature_shape['pfolio_info'], dtype=np.float32)
        # Variant time series: (20, 72, 3) per family — channels = [ewm_mean, ewm_winrate, ewm_sharpe]
        if params.SELECTOR_MODE == 'rl':
            self.family_metrics_ts = {fam: np.zeros((20, 72, 3), dtype=np.float32)
                                      for fam in params.STRAT_FAMILIES}
        elif params.SELECTOR_MODE == 'kalman':
            # Kalman state per variant: [mean, variance] for quality tracking
            # Q=process noise, R=observation noise
            self._kf_Q = 0.1   # how fast true quality changes
            self._kf_R = 1.0   # observation noise
            self._kf_state = {}  # {'{fam}_variants': (mean(72,), var(72,))}
            for fam in params.STRAT_FAMILIES:
                self._kf_state[f'{fam}_variants'] = (
                    np.zeros(72, dtype=np.float32), np.ones(72, dtype=np.float32))

        # Long-horizon hourly buffers for regime detection (one entry per consensus boundary).
        # Used to compute family-trajectory deltas at horizons [20, 60, 120, 200] hours back,
        # plus a global vol ratio at 20h. Independent of SELECTOR_MODE — always built.
        self._regime_horizons = (20, 60, 120, 200)
        self._regime_buf_len = max(self._regime_horizons) + 1
        self._regime_warmup_count = 0       # increments at each consensus boundary
        self.regime_ewm_mean_hist   = {fam: np.zeros(self._regime_buf_len, dtype=np.float32)
                                       for fam in params.STRAT_FAMILIES}
        self.regime_ewm_sharpe_hist = {fam: np.zeros(self._regime_buf_len, dtype=np.float32)
                                       for fam in params.STRAT_FAMILIES}
        self.regime_vol_hist = np.zeros(self._regime_buf_len, dtype=np.float32)

    @property
    def regime_warmed(self):
        """True when long-horizon buffers have ≥200 valid entries (no leading zeros for horizons)."""
        return self._regime_warmup_count >= max(self._regime_horizons)

    def _load_data(self, np_data):
        """Load pre-converted numpy data."""
        self._np_ohlc = np_data['ohlc']
        self._np_features = {k: v for k, v in np_data.items()
                             if k not in ('ohlc',)}
        self.n_steps = len(self._np_ohlc)
        self._np_day = self._np_ohlc[:, 5]  # day column for day-end detection

        # Data already stacked by convert_to_numpy — no re-stacking needed

        # No precomputed quality — temporal encoder consumes 2 channels directly:
        # ch0 = ewm_sharpe (col 2), ch1 = unr + rpnl (col 0 + col 4, both EWM-smoothed)

    def _get_begin_step(self):
        """Find the second boundary. Sets _prev_consensus_step to the first.
        This way consensus reward has a valid baseline from the very first step."""
        day_changes = np.where(np.diff(self._np_day) != 0)[0]
        start = max(int(day_changes[0]) if len(day_changes) > 0 else 60, 60)
        boundaries = []
        # Use the first family in STRAT_FAMILIES as the grid reference — all families share 60-min grid
        sample_key = f'{params.STRAT_FAMILIES[0]}_variants'
        if sample_key in self._np_features:
            for i in range(start, self.n_steps):
                curr = self._np_features[sample_key][i]
                if isinstance(curr, np.ndarray) and np.any(curr != 0):
                    boundaries.append(i)
                    if len(boundaries) == 2:
                        self._prev_consensus_step = boundaries[0]
                        return boundaries[1]
        # Fallback: use first boundary for both if only one found
        if boundaries:
            self._prev_consensus_step = boundaries[0]
            return boundaries[0]
        self._prev_consensus_step = start
        return start

    def _get_feature_shape(self):
        shapes = {
            'vol_60': (60, 6),         # 60 × 1min bars ending at consensus step
            'pfolio_info': (60, 18),   # 12 portfolio + 6 vol
            'session_summary': (13,),  # 9 base + target_mdd_ratio + target_size + nts + rth
        }
        for fam in params.STRAT_FAMILIES:
            shapes[f'{fam}_variants'] = (72, 5)   # [bar_pnl_norm, side, ewm_winrate, ewm_mean, ewm_sharpe]
        if params.SELECTOR_MODE == 'rl':
            for fam in params.STRAT_FAMILIES:
                shapes[f'{fam}_metrics_ts'] = (20, 72, 3)  # [ewm_mean, ewm_winrate, ewm_sharpe]
        # Regime features: per-family [4 ewm_mean deltas + 1 sharpe slope] + global [vol ratio]
        n_fam = len(params.STRAT_FAMILIES)
        shapes['regime_features'] = (n_fam * 5 + 1,)
        return shapes

    def _get_regime_features(self):
        """Compute (n_fam*5 + 1,) flat regime vector from long-horizon hourly buffers.

        Per family (5 features): 4 ewm_mean deltas at horizons [20,60,120,200] + sharpe slope (last 20).
        Global (1): log vol ratio (log(vol[-1] / vol[-21])) — bounded even at extreme ratios.
        All values clipped to [-3, 3] to keep network inputs in stable range.
        """
        out = []
        for fam in params.STRAT_FAMILIES:
            mh = self.regime_ewm_mean_hist[fam]
            sh = self.regime_ewm_sharpe_hist[fam]
            for h in self._regime_horizons:
                out.append(float(mh[-1] - mh[-1 - h]))
            last20 = sh[-20:]
            slope = float(np.polyfit(np.arange(len(last20)), last20, 1)[0]) if last20.size >= 2 else 0.0
            out.append(slope)
        vh = self.regime_vol_hist
        # log-ratio (robust to small denominator) instead of raw ratio
        log_vol = float(np.log((vh[-1] + 1e-3) / (vh[-21] + 1e-3)))
        out.append(log_vol)
        arr = np.array(out, dtype=np.float32)
        return np.clip(arr, -3.0, 3.0)

    @property
    def observation_space(self):
        return spaces.Dict({
            feat: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
            for feat, shape in self.feature_shape.items()
        })

    def _get_vol_obs(self):
        """Get 60 × 1min bars before consensus step (no peek)."""
        return self._np_features['vol_60'][self._consensus_step - 60:self._consensus_step]

    def _get_strategy_obs(self):
        """Get strategy variant observations (72, 5) per family from the last consensus boundary.
        Layout: [bar_pnl_raw/NORM_FACTOR, side, ewm_winrate, ewm_mean, ewm_sharpe].
        bar_pnl_raw normalized so all 5 features are NN-friendly scale."""
        obs = {}
        for fam in params.STRAT_FAMILIES:
            key = f'{fam}_variants'
            if key in self._np_features:
                # Cols 0/3/4 already z-scored in post_preprocessing.standardize_variant_signals
                # → no further /NORM_FACTOR needed.
                arr = self._np_features[key][self._consensus_step, :, :5].copy()
                obs[key] = arr
            else:
                obs[key] = np.zeros((72, 5), dtype=np.float32)
        return obs

    def _get_obs(self, session_summary_=None):
        """Build full observation dict (called at 60min only)."""
        strat_obs = self._get_strategy_obs()

        obs = {
            'vol_60': self._get_vol_obs(),
            'pfolio_info': self.pfolio_history.copy(),
            'session_summary': session_summary_ if session_summary_ is not None
                               else np.zeros(self.feature_shape['session_summary'], dtype=np.float32),
        }
        for fam in params.STRAT_FAMILIES:
            obs[f'{fam}_variants'] = strat_obs[f'{fam}_variants']
        if params.SELECTOR_MODE == 'rl':
            for fam in params.STRAT_FAMILIES:
                obs[f'{fam}_metrics_ts'] = self.family_metrics_ts[fam].copy()
        obs['regime_features'] = self._get_regime_features()

        if self.obs_noise_std > 0:
            for key in obs:
                if key not in ('pfolio_info', 'session_summary'):
                    obs[key] = obs[key] + np.random.normal(
                        0, self.obs_noise_std, obs[key].shape).astype(np.float32)

        # Extreme-value clip: guard NN inputs. Tightened to ±3 — values beyond
        # this range hurt training stability (gradient explosions, BN/LN drift).
        # ── TEMP DIAGNOSTIC ── remove once ill-scaled features are identified.
        # Tracks per-key clip events + max_abs seen; periodic dump every 5000 calls.
        if not hasattr(TradingEnvV2, '_clip_diag'):
            TradingEnvV2._clip_diag = {}   # {key: [n_clipped_total, max_abs_seen]}
            TradingEnvV2._clip_calls = 0
        TradingEnvV2._clip_calls += 1
        for key in obs:
            arr = obs[key]
            # NaN/Inf guard — must precede stats and clip (max of NaN-containing arr is NaN)
            if not np.all(np.isfinite(arr)):
                arr = np.nan_to_num(arr, nan=0.0, posinf=3.0, neginf=-3.0)
                obs[key] = arr
            max_abs = float(np.abs(arr).max()) if arr.size else 0.0
            n_clip = int((np.abs(arr) > 3.0).sum())
            stat = TradingEnvV2._clip_diag.setdefault(key, [0, 0.0])
            stat[0] += n_clip
            if max_abs > stat[1]:
                stat[1] = max_abs
            np.clip(arr, -3.0, 3.0, out=arr)
        if self.env_id == 0 and TradingEnvV2._clip_calls % 5000 == 0:
            top = sorted(TradingEnvV2._clip_diag.items(), key=lambda kv: -kv[1][0])[:8]
            logger.info(f'[CLIP DIAG @ call={TradingEnvV2._clip_calls}] '
                        + ' '.join(f'{k}: n={v[0]} max={v[1]:.2f}' for k, v in top if v[0] > 0))
        # ── END TEMP DIAGNOSTIC ──
        return obs

    def step(self, action):
        """Gym API: one call = one 60min decision. Runs ~60 × 1min bars internally.

        Action: [k_index, conf_per_family × len(STRAT_FAMILIES)] all ∈ (0,1)
        e.g. 6 families → 7 dims: [k_idx, conf_s4, conf_cgf, conf_rofs, conf_rsi, conf_bb, conf_s3]
        Returns: (next_obs, reward, done, truncated, infos)
        """
        # ── 1. Apply action: lock K + per-family confidences ─────
        k_idx = int(action[0])  # categorical index into K_OPTIONS
        self.current_k = self.K_OPTIONS[min(k_idx, len(self.K_OPTIONS) - 1)]
        for i, fam in enumerate(params.STRAT_FAMILIES):
            self.family_confs[fam] = float(action[1 + i])

        # ── 2. Update consensus weighted by per-family confidence ─
        self._prev_consensus_step = self._consensus_step
        self._consensus_step = self.current_step
        prev_consensus_sign = int(np.sign(self._last_consensus)) if self._last_consensus != 0 else 0
        self._update_consensus_from_variants()

        prev_side = self.ud_side
        # Reset U/D watermarks only when consensus direction changes
        if self._last_consensus != 0:
            target_side = int(np.sign(self._last_consensus))
            if target_side != prev_side:
                self.ud_side = target_side
                first_open = self._np_ohlc[self.current_step + 1, 1]
                if target_side == 1:
                    self.last_u = first_open
                    self.last_d = 0.0
                else:
                    self.last_d = first_open
                    self.last_u = 90000
        else:
            self.ud_side = 0
            self.last_u = 90000
            self.last_d = 0.0

        hour_step_pnl = 0.0
        hour_start_floating = 0.0
        self._hour_start_floating_per_con = 0.0
        self._hour_hold_entry = None
        self._hour_hold_pnl = 0.0
        self._hour_rpnl_per_con = 0.0
        self._hour_stop_count = 0  # track stops within hour for re-entry limit
        self._hour_stopped_at_px = None  # first stop price this hour (for vol reward)
        is_first_minute = True
        done = False

        while True:
            was_stale = self.stale_consensus
            self.stale_consensus = not self._any_strategy_has_signal() or self.pm.topstep.daily_locked
                # (self.mode != 'train_activation' and )

            # Reset pfolio + update consensus step when transitioning stale→active
            if was_stale and not self.stale_consensus:
                self.pfolio_history[:] = 0
                self._consensus_step = self.current_step

            # Run 1min bars for this hour
            while True:

                trade_info, bar_done = self._step_1min()

                if not self.stale_consensus:
                    if is_first_minute:
                        is_first_minute = False
                        # Capture floating after PM executes at open
                        hour_start_floating = self.pm.cal_floating_pnl(self._np_ohlc[self.current_step, 1])
                        pos = abs(self.pm.position_size_equivalent)
                        self._hour_start_floating_per_con = hour_start_floating / pos if pos > 0 else 0.0
                        # Include direction change close, exclude stop (previous hour's K)
                        pnl_ex_stop = trade_info.get('net_pnl', 0) - trade_info.get('stop_net_pnl', 0)
                        hour_step_pnl += pnl_ex_stop
                    else:
                        hour_step_pnl += trade_info.get('net_pnl', 0)
                        self._hour_rpnl_per_con += trade_info.get('net_pnl_per_con', 0)
                    # Track hold entry for vol reward (consensus direction, not PM side)
                    if self._hour_hold_entry is None and self._last_consensus != 0:
                        self._hour_hold_entry = (self._np_ohlc[self.current_step, 1], int(np.sign(self._last_consensus)))
                if bar_done:
                    done = True
                    break
                if self._is_next_60min():
                    break

            if done or not self.stale_consensus:
                break

            # Still stale — advance to next boundary and check again
            self._update_consensus_from_variants()


        # ── 4. Compute rewards per actor ───────────────────────────
        floating_delta = trade_info.get('floating_pnl', 0) - hour_start_floating
        hour_total = hour_step_pnl + floating_delta

        # Hold PnL per contract: entry→close for 1 micro, net of commission
        if self._hour_hold_entry is not None:
            entry_px, direction = self._hour_hold_entry
            close_px = self._np_ohlc[self.current_step, 4]
            self._hour_hold_pnl = direction * (close_px - entry_px) * self.pm.TICK_VALUE_MICRO - (self.pm.COMMISSION_MICRO/self.pm.MICRO_PER_MINI)

        r_pm = self._compute_porf_reward(trade_info, hour_total)
        r_consensus_display = self._compute_consensus_reward()            # always compute for comparison
        r_consensus = r_consensus_display if params.SELECTOR_MODE == 'rl' else 0.0  # only feed to PPO when rl
        r_conf = self._compute_conf_reward(trade_info)        # confidence-specific
        r_vol = self._compute_vol_reward(trade_info)                    # K-specific


        self._last_r_pm = r_pm
        self._last_r_selector = r_consensus  # 0 when not rl mode — PPO sees this
        self._last_r_selector_display = r_consensus_display  # always real — for logging
        self._last_r_confidence = r_conf  # raw conf only, PPO combines with normalized r_pm
        self._last_r_k = r_vol            # raw vol only, PPO combines with normalized r_pm
        reward = r_pm

        # ── 5. Build next obs ────────────────────────────────────
        kelly_stats = {**self._cached_kelly_ewm, 'mdd_buffer': self._cached_mdd_buffer}
        session_summary = self._build_session_summary(trade_info, kelly_stats)
        obs = self._get_obs(session_summary_=session_summary)

        infos = self._build_infos(trade_info, hour_total)
        # Snapshot end-of-episode topstep + env stats BEFORE auto-reset wipes them.
        # PPO reads info['final_episode_stats'] when done=True.
        if done:
            infos['final_episode_stats'] = self._snapshot_episode_stats()
        return obs, reward, done, False, infos

    def _snapshot_episode_stats(self):
        """Capture all stats PPO reads at end-of-rollout, before reset() zeros them."""
        ts = self.pm.topstep
        return {
            'exam_success': int(ts.exam_success),
            'exam_counts':  int(ts.exam_counts),
            'pass_exam':    bool(ts.pass_exam),
            'days_taken_to_pass_exams':         list(ts.days_taken_to_pass_exams),
            'days_taken_to_fail_exams':         list(ts.days_taken_to_fail_exams),
            'days_taken_to_fail_activation':    list(ts.days_taken_to_fail_activation),
            'exam_daily_target_hit_day_list':   list(ts.exam_daily_target_hit_day_list),
            'exam_daily_resets_list':           list(ts.exam_daily_resets_list),
            'payout_cnt':       int(ts.payout_cnt),
            'activation_cnt':   int(ts.activation_cnt),
            'payouts':          float(ts.payouts),
            'topstep_cost':     float(ts.topstep_cost),
            'acc_pnl':          float(ts.acc_pnl),
            'total_resets':     int(ts.total_resets),
            'total_commission': float(ts.total_commission),
            'final_pnl_list':   list(ts.final_pnl_list),
            'back_to_fund_list': list(ts.back_to_fund_list),
            'days_taken_to_payouts': list(ts.days_taken_to_payouts),
            'stop_loss_count':       int(self.stop_loss_count),
            'sim_net_pnl_per_con_list': list(self.pm.sim_net_pnl_per_con_list),
            'ttl_days':              int(self.ttl_days),
        }

    def _step_1min(self):
        """Internal: advance one 1min bar. No RL action.
        PM always called — handles close, topstep, day_end for all cases.
        Returns: (trade_info, done)
        """
        self.current_step += 1

        # Parse current bar
        _, open_px, high_px, low_px, close_px = self._np_ohlc[self.current_step, :5]
        # Detect bar gap
        # Bar gap detection (debug only)
        # if self.current_step > 0:
        #     prev_close = self._np_ohlc[self.current_step - 1, 4]
        #     prev_day = self._np_day[self.current_step - 1]
        #     curr_day = self._np_day[self.current_step]
        #     gap = abs(open_px - prev_close)
        #     gap_type = 'session' if prev_day != curr_day else '1min'
        #     if gap > 50:
        #         logger.warning(f'BAR GAP: step={self.current_step} ts={self._np_ohlc[self.current_step, 0]:.0f} '
        #                       f'prev_close={prev_close:.0f} open={open_px:.0f} gap={gap:.0f} '
        #                       f'type={gap_type} pos={self.pm.position_size_equivalent}')
        nts = self._np_ohlc[self.current_step, 9]
        rth = self._np_ohlc[self.current_step, 11]
        next_open = self._np_ohlc[min(self.current_step + 1, self.n_steps - 1), 1]
        hl = int(self._np_ohlc[self.current_step, 12])
        k_pts = max(self.current_k  , 10)

        # Detect day end: last bar of the day or end of data
        next_step = self.current_step + 1
        done = next_step >= self.n_steps
        day_end = done or (self._np_day[self.current_step] != self._np_day[next_step])

        # ── Determine side from U/D + consensus ─────────────────
        target_side = int(np.sign(self._last_consensus))
        target_size = abs(self._last_consensus)

        ud_matches = (self.ud_side == target_side) and (target_side != 0)

        # Lock: daily target hit or max resets or sim_daily_mdd breach — stay flat
        if self.pm.topstep.daily_locked: #self.mode != 'train_activation' and 
            new_side = 0
        elif self.sim_daily_locked:
            new_side = 0  # sim 1-micro daily MDD breached — done for the day
        elif self._hour_stop_count >= (self.MAX_REENTER_AFTER_STOP + 1) and self.side == 0:
            new_side = 0  # no re-entry after max stops
        else:
            new_side = target_side if ud_matches else 0

        if new_side != self.side or self.side != 0 or day_end:
            if new_side == 0:
                # Going flat (lock / close / day_end) — skip sizing, pm_step handles close_all
                final_size = 0
                self.size_capped = True
            else:
                final_size = self._get_final_size(target_size, day_end, k_pts)
            self._last_final_size = final_size
            self.side = new_side
            trade_info = self.pm.pm_step(
                open_px, high_px, low_px, close_px, hl,
                new_side, final_size, day_end, next_open,
                stop_win=99999, stop_loss=k_pts,
                last_u=self.last_u if self.last_u > 0 else None,
                last_d=self.last_d if self.last_d > 0 else None,
                rth=rth,
            )

            # Log position sanity check
            pos = self.pm.position_size_equivalent
            if abs(pos) > 0:
                logger.debug(f'POS: step={self.current_step} side={self.side} pos={pos} size={final_size:.1f} '
                            f'entry={self.pm.avg_entry_px:.0f} open={open_px:.0f} close={close_px:.0f} '
                            f'acc={self.pm.topstep.acc_pnl:.0f} mdd_buf={self._cached_mdd_buffer:.2f} '
                            f'K={k_pts:.0f} stopped={trade_info.get("was_stopped", False)}')

            if trade_info.get('was_stopped', False):
                self.side = 0
                self.stop_loss_count += 1
                self._hour_stop_count += 1
                # Capture the first stop price this hour for vol reward (1-micro comparison)
                if self._hour_stopped_at_px is None:
                    self._hour_stopped_at_px = trade_info.get('stop_px')

        else:
            trade_info = self._empty_trade_info()

        # ── Update U/D watermarks from current close ─────────
        if target_side != 0:
            if self.ud_side == 1:
                self.last_u = max(self.last_u, close_px)
                if self.last_u - close_px >= k_pts:
                    self.ud_side = -1
                    self.last_d = close_px
                    self.last_u = 0.0
            elif self.ud_side == -1:
                self.last_d = min(self.last_d, close_px) if self.last_d > 0 else close_px
                if close_px - self.last_d >= k_pts:
                    self.ud_side = 1
                    self.last_u = close_px
                    self.last_d = 0.0
            else:
                self.ud_side = target_side
                if target_side == 1:
                    self.last_u = close_px
                else:
                    self.last_d = close_px

        # ── Shadow 1-micro tracker: pm.execute_trade updates pm.sim_daily_pnl on
        # every transition. We just add current floating for MDD tracking.
        if self.pm.sim_entry_px is not None and self.pm.position_size_equivalent != 0:
            direction = 1 if self.pm.position_size_equivalent > 0 else -1
            sim_floating = (close_px - self.pm.sim_entry_px) * direction * self.pm.TICK_VALUE_MICRO
        else:
            sim_floating = 0.0
        self.sim_daily_pnl = self.pm.sim_daily_pnl + sim_floating
        self.sim_daily_mdd = min(self.sim_daily_mdd, self.sim_daily_pnl)
        if self.sim_daily_mdd < self.SIM_DAILY_MDD_THRESHOLD:
            self.sim_daily_locked = True

        if day_end:
            # PM close_all at day_end already realized into pm.sim_daily_pnl.
            # Reset sim state for next day.
            self.pm.sim_daily_pnl = 0.0
            self.sim_daily_pnl = 0.0
            self.sim_daily_mdd = 0.0
            self.sim_daily_locked = False
            self.ttl_days += 1

        # Reset sim state on exam reset too (not just day_end)
        if trade_info.get('reset_account', False):
            self.sim_daily_pnl = 0.0
            self.sim_daily_mdd = 0.0
            self.sim_daily_locked = False

        # ── Roll pfolio_info ─────────────────────────────────────
        self._cached_mdd_buffer, self._cached_humble_mdd_buffer = self._get_mdd_buffer(trade_info['floating_pnl'])
        kelly_stats = {**self._cached_kelly_ewm, 'mdd_buffer': self._cached_mdd_buffer}

        vol = self._np_features['vol_60'][self.current_step]
        pfolio_info = np.array([
            trade_info.get('floating_pnl', 0) / self.NORMALIZATION_FACTOR,
            trade_info.get('net_pnl_per_con', 0) / self.NORMALIZATION_FACTOR,
            trade_info.get('step_pnl', 0) / self.NORMALIZATION_FACTOR,
            trade_info.get('position_usage', 0),
            rth,
            nts,
            kelly_stats['kelly_f'],
            kelly_stats['mdd_buffer'],
            kelly_stats['confidence'],
            kelly_stats['win_rate'],
            self._last_consensus,  # [-1, +1]
            float(self.side),
            *vol,  # 6 vol features
        ], dtype=np.float32)

        self.pfolio_history[:-1] = self.pfolio_history[1:]
        self.pfolio_history[-1] = pfolio_info

        return trade_info, done

    def _is_next_60min(self):
        """Check if the current bar is a 60min boundary (has strategy data).
        Strategy data appears at the 59th minute (end of hour)."""
        if self.current_step >= self.n_steps - 1:
            return True
        # Use the first family in STRAT_FAMILIES as the reference — all families
        # share the same 60-min grid by construction.
        sample_key = f'{params.STRAT_FAMILIES[0]}_variants'
        if sample_key not in self._np_features:
            return False
        curr = self._np_features[sample_key][self.current_step]
        if isinstance(curr, np.ndarray):
            return np.any(curr != 0) and not np.any(np.isnan(curr))
        return False

    def _empty_trade_info(self):
        """Return a default trade_info dict when skipping (no PM activity)."""
        return {
            'floating_pnl': 0, 'per_contract_pnl': 0, 'step_pnl': 0,
            'net_pnl': 0, 'position_usage': 0,
            'acc_pnl': self.pm.topstep.acc_pnl,
            'mdd_limit': self.pm.topstep.mdd_limit,
            'success_activation_days': getattr(self.pm.topstep, 'success_activation_days', 0),
            'subscription_days': getattr(self.pm.topstep, 'subscription_days', 0),
        }

    def _build_infos(self, trade_info, hour_total):
        """Build info dict for step() return."""
        infos = {
            'acc_pnl': self.pm.topstep.acc_pnl,
            'payouts': self.pm.topstep.payouts,
            'topstep_cost': self.pm.topstep.topstep_cost,
            'exam_pass_rate': self.pm.topstep.exam_success / max(self.pm.topstep.exam_counts, 1),
            'exam_counts': self.pm.topstep.exam_counts,
            'total_commission': self.pm.topstep.total_commission,
            'consensus': self._last_consensus,
            'k_pts': self.current_k,
            'side': self.side,
            'position': self.pm.position_size_equivalent,
            'stop_loss_count': self.stop_loss_count,
            'hour_pnl': hour_total,
            'r_pm': self._last_r_pm,
            'r_selector': self._last_r_selector,
            'r_selector_display': self._last_r_selector_display,
            'r_confidence': self._last_r_confidence,
            'r_k': self._last_r_k,
            'final_size': self._last_final_size,
        }
        # Per-family confidences (conf_s4, conf_cgf, conf_rofs, conf_rsi, conf_bb, conf_s3)
        for fam in params.STRAT_FAMILIES:
            infos[f'conf_{fam.lower()}'] = self.family_confs[fam]
        return infos

    def _any_strategy_has_signal(self):
        """Check if ANY strategy variant (across all families) has a non-zero signal at current step."""
        for fam in params.STRAT_FAMILIES:
            key = f'{fam}_variants'
            if key not in self._np_features:
                continue
            signals = self._np_features[key][self.current_step, :, 1]  # (72,) signal column
            if np.any(signals != 0):
                return True
        return False

    def _update_consensus_from_variants(self):
        """Consensus weighted by per-family confidence and per-variant attention.
        - rl mode: attn comes from network's full softmax over all 72 variants
                   (set externally via env.family_attns[fam] = network_attn)
        - kalman / deterministic modes: top-5 attn computed here from ewm_sharpe
        Per-family signal = (attn · variant_signals) is then scaled by confidence and
        summed across all params.STRAT_FAMILIES into self._last_consensus ∈ [-1, +1]."""
        total_weighted = 0.0
        total_weight = 0.0

        for fam in params.STRAT_FAMILIES:
            key = f'{fam}_variants'
            if key not in self._np_features:
                continue
            conf = self.family_confs[fam]
            attn = self.family_attns[fam]

            feats = self._np_features[key][self.current_step]  # (72, 5)
            # Layout: [bar_pnl_raw, side, ewm_winrate, ewm_mean, ewm_sharpe]
            signals       = feats[:, 1]
            ewm_winrate   = feats[:, 2]
            ewm_mean      = feats[:, 3]
            ewm_sharpe    = feats[:, 4]

            if params.SELECTOR_MODE == 'rl':
                # Roll 3-channel time series buffer: [mean, winrate, sharpe]
                ts_buf = self.family_metrics_ts[fam]               # (20, 72, 3)
                ts_buf[:-1] = ts_buf[1:]
                ts_buf[-1, :, 0] = ewm_mean
                ts_buf[-1, :, 1] = ewm_winrate
                ts_buf[-1, :, 2] = ewm_sharpe
            elif params.SELECTOR_MODE == 'kalman':
                # Kalman filter on ewm_sharpe (canonical risk-adjusted score)
                kf_mean, kf_var = self._kf_state[key]
                kf_var_pred = kf_var + self._kf_Q
                K = kf_var_pred / (kf_var_pred + self._kf_R)
                kf_mean = kf_mean + K * (ewm_sharpe - kf_mean)
                kf_var = (1 - K) * kf_var_pred
                self._kf_state[key] = (kf_mean, kf_var)
                kf_score = kf_mean / np.sqrt(kf_var + 1e-8)
                top5_idx = np.argsort(kf_score)[-5:]
                attn = np.zeros(72, dtype=np.float32)
                attn[top5_idx] = 1.0 / 5.0
                self.family_attns[fam] = attn
            else:
                # Deterministic top-5 by ewm_sharpe
                top5_idx = np.argsort(ewm_sharpe)[-5:]
                attn = np.zeros(72, dtype=np.float32)
                attn[top5_idx] = 1.0 / 5.0
                self.family_attns[fam] = attn

            # Long-horizon regime buffers — family-level aggregates (mean across 72 variants).
            # One entry per consensus boundary; used to compute multi-horizon trajectory deltas.
            mh = self.regime_ewm_mean_hist[fam]
            sh = self.regime_ewm_sharpe_hist[fam]
            mh[:-1] = mh[1:]
            mh[-1] = float(ewm_mean.mean())
            sh[:-1] = sh[1:]
            sh[-1] = float(ewm_sharpe.mean())

            # Env-side momentum safety net. Cols 0/3/4 are z-scored (post_preprocessing)
            # so all thresholds are in sigma units (positive = above non-zero mean).
            # ewm_mean / ewm_sharpe thresholds raised to 0.01 → require meaningfully
            # positive signal (not just barely above zero) for full keep.
            bar_pnl_raw = feats[:, 0]
            mask_ewm    = np.where(ewm_mean    >= 0.01, 1.0, 0.10).astype(np.float32)
            mask_bar    = np.where(bar_pnl_raw >= 0,    1.0, 0.50).astype(np.float32)
            mask_sharpe = np.where(ewm_sharpe  >= 0.01, 1.0, 0.70).astype(np.float32)
            attn = attn * (mask_ewm * mask_bar * mask_sharpe)
            if attn.sum() > 1e-8:
                attn = attn / attn.sum()
            self.family_attns[fam] = attn

            attn_sum = attn.sum() + 1e-8
            avg_fam_signal = (attn * signals).sum() / attn_sum
            effective_conf = conf if conf >= self.CONFIDENCE_THRESHOLD else 0.0
            total_weighted += avg_fam_signal * effective_conf
            total_weight += effective_conf

        if total_weight > 0:
            self._last_consensus = total_weighted / total_weight  # [-1, +1]
        else:
            self._last_consensus = 0.0

        # Global hourly vol regime buffer — one entry per consensus boundary.
        # Use mean of 6 vol features × 60 1min bars at this consensus step.
        vol_window = self._np_features['vol_60'][self._consensus_step - 60:self._consensus_step]
        vol_hour = float(vol_window.mean()) if vol_window.size else 0.0
        self.regime_vol_hist[:-1] = self.regime_vol_hist[1:]
        self.regime_vol_hist[-1] = vol_hour
        self._regime_warmup_count += 1   # one consensus boundary processed

    # Consensus reward multiplier (single signal: mean-centered bar_pnl_raw / NORM_FACTOR).
    # Gates do most of the magnitude work now — keep MULT modest (1.0).
    CONSENSUS_REWARD_MULT = 1.0

    def _compute_consensus_reward(self):
        """Reward for model selection: mean-centered advantage on bar_pnl_raw.

        Per family:
          q_bar = (bar_pnl_raw / NORM_FACTOR) - mean(bar_pnl_raw / NORM_FACTOR)
          fam_r = MULT × (attn · q_bar)

        Why bar_pnl_raw only:
        - High cross-variant variance (each variant holds different positions per bar)
        - ewm_mean and ewm_sharpe are smoothed → low cross-variant variance → mostly noise
        - Single clean signal is cleaner gradient than 3-signal weighted sum
        - Mean-centered: uniform attn → 0; selector picking above-mean variants → positive rs
        """
        if self.sim_daily_locked:
            return 0.0

        total_reward = 0.0
        for fam in params.STRAT_FAMILIES:
            key = f'{fam}_variants'
            if key not in self._np_features:
                continue
            attn = self.family_attns[fam]
            feats = self._np_features[key][self._consensus_step]  # (72, 5)

            q_raw = feats[:, 0]   # already z-scored in post_preprocessing
            if not np.any(q_raw != 0):
                continue

            q = q_raw - q_raw.mean()
            fam_reward = self.CONSENSUS_REWARD_MULT * float((attn * q).sum())
            total_reward += fam_reward
            logger.debug(f'sel_reward {fam}: q_std={q.std():.3f} r={fam_reward:+.4f}')
        logger.debug(f'sel_reward total={total_reward:+.4f}')
        return total_reward

    def _compute_porf_reward(self, trade_info, hour_total):
        if self.mode == 'train_activation':
            return self._compute_porf_reward_activation(trade_info, hour_total)
        return self._compute_porf_reward_exam(trade_info, hour_total)

    def _compute_porf_reward_exam(self, trade_info, hour_total):
        """Exam reward: unified equity position + live p_survive + delta shaping."""
        import math
        ts = self.pm.topstep
        _EXP2 = math.e ** 2 - 1.0

        # Real equity for reward (humble is only used for sizing gate)
        equity = trade_info.get('acc_pnl', 0) + trade_info.get('floating_pnl', 0)

        # 1. r_pnl: per-contract when capped at 1 micro, total (with size) otherwise
        floating_per_con = trade_info.get('floating_pnl_per_con', 0) - self._hour_start_floating_per_con
        pnl_per_con = self._hour_rpnl_per_con + floating_per_con
        if self.size_capped:
            r_pnl = pnl_per_con / 10
        else:
            r_pnl = hour_total / self.NORMALIZATION_FACTOR
        r_pnl = max(min(r_pnl, 4.0), -2.0)

        # 2. r_position: equity position, origin=0, range -1 to +1
        day_start_pnl = equity - (ts.daily_pnl + trade_info.get('floating_pnl', 0))
        today_max = day_start_pnl + ts.EXAM_TARGET * ts.CONSISTENCY_THRESHOLD
        effective_target = min(today_max, ts.EXAM_TARGET)
        if equity >= 0:
            position = equity / max(effective_target, 1.0)
        else:
            position = equity / max(abs(ts.mdd_limit), 1.0)
        position = min(max(position, -1.0), 1.0)
        abs_pos = abs(position)
        r_magnitude = (math.exp(2.0 * abs_pos) - 1.0) / _EXP2
        r_position = r_magnitude if position >= 0 else -r_magnitude  # -1 to +1

        # 3. r_ev: survival probability, -1 to +1 (uses sim 1-micro trades for pure signal)
        remaining_target = max(effective_target - equity, 1.0)
        remaining_mdd = max(equity - ts.mdd_limit, 1.0)
        n_trades = self.pm.sim_net_pnl_per_con_list
        mu, sigma = ts._ewm_mu_sigma(n_trades) if len(n_trades) >= 10 else (0, 1)
        p_survive = ts._gambler_ruin(mu, sigma, remaining_target, remaining_mdd, ts.max_contracts) if mu > 0 else 0.1
        r_ev = (2 * p_survive - 1) if len(n_trades) >= 10 else 0.0  # -1 to +1, neutral when no data

        # 4. time_pressure: -1 to 0 (always penalty)
        r_time = -(ts.subscription_days / ts.SUBSCRIPTION_RENEWAL_DAYS)  # 0 to -1

        # Reset pressure: amplifies negative r_position (1.0 to 1.5 based on resets)
        reset_pressure = 1.0 + 0.5 * ts.daily_resets / max(ts.MAX_DAILY_RESETS, 1)
        if r_position < 0:
            r_position = r_position * reset_pressure

        logger.debug(f'exam_reward: capped={self.size_capped} pnl_pc={pnl_per_con:+.2f} '
                     f'hour_total={hour_total:+.2f} r_pnl={r_pnl:+.3f} r_pos={r_position:+.3f} r_ev={r_ev:+.3f} '
                     f'r_time={r_time:+.3f} rst_p={reset_pressure:.2f}')
        # State reward: slow-moving components, delta'd to get change signal
        r_state = r_position + r_ev  # r_time removed: already in obs, constant drag biases Rp negative
        # Reset delta baseline on exam reset (acc_pnl jumps to 0)
        if ts.acc_pnl == 0 and self._prev_r_state != 0:
            self._prev_r_state = 0.0
        delta_state = r_state - self._prev_r_state
        self._prev_r_state = r_state

        # Final: instantaneous pnl + state change
        pm_r = r_pnl * self.W_PNL + delta_state * self.W_DELTA
        return pm_r


    def _compute_porf_reward_activation(self, trade_info, hour_total):
        """Activation reward (mirrors exam structure): r_pnl + delta(r_position + r_ev).
        Daily target = $1000 when success_days < 5 (push for big consistent days);
        once success_days >= 5, payout fires (handled in _handle_new_day)."""
        import math
        ts = self.pm.topstep
        _EXP2 = math.e ** 2 - 1.0

        # Real equity for reward (humble is only used for sizing gate)
        equity = trade_info.get('acc_pnl', 0) + trade_info.get('floating_pnl', 0)

        # 1. r_pnl: per-contract when capped at 1 micro, total (with size) otherwise
        floating_per_con = trade_info.get('floating_pnl_per_con', 0) - self._hour_start_floating_per_con
        pnl_per_con = self._hour_rpnl_per_con + floating_per_con
        if self.size_capped:
            r_pnl = pnl_per_con / 10
        else:
            r_pnl = hour_total / self.NORMALIZATION_FACTOR
        r_pnl = max(min(r_pnl, 4.0), -2.0)

        # 2. r_position: equity position toward effective_target, origin=0, range -1 to +1
        # Daily target = $1000 when success_days < 5, else full ACTIVATION_PNL_TARGET (rarely caps)
        daily_target = 1500 if ts.success_activation_days < 5 else ts.ACTIVATION_PNL_TARGET
        day_start_pnl = equity - (ts.daily_pnl + trade_info.get('floating_pnl', 0))
        today_max = day_start_pnl + daily_target
        effective_target = min(today_max, ts.ACTIVATION_PNL_TARGET)
        if equity >= 0:
            position = equity / max(effective_target, 1.0)
        else:
            position = equity / max(abs(ts.mdd_limit), 1.0)
        position = min(max(position, -1.0), 1.0)
        abs_pos = abs(position)
        r_magnitude = (math.exp(2.0 * abs_pos) - 1.0) / _EXP2
        r_position = r_magnitude if position >= 0 else -r_magnitude

        # 3. r_ev: survival probability, -1 to +1
        remaining_target = max(effective_target - equity, 1.0)
        remaining_mdd = max(equity - ts.mdd_limit, 1.0)
        n_trades = self.pm.sim_net_pnl_per_con_list
        mu, sigma = ts._ewm_mu_sigma(n_trades) if len(n_trades) >= 10 else (0, 1)
        p_survive = ts._gambler_ruin(mu, sigma, remaining_target, remaining_mdd, ts.max_contracts) if mu > 0 else 0.1
        r_ev = (2 * p_survive - 1) if len(n_trades) >= 10 else 0.0

        logger.debug(f'act_reward: capped={self.size_capped} pnl_pc={pnl_per_con:+.2f} '
                     f'hour_total={hour_total:+.2f} r_pnl={r_pnl:+.3f} r_pos={r_position:+.3f} r_ev={r_ev:+.3f} '
                     f'daily_target={daily_target} success_days={ts.success_activation_days}')
        # State reward: slow-moving components, delta'd to get change signal
        r_state = r_position + r_ev
        # Reset delta baseline on activation reset (acc_pnl jumps to 0)
        if ts.acc_pnl == 0 and self._prev_r_state != 0:
            self._prev_r_state = 0.0
        delta_state = r_state - self._prev_r_state
        self._prev_r_state = r_state

        pm_r = r_pnl * self.W_PNL + delta_state * self.W_DELTA
        return pm_r


    # def _compute_porf_reward_activation(self, trade_info, hour_total):
    #     """OLD: Activation reward with EV-dollars + r_success. Commented out for reference.
    #     Replaced with exam-mirror version above (daily target $1000 when success_days<5)."""
    #     import math
    #     ts = self.pm.topstep
    #     _EXP2 = math.e ** 2 - 1.0
    #
    #     equity = trade_info.get('acc_pnl', 0) + trade_info.get('floating_pnl', 0)
    #
    #     # 1. r_pnl
    #     floating_per_con = trade_info.get('floating_pnl_per_con', 0) - self._hour_start_floating_per_con
    #     pnl_per_con = self._hour_rpnl_per_con + floating_per_con
    #     if self.size_capped:
    #         r_pnl = pnl_per_con / 10
    #     else:
    #         r_pnl = hour_total / self.NORMALIZATION_FACTOR
    #     r_pnl = max(min(r_pnl, 4.0), -2.0)
    #
    #     # 2. r_position
    #     if equity >= 0:
    #         position = equity / max(ts.ACTIVATION_PNL_TARGET, 1.0)
    #     else:
    #         position = equity / max(abs(ts.mdd_limit), 1.0)
    #     position = max(position, -1.0)
    #     abs_pos = min(abs(position), 3.0)
    #     r_magnitude = (math.exp(2.0 * abs_pos) - 1.0) / _EXP2
    #     r_position = r_magnitude if position >= 0 else -r_magnitude
    #
    #     # 3. r_ev: dollar-based EV (sign-aware normalization)
    #     remaining_target = max(ts.ACTIVATION_PNL_TARGET - equity, 1.0)
    #     remaining_mdd = max(equity - ts.mdd_limit, 1.0)
    #     n_trades = self.pm.sim_net_pnl_per_con_list
    #     mu, sigma = ts._ewm_mu_sigma(n_trades) if len(n_trades) >= 10 else (0, 1)
    #     p_survive = ts._gambler_ruin(mu, sigma, remaining_target, remaining_mdd, ts.max_contracts) if mu > 0 else 0.1
    #     expected_payout = ts.ACTIVATION_PNL_TARGET * 0.5 * ts.DISCOUNT_FACTOR
    #     reset_cost = ts.reactivation_fees if ts.reactivation_fees > 0 else 599
    #     ev_dollars = p_survive * expected_payout - (1 - p_survive) * reset_cost
    #     if ev_dollars >= 0:
    #         r_ev = (ev_dollars / expected_payout) if len(n_trades) >= 10 else 0.0
    #     else:
    #         r_ev = (ev_dollars / reset_cost) if len(n_trades) >= 10 else 0.0
    #     r_ev = max(min(r_ev, 1.5), -1.5)
    #
    #     # 4. r_success: stacked discrete bonus per success day
    #     current_succ = ts.success_activation_days
    #     prev_succ = getattr(self, '_prev_success_days_cnt', 0)
    #     succ_delta = max(current_succ - prev_succ, 0)
    #     if current_succ == 0 and prev_succ != 0:
    #         self._prev_success_days_cnt = 0
    #     else:
    #         self._prev_success_days_cnt = current_succ
    #     r_success = succ_delta * current_succ  # 1, 4, 9, 16, 25
    #
    #     r_state = r_position + r_ev
    #     if ts.acc_pnl == 0 and self._prev_r_state != 0:
    #         self._prev_r_state = 0.0
    #     delta_state = r_state - self._prev_r_state
    #     self._prev_r_state = r_state
    #
    #     pm_r = r_pnl * self.W_PNL + delta_state * self.W_DELTA + r_success
    #     return pm_r

    def _compute_conf_reward(self, trade_info):
        """Confidence reward per-family: signal alignment with market (raw $/micro).
        Three branches:
          - Dead zone (tiny move): no reward
          - Above confidence threshold: reward = alignment × conf (amplified)
          - Below threshold (sitting out): reward = -alignment (regret/dodge)
        Uses hour_open (from _hour_hold_entry) as reference, raw $ via TICK_VALUE_MICRO.
        """
        if self.sim_daily_locked:
            return 0.0

        # Reference open: hour_hold_entry if active, else consensus step open
        if self._hour_hold_entry is not None:
            hour_open = self._hour_hold_entry[0]
        else:
            hour_open = self._np_ohlc[self._consensus_step, 1]
        close_px = self._np_ohlc[self.current_step, 4]
        # Raw $/micro (typical hour move ≈ 50-100 pts ≈ ±$25-50)
        market_delta = (close_px - hour_open) * self.pm.TICK_VALUE_MICRO

        total = 0.0
        for fam in params.STRAT_FAMILIES:
            key = f'{fam}_variants'
            if key not in self._np_features:
                continue
            conf = self.family_confs[fam]
            attn = self.family_attns[fam]
            signals = self._np_features[key][self._consensus_step, :, 1]  # (72,) in {-1, 0, +1}
            avg_fam_signal = (attn * signals).sum() / (attn.sum() + 1e-8)  # [-1, +1]
            signal_alignment = avg_fam_signal * market_delta  # normalized $, signed

            if conf >= self.CONFIDENCE_THRESHOLD:
                # Above threshold: reward scales with conf and alignment
                fam_reward = signal_alignment * conf
            else:
                # Sitting out (low conf): damped regret/dodge (factor = threshold)
                fam_reward = -signal_alignment * self.CONFIDENCE_THRESHOLD
            total += fam_reward
            logger.debug(f'conf_reward {fam}: conf={conf:.2f} sig={avg_fam_signal:+.3f} align={signal_alignment:+.2f} r={fam_reward:+.4f}')

        logger.debug(f'conf_reward total={total:+.4f} market_delta={market_delta:+.2f}')
        return total

    def _compute_vol_reward(self, trade_info):
        """Vol/K-specific reward: compares 1-micro 'actual' (hour_open → stop_px or close)
        vs 'hold' (hour_open → close_px, no stops). Pure per-micro, no size contamination.
        Returns 0 when sim_daily_locked or no entry was tracked this hour."""
        if self.sim_daily_locked or self._hour_hold_entry is None:
            return 0.0
        hour_open, direction = self._hour_hold_entry
        close_px = self._np_ohlc[self.current_step, 4]
        commission = self.pm.COMMISSION_MICRO  # per-micro round-trip proxy
        hold = direction * (close_px - hour_open) * self.pm.TICK_VALUE_MICRO - commission
        if self._hour_stopped_at_px is not None:
            actual = direction * (self._hour_stopped_at_px - hour_open) * self.pm.TICK_VALUE_MICRO - commission
        else:
            actual = hold  # no stop this hour → identical → diff = 0
        diff = actual - hold  # positive = stop helped vs holding to close
        vol_r = symlog(diff)
        logger.debug(f'vol_reward: actual={actual:+.2f} hold={hold:+.2f} diff={diff:+.2f} r={vol_r:+.4f}')
        return vol_r

    def _get_final_size(self, target_size, day_end, k_pts):
        """Compute position size in micros, capped by MDD/stop."""
        if day_end:
            self._update_kelly_ewm()
        # Scale max_contracts with training_progress: 1 early → full late (training only)
        if self.mode != 'eval':
            scaled_contracts = max(1, int(self.pm.topstep.max_contracts * self.training_progress)) if self.training_progress < 1.0 else self.pm.topstep.max_contracts
        else:
            scaled_contracts = self.pm.topstep.max_contracts

        max_micro = int(scaled_contracts or 1) * self.pm.MICRO_PER_MINI / self.MICRO_DIVISOR
        kelly_confidence = self._cached_kelly_ewm['confidence']
        final_size = target_size * kelly_confidence * max_micro  # in micros

        # Cap by MDD room (humble) so stop-loss doesn't blow through limit
        mdd_room = self._cached_humble_mdd_buffer * abs(self.pm.topstep.MDD_THRESHOLD)
        loss_per_micro = k_pts * self.pm.TICK_VALUE_MICRO
        max_pos_from_stop = (mdd_room * 0.5) / max(loss_per_micro, 0.01)  # in micros
        final_size = min(final_size, max_pos_from_stop, max_micro)  # all in micros

        if self._cached_humble_mdd_buffer < self.MDD_BUFFER_CAP:
            final_size = min(1, final_size)
            self.size_capped = True
        else:
            final_size = max(final_size, 1)  # at least 1 micro
            self.size_capped = False

        logger.debug(f'SIZE: target={target_size:.3f} kelly={kelly_confidence:.3f} max_micro={max_micro:.0f} '
                    f'mdd_buf={self._cached_mdd_buffer:.2f} humble={self._cached_humble_mdd_buffer:.2f} mdd_cap={max_pos_from_stop:.1f} final={final_size:.1f} K={k_pts:.0f}')
        return final_size

    def _get_mdd_buffer(self, floating_pnl):
        """Returns (real_buffer, humble_buffer).
        real_buffer   : based on acc_pnl + floating (true MDD room)
        humble_buffer : min(acc, daily) + floating, wrapped by cycle_target
                        cycle_target = EXAM_TARGET/2 (exam) or ACTIVATION_PNL_TARGET/2 (activation)
        Both are clamped to [0, 2] and normalized by abs(MDD_THRESHOLD).
        """
        ts = self.pm.topstep
        mdd_threshold = abs(ts.MDD_THRESHOLD)
        if mdd_threshold <= 0:
            return 1.0, 1.0

        # Real
        real_equity = ts.acc_pnl + floating_pnl
        real_room = real_equity - ts.mdd_limit
        real_buffer = min(max(real_room / mdd_threshold, 0.0), 2)

        # Humble: min(acc, daily) + floating, then wrap positive by cycle_target
        acc_equity = ts.acc_pnl + floating_pnl
        daily_equity = ts.daily_pnl + floating_pnl
        humble_equity = min(acc_equity, daily_equity)
        if self.mode == 'train_exam' or (self.mode == 'eval' and not ts.pass_exam):
            cycle_target = ts.EXAM_TARGET / self.EXAM_CYCLE_DIV
        else:
            cycle_target = ts.ACTIVATION_PNL_TARGET / self.ACTIVATION_CYCLE_DIV
        if humble_equity > 0 and cycle_target > 0:
            humble_equity = humble_equity % cycle_target
        humble_room = humble_equity - ts.mdd_limit
        humble_buffer = min(max(humble_room / mdd_threshold, 0.0), 2)

        return real_buffer, humble_buffer

    @staticmethod
    def _ewm_kelly(returns):
        """EWM-weighted mu and downside semi-sigma for Kelly."""
        n = len(returns)
        span = 1000
        alpha = 2 / (span + 1)
        w = [(1 - alpha) ** (n - 1 - i) for i in range(n)]
        w_sum = sum(w)
        mu = sum(wi * ri for wi, ri in zip(w, returns)) / w_sum
        sigma_sq = sum(wi * min(ri, 0.0) ** 2 for wi, ri in zip(w, returns)) / w_sum
        sigma = sigma_sq ** 0.5
        win_rate = sum(wi for wi, ri in zip(w, returns) if ri > 0) / w_sum
        return mu, sigma, win_rate

    def _update_kelly_ewm(self):
        """Recompute Kelly EWM stats using sim 1-micro per-trade PnL (pure signal, no sizing contamination)."""
        pnl_list = self.pm.sim_net_pnl_per_con_list[-1000:]
        n = len(pnl_list)
        kelly_f, mu_norm, sigma_norm, win_rate = 0.0, 0.0, 0.0, 0.0

        if n >= 30:
            returns = [x / self.NORMALIZATION_FACTOR for x in pnl_list]
            mu_norm, sigma_norm, win_rate = self._ewm_kelly(returns)
            if sigma_norm > 0 and mu_norm > 0:
                kelly_f = mu_norm / (sigma_norm ** 2)
                kelly_f = min(kelly_f, 2.0) * 0.5  # half Kelly

        if n < 30:
            confidence = 0.3
        elif n < 100:
            confidence = 0.3 + ((n - 30) / 70) * kelly_f
        else:
            confidence = kelly_f

        self._cached_kelly_ewm = {
            'kelly_f': kelly_f, 'win_rate': win_rate,
            'mu_norm': mu_norm, 'sigma_norm': sigma_norm,
            'confidence': confidence, 'n_trades': n,
        }

    def _build_session_summary(self, trade_info, kelly_stats):
        """Build session summary observation."""
        ts = self.pm.topstep
        mdd_threshold = abs(ts.MDD_THRESHOLD)

        acc_pnl = trade_info.get('acc_pnl', 0)
        equity = acc_pnl + trade_info.get('floating_pnl', 0)
        mdd_limit = ts.mdd_limit

        # Effective target / mdd_limit ratio — risk/reward of current situation
        if self.mode == 'train_exam' or (self.mode == 'eval' and not ts.pass_exam):
            day_start_pnl = equity - (ts.daily_pnl + trade_info.get('floating_pnl', 0))
            today_max = day_start_pnl + ts.EXAM_TARGET * ts.CONSISTENCY_THRESHOLD
            effective_target = min(today_max, ts.EXAM_TARGET)
        else:  # train_activation or eval activation
            effective_target = ts.ACTIVATION_PNL_TARGET
        target_mdd_ratio = effective_target / max(abs(mdd_limit), 1.0)

        target = ts.EXAM_TARGET if not ts.pass_exam else ts.ACTIVATION_PNL_TARGET

        # EWM helper
        def ewm_mean(lst, default=0.0):
            if not lst:
                return default
            alpha = 0.9
            w = [alpha ** (len(lst) - 1 - i) for i in range(len(lst))]
            return sum(wi * vi for wi, vi in zip(w, lst)) / sum(w)

        # Slots 3-5: mode-dependent (exam performance or kelly stats)
        use_exam = (self.mode == 'train_exam') or (self.mode == 'eval' and not ts.pass_exam)
        if use_exam:
            avg_days = ewm_mean(ts.days_taken_to_pass_exams, default=20)
            slot3 = max(1.0 - (avg_days - 2) / 18, -1.0)            # speed: 2→+1, 20→-1
            slot4 = 2 * ewm_mean(ts.exam_daily_target_hit_day_list, default=0.5) - 1  # hit_rate: -1→+1
            slot5 = -min(ewm_mean(ts.exam_daily_resets_list, default=0) / 2, 1.0)     # reset: 0→-1
        else:  # train_activation or eval activation
            slot3 = kelly_stats['mu_norm']
            slot4 = kelly_stats['sigma_norm']
            slot5 = min(max(kelly_stats.get('mu_norm', 0) / max(kelly_stats.get('sigma_norm', 1e-8), 1e-8), -5), 5)  # sharpe

        return np.array([
            equity / max(target, 1),
            (equity - mdd_limit) / max(mdd_threshold, 1),
            kelly_stats['win_rate'],
            slot3,
            slot4,
            slot5,
            getattr(ts, 'subscription_days', 0) / 20,
            float(getattr(ts, 'pass_exam', False)),
            getattr(ts, 'success_activation_days', 0) / 5,
            target_mdd_ratio,
            min(abs(self._last_consensus), 1.0),
            self._np_ohlc[self.current_step, 9],  # nts
            self._np_ohlc[self.current_step, 11],  # rth
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if len(self.data_pool) > 1:
            self._load_data(self.data_pool[self.chunk_idx])
            self.chunk_idx = (self.chunk_idx + self.num_envs) % len(self.data_pool)

        self.current_step = self._get_begin_step()
        self.side = 0
        self.last_u = 0.0
        self.last_d = 0.0
        self.current_k = 50.0
        for fam in params.STRAT_FAMILIES:
            self.family_confs[fam] = 0.5
            self.family_attns[fam] = np.ones(72, dtype=np.float32) / 72.0
        self._hour_start_acc_pnl = 0.0
        self._last_final_size = 0.0
        self._cached_mdd_buffer = 1.0
        self._cached_humble_mdd_buffer = 1.0
        self.ud_side = 0
        self.stop_loss_count = 0
        self.stale_consensus = False
        self.size_capped = False
        self.sim_daily_pnl = 0.0
        self.sim_daily_mdd = 0.0
        self.sim_daily_locked = False
        self.ttl_days = 0   # zero on every reset; snapshot captured pre-reset preserves the count
        self.pm.sim_entry_px = None
        self.pm.sim_daily_pnl = 0.0
        self._consensus_step = self.current_step
        if params.SELECTOR_MODE == 'rl':
            for fam in params.STRAT_FAMILIES:
                self.family_metrics_ts[fam][:] = 0
        elif params.SELECTOR_MODE == 'kalman':
            for key in self._kf_state:
                self._kf_state[key] = (np.zeros(72, dtype=np.float32), np.ones(72, dtype=np.float32))

        self._last_consensus = 0.0
        self._last_size = 0.0
        self._cached_kelly_ewm = {
            'kelly_f': 0.0, 'win_rate': 0.0, 'mu_norm': 0.0,
            'sigma_norm': 0.0, 'confidence': 0.3, 'n_trades': 0,
        }
        self._prev_pm_r = 0.0
        self._prev_r_state = 0.0

        # PM reset
        self.pm.net_pnl_per_con_list = self.pm.net_pnl_per_con_list[-1000:]
        self.pm.sim_net_pnl_per_con_list = self.pm.sim_net_pnl_per_con_list[-1000:]
        self.pm.mini_contracts = 0
        self.pm.micro_contracts = 0
        self.pm.avg_entry_px = None
        self.pm._mini_traded = 0
        self.pm._micro_traded = 0
        self.pm._rt_comm = 0.51
        self.pm._trade_pnl_accum = 0.0
        self.pm._trade_closed_accum = 0
        self.pm.topstep.maybe_resize_account(self.pm.net_pnl_per_con_list)
        self.pm.topstep.get_plan()
        if self.mode != 'train_activation':
            self.pm.topstep.pass_exam = False
        else:
            self.pm.topstep.pass_exam = True
        self.pm.topstep.back_to_fund_cnt = 0
        self.pm.topstep.exam_passed_today = False
        self.pm.topstep.final_pnl_list = []
        self.pm.topstep.topstep_cost = 0
        self.pm.topstep.payouts = 0
        self.pm.topstep.exam_counts = 0
        self.pm.topstep.exam_success = 0
        self.pm.topstep.success_activation_days = 0
        self.pm.topstep.subscription_days= 0
        self.pm.topstep.days_taken_to_pass_exams = self.pm.topstep.days_taken_to_pass_exams[-30:]
        self.pm.topstep.days_taken_to_fail_exams = self.pm.topstep.days_taken_to_fail_exams[-30:]
        self.pm.topstep.activation_day_counter = 0
        self.pm.topstep.days_taken_to_payouts = self.pm.topstep.days_taken_to_payouts[-30:]
        self.pm.topstep.days_taken_to_fail_activation = self.pm.topstep.days_taken_to_fail_activation[-30:]
        self.pm.topstep.exam_daily_target_hit_days = 0
        self.pm.topstep.exam_daily_target_hit_day_list = self.pm.topstep.exam_daily_target_hit_day_list[-30:]
        self.pm.topstep.exam_daily_resets_list = self.pm.topstep.exam_daily_resets_list[-30:]
        self.pm.topstep.back_to_fund_list = []
        self.pm.topstep.total_resets = 0
        self.pm.topstep.payout_cnt = 0
        self.pm.topstep.activation_cnt = 1 if self.mode == 'train_activation' else 0
        self.pm.topstep._cached_sharpe = 0.0
        self.pm.topstep._cached_sharpe_n = 0
        #self.pm.net_pnl_per_con_list = []
        self.pm.topstep._reset_exam()
        self.pfolio_history = np.zeros(self.feature_shape['pfolio_info'], dtype=np.float32)
        return self._get_obs(), {}

    def close(self):
        super().close()