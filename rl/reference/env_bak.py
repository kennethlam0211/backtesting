import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
import numpy as np
from gym import spaces
import math
from params import FREQS
from pm import PM
from utils.app_logger import get_logger
import datetime

logger = get_logger('env', level=20)  # INFO=20, skip DEBUG logging during training



import params
class TradingEnv(gym.Env):

    NORMALIZATION_FACTOR = params.NORM_FACTOR
    REWARD_DISCOUNT = 10
    TRADING_COST = 0.28  # micro commission in points (1.4 / 5/5)

    # --- PM Reward Weights ---
    W_STEP = 5.0         # per-bar P&L (~0.2 raw, ×5 → ~1.0)
    W_STOP = 0.3         # stop loss hit penalty (-1 raw, ×0.3 → -0.3)
    W_STATE = 1.0        # EV of current state (upside - downside*cost_ratio, ~[-0.11, 1.0])
    W_SIZING = 15.0      # sizing confidence alignment (~0.05 raw, ×15 → ~0.75)

    # --- Direction Reward Weights ---
    W_DIR_MOVE = 1.0     # bar move * signal quality
    W_DIR_NOGO = 0.3     # penalty for big moves during no-go (regime shift warning)

    MAGNITUDE_FACTOR = 100        # for stop sizing (per-minute scale)
    NORM_FACTOR = 15              # for reward/obs normalization (per-step scale)
    FEATURE_WINDOW_SIZE_DICT = {
                        # 'movement': 21,  # removed — zero correlation
                        '20_UD'   : 21,
                        # 'day_targets' : 20,  # merged into ta
                        'vol' : 20,
                        # 'px_session' : 20,   # merged into ta
                        'ta' : 20,             # replaces tas + px_session + directional + day_targets
                        # 'directional' : 20,  # merged into ta
                        }

    # ohlc column order in _np_ohlc
    # ts(0) open(1) high(2) low(3) close(4) signal(5) original_signal(6)
    # reward(7) session_end(8) nts(9) n_px_1(10) rth(11) hl(12)

    def __init__(self, data, data_pool=None, skip_night=False, env_id=0, num_envs=12, obs_noise_std=0.0):
        """
        Args:
            data: numpy dict from trading_data_np.pkl (or convert_to_numpy output)
            data_pool: list of numpy dicts for chunk cycling
        """
        super(TradingEnv, self).__init__()
        self.data_pool = data_pool if data_pool is not None else [data]
        self.pm = PM()
        self.env_id = env_id
        self.num_envs = num_envs
        self.obs_noise_std = obs_noise_std
        self.chunk_idx = env_id % len(self.data_pool)
        self.played_chunk_idx = self.chunk_idx
        self.features_cols = {}
        self.skip_night = skip_night
        self._load_np_data(data)
        self.current_step = self.get_begin_step()
        print(f"TradingEnv initialized: {self.n_steps} steps")

        # State
        self.side = 0
        self.stop_win_count = 0
        self.stop_loss_count = 0
        self._cached_kelly_ewm = {
            'kelly_f': 0.0, 'win_rate': 0.0, 'mu_norm': 0.0,
            'sigma_norm': 0.0, 'confidence': 0.1, 'n_trades': 0,
        }

        # Spaces
        self.feature_shape = self.get_feature_shape()
        # action: [direction(0=buy,1=sell), size_confid(0-1), magnitude(0-1)]
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.feature_obs_space = self.get_feature_space()
        self.pfolio_history = np.zeros(self.feature_shape['pfolio_info'], dtype=np.float32)

    def _load_np_data(self, np_data):
        """Unpack pre-converted numpy dict directly."""
        self._np_ohlc = np_data['ohlc']
        self._np_session_end = self._np_ohlc[:, 8].astype(bool)
      #  self._np_skip_night = np_data.get('skip_night', None)
        self._np_features = {k: v for k, v in np_data.items()
                             if k not in ('ohlc', 'skip_night')}
        self.n_steps = len(self._np_ohlc)
        self._prestacked = set()
        self._prestack_features()

    @property
    def observation_space(self):
        return spaces.Dict(self.feature_obs_space)

    def get_feature_space(self):
        return {feature: spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=shape, dtype=np.float32
                ) for feature, shape in self.feature_shape.items()}

    def get_begin_step(self):
        return max(int(np.argmax(self._np_session_end)), 20)

    def get_feature_shape(self):
        return {
            # movement removed — zero correlation
            '20_UD_720': (21,),
            '20_UD_60': (21,),
            '20_UD_15': (21,),
            '20_UD_1': (21,),
            # day_targets, px_session, directional merged into ta
            'vol_720': (20, 9),
            'vol_60': (20, 9),
            'vol_15': (20, 9),
            'vol_1': (20, 9),
            'ta_720': (20, 12),  # tas(6) + px_session(4) + directional(2)
            'ta_60': (20, 12),
            'ta_15': (20, 12),
            'ta_1': (20, 22),   # tas(8) + px_session(2) + day_targets(10) + directional(2)
            'pfolio_info': (60, 11),
            'session_summary': (14,),
        }

    def _prestack_features(self):
        """Convert object arrays to proper numeric arrays for fast slicing."""
        stack_keys = set()
        for feature, window_size in self.FEATURE_WINDOW_SIZE_DICT.items():
            key = f'{feature}_1'
            if key in self._np_features and feature != '20_UD':
                stack_keys.add(key)

        for key in stack_keys:
            obj_arr = self._np_features[key]
            elem_shape = None
            for elem in obj_arr:
                if isinstance(elem, np.ndarray):
                    elem_shape = elem.shape
                    break
            if elem_shape is None:
                continue
            stacked = np.zeros((len(obj_arr), *elem_shape), dtype=np.float32)
            for i, elem in enumerate(obj_arr):
                if isinstance(elem, np.ndarray):
                    stacked[i] = elem
            self._np_features[key] = stacked
            self._prestacked.add(key)

    def get_stack(self, col_names, window_size):
        start = self.current_step - (window_size - 1)
        end = self.current_step + 1
        arr = self._np_features[col_names][start:end]
        if col_names in self._prestacked:
            return arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr
        return np.vstack(arr)

    def _get_obs(self, pfolio_info_=None, session_summary_=None):
        """Get observation as a dict."""

        # pfolio_info — 60-step rolling window
        if pfolio_info_ is None:
            self.pfolio_history = np.zeros(self.feature_shape['pfolio_info'], dtype=np.float32)
        else:
            self.pfolio_history[:-1] = self.pfolio_history[1:]
            self.pfolio_history[-1] = pfolio_info_
        self.features_cols['pfolio_info'] = self.pfolio_history.copy()

        # session_summary — flat, passed from step() or zeros on reset
        if session_summary_ is not None:
            self.features_cols['session_summary'] = session_summary_
        elif 'session_summary' not in self.features_cols:
            self.features_cols['session_summary'] = np.zeros(self.feature_shape['session_summary'], dtype=np.float32)

        # Populate market features
        for feature, window_size in self.FEATURE_WINDOW_SIZE_DICT.items():
            for freq in FREQS:
                    key = f'{feature}_{freq}'
                    if key in self._np_features:
                        if freq == '1' and feature != '20_UD':
                            self.features_cols[key] = self.get_stack(key, window_size=window_size)
                        else:
                            val = self._np_features[key][self.current_step]
                            if isinstance(val, np.ndarray):
                                self.features_cols[key] = val
                            elif self.features_cols.get(key) is None:
                                # Forward-fill: find last valid value
                                col_arr = self._np_features[key][:self.current_step + 1]
                                res = None
                                for i in range(len(col_arr) - 1, -1, -1):
                                    if isinstance(col_arr[i], np.ndarray):
                                        res = col_arr[i]
                                        break
                                if res is None:
                                    res = np.zeros(self.feature_shape[key], dtype=np.float32)
                                self.features_cols[key] = res

        obs = {key: self.features_cols[key]
               for key in self.feature_shape if key in self.features_cols}
        if self.obs_noise_std > 0:
            for key in obs:
                if key not in ('pfolio_info', 'session_summary'):
                    obs[key] = obs[key] + np.random.normal(0, self.obs_noise_std, obs[key].shape).astype(np.float32)
        return obs

    def get_final_metrics(self):
        """Compute avg_net and total_net from topstep final_pnl_list (callable any time)."""
        pnl_list = self.pm.topstep.final_pnl_list
        if not pnl_list:
            return 0.0, 0.0
        pnl_arr = np.array(pnl_list)
        avg_net = float(pnl_arr.mean())
        total_net = float(pnl_arr.sum())
        return avg_net, total_net

    def capped_growth(self, x, threshold=300, cap=5, cap_x=3000, k=0.001):
        if x < threshold:
            return 0.0
        if x >= cap_x:
            return cap
        a = cap / (math.exp(k * (cap_x - threshold)) - 1)
        return a * (math.exp(k * (x - threshold)) - 1)


    def _direction_actor_step(self, action, pre_side, open, close, signal, expected_reward):
        """Compute reward for direction actor (binary: long/short)."""
        step_pnl = (close - open) * action

        if action != pre_side:
            step_pnl -= self.TRADING_COST * abs(action - pre_side)

        direction_reward = step_pnl / self.NORMALIZATION_FACTOR

        # signal is always ±1 (original_signal)
        if action == signal:
            direction_reward *= (1.0 + expected_reward)
        else:
            direction_reward = -abs(direction_reward) * (1.0 + expected_reward)

        return direction_reward


    def _build_session_summary(self, trade_info, kelly_stats):
        """Build session_summary array (14,) reusing trade_info + kelly_stats."""
        ts = self.pm.topstep
        acc_pnl = trade_info['acc_pnl']
        mdd_limit = abs(trade_info['mdd_limit'])
        n = kelly_stats['n_trades']
        has_enough = n >= 10

        # sharpe — cached, only recompute when pnl list grows
        n_pnl = len(ts.all_pnl_per_contract_list)
        if has_enough and n_pnl >= 2 and n_pnl != ts._cached_sharpe_n:
            returns = np.array(ts.all_pnl_per_contract_list)
            std = returns.std()
            ts._cached_sharpe = float(returns.mean() / std) if std > 1e-8 else 0.0
            ts._cached_sharpe_n = n_pnl
        sharpe = ts._cached_sharpe
        sharpe = min(max(sharpe, -5.0), 5.0)

        target = ts.EXAM_TARGET if not trade_info['pass_exam'] else ts.ACTIVATION_PNL_TARGET
        mdd_threshold = abs(ts.MDD_THRESHOLD)

        # account_size one-hot: [50k, 100k, 150k]
        acct_oh = [0.0, 0.0, 0.0]
        acct_oh[{'50k': 0, '100k': 1, '150k': 2}.get(trade_info['account_size'], 0)] = 1.0
        # plan one-hot: [exam_plan, no_activation_plan]
        has_activation_fee = ts.activation_fees > 0
        plan_oh = [float(has_activation_fee), float(not has_activation_fee)]

        return np.array([
            acc_pnl / target,
            (acc_pnl-mdd_limit) / mdd_threshold,
            kelly_stats['win_rate'],
            kelly_stats['mu_norm'],
            kelly_stats['sigma_norm'],
            sharpe,
            trade_info['subscription_days'],
            float(trade_info['pass_exam']),
            trade_info['success_activation_days'],
            *acct_oh,
            *plan_oh,
        ], dtype=np.float32)

    def _compute_pm_reward(self, trade_info,  size_confid, final_size):
        """Compute PM reward — 3 components: r_step, r_sizing, r_state.

        Dense signals (per-step): r_step, r_sizing — teaches moment-to-moment trading
        State signal (EV):        r_state = upside - downside × cost_ratio
          upside  = P(getting payout) based on goal progress + activation days
          downside = P(paying exam again) based on MDD danger + time pressure
          cost_ratio = cost/payout (~0.11 for 50k) bakes in the 9:1 asymmetry
        """

        ts = self.pm.topstep
        _EXP2 = math.e ** 2 - 1.0  # normalizer: exp(2x)-1 → [0,1]

        # --- Dense signals ---

        # 1. Stop loss count (no reward, just tracking)
        if trade_info['was_stopped']:
            self.stop_loss_count += 1

        # 2. Per-bar PnL per contract
        r_step = (trade_info['step_pnl_per_con'] + trade_info['net_pnl_per_con']) / self.NORMALIZATION_FACTOR

        # 3. Sizing quality: kelly discount factor × outcome
        effective_size = max(final_size, 1e-6) / max(size_confid, 1e-6)  # [0,1]
        r_sizing = (effective_size - 0.5) * 2.0 * r_step

        # --- State EV signal ---

        # Real money at stake
        payout = ts.ACTIVATION_PNL_TARGET * 0.45
        cost = ts.exam_fees + ts.activation_fees
        cost_ratio = cost / max(payout, 1.0)  # ~0.11 for 50k

        equity = trade_info['acc_pnl'] + trade_info['floating_pnl']

        # Upside: P(getting cash) [0, 1]
        target = ts.EXAM_TARGET if not ts.pass_exam else ts.ACTIVATION_PNL_TARGET
        goal_progress = (math.exp(2.0 * max(equity / target, 0.0)) - 1.0) / _EXP2
        if not ts.pass_exam:
            upside = goal_progress * 0.5  # exam is only step 1 of 2
        else:
            act_progress = (math.exp(2.0 * min(trade_info['success_activation_days'], 1.0)) - 1.0) / _EXP2
            upside = (goal_progress*0.7 + act_progress*0.3)

        # Downside: P(paying again) [0, 1]
        mdd_threshold = abs(ts.MDD_THRESHOLD)
        buffer = min(max((equity - trade_info['mdd_limit']) / mdd_threshold, 0.0), 1.0)
        danger = (math.exp(2.0 * (1.0 - buffer)) - 1.0) / _EXP2
        if not ts.pass_exam:
            time_pressure = 0.5 * (math.exp(2.0 * max(min(trade_info['subscription_days'], 1.0), 0.0)) - 1.0) / _EXP2
            downside = max(danger, time_pressure)  # whichever is worse
        else:
            downside = danger  # activation: only MDD matters

        r_state = upside - downside * cost_ratio

        return (self.W_STEP * r_step
                + self.W_SIZING * r_sizing
                + self.W_STATE * r_state)

    @staticmethod
    def _ewm_kelly(returns):
        """EWM-weighted mu and downside semi-sigma for Kelly.
        Recent trades matter more — old trades decay exponentially."""
        n = len(returns)
        span = 10000
        alpha = 2 / (span + 1)
        # weights: most recent = highest weight
        w = [(1 - alpha) ** (n - 1 - i) for i in range(n)]
        w_sum = sum(w)

        mu = sum(wi * ri for wi, ri in zip(w, returns)) / w_sum
        # Downside semi-variance: only losses contribute
        sigma_sq = sum(wi * min(ri, 0.0) ** 2 for wi, ri in zip(w, returns)) / w_sum
        sigma = sigma_sq ** 0.5
        # EWM-weighted win rate: sum of weights where return > 0
        win_rate = sum(wi for wi, ri in zip(w, returns) if ri > 0) / w_sum
        return mu, sigma, win_rate

    def _update_kelly_ewm(self):
        """Recompute Kelly EWM stats. Called only on session_end."""
        pnl_list = self.pm.net_pnl_per_con_list[-10000:]
        n = len(pnl_list)
        #ts = self.pm.topstep
        #mdd_threshold = abs(ts.MDD_THRESHOLD)

        kelly_f, mu_norm, sigma_norm, win_rate = 0.0, 0.0, 0.0, 0.0

        if n >= 30:
            returns = [x /self.NORMALIZATION_FACTOR for x in pnl_list]
            mu_norm, sigma_norm, win_rate = self._ewm_kelly(returns)
            if sigma_norm > 0 and mu_norm > 0:
                kelly_f = mu_norm / (sigma_norm ** 2)
                kelly_f = min(kelly_f, 2.0) * 0.75  # 75%-Kelly — cheap resets justify more aggression

        if n < 30:
            confidence = 0.1
        elif n < 100:
            confidence = 0.1 + ((n - 30) / 70) * kelly_f
        else:
            confidence = kelly_f

        return {
            'kelly_f': kelly_f, 'win_rate': win_rate,
            'mu_norm': mu_norm, 'sigma_norm': sigma_norm,
            'confidence': confidence, 'n_trades': n,
        }

    def get_kelly(self, size_confid, session_end):
        """Combine Kelly criterion with RL's size_confid, discounted by MDD proximity."""
        if session_end:
            self._cached_kelly_ewm = self._update_kelly_ewm()

        # MDD buffer: live every step
        ts = self.pm.topstep
        mdd_threshold = abs(ts.MDD_THRESHOLD)
        mdd_buffer = min(max((ts.acc_pnl - ts.mdd_limit) / mdd_threshold, 0.0), 1.0) #FXIME back to add floating_PNL

        confidence = self._cached_kelly_ewm['confidence']
        mdd_buffer_adj = mdd_buffer ** 0.5  # sqrt: stay aggressive longer (0.5 room → 0.71 sizing)
        final_size = min(max(size_confid * max(confidence, 0.1) * mdd_buffer_adj, 0.0), 1.0)

        kelly_stats = {**self._cached_kelly_ewm, 'mdd_buffer': mdd_buffer}
        return final_size, kelly_stats
    

    def _portfolio_manager_step(self, side, open, high, low, close, hl, session_end, size_confid, stop_win=0.0, stop_loss=0.0):
        """Execute PM trade via PM class."""
        trade_info = self.pm.pm_step(open, high, low, close, hl, side, size_confid, session_end,
                                     stop_win=stop_win, stop_loss=stop_loss)
        return trade_info
    

    def step(self, action):
        """Gym step — action is [direction_action, size_confid, magnitude] array."""
        direction_action = int(action[0])
        size_confid = float(action[1])
        magnitude_raw = float(action[2])
        self.current_step += 1

        # Binary: 0=buy(+1), 1=sell(-1)
        dir_action = 1 if direction_action == 0 else -1

        _, open, high, low, close, signal, original_signal, expected_reward, session_end, nts, n_px_1, rth, hl = self._np_ohlc[self.current_step]
        timestamp = self._np_ohlc[self.current_step, 0]

        magnitude_pts = magnitude_raw * self.MAGNITUDE_FACTOR


        stop_loss = max(magnitude_pts, 10) 
        stop_win = 99999  # set impossibly high — exit via direction change only


        direction_reward = self._direction_actor_step(
            dir_action, self.side, open, close, original_signal, expected_reward
        )

        final_size, kelly_stats = self.get_kelly(size_confid, bool(session_end))

        # Compute stop levels from magnitude + ratio

        # Cap final_size so worst-case stop loss doesn't blow through MDD
        # max_loss_per_micro = stop_loss * TICK_VALUE_MICRO * 0.5 (safety margin)
        ts_obj = self.pm.topstep
        prev_close = self._np_ohlc[self.current_step-1][4]
        floating_pnl = self.pm.cal_floating_pnl(prev_close)
        mdd_room = abs((ts_obj.acc_pnl + floating_pnl) - ts_obj.mdd_limit)
        if mdd_room > 0 and stop_loss > 0:
            # Use half mdd_room so a stop hit still leaves half the room
            loss_per_micro = stop_loss * self.pm.TICK_VALUE_MICRO
            max_micro = int(ts_obj.max_contracts) * self.pm.MICRO_PER_MINI
            max_pos_from_stop = (mdd_room * 0.7) / loss_per_micro
            stop_cap = min(max_pos_from_stop / max_micro, 1.0) if max_micro > 0 else 1.0
            final_size = min(final_size, stop_cap)

        self.side = dir_action
        trade_info = self._portfolio_manager_step(
            dir_action, open, high, low, close, int(hl), session_end, final_size,
            stop_win=stop_win, stop_loss=stop_loss
        )

        pm_reward = self._compute_pm_reward(trade_info,  size_confid,final_size)

        done = self.current_step >= self.n_steps - 1

        pfolio_info = np.array([
            trade_info['floating_pnl'] / self.NORMALIZATION_FACTOR,
            trade_info['per_contract_pnl'] /self.NORMALIZATION_FACTOR,
            trade_info['step_pnl'] / self.NORMALIZATION_FACTOR,
            trade_info['position_usage'],
            rth,
            nts,
            kelly_stats['kelly_f'],
            kelly_stats['mdd_buffer'],
            kelly_stats['confidence'],
            kelly_stats['win_rate'],
        ], dtype=np.float32)

        session_summary = self._build_session_summary(trade_info, kelly_stats)

        # infos: only what ppo.py actually reads — keep buffer light
        infos = {
            'direction_reward': direction_reward,
            'signal': original_signal,
            'expected_reward': expected_reward,
            'topstep_cost': trade_info['topstep_cost'],
            'acc_pnl': trade_info['acc_pnl'],
            'payouts': trade_info['payouts'],
            'exam_pass_rate': self.pm.topstep.exam_success / max(self.pm.topstep.exam_counts, 1),
            'exam_counts': self.pm.topstep.exam_counts,
            'stop_loss_count': self.stop_loss_count,
            'total_commission': self.pm.topstep.total_commission,
            'position': self.pm.position_size_equivalent,
        }

        
        if logger.isEnabledFor(10):  # DEBUG=10
            all_infos = {
                'timestamp': timestamp,
                'magnitude_raw': magnitude_raw,
                **kelly_stats,
                **trade_info,
                **infos,
                'size_confid': size_confid,
                'final_size': final_size,
                'pm_reward': pm_reward,
            }
            logger.debug(all_infos)

        return self._get_obs(pfolio_info_=pfolio_info, session_summary_=session_summary), pm_reward, done, False, infos


    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Cycle through chunks sequentially
        if len(self.data_pool) > 1:
            self.played_chunk_idx = self.chunk_idx
            self._load_np_data(self.data_pool[self.chunk_idx])
            self.chunk_idx = (self.chunk_idx + self.num_envs) % len(self.data_pool)

        self.current_step = self.get_begin_step()
        self.side = 0
        self.stop_win_count = 0
        self.stop_loss_count = 0
        self._cached_kelly_ewm = {
            'kelly_f': 0.0, 'win_rate': 0.0, 'mu_norm': 0.0,
            'sigma_norm': 0.0, 'confidence': 0.1, 'n_trades': 0,
        }
        # Full PM + TopStep reset for clean episode
        self.pm.mini_contracts = 0
        self.pm.micro_contracts = 0
        self.pm.avg_entry_px = None
        self.pm._mini_traded = 0
        self.pm._micro_traded = 0
        self.pm._rt_comm = 0.51
        self.pm._trade_pnl_accum = 0.0
        self.pm._trade_closed_accum = 0
        self.pm.topstep.final_pnl_list = []
        self.pm.topstep.topstep_cost = 0
        self.pm.topstep.payouts = 0
        self.pm.topstep.exam_counts = 0
        self.pm.topstep.exam_success = 0
        self.pm.topstep.all_pnl_per_contract_list = []
        self.pm.topstep.pass_exam = False
        self.pm.topstep.success_activation_days = 0
        self.pm.topstep._cached_sharpe = 0.0
        self.pm.topstep._cached_sharpe_n = 0
        self.pm.net_pnl_per_con_list = []
        self.pm.topstep._reset_exam()
        self.pfolio_history = np.zeros(self.feature_shape['pfolio_info'], dtype=np.float32)

        return self._get_obs(), {}

    def close(self):
        self._closed = True
        super().close()
