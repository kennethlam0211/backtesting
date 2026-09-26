import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topstep import ExamSimulator
from utils.app_logger import get_logger
import params
logger = get_logger('pm', level=params.LOG_LEVEL)


class PM:
    # 10 micro = 1 mini
    MICRO_PER_MINI = 10
    # commission per contract
    COMMISSION_MINI = 1.4 + 0.5
    COMMISSION_MICRO = 0.37 + 0.25
    # tick value: YM mini = $5/tick, micro = $0.50/tick
    TICK_VALUE_MINI = 5.0
    TICK_VALUE_MICRO = 0.5

    def __init__(self, account_size=params.ACCOUNT_SIZE, mode='eval', exam_success=0, exam_counts=0):
        self.mode = mode
        self.topstep = ExamSimulator(account_size, mode=mode, exam_success=exam_success, exam_counts=exam_counts)
        self.topstep._on_reset = self._reset_sim_state  # callback: topstep._reset_exam fires this
        # actual contract holdings (signed: +long, -short)
        self.mini_contracts = 0
        self.micro_contracts = 0
        # avg entry price tracking
        self.avg_entry_px = None
        # stop levels (updated every step by RL agent)
        self.stop_win = 0.0
        self.stop_loss = 0.0
        # net realised PnL per closed contract (for Kelly)
        self.net_pnl_per_con_list = []
        self._trade_pnl_accum = 0.0      # accumulated net PnL for current same-side trade
        self._trade_closed_accum = 0      # accumulated closed contracts for current same-side trade
        # mini-to-micro ratio for blended commission estimate (reset per episode)
        self._mini_traded = 0            # |d_mini| accumulated
        self._micro_traded = 0           # |d_micro| accumulated
        self._rt_comm = 0.81             # cached round-trip commission per micro-equiv
        # Shadow 1-micro state (updated inside execute_trade on transitions)
        self.sim_entry_px = None         # entry price of shadow 1-micro position
        self.sim_daily_pnl = 0.0         # realized 1-micro daily PnL (env resets daily)
        self.sim_net_pnl_per_con_list = []  # per-trade 1-micro net PnL (for Kelly / p_survive)

    def _reset_sim_state(self):
        """Called by topstep._reset_exam via callback to reset shadow 1-micro state."""
        self.sim_entry_px = None
        self.sim_daily_pnl = 0.0

    @property
    def position_size_equivalent(self):
        """Net position in micro units."""
        return self.mini_contracts * self.MICRO_PER_MINI + self.micro_contracts

    def _update_rt_comm(self):
        """Recompute cached round-trip commission. Called on trade.
        default /2: 2 * (0.5 * 0.37 + 0.5 * 1.4/10) = 0.51"""
        total = self._mini_traded + self._micro_traded
        if total >= 100:
            r = self._mini_traded / total
            self._rt_comm = 2 * (r * self.COMMISSION_MINI + (1 - r) * self.COMMISSION_MICRO / self.MICRO_PER_MINI)

    # ------------------------------------------------------------------ PnL-----------------------------------------------------------------------
    def cal_floating_pnl(self, current_price):
        """Unrealised PnL across all open entries."""
        pos = self.position_size_equivalent
        if pos == 0:
            return 0.0
        direction = 1 if pos > 0 else -1
        return abs(pos) * (current_price - self.avg_entry_px) * direction * self.TICK_VALUE_MICRO

    def cal_trading_fees(self, mini_qty, micro_qty):
        """Commission cost for a trade (always negative)."""
        return -(abs(mini_qty) * self.COMMISSION_MINI
                 + abs(micro_qty) * self.COMMISSION_MICRO)

    def execute_trade(self, price, mini_qty, micro_qty):
        """Execute a trade using avg entry price.

        When increasing position (same direction): update avg_entry_px.
        When reducing position (same direction): realise PnL at avg_entry_px.
        When flipping direction: close all first, realise PnL, then open new.

        mini_qty / micro_qty: signed (+buy, -sell).
        Returns (realised_pnl_this_trade, trading_fees_this_trade).
        """
        trade_micro = mini_qty * self.MICRO_PER_MINI + micro_qty
        fees = self.cal_trading_fees(mini_qty, micro_qty)

        old_pos = self.position_size_equivalent
        rpnl = 0.0

        if old_pos == 0:
            # no position, just open
            self.avg_entry_px = price

        elif (old_pos > 0 and trade_micro > 0) or (old_pos < 0 and trade_micro < 0):
            # increasing position same direction → update avg entry px
            assert self.avg_entry_px is not None
            self.avg_entry_px = (self.avg_entry_px * abs(old_pos) + price * abs(trade_micro)) / (abs(old_pos) + abs(trade_micro))

        elif (old_pos > 0 and trade_micro < 0) or (old_pos < 0 and trade_micro > 0):
            # opposite direction
            direction = 1 if old_pos > 0 else -1
            close_qty = min(abs(trade_micro), abs(old_pos))
            rpnl = close_qty * (price - self.avg_entry_px) * direction * self.TICK_VALUE_MICRO

            remaining = abs(trade_micro) - close_qty
            if remaining > 0:
                # flipped direction, new position at new price
                self.avg_entry_px = price
            elif abs(old_pos) == close_qty:
                # fully closed, no remaining position
                self.avg_entry_px = None

        self.mini_contracts += mini_qty
        self.micro_contracts += micro_qty
        self._mini_traded += abs(mini_qty)
        self._micro_traded += abs(micro_qty)

        new_pos = self.position_size_equivalent

        # ── Shadow 1-micro state machine (transitions only) ──
        if old_pos == 0 and new_pos != 0:
            # Flat → open
            self.sim_entry_px = price
            self.sim_daily_pnl -= self.COMMISSION_MICRO
        elif old_pos != 0 and new_pos == 0:
            # Fully closed
            if self.sim_entry_px is not None:
                direction = 1 if old_pos > 0 else -1
                gross = (price - self.sim_entry_px) * direction * self.TICK_VALUE_MICRO
                self.sim_daily_pnl += gross
                self.sim_daily_pnl -= self.COMMISSION_MICRO
                trade_net = gross - 2 * self.COMMISSION_MICRO  # round-trip commission
                self.sim_net_pnl_per_con_list.append(trade_net)
                self.sim_entry_px = None
        elif old_pos != 0 and new_pos != 0 and (old_pos > 0) != (new_pos > 0):
            # Flip: realize old at price, reopen new at price
            if self.sim_entry_px is not None:
                direction = 1 if old_pos > 0 else -1
                gross = (price - self.sim_entry_px) * direction * self.TICK_VALUE_MICRO
                self.sim_daily_pnl += gross
                self.sim_daily_pnl -= self.COMMISSION_MICRO
                trade_net = gross - 2 * self.COMMISSION_MICRO
                self.sim_net_pnl_per_con_list.append(trade_net)
            self.sim_entry_px = price
            self.sim_daily_pnl -= self.COMMISSION_MICRO
        # else: scaling same direction — sim holds original 1 micro, no change

        logger.debug(f'TRADE: px={price:.0f} old={old_pos} trade={trade_micro:+d} new={new_pos} '
                    f'rpnl={rpnl:.1f} fees={fees:.2f} entry={self.avg_entry_px}')

        return rpnl, fees

    # --------------------------------------------------------- position sizing
    def get_position_size(self,target_equiv):
        """Compute the least-action trade to reach desired position.

        Args:
            size_confid: float 0-1, fraction of max capacity.
            side: +1 for long, -1 for short, 0 for flat.

        Returns:
            (delta_mini, delta_micro) signed trade to execute.

        Examples:
            Current: -2 mini, +3 micro (equiv -17)
            Target equiv 0  → delta: +2 mini, -3 micro  (5 contracts traded)
            Target equiv 25 → delta: +4 mini, -1 micro  (optimised split)
        """
 

        # try floor and ceil mini splits, pick least cost
        # e.g. target 17: floor → (1 mini, 7 micro), ceil → (2 mini, -3 micro)
        if target_equiv >= 0:
            floor_mini = target_equiv // self.MICRO_PER_MINI
            ceil_mini = floor_mini + 1
        else:
            floor_mini = -(abs(target_equiv) // self.MICRO_PER_MINI)
            ceil_mini = floor_mini - 1

        best = (0, 0)
        best_cost = float('inf')
        for t_mini in (floor_mini, ceil_mini):
            t_micro = target_equiv - t_mini * self.MICRO_PER_MINI
            d_mini = t_mini - self.mini_contracts
            d_micro = t_micro - self.micro_contracts
            cost = abs(d_mini) * self.COMMISSION_MINI + abs(d_micro) * self.COMMISSION_MICRO
            if cost < best_cost:
                best_cost = cost
                best = (d_mini, d_micro)

        if best == (0, 0):
            return 0, 0
        return best 
    
    def get_size_equiv(self, size_confid, side):
        max_micro = self.topstep.max_contracts * self.MICRO_PER_MINI
        target_equiv = int(max_micro * size_confid) * side  # truncate down, never exceed intended size
        # At least 1 micro if there's a side
        if side != 0 and target_equiv == 0:
            target_equiv = side  # +1 or -1 micro
        return target_equiv
    
    def close_all(self, price):
        """Close all positions at day end. Returns (rpnl, fees)."""
        if self.mini_contracts == 0 and self.micro_contracts == 0:
            return 0.0, 0.0
        old_pos = self.position_size_equivalent
        d_mini = -self.mini_contracts
        d_micro = -self.micro_contracts
        rpnl, fees = self.execute_trade(price, d_mini, d_micro)
        logger.debug(f'CLOSE_ALL: px={price:.0f} pos={old_pos} rpnl={rpnl:.1f} fees={fees:.2f}')
        return rpnl, fees
    

    def check_stops(self, high, low, hl, last_u=None, last_d=None, gp_stop_check=None):
        """Check position-level stops against intrabar high/low.
        Stop-loss based on watermark (last_u for long, last_d for short).
        hl: 1 = high first, 0 = low first (from preprocessing).
        Returns (rpnl, fees, was_stopped, stop_px). stop_px is None if not stopped."""
        pos = self.position_size_equivalent
        if pos == 0 or self.avg_entry_px is None:
            return 0.0, 0.0, False, None
        if self.stop_win <= 0 and self.stop_loss <= 0:
            return 0.0, 0.0, False, None

        # Stop-loss from watermark: long exits K below last_u, short exits K above last_d
        if pos > 0:
            ref = last_u if last_u is not None else self.avg_entry_px
            sl_px = ref - self.stop_loss
        else:
            ref = last_d if last_d is not None else self.avg_entry_px
            sl_px = ref + self.stop_loss

        sw_px = self.avg_entry_px + self.stop_win if pos > 0 else self.avg_entry_px - self.stop_win

        # Clamp stop_loss so it can't exceed stop_win target (prevent overshoot)
        if self.stop_win > 0:
            if pos > 0:
                sl_px = min(sl_px, sw_px)  # long: sl can't be above sw
            else:
                sl_px = max(sl_px, sw_px)  # short: sl can't be below sw

        sl_hit = (low <= sl_px) if pos > 0 else (high >= sl_px)
        sw_hit = (high >= sw_px) if pos > 0 else (low <= sw_px)

        if not sl_hit and gp_stop_check is not None:
            sl_hit = (gp_stop_check <= sl_px) if pos > 0 else (gp_stop_check >= sl_px)

        if not sw_hit and not sl_hit and gp_stop_check is not None:
            sw_hit = (gp_stop_check >= sw_px) if pos > 0 else (gp_stop_check <= sw_px)

        # if sl_hit or sw_hit:
        #     logger.debug(f'STOP: pos={pos} sl_px={sl_px:.0f} sw_px={sw_px:.0f} sl_hit={sl_hit} sw_hit={sw_hit} '
        #                 f'ref={ref:.0f} H={high:.0f} L={low:.0f} gap={gp_stop_check}')
        
        # if sl_hit or sw_hit:
        #     close_px = sw_px if sw_hit and not sl_hit else sl_px
        #     est_rpnl = pos * (close_px - self.avg_entry_px) * self.TICK_VALUE_MICRO
        #     ts = self.topstep
        #     if ts.daily_pnl + est_rpnl > ts.EXAM_TARGET * ts.CONSISTENCY_THRESHOLD+100 or ts.acc_pnl + est_rpnl > ts.EXAM_TARGET+100:
        #         logger.warning(f'[STOP OVERSHOOT] sl={sl_hit} sw={sw_hit} pos={pos} sl_px={sl_px:.0f} sw_px={sw_px:.0f} '
        #                       f'entry={self.avg_entry_px:.0f} hl={hl} est_rpnl={est_rpnl:.0f} '
        #                       f'acc={ts.acc_pnl:.0f} daily={ts.daily_pnl:.0f} stop_win={self.stop_win:.0f} stop_loss={self.stop_loss:.0f} '
        #                       f'last_u={last_u} last_d={last_d}')
        
        if sl_hit and sw_hit:
            if pos > 0:
                return self._close_at(sw_px) if hl == 1 else self._close_at(sl_px)
            else:
                return self._close_at(sl_px) if hl == 1 else self._close_at(sw_px)
        elif sl_hit:
            return self._close_at(sl_px)
        elif sw_hit:
            return self._close_at(sw_px)

        return 0.0, 0.0, False, None

    def _close_at(self, price):
        """Close all at given price. Returns (rpnl, fees, True, stop_px)."""
        rpnl, fees = self.close_all(price)
        return rpnl, fees, True, price

    def pm_step(self, open, high, low, close, hl, side, size_confid, day_end, next_open,
                stop_win=0.0, stop_loss=0.0, last_u=None, last_d=None, rth=1):
        # Update stop levels (they change every step)
        self.stop_win = stop_win
        self.stop_loss = stop_loss

        rpnl, fees = 0.0, 0.0

        # 0. If closing at open would hit daily/exam target → force close
        old_pos = self.position_size_equivalent
        if self.mode != 'train_activation' and not self.topstep.pass_exam and old_pos != 0 and self.avg_entry_px is not None:#and not self.topstep.daily_locked:
            floating_at_open = old_pos * (open - self.avg_entry_px) * self.TICK_VALUE_MICRO
            ts = self.topstep
            daily_check = ts.daily_pnl + floating_at_open
            exam_check = ts.acc_pnl + floating_at_open
            if (daily_check >= ts.EXAM_TARGET * ts.CONSISTENCY_THRESHOLD or
                exam_check >= ts.EXAM_TARGET):
                #print(f'[STEP0 CLOSE] daily={daily_check:.0f} exam={exam_check:.0f} float={floating_at_open:.0f} pos={old_pos} entry={self.avg_entry_px:.0f} open={open:.0f}')
                rpnl, fees = self.close_all(open)
                side = 0

        # 1. Adjust position at open (trim/close/resize)
        old_pos = self.position_size_equivalent

        target_equiv = self.get_size_equiv(size_confid, side)

        if day_end or side == 0:
            rpnl, fees = self.close_all(open)
        else:
            d_mini, d_micro = self.get_position_size(target_equiv)
            if d_mini != 0 or d_micro != 0:
                rpnl, fees = self.execute_trade(open, d_mini, d_micro)

        pos_after_resize = self.position_size_equivalent
        if pos_after_resize != old_pos:
            logger.debug(f'PM: side={side} target={target_equiv:.1f} old={old_pos} new={pos_after_resize} '
                        f'open={open:.0f} rpnl={rpnl:.1f} fees={fees:.1f}')

        # 2. stop_win: close at exact price that hits today's effective target.
        # Phase-based, not mode-based — works for train_exam, train_activation, and eval (both phases).
        ts = self.topstep
        if pos_after_resize != 0 and self.avg_entry_px is not None and not ts.daily_locked:
            exit_fees = abs(self.mini_contracts) * self.COMMISSION_MINI + abs(self.micro_contracts) * self.COMMISSION_MICRO
            floating_at_open = pos_after_resize * (open - self.avg_entry_px) * self.TICK_VALUE_MICRO

            if not ts.pass_exam:
                # Exam phase: target = min(daily, exam) cap
                daily_remaining = ts.EXAM_TARGET * ts.CONSISTENCY_THRESHOLD - (ts.daily_pnl + rpnl + fees + floating_at_open) + exit_fees
                exam_remaining = ts.EXAM_TARGET - (ts.acc_pnl + rpnl + fees + floating_at_open) + exit_fees
                remaining_with_float = min(daily_remaining, exam_remaining)
                if remaining_with_float > 0:
                    daily_clean = ts.EXAM_TARGET * ts.CONSISTENCY_THRESHOLD - (ts.daily_pnl + rpnl + fees) + exit_fees
                    exam_clean = ts.EXAM_TARGET - (ts.acc_pnl + rpnl + fees) + exit_fees
                    remaining_clean = min(daily_clean, exam_clean)
                    self.stop_win = remaining_clean / (abs(pos_after_resize) * self.TICK_VALUE_MICRO)
                    logger.debug(f'[SW SET EXAM] sw={self.stop_win:.1f} rem={remaining_clean:.0f} '
                                f'daily_rem={daily_clean:.0f} exam_rem={exam_clean:.0f} pos={pos_after_resize}')
                else:
                    close_rpnl, close_fees = self.close_all(open)
                    rpnl += close_rpnl
                    fees += close_fees
            else:
                # Activation phase: tier-based daily target (mirrors topstep._handle_activation_phase)
                #   success_days < 4 + acc >= target → tier 3: $225 (1.5 × daily_target)
                #   success_days < 4 + acc <  target → tier 1: $1000
                #   success_days >= 4                → tier 2: 3 × target (effectively off)
                if ts.success_activation_days < 4:
                    if ts.acc_pnl >= ts.ACTIVATION_PNL_TARGET:
                        daily_target = 1.5 * ts.DAILY_ACTIVATION_TARGET
                    else:
                        daily_target = 1500.0
                else:
                    daily_target = 3 * ts.ACTIVATION_PNL_TARGET
                daily_remaining = daily_target - (ts.daily_pnl + rpnl + fees + floating_at_open) + exit_fees
                if daily_remaining > 0:
                    daily_remaining_clean = daily_target - (ts.daily_pnl + rpnl + fees) + exit_fees
                    self.stop_win = daily_remaining_clean / (abs(pos_after_resize) * self.TICK_VALUE_MICRO)
                    logger.debug(f'[SW SET ACT] sw={self.stop_win:.1f} rem={daily_remaining_clean:.0f} '
                                f'daily_target={daily_target:.0f} success_days={ts.success_activation_days} '
                                f'acc={ts.acc_pnl:.0f} daily={ts.daily_pnl:.0f} pos={pos_after_resize}')
                else:
                    close_rpnl, close_fees = self.close_all(open)
                    rpnl += close_rpnl
                    fees += close_fees

        # 3. Check stops using watermark (last_u/last_d) against intrabar high/low
        stop_rpnl, stop_fees, was_stopped, stop_px = self.check_stops(high, low, hl, last_u=last_u, last_d=last_d, gp_stop_check=next_open)
        rpnl += stop_rpnl
        fees += stop_fees

        pos = self.position_size_equivalent
        floating_pnl = 0.0 if was_stopped else self.cal_floating_pnl(close)

        abs_pos = abs(pos)
        if abs_pos > 0:
            adverse_px = low if pos > 0 else high
            max_risk = pos * (adverse_px - self.avg_entry_px) * self.TICK_VALUE_MICRO
            step_pnl = (close - open) * pos * self.TICK_VALUE_MICRO
            step_pnl_per_con = step_pnl / abs_pos
            floating_pnl_per_con = floating_pnl / abs_pos
        else:
            max_risk = 0.0
            step_pnl = 0.0
            step_pnl_per_con = 0.0
            floating_pnl_per_con = 0.0

        # Closed quantity from both phases (trade + stop)
        closed_pos = 0
        # Phase 1: trade closed portion (old_pos → pos_after_resize)
        if old_pos != 0 and pos_after_resize != old_pos:
            if (old_pos > 0) != (pos_after_resize > old_pos) or pos_after_resize == 0:
                closed_pos += min(abs(pos_after_resize - old_pos), abs(old_pos))
        # Phase 2: stop closed portion (pos_after_resize → pos)
        if pos_after_resize != 0 and pos != pos_after_resize:
            if (pos_after_resize > 0) != (pos > pos_after_resize) or pos == 0:
                closed_pos += min(abs(pos - pos_after_resize), abs(pos_after_resize))
        
        trade_info = self.topstep.update(rpnl, max_risk, day_end, fees ,rth=rth)

        if trade_info['reset_account']:
            # Flush any in-progress trade before wiping state
            if self._trade_closed_accum > 0:
                self.net_pnl_per_con_list.append(self._trade_pnl_accum / self._trade_closed_accum)
                self._trade_pnl_accum = 0.0
                self._trade_closed_accum = 0
            # Note: sim state (sim_entry_px, sim_daily_pnl) already reset via topstep._on_reset callback
            old_pos = 0
            self.mini_contracts = 0
            self.micro_contracts = 0
            if pos != 0:
                self.avg_entry_px = close
                d_mini, d_micro = self.get_position_size(target_equiv)
                rpnl, fees = self.execute_trade(close, d_mini, d_micro)
            pos = self.position_size_equivalent
            floating_pnl = 0.0  # just opened at close
            floating_pnl_per_con = 0.0
            step_pnl = 0.0
            step_pnl_per_con = 0.0
            net_pnl_per_con = 0.0
            net_pnl = 0.0
        else:
            # rpnl already in cash (has TICK_VALUE_MICRO baked in)
            rpnl_per_con = rpnl / closed_pos if closed_pos > 0 else 0.0

            # Accumulate partial closes: cash PnL per contract minus round-trip commission
            if closed_pos > 0:
                self._update_rt_comm()
                self._trade_pnl_accum += (rpnl_per_con - self._rt_comm)
                self._trade_closed_accum += closed_pos

            # Trade ended: went flat or flipped direction → flush to kelly list
            if self._trade_closed_accum > 0 and (pos == 0 or (old_pos != 0 and (old_pos > 0) != (pos > 0))):
                self.net_pnl_per_con_list.append(self._trade_pnl_accum / self._trade_closed_accum)
                self._trade_pnl_accum = 0.0
                self._trade_closed_accum = 0

            net_pnl = rpnl + fees
            net_pnl_per_con = net_pnl / closed_pos if closed_pos > 0 else 0.0
        
        self.topstep.total_commission += fees

        trade_info = {
                        **trade_info,
                        'net_pnl': net_pnl,
                        'net_pnl_per_con': net_pnl_per_con,
                        'rpnl': rpnl,
                        'fees': fees,
                        'closed_pos': closed_pos,
                        'step_pnl': step_pnl,
                        'step_pnl_per_con': step_pnl_per_con,
                        'floating_pnl_per_con': floating_pnl_per_con,
                        'floating_pnl': floating_pnl,
                        'position': self.position_size_equivalent,#current_pos
                        'was_stopped':was_stopped,
                        'stop_px': stop_px,
                        'stop_net_pnl': stop_rpnl + stop_fees,
                        'position_usage': self.position_size_equivalent/(self.topstep.max_contracts*self.MICRO_PER_MINI), 
                        'avg_entry_px': self.avg_entry_px,
                    }


        return trade_info

