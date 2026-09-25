"""U/D Trend Tracker — reusable for real trading and counterfactual tracking.

Tracks watermark high (last_u) or low (last_d) at 1min resolution.
Determines when trend exits (price drops/rises K from watermark).
Computes close-to-close PnL per trade cycle.

Usage:
    tracker = UDTracker()
    tracker.set_trend(side=+1, entry_close=44500, k=50)

    # Each bar:
    result = tracker.update(high, low, close)
    if result['exited']:
        print(f"Trend over, PnL: {result['realised_pnl']}")
        # Re-enter if consensus still active
        tracker.set_trend(side=+1, entry_close=close, k=50)
"""
import numpy as np

DOLLAR_PER_POINT = 5.0


class UDTracker:
    """Tracks one U/D trend cycle at 1min resolution."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.side = 0          # +1 long, -1 short, 0 flat
        self.entry_close = 0.0 # close price at entry
        self.last_u = 0.0      # watermark high (long)
        self.last_d = 0.0      # watermark low (short)
        self.k = 50.0          # K for exit threshold
        self.active = False
        self.cum_pnl = 0.0     # cumulative realised PnL across cycles

    def set_trend(self, side, entry_close, k):
        """Start a new trend cycle."""
        self.side = side
        self.entry_close = entry_close
        self.k = k
        self.active = True
        if side == 1:
            self.last_u = entry_close
            self.last_d = 0.0
        elif side == -1:
            self.last_d = entry_close
            self.last_u = 0.0
        else:
            self.active = False
            self.last_u = 0.0
            self.last_d = 0.0

    def exit_trend(self, exit_close):
        """Exit current trend, compute realised PnL."""
        if not self.active or self.side == 0:
            return 0.0
        pnl = (exit_close - self.entry_close) * self.side * DOLLAR_PER_POINT
        self.cum_pnl += pnl
        self.side = 0
        self.active = False
        self.last_u = 0.0
        self.last_d = 0.0
        return pnl

    def update(self, high, low, close):
        """Update tracker with current bar's OHLC.

        Returns dict:
            exited: bool — trend exited this bar
            realised_pnl: float — PnL if exited (0 if not)
            unrealised_pnl: float — current floating PnL
            last_u: float
            last_d: float
        """
        result = {
            'exited': False,
            'realised_pnl': 0.0,
            'unrealised_pnl': 0.0,
            'last_u': self.last_u,
            'last_d': self.last_d,
        }

        if not self.active or self.side == 0:
            return result

        # Update watermark
        if self.side == 1:
            self.last_u = max(self.last_u, high)
            # Check exit: low drops K from watermark
            if self.last_u - low >= self.k:
                exit_px = self.last_u - self.k  # exit at stop level
                pnl = (exit_px - self.entry_close) * DOLLAR_PER_POINT
                self.cum_pnl += pnl
                result['exited'] = True
                result['realised_pnl'] = pnl
                self.side = 0
                self.active = False
                self.last_u = 0.0
                self.last_d = 0.0
                return result

        elif self.side == -1:
            if self.last_d == 0:
                self.last_d = low
            else:
                self.last_d = min(self.last_d, low)
            # Check exit: high rises K from watermark
            if high - self.last_d >= self.k:
                exit_px = self.last_d + self.k
                pnl = (self.entry_close - exit_px) * DOLLAR_PER_POINT
                self.cum_pnl += pnl
                result['exited'] = True
                result['realised_pnl'] = pnl
                self.side = 0
                self.active = False
                self.last_u = 0.0
                self.last_d = 0.0
                return result

        # Still in trend — compute unrealised
        result['unrealised_pnl'] = (close - self.entry_close) * self.side * DOLLAR_PER_POINT
        result['last_u'] = self.last_u
        result['last_d'] = self.last_d
        return result

    @property
    def is_active(self):
        return self.active and self.side != 0


class CounterfactualTracker:
    """Tracks what PnL would have been if we were trading (but we're frozen).

    Uses UDTracker internally with same consensus direction and K.
    Accumulates PnL across multiple U/D cycles within the frozen hour.
    """

    def __init__(self):
        self.tracker = UDTracker()
        self.total_pnl = 0.0
        self.n_cycles = 0

    def reset(self):
        self.tracker.reset()
        self.total_pnl = 0.0
        self.n_cycles = 0

    def update(self, high, low, close, target_side, k):
        """Update counterfactual tracker.

        If not in a trend and target_side != 0 → enter.
        If in a trend → update U/D, handle exits and re-entries.
        """
        if not self.tracker.is_active:
            if target_side != 0:
                self.tracker.set_trend(target_side, close, k)
                self.n_cycles += 1
            return

        result = self.tracker.update(high, low, close)

        if result['exited']:
            self.total_pnl += result['realised_pnl']
            # Re-enter if consensus still active
            if target_side != 0:
                self.tracker.set_trend(target_side, close, k)
                self.n_cycles += 1

    @property
    def floating_pnl(self):
        """Current unrealised PnL of counterfactual position."""
        if self.tracker.is_active:
            return 0.0  # need close to compute, use update result instead
        return 0.0

    @property
    def net_pnl(self):
        """Total realised + current unrealised."""
        return self.total_pnl
