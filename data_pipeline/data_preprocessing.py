"""
Bar features for the RL agent: the polars version of reference/preprocessing_pandas.py. Runs after step 2:

    python -m data_pipeline.data_preprocessing                  # step 2's default output -> training_data.parquet in it
    python -m data_pipeline.data_preprocessing --data-dir data/processed_2024 --out data/features_2024.parquet

Reads {data-dir}/{freq}_ohlcv.parquet for every freq in params.FREQS and adds, per freq (window 20): SMA, std,
Bollinger bands (2 sigma), ATR, SMA-RSI, FVG, bar-to-bar moves, bar score and the U/D levels. Every higher
freq is then joined onto the 1-min bars: a bar appears on the 1-min row during which it closes and stays
until the next one closes, so a row holds nothing from the future once that 1-min bar has closed.
Prices are in ticks (x4), like the bar files.
"""
import argparse
import os
import time

import numpy as np
import polars as pl
from numba import njit

from data_pipeline.to_dat import DEFAULT_OUT as DATA_DIR
from params import FREQS, NORM_FACTOR

# A session is [00:00, 23:00) on the shifted clock (New York + 6h): no bar runs past 23:00
SESSION_SECONDS = 23 * 3600

@njit(cache=True)
def _calc_ud_levels(px, kv):
    n = len(px)
    U_arr = np.full(n, np.nan)
    D_arr = np.full(n, np.nan)
    flag_arr = np.full(n, -1, dtype=np.int8)

    U_last = 0.0
    D_last = 0.0

    for i in range(n):
        if i == 0:
            # Start on the D side. The reference picked the side from 10 bars ahead (px[i+10]): a peek at the
            # future, so it is left out; the levels only differ from the reference until the state first resets.
            U_last = 0.0
            D_last = px[i]

        U_update = U_last
        D_update = D_last

        if U_last == 0 and px[i] - D_last > kv[i]:  # leave_d
            U_update = px[i]
            flag_arr[i] = 1
        elif U_last == 0 or U_last - px[i] > kv[i]:  # leave_u
            if i > 0:
                U_arr[i-1] = U_last
            U_update = 0.0
        elif px[i] > U_last:
            U_update = px[i]
            flag_arr[i] = 1
        else:
            U_update = U_last
            flag_arr[i] = 1

        if D_last == 0 and U_last - px[i] > kv[i]:  # leave_u
            D_update = px[i]
        elif D_last == 0 or px[i] - D_last > kv[i]:  # leave_d
            if i > 0:
                D_arr[i-1] = D_last
            D_update = 0.0
        elif px[i] < D_last:
            D_update = px[i]
        else:
            D_update = D_last

        U_last = U_update
        D_last = D_update

    # A level of 0 means "no level": NaN, as in the reference (not a price of 0)
    U_arr[U_arr == 0] = np.nan
    D_arr[D_arr == 0] = np.nan

    # Shift forward by 1 to prevent look-ahead (the level at i-1 is only known at bar i)
    U_out = np.concatenate((np.array([np.nan]), U_arr[:-1]))
    D_out = np.concatenate((np.array([np.nan]), D_arr[:-1]))
    flag_out = np.concatenate((np.array([-1], dtype=np.int8), flag_arr[:-1]))

    return U_out, D_out, flag_out


def calc_ud_levels_polars(df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Wrapper to run the Numba UD level calculation inside Polars."""
    px = df['close'].to_numpy().astype(np.float64)
    # The first window-1 bars have no std: NaN, so every comparison with it is False, as in the reference
    kv = df[f'{window}_std'].to_numpy().astype(np.float64)

    u, d, flag = _calc_ud_levels(px, kv)

    return df.with_columns([
        pl.Series(f"{window}_U", u),
        pl.Series(f"{window}_D", d),
        pl.Series(f"{window}_UD_flag", flag)
    ])


class DataPreprocessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def add_technical_indicators(self, lf: pl.LazyFrame, window: int = 20) -> pl.LazyFrame:
        """Add standard indicators (SMA, STD, ATR, RSI, Bollinger Bands, FVG)."""

        lf = lf.with_columns([
            # 1. SMA & STD
            pl.col('close').rolling_mean(window_size=window).alias(f'{window}_sma'),
            pl.col('close').rolling_std(window_size=window, ddof=0).alias(f'{window}_std'),

            # True Range for ATR
            pl.max_horizontal([
                pl.col('high') - pl.col('low'),
                (pl.col('high') - pl.col('close').shift(1)).abs(),
                (pl.col('low') - pl.col('close').shift(1)).abs()
            ]).alias('_tr')
        ])

        # 2. Bollinger Bands, 2 sigma (the pandas reference used 4)
        lf = lf.with_columns([
            ((pl.col('close') - pl.col(f'{window}_sma')) / (2 * pl.col(f'{window}_std'))).alias(f'{window}_bbands'),
            pl.col(f'{window}_sma').alias(f'{window}_bband_mid'),
            (pl.col(f'{window}_sma') + pl.col(f'{window}_std') * 2).alias(f'{window}_bband_upper'),
            (pl.col(f'{window}_sma') - pl.col(f'{window}_std') * 2).alias(f'{window}_bband_lower')
        ])

        # 3. ATR (Wilder's smoothing over 14 bars, whatever the window, as in the reference)
        lf = lf.with_columns(
            pl.col('_tr').ewm_mean(com=13, min_samples=14, ignore_nulls=True, adjust=False).alias(f'{window}_atr')
        ).drop('_tr')

        # 4. SMA RSI (Custom Logic)
        lf = lf.with_columns([
            (pl.col('close') - pl.col('close').shift(1)).alias('_change')
        ]).with_columns([
            # clip keeps the first bar's missing change missing, as pandas does
            pl.col('_change').clip(lower_bound=0).rolling_mean(14).alias('_avg_gain'),
            (-pl.col('_change')).clip(lower_bound=0).rolling_mean(14).alias('_avg_loss')
        ]).with_columns([
            (pl.col('_avg_gain') / pl.col('_avg_loss')).alias('_rs')
        ]).with_columns([
            # If avg_loss is 0, RSI is 100
            pl.when(pl.col('_avg_loss') == 0).then(100)
              .otherwise(100 - (100 / (1 + pl.col('_rs')))).alias('_rsi_0_100')
        ]).with_columns([
            # Scale to [-1, 1] and invert
            (((pl.col('_rsi_0_100') / 100) * 2 - 1) * -1).alias(f'{window}_sma_rsi')
        ]).drop(['_change', '_avg_gain', '_avg_loss', '_rs', '_rsi_0_100'])

        # 5. FVG (Fair Value Gap)
        # c3.low > c1.high (Bullish) or c3.high < c1.low (Bearish)
        lf = lf.with_columns([
            pl.col('high').shift(2).alias('_c1_high'),
            pl.col('low').shift(2).alias('_c1_low'),
            pl.col('open').shift(1).alias('_c2_open'),
            pl.col('close').shift(1).alias('_c2_close'),
            (pl.col('high').shift(1) - pl.col('low').shift(1)).alias('_c2_range')
        ]).with_columns([
            pl.when(
                (pl.col('low') > pl.col('_c1_high')) &  # Gap
                (pl.col('_c2_close') > pl.col('_c2_open')) & # Momentum
                ((pl.col('high') - pl.col('low')) >= 1.5 * ((pl.col('_c1_high') - pl.col('_c1_low') + pl.col('_c2_range')) / 2)) # Size
            ).then(1)
            .when(
                (pl.col('high') < pl.col('_c1_low')) &  # Gap
                (pl.col('_c2_close') < pl.col('_c2_open')) & # Momentum
                ((pl.col('high') - pl.col('low')) >= 1.5 * ((pl.col('_c1_high') - pl.col('_c1_low') + pl.col('_c2_range')) / 2)) # Size
            ).then(-1)
            .otherwise(0).alias(f'{window}_fvg')
        ]).drop(['_c1_high', '_c1_low', '_c2_open', '_c2_close', '_c2_range'])

        # 6. Price Session Coordinate Shifts
        lf = lf.with_columns([
            (pl.col('open') - pl.col('open').shift(1)).alias('HO'),
            (pl.col('high') - pl.col('high').shift(1)).alias('HH'),
            (pl.col('low') - pl.col('low').shift(1)).alias('HL'),
            (pl.col('close') - pl.col('close').shift(1)).alias('HC')
        ])

        # 7. Bar Score
        lf = lf.with_columns([
            (
                (pl.max_horizontal(pl.col('high') - pl.col('open'), pl.col('open') - pl.col('low')) -
                 pl.min_horizontal(pl.col('high') - pl.col('open'), pl.col('open') - pl.col('low')) * 2 +
                 (pl.col('close') - pl.col('open')).abs()) / NORM_FACTOR
            ).alias('bar_score')
        ])

        return lf

    def process_frequency(self, freq: str) -> pl.DataFrame:
        """Load and process a single timeframe."""
        file_path = os.path.join(self.data_dir, f'{freq}_ohlcv.parquet')
        lf = pl.scan_parquet(file_path)

        # Ensure sorted by time
        lf = lf.sort('ts')

        # Add indicators
        lf = self.add_technical_indicators(lf, window=20)

        # Collect to DataFrame because UD levels requires Numba (can't be lazy)
        df = lf.collect()

        # Calculate stateful UD levels
        df = calc_ud_levels_polars(df, window=20)

        # Rename columns to have frequency suffix (except ts and join keys)
        rename_dict = {col: f"{col}_{freq}" for col in df.columns if col not in ['ts', 'start_ind', 'rth', 'session', 'hour']}
        df = df.rename(rename_dict)

        return df

    def build_merged_dataset(self) -> pl.DataFrame:
        """Load 1m data and asof join all higher timeframes onto it."""
        print("Processing 1m base data...")
        df_1m = self.process_frequency('1')

        for freq in FREQS:
            if freq == '1':
                continue

            print(f"Processing and joining {freq if freq == 'day' else freq + 'm'} data...")
            df_htf = self.process_frequency(freq)

            # No look-ahead: a bar appears on the 1-min row during which it closes, i.e. its ts moves to its
            # close minus 1 minute. The 15-min bar 10:00-10:14:59 lands on the 10:14 row, whose close is 10:15:00.
            # A bar closes after its length, or at the session end (23:00) if that comes first (the day bar).
            bar_seconds = 24 * 3600 if freq == 'day' else int(freq) * 60
            session_end = (pl.col('ts') // 86400) * 86400 + SESSION_SECONDS
            df_htf = df_htf.with_columns(
                (pl.min_horizontal(pl.col('ts') + bar_seconds, session_end) - 60).alias('ts')
            )

            # ASOF JOIN: For every 1-minute tick, find the MOST RECENT *completed* higher timeframe bar.
            # strategy='backward' ensures absolutely zero lookahead bias.
            df_1m = df_1m.join_asof(
                df_htf.drop(['start_ind', 'rth', 'session', 'hour'], strict=False),
                on='ts',
                strategy='backward'
            )

        return df_1m

def main(argv=None):
    parser = argparse.ArgumentParser(description="Bar features, higher freqs joined onto the 1-min bars.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="step 2's output folder with the {freq}_ohlcv.parquet files")
    parser.add_argument("--out", default=None, help="output parquet (default: training_data.parquet in --data-dir)")
    args = parser.parse_args(argv)
    out = args.out or os.path.join(args.data_dir, "training_data.parquet")

    t0 = time.time()
    final_df = DataPreprocessor(data_dir=args.data_dir).build_merged_dataset()
    print(f"\nPipeline completed in {time.time() - t0:.2f} seconds!")
    print(f"Final shape: {final_df.shape}")
    print("Sample of merged data:")
    print(final_df.tail())

    final_df.write_parquet(out)
    print(f"Data saved to: {out}")


if __name__ == "__main__":
    main()
