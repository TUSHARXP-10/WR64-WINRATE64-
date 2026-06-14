"""
walk_forward.py
================
Walk-forward validation harness for the candidate strategies.

Instead of one in-sample/out-of-sample split (which can look great or terrible
purely by luck of where the split falls), this fetches a longer history per
symbol, computes each candidate's signal ONCE on the full series (so slow
indicators like EMA300 get proper warm-up), then evaluates performance across
several SEQUENTIAL, non-overlapping windows. A strategy with a genuine edge
should be profitable (or at least not lose) in most windows across most
symbols -- not just in one lucky slice.

Costs/fees/slippage/funding are taken from btc_backtest.Config. Leverage is
fixed at 1x here on purpose: the goal is to judge signal quality in isolation
from leverage/liquidation effects (those are covered separately by
btc_backtest's risk-of-ruin table).

v2 round: every v1 strategy gets a "_v2" counterpart that adds an ADX-based
market-regime filter (only trend-trade when ADX shows a real trend, only
mean-revert when ADX shows a range), widens R:R slightly to reduce the drag
of per-trade costs, and -- for the SMC strategy -- adds a liquidity-sweep
requirement and a minimum Fair-Value-Gap size filter (closer to how SMC is
actually used). Timeframe moved from 1h to 4h: SMC/FVG concepts and ADX
regime filters are noisier and less meaningful on 1h.

Run:  python walk_forward.py
"""
import numpy as np
import pandas as pd

from btc_backtest import (
    Config, fetch_binance_ohlcv, backtest,
    calculate_ema, calculate_atr, calculate_rsi,
)


# --------------------------------------------------------------------------- #
# ADX (Wilder) -- market-regime filter
# --------------------------------------------------------------------------- #
def calculate_adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))

    atr_s = np.zeros(n)
    plus_dm_s = np.zeros(n)
    minus_dm_s = np.zeros(n)

    if n <= period:
        return np.zeros(n)

    atr_s[period] = tr[1:period+1].sum()
    plus_dm_s[period] = plus_dm[1:period+1].sum()
    minus_dm_s[period] = minus_dm[1:period+1].sum()

    for i in range(period+1, n):
        atr_s[i] = atr_s[i-1] - atr_s[i-1]/period + tr[i]
        plus_dm_s[i] = plus_dm_s[i-1] - plus_dm_s[i-1]/period + plus_dm[i]
        minus_dm_s[i] = minus_dm_s[i-1] - minus_dm_s[i-1]/period + minus_dm[i]

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    for i in range(period, n):
        if atr_s[i] > 0:
            plus_di[i] = 100 * plus_dm_s[i] / atr_s[i]
            minus_di[i] = 100 * minus_dm_s[i] / atr_s[i]
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum > 0 else 0.0

    adx = np.zeros(n)
    start = period * 2
    if start < n:
        adx[start] = dx[period:start+1].mean()
        for i in range(start+1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    return adx


# --------------------------------------------------------------------------- #
# Candlestick pattern helpers (operate on numpy OHLC arrays at index i)
# --------------------------------------------------------------------------- #
def _is_hammer(o, h, l, c, i):
    rng = h[i] - l[i]
    if rng <= 0:
        return False
    body = abs(c[i] - o[i])
    lower_wick = min(o[i], c[i]) - l[i]
    upper_wick = h[i] - max(o[i], c[i])
    return body <= 0.3 * rng and lower_wick >= 2 * body and upper_wick <= 0.15 * rng


def _is_shooting_star(o, h, l, c, i):
    rng = h[i] - l[i]
    if rng <= 0:
        return False
    body = abs(c[i] - o[i])
    upper_wick = h[i] - max(o[i], c[i])
    lower_wick = min(o[i], c[i]) - l[i]
    return body <= 0.3 * rng and upper_wick >= 2 * body and lower_wick <= 0.15 * rng


def _is_bullish_engulfing(o, h, l, c, i):
    return c[i-1] < o[i-1] and c[i] > o[i] and c[i] >= o[i-1] and o[i] <= c[i-1]


def _is_bearish_engulfing(o, h, l, c, i):
    return c[i-1] > o[i-1] and c[i] < o[i] and c[i] <= o[i-1] and o[i] >= c[i-1]


def _is_three_white_soldiers(o, h, l, c, i):
    return (c[i-2] > o[i-2] and c[i-1] > o[i-1] and c[i] > o[i] and
            c[i] > c[i-1] > c[i-2] and
            o[i-1] > o[i-2] and o[i-1] < c[i-2] and
            o[i] > o[i-1] and o[i] < c[i-1])


def _is_three_black_crows(o, h, l, c, i):
    return (c[i-2] < o[i-2] and c[i-1] < o[i-1] and c[i] < o[i] and
            c[i] < c[i-1] < c[i-2] and
            o[i-1] < o[i-2] and o[i-1] > c[i-2] and
            o[i] < o[i-1] and o[i] > c[i-1])


# --------------------------------------------------------------------------- #
# Trend-following family (v1 = no regime filter, v2 = ADX-filtered + wider R:R)
# --------------------------------------------------------------------------- #
def _trend_strategy(df, fast, slow, confirm_atr, sl_atr, tp_atr, rsi_lo, rsi_hi,
                     require_cross=True, atr_period=14, adx_min=0.0, adx_period=14):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    ema_f = calculate_ema(close, fast)
    ema_s = calculate_ema(close, slow)
    atr = calculate_atr(df, atr_period)
    rsi = calculate_rsi(df, atr_period)
    adx = calculate_adx(df, adx_period) if adx_min > 0 else None

    pos = np.zeros(n)
    current_pos = 0
    stop_loss = take_profit = np.nan
    start = max(slow, atr_period, (adx_period * 2 if adx_min > 0 else 0)) + 1

    for i in range(start, n):
        if current_pos != 0:
            if current_pos == 1:
                if low[i] <= stop_loss or high[i] >= take_profit:
                    current_pos = 0
            else:
                if high[i] >= stop_loss or low[i] <= take_profit:
                    current_pos = 0

        if current_pos == 0:
            trend_ok = True if adx is None else adx[i] >= adx_min

            long_cond = (close[i] > ema_s[i] and ema_f[i] > ema_f[i-1] and
                         close[i] > ema_f[i] + confirm_atr * atr[i] and
                         rsi[i] < rsi_hi and trend_ok)
            short_cond = (close[i] < ema_s[i] and ema_f[i] < ema_f[i-1] and
                           close[i] < ema_f[i] - confirm_atr * atr[i] and
                           rsi[i] > rsi_lo and trend_ok)
            if require_cross:
                long_cond = long_cond and close[i-1] <= ema_f[i-1]
                short_cond = short_cond and close[i-1] >= ema_f[i-1]

            if long_cond:
                current_pos = 1
                entry = close[i]
                stop_loss = entry - sl_atr * atr[i]
                take_profit = entry + tp_atr * atr[i]
            elif short_cond:
                current_pos = -1
                entry = close[i]
                stop_loss = entry + sl_atr * atr[i]
                take_profit = entry - tp_atr * atr[i]

        pos[i] = current_pos

    return pd.Series(pos, index=df.index)


def strat_original(df):
    """EMA20/EMA300 trend filter, same-bar crossover + 0.3xATR confirm, 0.7/2.1 ATR SL/TP, RSI 38-62"""
    return _trend_strategy(df, fast=20, slow=300, confirm_atr=0.3, sl_atr=0.7, tp_atr=2.1,
                            rsi_lo=38, rsi_hi=62, require_cross=True)


def strat_original_v2(df):
    """v1 + ADX>=20 trend-strength filter, wider 1.0/2.5 ATR SL/TP for better R:R after costs"""
    return _trend_strategy(df, fast=20, slow=300, confirm_atr=0.3, sl_atr=1.0, tp_atr=2.5,
                            rsi_lo=38, rsi_hi=62, require_cross=True, adx_min=20)


def strat_improved(df):
    """EMA20/EMA50, 0.1xATR confirm, 1.0/2.0 ATR SL/TP, RSI 30-70"""
    return _trend_strategy(df, fast=20, slow=50, confirm_atr=0.1, sl_atr=1.0, tp_atr=2.0,
                            rsi_lo=30, rsi_hi=70, require_cross=True)


def strat_improved_v2(df):
    """v1 + ADX>=18 trend-strength filter, wider 1.2/2.5 ATR SL/TP"""
    return _trend_strategy(df, fast=20, slow=50, confirm_atr=0.1, sl_atr=1.2, tp_atr=2.5,
                            rsi_lo=30, rsi_hi=70, require_cross=True, adx_min=18)


def strat_relaxed_trend(df):
    """EMA20/EMA100, no same-bar-crossover requirement (re-enter freely while flat), 1.0/2.0 ATR SL/TP, RSI 30-70"""
    return _trend_strategy(df, fast=20, slow=100, confirm_atr=0.2, sl_atr=1.0, tp_atr=2.0,
                            rsi_lo=30, rsi_hi=70, require_cross=False)


def strat_relaxed_v2(df):
    """v1 + ADX>=15 filter (cuts re-entries in chop), wider 1.2/2.5 ATR SL/TP"""
    return _trend_strategy(df, fast=20, slow=100, confirm_atr=0.2, sl_atr=1.2, tp_atr=2.5,
                            rsi_lo=30, rsi_hi=70, require_cross=False, adx_min=15)


# --------------------------------------------------------------------------- #
# Mean reversion family
# --------------------------------------------------------------------------- #
def strat_mean_reversion(df, rsi_period=14, rsi_lo=30, rsi_hi=70, sl_atr=1.2, tp_atr=1.5,
                          adx_max=100.0, adx_period=14):
    """RSI-extreme mean reversion: fade into oversold/overbought, target EMA20.
    adx_max < 100 restricts entries to range-bound regimes (low ADX)."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    atr = calculate_atr(df, 14)
    rsi = calculate_rsi(df, rsi_period)
    ema_mid = calculate_ema(close, 20)
    adx = calculate_adx(df, adx_period) if adx_max < 100 else None

    pos = np.zeros(n)
    current_pos = 0
    stop_loss = take_profit = np.nan
    start = max(21, (adx_period * 2 if adx is not None else 0) + 1)

    for i in range(start, n):
        if current_pos != 0:
            if current_pos == 1:
                if low[i] <= stop_loss or high[i] >= take_profit:
                    current_pos = 0
            else:
                if high[i] >= stop_loss or low[i] <= take_profit:
                    current_pos = 0

        if current_pos == 0:
            range_ok = True if adx is None else adx[i] <= adx_max

            if rsi[i] < rsi_lo and close[i] < ema_mid[i] and range_ok:
                current_pos = 1
                entry = close[i]
                stop_loss = entry - sl_atr * atr[i]
                tp_candidate = min(ema_mid[i], entry + tp_atr * atr[i])
                take_profit = tp_candidate if tp_candidate > entry else entry + tp_atr * atr[i]
            elif rsi[i] > rsi_hi and close[i] > ema_mid[i] and range_ok:
                current_pos = -1
                entry = close[i]
                stop_loss = entry + sl_atr * atr[i]
                tp_candidate = max(ema_mid[i], entry - tp_atr * atr[i])
                take_profit = tp_candidate if tp_candidate < entry else entry - tp_atr * atr[i]

        pos[i] = current_pos

    return pd.Series(pos, index=df.index)


def strat_mean_reversion_v2(df):
    """v1 + ADX<=18 range filter (only fade in chop), tighter RSI 25/75, 1.0/1.5 ATR SL/TP"""
    return strat_mean_reversion(df, rsi_lo=25, rsi_hi=75, sl_atr=1.0, tp_atr=1.5, adx_max=18.0)


# --------------------------------------------------------------------------- #
# SMC / Fair-Value-Gap / candlestick family
# --------------------------------------------------------------------------- #
def strat_smc_candles(df, atr_period=14, trend_fast=50, trend_slow=200,
                       fvg_lookback=30, sl_atr=1.0, tp_atr=2.5,
                       fvg_min_atr=0.0, sweep_lookback=0):
    """
    Smart-Money-Concepts-flavored strategy:
    - Trend bias from EMA50 vs EMA200
    - Entry requires price trading back inside a recent Fair Value Gap
      (3-candle imbalance) in the direction of the bias
    - Entry trigger is a candlestick reversal pattern at the FVG:
      hammer / bullish engulfing / three white soldiers for longs,
      shooting star / bearish engulfing / three black crows for shorts
    - SL beyond the pattern's extreme (with ATR buffer), TP = tp_atr x ATR

    fvg_min_atr > 0 requires the FVG to be at least that many ATRs wide
    (filters out noise gaps). sweep_lookback > 0 additionally requires a
    liquidity sweep -- the entry bar makes a fresh local high/low vs the
    prior `sweep_lookback` bars before the reversal pattern confirms.
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)

    ema_fast = calculate_ema(c, trend_fast)
    ema_slow = calculate_ema(c, trend_slow)
    atr = calculate_atr(df, atr_period)

    pos = np.zeros(n)
    current_pos = 0
    stop_loss = take_profit = np.nan

    bull_zones = []  # list of (bottom, top, created_idx)
    bear_zones = []

    start = max(trend_slow, atr_period, sweep_lookback) + 3

    for i in range(start, n):
        # --- update Fair Value Gap zones (3-candle imbalance) ---
        if l[i] > h[i-2] and (l[i] - h[i-2]) >= fvg_min_atr * atr[i]:
            bull_zones.append((h[i-2], l[i], i))
        if h[i] < l[i-2] and (l[i-2] - h[i]) >= fvg_min_atr * atr[i]:
            bear_zones.append((h[i], l[i-2], i))
        bull_zones = [z for z in bull_zones if i - z[2] <= fvg_lookback]
        bear_zones = [z for z in bear_zones if i - z[2] <= fvg_lookback]

        # --- manage open position ---
        if current_pos != 0:
            if current_pos == 1:
                if l[i] <= stop_loss or h[i] >= take_profit:
                    current_pos = 0
            else:
                if h[i] >= stop_loss or l[i] <= take_profit:
                    current_pos = 0

        # --- look for new entry ---
        if current_pos == 0:
            bull_bias = ema_fast[i] > ema_slow[i]
            bear_bias = ema_fast[i] < ema_slow[i]

            in_bull_zone = any(z[0] <= c[i] <= z[1] for z in bull_zones)
            in_bear_zone = any(z[0] <= c[i] <= z[1] for z in bear_zones)

            bull_pattern = (_is_hammer(o, h, l, c, i) or
                            _is_bullish_engulfing(o, h, l, c, i) or
                            _is_three_white_soldiers(o, h, l, c, i))
            bear_pattern = (_is_shooting_star(o, h, l, c, i) or
                            _is_bearish_engulfing(o, h, l, c, i) or
                            _is_three_black_crows(o, h, l, c, i))

            swept_low = swept_high = True
            if sweep_lookback > 0:
                swept_low = l[i] < np.min(l[i-sweep_lookback:i])
                swept_high = h[i] > np.max(h[i-sweep_lookback:i])

            if bull_bias and in_bull_zone and bull_pattern and swept_low:
                current_pos = 1
                entry = c[i]
                stop_loss = min(l[i], l[i-1]) - 0.1 * atr[i]
                if stop_loss >= entry:
                    stop_loss = entry - sl_atr * atr[i]
                take_profit = entry + tp_atr * atr[i]
            elif bear_bias and in_bear_zone and bear_pattern and swept_high:
                current_pos = -1
                entry = c[i]
                stop_loss = max(h[i], h[i-1]) + 0.1 * atr[i]
                if stop_loss <= entry:
                    stop_loss = entry + sl_atr * atr[i]
                take_profit = entry - tp_atr * atr[i]

        pos[i] = current_pos

    return pd.Series(pos, index=df.index)


def strat_smc_v2(df):
    """v1 + liquidity-sweep requirement + minimum FVG size (0.3xATR) -- closer to real ICT usage"""
    return strat_smc_candles(df, fvg_min_atr=0.3, sweep_lookback=10)


STRATEGIES = {
    "original_ema20_300": strat_original,
    "original_v2_adx": strat_original_v2,
    "improved_ema20_50": strat_improved,
    "improved_v2_adx": strat_improved_v2,
    "relaxed_ema20_100": strat_relaxed_trend,
    "relaxed_v2_adx": strat_relaxed_v2,
    "mean_reversion_rsi": strat_mean_reversion,
    "mean_reversion_v2_adx": strat_mean_reversion_v2,
    "smc_fvg_candles": strat_smc_candles,
    "smc_v2_sweep": strat_smc_v2,
}


# --------------------------------------------------------------------------- #
# Walk-forward evaluation
# --------------------------------------------------------------------------- #
def _window_metrics(bar_ret, held, bpy):
    r = np.asarray(bar_ret, dtype=float)
    held = np.asarray(held, dtype=float)
    total_return = float(np.prod(1.0 + r) - 1.0)
    sharpe = float((r.mean() * bpy) / (r.std() * np.sqrt(bpy))) if r.std() > 0 else 0.0
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float(len(wins) / max(1, len(wins) + len(losses)))
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    trades = int((np.abs(np.diff(np.sign(held), prepend=0.0)) > 0).sum())
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "profit_factor": pf,
        "trades": trades,
    }


def walk_forward(df, pos, cfg, n_windows=5, warmup=310):
    res = backtest(df, pos, cfg)
    bar_ret = res.bar_returns.to_numpy()
    tgt = pos.to_numpy(float)
    held = np.zeros(len(tgt))
    held[1:] = tgt[:-1]

    usable = np.arange(warmup, len(df))
    edges = np.array_split(usable, n_windows)

    windows = []
    for w, idx in enumerate(edges):
        if len(idx) == 0:
            continue
        m = _window_metrics(bar_ret[idx], held[idx], cfg.bars_per_year)
        m["window"] = w + 1
        m["bars"] = (int(idx[0]), int(idx[-1]))
        windows.append(m)
    return windows, res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    symbols = ["BTCUSDT", "BNBUSDT", "ETHUSDT"]
    interval = "4h"
    total_bars = 3000
    n_windows = 5
    warmup = 310
    cfg = Config(leverage=1.0)  # isolate signal quality from leverage/liquidation

    summary_rows = []
    detail_rows = []

    for symbol in symbols:
        print(f"Fetching {symbol} ({total_bars} x {interval} bars)...")
        df = fetch_binance_ohlcv(symbol, interval, total_bars)

        for name, fn in STRATEGIES.items():
            pos = fn(df)
            windows, res = walk_forward(df, pos, cfg, n_windows=n_windows, warmup=warmup)

            n_w = len(windows)
            mean_sharpe = float(np.mean([w["sharpe"] for w in windows])) if n_w else 0.0
            mean_return = float(np.mean([w["total_return"] for w in windows])) if n_w else 0.0
            pct_pos = float(np.mean([w["total_return"] > 0 for w in windows])) if n_w else 0.0
            total_trades = int(sum(w["trades"] for w in windows))
            traded_windows = [w for w in windows if w["trades"] > 0]
            mean_win_rate = float(np.mean([w["win_rate"] for w in traded_windows])) if traded_windows else float("nan")

            summary_rows.append({
                "symbol": symbol,
                "strategy": name,
                "windows": n_w,
                "pct_windows_profitable": pct_pos,
                "mean_window_sharpe": mean_sharpe,
                "mean_window_return": mean_return,
                "total_trades": total_trades,
                "mean_win_rate": mean_win_rate,
            })

            for w in windows:
                detail_rows.append({
                    "symbol": symbol,
                    "strategy": name,
                    "window": w["window"],
                    "bars": f"{w['bars'][0]}-{w['bars'][1]}",
                    "return": w["total_return"],
                    "sharpe": w["sharpe"],
                    "trades": w["trades"],
                    "win_rate": w["win_rate"],
                    "profit_factor": w["profit_factor"],
                })

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)

    pd.set_option("display.width", 200)

    print("\n" + "=" * 100)
    print("PER-WINDOW DETAIL (sequential, non-overlapping windows over the same period for every strategy)")
    print("=" * 100)
    print(detail.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print(f"WALK-FORWARD SUMMARY ({interval} bars, leverage=1x, costs included -- isolates raw signal quality)")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("AGGREGATE ACROSS SYMBOLS (per strategy)")
    print("=" * 100)
    agg = summary.groupby("strategy").agg(
        pct_windows_profitable=("pct_windows_profitable", "mean"),
        mean_window_sharpe=("mean_window_sharpe", "mean"),
        mean_window_return=("mean_window_return", "mean"),
        total_trades=("total_trades", "sum"),
    ).reset_index().sort_values("mean_window_sharpe", ascending=False)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
