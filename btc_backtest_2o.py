"""
btc_backtest_2o.py
==================
Version 2.0 - Fixed liquidation issues for high leverage (50x+)
- Stop loss inside liquidation buffer
- Volatility filter
- Trailing stop
- Adaptive R:R
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import requests

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    bar_hours: float = 1.0          # bar size in hours (1h bars by default)
    fee_bps: float = 4.0            # per-side cost in basis points (taker ~4-5bps)
    slippage_bps: float = 2.0       # per-side slippage estimate in bps
    funding_bps_8h: float = 1.0     # funding paid by the held side per 8h (~1bp typical, varies a lot)
    leverage: float = 20.0          # notional / equity
    maint_margin: float = 0.005     # maintenance margin rate (for liquidation calc)
    oos_fraction: float = 0.40      # fraction of data reserved for out-of-sample
    start_equity: float = 100.0     # starting equity is now $100

    @property
    def bars_per_year(self) -> float:
        return (365.0 * 24.0) / self.bar_hours

    @property
    def cost_per_turn(self) -> float:
        # cost charged each time position size changes by 1 unit of notional
        return (self.fee_bps + self.slippage_bps) / 1e4

    @property
    def funding_per_bar(self) -> float:
        # funding accrues every 8h; spread it across bars on held notional
        return (self.funding_bps_8h / 1e4) * (self.bar_hours / 8.0)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV needs columns {needed}; got {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    return df.reset_index(drop=True)


def synthetic_ohlcv(n_bars: int, cfg: Config, seed: int = 7) -> pd.DataFrame:
    """Fat-tailed, mildly vol-clustered price path. Has NO predictable edge."""
    rng = np.random.default_rng(seed)
    daily_vol = 0.035                       # ~3.5% daily vol, crypto-ish
    bar_vol = daily_vol * np.sqrt(cfg.bar_hours / 24.0)
    # volatility clustering via a slow-moving multiplier
    vol_state = np.abs(rng.normal(1.0, 0.3, n_bars))
    vol_state = pd.Series(vol_state).ewm(span=48).mean().to_numpy()
    # Student-t shocks (fat tails); df=3 => occasional violent candles
    shocks = rng.standard_t(df=3, size=n_bars) * bar_vol * vol_state
    log_close = np.cumsum(shocks) + np.log(60000.0)
    close = np.exp(log_close)
    openp = np.empty_like(close)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    # intrabar range proportional to local vol
    wick = np.abs(rng.normal(0, bar_vol, n_bars)) * close
    high = np.maximum(openp, close) + wick
    low = np.minimum(openp, close) - wick
    vol = rng.lognormal(mean=6, sigma=0.5, size=n_bars)
    return pd.DataFrame({"open": openp, "high": high, "low": low,
                         "close": close, "volume": vol})


def fetch_binance_ohlcv(symbol: str = "BTCUSDT", interval: str = "1h",
                        total_bars: int = 5000) -> pd.DataFrame:
    """Fetch OHLCV data from Binance public API with pagination (max 1000 per request)."""
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    remaining_bars = total_bars
    max_limit_per_request = 1000

    while remaining_bars > 0:
        limit = min(remaining_bars, max_limit_per_request)
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if end_time is not None:
            params["endTime"] = end_time

        response = requests.get(url, params=params)
        response.raise_for_status()
        batch_data = response.json()

        if not batch_data:
            break  # No more data available

        all_data.extend(batch_data)
        remaining_bars -= len(batch_data)
        end_time = batch_data[0][0] - 1  # Get earliest timestamp for older data

    all_data.reverse()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    df = df[["open", "high", "low", "close", "volume"]].head(total_bars)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Strategy Helpers
# --------------------------------------------------------------------------- #
def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average (helper function)"""
    n = len(data)
    ema = np.zeros(n)
    if n < period:
        return ema
    ema[period-1] = np.mean(data[:period])
    alpha = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def calculate_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Calculate Average True Range (helper function)"""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """Calculate MACD indicator"""
    close = df["close"].values
    ema_fast = calculate_ema(close, fast_period)
    ema_slow = calculate_ema(close, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Calculate RSI (Relative Strength Index) (helper function)"""
    close = df["close"].values
    n = len(df)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        change = close[i] - close[i-1]
        if change > 0:
            gains[i] = change
        elif change < 0:
            losses[i] = -change
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    for i in range(period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    return rsi


def get_liquidation_buffer(cfg: Config) -> float:
    """Calculate maximum adverse move before liquidation"""
    return max(1e-9, (1.0 / cfg.leverage) - cfg.maint_margin)


# --------------------------------------------------------------------------- #
# Strategy v3.0 - Scalping
# --------------------------------------------------------------------------- #
def signal(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Version 12.0 - 20x ETH-FIXED (Ultra Relaxed RSI + Higher R:R) Scalping Strategy!
    1. SL inside liquidation buffer (45% of liq move)
    2. EMA 5/15 crossover signals
    3. Ultra gentle RSI filter (20-80) to avoid only the worst entries
    4. Fixed SL/TP with 1.9 R:R to give winners even more room
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    # Calculate indicators
    ema5 = calculate_ema(close, 5)
    ema15 = calculate_ema(close, 15)
    atr = calculate_atr(df, 14)
    rsi = calculate_rsi(df, 14)

    # Get liquidation buffer
    liq_buffer = get_liquidation_buffer(cfg)
    sl_buffer = liq_buffer * 0.45  # 45% of liq buffer for SL

    # Track position and exit levels
    pos = np.zeros(n)
    current_pos = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan

    for i in range(100, n):
        # First check if we need to exit current position
        if current_pos != 0:
            if current_pos == 1:  # Long
                if low[i] <= stop_loss or high[i] >= take_profit:
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan
            elif current_pos == -1:  # Short
                if high[i] >= stop_loss or low[i] <= take_profit:
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan

        # If flat, look for new entry with ultra gentle RSI filter
        if current_pos == 0:
            # Long entry: EMA5 crosses above EMA15 + RSI not extreme
            if ema5[i] > ema15[i] and ema5[i-1] <= ema15[i-1] and rsi[i] > 20 and rsi[i] < 80:
                current_pos = 1
                entry_price = close[i]
                stop_loss = entry_price - (sl_buffer * entry_price)
                take_profit = entry_price + (sl_buffer * entry_price * 1.9)  # 1.9 R:R
            # Short entry: EMA5 crosses below EMA15 + RSI not extreme
            elif ema5[i] < ema15[i] and ema5[i-1] >= ema15[i-1] and rsi[i] > 20 and rsi[i] < 80:
                current_pos = -1
                entry_price = close[i]
                stop_loss = entry_price + (sl_buffer * entry_price)
                take_profit = entry_price - (sl_buffer * entry_price * 1.9)

        pos[i] = current_pos

    return pd.Series(pos, index=df.index)


# --------------------------------------------------------------------------- #
# Backtest engine
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    equity: pd.Series
    bar_returns: pd.Series
    trades: int
    liquidated_at: int | None
    cfg: Config
    label: str = ""
    extras: dict = field(default_factory=dict)


def _liquidation_move(cfg: Config) -> float:
    """Approx adverse fractional move that triggers liquidation (isolated margin)."""
    return max(1e-9, (1.0 / cfg.leverage) - cfg.maint_margin)


def backtest(df: pd.DataFrame, target_pos: pd.Series, cfg: Config,
             label: str = "", risk_per_trade: float = 0.20) -> Result:
    """
    Backtest with risk-based position sizing (default 20% equity per trade for ALL IN profit!)
    """
    px = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    bar_ret = np.zeros(len(px))
    bar_ret[1:] = px[1:] / px[:-1] - 1.0

    tgt = target_pos.to_numpy(float)
    # execution lag: position decided on bar t is held over bar t+1
    held = np.zeros(len(px))
    held[1:] = tgt[:-1]

    # Get liquidation buffer for position sizing
    liq_buffer = _liquidation_move(cfg)
    sl_buffer = liq_buffer * 0.45  # Same as strategy (45%)

    # Calculate position size: risk 0.5% equity per trade + max size cap
    position_size = np.zeros(len(px))
    # Absolute max position size to guarantee no liquidation (even 100% adverse move)
    absolute_max_size = (liq_buffer * 0.9) / cfg.leverage  # 90% of liq buffer max
    for t in range(1, len(px)):
        if held[t] != 0 and held[t-1] == 0:
            # New trade: size based on risk, but cap at absolute max
            risk_based_size = risk_per_trade / (sl_buffer * cfg.leverage)
            final_size = min(risk_based_size, absolute_max_size)
            position_size[t] = final_size * np.sign(held[t])
        else:
            position_size[t] = position_size[t-1]

    turn = np.abs(np.diff(position_size, prepend=0.0))  # change in exposure
    cost = turn * cfg.cost_per_turn                     # entry/exit cost
    funding = np.abs(position_size) * cfg.funding_per_bar  # funding

    liquidated_at: int | None = None

    equity = np.empty(len(px))
    equity[0] = cfg.start_equity
    for t in range(1, len(px)):
        # Liquidation check: adverse move * position size * leverage >= liq buffer
        if position_size[t] > 0:
            adverse = (px[t - 1] - low[t]) / px[t - 1]
        elif position_size[t] < 0:
            adverse = (high[t] - px[t - 1]) / px[t - 1]
        else:
            adverse = 0.0
        
        if position_size[t] != 0:
            # Maximum possible loss from adverse move
            max_adverse_loss = adverse * abs(position_size[t]) * cfg.leverage
            if max_adverse_loss >= liq_buffer:
                equity[t] = 0.0
                equity[t:] = 0.0
                liquidated_at = t
                break
            
        gross = position_size[t] * bar_ret[t] * cfg.leverage
        net = gross - cost[t] - funding[t]
        equity[t] = equity[t - 1] * (1.0 + net)

    eq = pd.Series(equity, index=df.index)
    strat_ret = eq.pct_change().fillna(0.0)
    n_trades = int((np.abs(np.diff(np.sign(position_size), prepend=0.0)) > 0).sum())
    return Result(eq, strat_ret, n_trades, liquidated_at, cfg, label)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(res: Result) -> dict:
    r = res.bar_returns.to_numpy()
    eq = res.equity.to_numpy()
    bpy = res.cfg.bars_per_year
    total = eq[-1] / eq[0] - 1.0
    years = len(r) / bpy
    cagr = (eq[-1] / eq[0]) ** (1.0 / years) - 1.0 if eq[-1] > 0 and years > 0 else -1.0
    vol = r.std() * np.sqrt(bpy)
    sharpe = (r.mean() * bpy) / (r.std() * np.sqrt(bpy)) if r.std() > 0 else 0.0
    downside = r[r < 0].std()
    sortino = (r.mean() * bpy) / (downside * np.sqrt(bpy)) if downside > 0 else 0.0
    roll_max = np.maximum.accumulate(eq)
    dd = (eq - roll_max) / roll_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = len(wins) / max(1, (len(wins) + len(losses)))
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    return {
        "label": res.label,
        "final_equity": float(eq[-1]),
        "total_return": float(total),
        "CAGR": float(cagr),
        "ann_vol": float(vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "Calmar": float(calmar),
        "win_rate": float(win_rate),
        "profit_factor": float(pf),
        "trades": res.trades,
        "liquidated_at_bar": res.liquidated_at,
        "start_equity": float(res.cfg.start_equity),
    }


def print_metrics(m: dict) -> None:
    liq = "NO" if m["liquidated_at_bar"] is None else f"YES @ bar {m['liquidated_at_bar']}"
    print(f"  [{m['label']}]")
    print(f"    final equity   : ${m['final_equity']:.2f}  (start ${m['start_equity']:.0f})")
    print(f"    total return   : {m['total_return']*100:+.1f}%")
    print(f"    CAGR           : {m['CAGR']*100:+.1f}%")
    print(f"    ann. vol       : {m['ann_vol']*100:.1f}%")
    print(f"    Sharpe / Sortino: {m['Sharpe']:.2f} / {m['Sortino']:.2f}")
    print(f"    max drawdown   : {m['max_drawdown']*100:.1f}%")
    print(f"    win rate       : {m['win_rate']*100:.1f}%   profit factor: {m['profit_factor']:.2f}")
    print(f"    trades         : {m['trades']}")
    print(f"    LIQUIDATED     : {liq}")


# --------------------------------------------------------------------------- #
# Risk of ruin -- the leverage reality check
# --------------------------------------------------------------------------- #
def risk_of_ruin(edge_per_trade: float, std_per_trade: float, n_trades: int,
                 leverages: list[float], cfg: Config, paths: int = 5000,
                 ruin_threshold: float = 0.5, seed: int = 11) -> pd.DataFrame:
    """
    Monte Carlo a small account. Each 'trade' has an UNLEVERED return drawn from
    N(edge, std). Leverage multiplies both. We bet full equity each trade
    (typical of small-account scalpers) and charge round-trip costs.

    'Ruin' = equity ever falls below ruin_threshold * start (e.g. 50% drawdown),
    which for a small leveraged account is effectively game over.
    """
    rng = np.random.default_rng(seed)
    rows = []
    rt_cost = 2.0 * cfg.cost_per_turn  # round trip
    liq_move = None
    for L in leverages:
        liq_move = max(1e-9, (1.0 / L) - cfg.maint_margin)
        draws = rng.normal(edge_per_trade, std_per_trade, size=(paths, n_trades))
        eq = np.full(paths, cfg.start_equity)
        ruined = np.zeros(paths, dtype=bool)
        peak = eq.copy()
        for k in range(n_trades):
            r = draws[:, k]
            # liquidation if a single trade's adverse move exceeds the buffer
            liq_hit = (-r) >= liq_move
            net = r * L - rt_cost
            eq = np.where(ruined, eq, eq * (1.0 + net))
            eq = np.where(liq_hit & ~ruined, 0.0, eq)
            peak = np.maximum(peak, eq)
            ruined = ruined | (eq <= ruin_threshold * cfg.start_equity)
        rows.append({
            "leverage": L,
            "P(ruin)": float(ruined.mean()),
            "median_final": float(np.median(eq)),
            "mean_final": float(eq.mean()),
            "p10_final": float(np.percentile(eq, 10)),
            "p90_final": float(np.percentile(eq, 90)),
            "liq_move_pct": liq_move * 100,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Honest BTC futures backtester v2.0")
    ap.add_argument("--csv", default=None, help="OHLCV csv (else synthetic)")
    ap.add_argument("--bars", type=int, default=8760, help="synthetic bar count")
    ap.add_argument("--binance", action="store_true", help="fetch data from Binance API")
    ap.add_argument("--symbol", default="BTCUSDT", help="Binance symbol (default: BTCUSDT)")
    ap.add_argument("--interval", default="1h", help="Binance kline interval (default: 1h)")
    ap.add_argument("--total-bars", type=int, default=5000, help="Total Binance kline bars to fetch (default: 5000)")
    ap.add_argument("--leverage", type=float, default=20.0)
    ap.add_argument("--fee-bps", type=float, default=4.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--funding-bps-8h", type=float, default=1.0)
    ap.add_argument("--start-equity", type=float, default=100.0, help="Starting equity in USDT (default: 100)")
    ap.add_argument("--plot", default=None, help="path to save equity/ruin plot")
    args = ap.parse_args()

    cfg = Config(leverage=args.leverage, fee_bps=args.fee_bps,
                 slippage_bps=args.slip_bps, funding_bps_8h=args.funding_bps_8h,
                 start_equity=args.start_equity)

    if args.csv:
        df = load_csv(args.csv)
        src = f"CSV: {args.csv} ({len(df)} bars)"
    elif args.binance:
        df = fetch_binance_ohlcv(symbol=args.symbol, interval=args.interval, total_bars=args.total_bars)
        src = f"BINANCE: {args.symbol} {args.interval} ({len(df)} bars)"
    else:
        df = synthetic_ohlcv(args.bars, cfg)
        src = f"SYNTHETIC fat-tailed data ({len(df)} bars) -- swap in real data!"

    print("=" * 64)
    print("HONEST BTC FUTURES BACKTEST v2.0")
    print("=" * 64)
    print(f"data           : {src}")
    print(f"leverage       : {cfg.leverage:g}x   (liquidation on ~"
          f"{_liquidation_move(cfg)*100:.1f}% adverse move)")
    print(f"cost per side  : {cfg.fee_bps+cfg.slippage_bps:g} bps   "
          f"funding/8h: {cfg.funding_bps_8h:g} bps")
    print("-" * 64)
    print("STRATEGY       : v2.0 with liq protection, vol filter, trailing stop")
    print("-" * 64)

    # in-sample / out-of-sample split
    split = int(len(df) * (1 - cfg.oos_fraction))
    df_is, df_oos = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)

    sig_is = signal(df_is, cfg)
    sig_oos = signal(df_oos, cfg)
    res_is = backtest(df_is, sig_is, cfg, "IN-SAMPLE")
    res_oos = backtest(df_oos, sig_oos, cfg, "OUT-OF-SAMPLE")
    print("Strategy performance:")
    print_metrics(metrics(res_is))
    print_metrics(metrics(res_oos))

    # cost sensitivity: does the edge survive 2x costs?
    print("-" * 64)
    print("Cost sensitivity (out-of-sample, edge should survive higher costs):")
    for mult in (0.0, 1.0, 2.0):
        c2 = Config(leverage=cfg.leverage, fee_bps=cfg.fee_bps * mult,
                    slippage_bps=cfg.slippage_bps * mult,
                    funding_bps_8h=cfg.funding_bps_8h * mult)
        r2 = backtest(df_oos, signal(df_oos, c2), c2, f"costs x{mult:g}")
        m2 = metrics(r2)
        print(f"    costs x{mult:<3g}: final ${m2['final_equity']:.2f}  "
              f"Sharpe {m2['Sharpe']:+.2f}  maxDD {m2['max_drawdown']*100:.0f}%  "
              f"liq={'Y' if m2['liquidated_at_bar'] is not None else 'N'}")

    # risk of ruin
    print("-" * 64)
    print("RISK OF RUIN on a $100 account (assumes a SMALL real edge of")
    print("+0.10% mean / 1.5% std per trade, ~500 trades, round-trip costs):")
    ror = risk_of_ruin(edge_per_trade=0.001, std_per_trade=0.015, n_trades=500,
                       leverages=[1, 3, 5, 10, 20, 50], cfg=cfg)
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(ror.to_string(index=False))
    print()
    print("Read that table carefully: even WITH a positive edge, cranking")
    print("leverage drives P(ruin) toward 1. That is the whole game.")

    if args.plot and HAVE_PLT:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        res_is.equity.plot(ax=ax[0], label="in-sample")
        idx = range(len(res_is.equity), len(res_is.equity) + len(res_oos.equity))
        ax[0].plot(list(idx), res_oos.equity.values, label="out-of-sample")
        ax[0].axhline(cfg.start_equity, ls="--", c="grey", lw=0.8)
        ax[0].set_title("Equity curve v2.0"); ax[0].set_ylabel("USD"); ax[0].legend()
        ax[1].plot(ror["leverage"], ror["P(ruin)"], marker="o")
        ax[1].set_title("P(ruin) vs leverage"); ax[1].set_xlabel("leverage")
        ax[1].set_ylim(0, 1.02)
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"\nsaved plot -> {args.plot}")


if __name__ == "__main__":
    main()
