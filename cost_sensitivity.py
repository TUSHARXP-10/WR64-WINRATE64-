"""
cost_sensitivity.py
====================
The walk-forward run showed smc_fvg_candles (and improved_v2_adx) hovering
near breakeven under the generic Config defaults (fee_bps=4, slippage_bps=2
=> 6bps per side, 12bps round trip). This checks how much of that gap is
"costs" vs "signal" by re-running the SAME signals (computed once, unchanged)
through several realistic Binance Futures cost scenarios.

Binance USDT-M perpetual fee schedule (VIP0):
    maker 0.0200% (2bps)   taker 0.0500% (5bps)
    with 10% BNB discount: maker 0.0180%, taker 0.0450%
The live bots use MARKET entries + STOP_MARKET/TAKE_PROFIT_MARKET exits,
i.e. taker on both legs -> ~9-10bps round trip from fees alone, before
slippage.

Scenarios (fee_bps, slippage_bps -> cost_per_turn, round-trip):
    current_default        : 4.0, 2.0 -> 6bps/side, 12bps RT  (what walk_forward.py used)
    taker_no_discount       : 5.0, 1.0 -> 6bps/side, 12bps RT  (real taker + modest slippage)
    taker_bnb_discount       : 4.5, 1.0 -> 5.5bps/side, 11bps RT
    taker_bnb_discount_tight : 4.5, 0.5 -> 5bps/side, 10bps RT (tight slippage, liquid majors)
    zero_cost                : 0.0, 0.0 -> 0bps (theoretical raw signal edge)

Run:  python cost_sensitivity.py
"""
import numpy as np
import pandas as pd

from btc_backtest import Config, fetch_binance_ohlcv
from walk_forward import (
    STRATEGIES, walk_forward,
)

SCENARIOS = {
    "current_default (6/side,12RT)":        dict(fee_bps=4.0, slippage_bps=2.0, funding_bps_8h=1.0),
    "taker_no_discount (6/side,12RT)":      dict(fee_bps=5.0, slippage_bps=1.0, funding_bps_8h=1.0),
    "taker_bnb_discount (5.5/side,11RT)":   dict(fee_bps=4.5, slippage_bps=1.0, funding_bps_8h=1.0),
    "taker_bnb_discount_tight (5/side,10RT)": dict(fee_bps=4.5, slippage_bps=0.5, funding_bps_8h=1.0),
    "zero_cost (theoretical)":              dict(fee_bps=0.0, slippage_bps=0.0, funding_bps_8h=0.0),
}

STRATS_TO_CHECK = ["smc_fvg_candles", "improved_v2_adx", "original_v2_adx"]


def main():
    symbols = ["BTCUSDT", "BNBUSDT", "ETHUSDT"]
    interval = "4h"
    total_bars = 3000
    n_windows = 5
    warmup = 310

    # compute each strategy's position signal ONCE per symbol (cost-independent)
    data = {}
    positions = {}
    for symbol in symbols:
        print(f"Fetching {symbol} ({total_bars} x {interval} bars)...")
        df = fetch_binance_ohlcv(symbol, interval, total_bars)
        data[symbol] = df
        for name in STRATS_TO_CHECK:
            positions[(symbol, name)] = STRATEGIES[name](df)

    rows = []
    for scenario_name, cfg_kwargs in SCENARIOS.items():
        for name in STRATS_TO_CHECK:
            sharpes, returns, pos_pct, trades_total = [], [], [], 0
            for symbol in symbols:
                df = data[symbol]
                pos = positions[(symbol, name)]
                cfg = Config(leverage=1.0, **cfg_kwargs)
                windows, _ = walk_forward(df, pos, cfg, n_windows=n_windows, warmup=warmup)
                sharpes.append(np.mean([w["sharpe"] for w in windows]))
                returns.append(np.mean([w["total_return"] for w in windows]))
                pos_pct.append(np.mean([w["total_return"] > 0 for w in windows]))
                trades_total += sum(w["trades"] for w in windows)

            rows.append({
                "scenario": scenario_name,
                "strategy": name,
                "mean_window_sharpe": float(np.mean(sharpes)),
                "mean_window_return": float(np.mean(returns)),
                "pct_windows_profitable": float(np.mean(pos_pct)),
                "total_trades": trades_total,
            })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("COST SENSITIVITY (4h bars, walk-forward aggregate across BTC/BNB/ETH)")
    print("=" * 100)
    for name in STRATS_TO_CHECK:
        sub = out[out["strategy"] == name].drop(columns="strategy")
        print(f"\n--- {name} ---")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
