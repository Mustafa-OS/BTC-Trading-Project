# BTC Trading Dashboard

Live BTC options vol surface, futures curve, portfolio risk analytics, and trade execution — all in one Python app.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Deribit](https://img.shields.io/badge/Deribit-Testnet%20%2B%20Production-green)

## Features

| Tab | Description |
|-----|-------------|
| **Vol Smile** | SVI model calibrated to OTM option mark IVs. Displays 5 parameters, RMSE, ATMF vol, and Black-76 call/put premiums in BTC. |
| **Futures Curve** | Short-rate model fitted to inverse futures. Interpolates forward prices for any maturity. |
| **Risk** | Portfolio greeks (BS delta, smile delta, gamma 1%, vega $, theta $) per position and total. Scenario analysis across spot bumps. |
| **Hedge** | Compute delta/gamma/vega hedges. Execute suggested trades directly. Close individual or all positions. |
| **P&L** | Live equity, balance, unrealised P&L, and spot/perp price charts. Persists to CSV across restarts. |
| **Recon** | Transaction reconciliation — all trades, settlements, funding payments. Verifies calculated equity matches Deribit API. |

## Quick Start

```bash
# Install dependencies
pip install flask requests websocket-client numpy scipy

# Run (opens login page at http://localhost:5050)
python3 svi_ui.py
```

Enter your Deribit API credentials on the login page. Choose **Testnet** or **Production**.

Get API keys from [Deribit Testnet](https://test.deribit.com/account/BTC/api) or [Deribit Production](https://www.deribit.com/account/BTC/api).

### Environment Variables (optional)

Skip the login page by setting env vars:

```bash
export DERIBIT_CLIENT_ID="your_id"
export DERIBIT_CLIENT_SECRET="your_secret"
export DERIBIT_NETWORK="testnet"  # or "mainnet"
python3 svi_ui.py
```

## Browser-Only Version

The `index.html` file is a standalone client-side version (Vol Smile + Futures Curve only, no private API needed). Open it directly or visit [mustafa-os.github.io/BTC-Trading-Project](https://mustafa-os.github.io/BTC-Trading-Project/).

## How It Works

- Real-time data via Deribit WebSocket API (`.agg2` channels)
- SVI calibration using L-BFGS-B with multiple starting points
- Black-76 pricing with r=0 (BTC as base asset, inverse contracts)
- OTM options only: puts below forward, calls above — moneyness filter |k| < 0.25
- Short-rate model (cubic spline) fitted to BTC/USD inverse futures
- Portfolio greeks computed via finite differences with proper inverse option conventions
- Smile delta: IV stays sticky to delta (not strike) when bumping spot
- Orders executed via authenticated WebSocket for low latency

## Tech Stack

- **Backend**: Python, Flask, NumPy, SciPy
- **Data**: Deribit REST + WebSocket API
- **Charts**: Plotly.js
- **Models**: SVI, Black-76, short-rate interpolation

---

# MAG-7 ETF Arbitrage Strategy

A statistical arbitrage backtest that exploits mispricings between a synthetic Magnificent 7 ETF and its theoretical net asset value (NAV).

## How It Works

1. **NAV Construction** — A weighted basket of 7 stocks (MSFT, AAPL, META, AMZN, GOOGL, NVDA, TSLA) forms the theoretical fair value. Weights are fixed and sum to 1.

2. **ETF Price** — Simulated as the NAV plus a mean-reverting AR(1) mispricing process. This models the real-world phenomenon where ETF market prices temporarily deviate from their underlying value.

3. **Signal Generation** — When the percentage spread (ETF vs NAV) exceeds an entry threshold, the strategy enters a position:
   - ETF overpriced vs NAV → short ETF, long basket
   - ETF underpriced vs NAV → long ETF, short basket
   - Position exits when the spread reverts near zero

4. **PnL & Costs** — Daily PnL is computed from spread changes. Transaction costs (0.005% per trade) are deducted on every entry/exit.

## Data

Real historical closing prices are pulled via `yfinance` for the 2024 calendar year (~251 trading days). The mispricing layer is synthetic (AR(1) noise), since no actual MAG-7 ETF exists to source real spread data from.

## Quick Start

```bash
pip install numpy pandas matplotlib yfinance
python3 etf_arb.py
```

## Output

- Spread statistics (mean, std, min/max)
- Performance metrics: total PnL, Sharpe ratio, max drawdown, trade count, cost drag
- 4-panel chart: ETF vs NAV, % spread with thresholds, position timeline, cumulative PnL

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phi` | 0.85 | AR(1) persistence of mispricing |
| `ENTRY_THRESHOLD` | 0.02% | Spread level to enter a trade |
| `EXIT_THRESHOLD` | 0.005% | Spread level to exit a trade |
| `COST_PER_TRADE` | 0.005% | Transaction cost per round trip |
