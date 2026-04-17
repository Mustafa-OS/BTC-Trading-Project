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
