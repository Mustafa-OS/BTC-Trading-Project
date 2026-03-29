# BTC Vol Surface

Live implied volatility surface for BTC options on Deribit, running entirely in the browser.

![image](https://img.shields.io/badge/Deribit-Testnet%20%2B%20Production-blue)

## Features

**Vol Smile** — Calibrates an SVI (Stochastic Volatility Inspired) model to OTM option mark implied vols via WebSocket. Displays the 5 SVI parameters, RMSE, ATMF implied vol, and Black-76 call/put premiums in BTC. Toggle between expiry dates.

**Futures Curve** — Fits a short-rate model to the BTC inverse futures term structure. Interpolates forward prices for any maturity using the calibrated rate curve.

## How It Works

- Connects directly to the Deribit WebSocket API for real-time option and futures ticks
- SVI calibration runs client-side using Nelder-Mead optimization in JavaScript
- Black-76 pricing with zero interest rate (BTC as base asset)
- OTM options only: puts below forward, calls above forward
- Moneyness filter |k| < 0.25 for robust fits
- Re-calibrates every ~2 seconds as market ticks

## Usage

Open **[mustafa-os.github.io/BTC-Trading-Project](https://mustafa-os.github.io/BTC-Trading-Project/)** in your browser. No installation required.

Or run locally:
```
git clone https://github.com/Mustafa-OS/BTC-Trading-Project.git
open index.html
```

Use the toggle in the top-right to switch between **Testnet** and **Production** Deribit.

## Tech

Single `index.html` — no server, no build step, no API keys. Uses:
- Deribit public WebSocket + REST API
- Plotly.js for charts
- SVI model calibration (Nelder-Mead)
- Black-76 option pricing
- Short-rate model for futures term structure
