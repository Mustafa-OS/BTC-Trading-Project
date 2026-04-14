"""
SVI Implied Vol Surface + Portfolio Risk — Live Web UI with Futures Term Structure

Run:  python3 svi_ui.py
Open: http://localhost:5050

Tabs:
  1. Vol Smile  — SVI calibration to OTM option mark IVs
  2. Futures Curve — short-rate model fitted to inverse futures
  3. Risk — portfolio greeks (BS delta, smile delta, gamma, vega, theta) + scenario analysis
"""

import json
import math
import time
import threading
import traceback
import csv
import os
import requests
import websocket
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, render_template_string, request
from collections import deque

# ====== CONFIG (set at runtime via login page) ======
CLIENT_ID = None
CLIENT_SECRET = None
BASE_URL = None
WS_URL = None
MIN_UPDATE_INTERVAL = 2.0

NETWORKS = {
    "testnet": {
        "rest": "https://test.deribit.com/api/v2",
        "ws": "wss://test.deribit.com/ws/api/v2",
    },
    "mainnet": {
        "rest": "https://www.deribit.com/api/v2",
        "ws": "wss://www.deribit.com/ws/api/v2",
    },
}


# ====== AUTH ======
def authenticate():
    try:
        r = requests.get(f"{BASE_URL}/public/auth", params={
            "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
            "grant_type": "client_credentials"
        }, timeout=10)
        data = r.json()
        if "result" in data:
            return data["result"]["access_token"]
    except Exception as e:
        print(f"[Auth] Error: {e}")
    return None


def api_call(endpoint, params, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, headers=headers, timeout=10)
        data = r.json()
        if "result" in data:
            return data["result"]
    except Exception:
        pass
    return None


# ====== SVI MODEL ======
def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_implied_vol(k, T, a, b, rho, m, sigma):
    w = svi_total_variance(k, a, b, rho, m, sigma)
    w = np.maximum(w, 1e-8)
    return np.sqrt(w / T)


def calibrate_svi(k_arr, iv_arr, T):
    def objective(params):
        a, b, rho, m, sigma = params
        w_model = svi_total_variance(k_arr, a, b, rho, m, sigma)
        w_model = np.maximum(w_model, 1e-10)
        iv_model = np.sqrt(w_model / T)
        return np.sum((iv_model - iv_arr) ** 2)

    w_market = iv_arr ** 2 * T
    atm_var = np.interp(0.0, k_arr, w_market)
    bounds = [(-1.0, 5.0), (1e-6, 5.0), (-0.999, 0.999), (-1.0, 1.0), (1e-6, 5.0)]
    starting_points = [
        [atm_var, 0.05, -0.2, 0.0, 0.1],
        [atm_var, 0.1, -0.5, 0.0, 0.2],
        [atm_var, 0.2, 0.0, 0.0, 0.05],
        [atm_var * 0.8, 0.15, -0.3, -0.05, 0.15],
    ]
    best_result, best_fun = None, np.inf
    for sp in starting_points:
        result = minimize(objective, sp, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 5000, "ftol": 1e-16})
        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result
    if best_result is not None and best_fun < 1.0:
        return tuple(best_result.x)
    return None


# ====== BLACK-76 (r=0 for BTC base asset) ======
def black76_call(F, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def black76_put(F, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - F, 0.0)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)


# ====== GREEK COMPUTATION ======
# Delta convention: dV_usd/dF = BTC-equivalent exposure (matches Deribit)
#   - For calls: ≈ N(d1) per contract
#   - For puts:  ≈ N(d1)-1 per contract
# Gamma: change in delta for 1% spot move
# Vega:  USD P&L for 1 vol point (0.01 in σ) move
# Theta: USD P&L per calendar day

ZERO_GREEKS = {"bs_delta": 0.0, "smile_delta": 0.0, "gamma_1pct": 0.0, "vega_usd": 0.0, "theta_usd": 0.0}


def compute_option_greeks(F, K, T, sigma, option_type, size, direction,
                          svi_params=None, svi_T=None):
    sign = 1.0 if direction == "buy" else -1.0
    bs_fn = black76_call if option_type == "call" else black76_put

    if T <= 1e-6 or sigma <= 0 or F <= 0 or K <= 0:
        return dict(ZERO_GREEKS)

    eps = F * 0.0005

    # Current USD value per contract
    v_usd_0 = bs_fn(F, K, T, sigma)

    # ---- BS Delta (sticky strike): dV_usd/dF ----
    v_usd_up = bs_fn(F + eps, K, T, sigma)
    v_usd_dn = bs_fn(F - eps, K, T, sigma)
    bs_delta = sign * size * (v_usd_up - v_usd_dn) / (2.0 * eps)

    # ---- Smile Delta (sticky delta): IV moves with moneyness via SVI ----
    if svi_params and svi_T and svi_T > 0:
        k_up = math.log(K / (F + eps))
        k_dn = math.log(K / (F - eps))
        sig_up = max(float(svi_implied_vol(np.array([k_up]), svi_T, *svi_params)[0]), 0.01)
        sig_dn = max(float(svi_implied_vol(np.array([k_dn]), svi_T, *svi_params)[0]), 0.01)
        v_usd_up_s = bs_fn(F + eps, K, T, sig_up)
        v_usd_dn_s = bs_fn(F - eps, K, T, sig_dn)
        smile_delta = sign * size * (v_usd_up_s - v_usd_dn_s) / (2.0 * eps)
    else:
        smile_delta = bs_delta

    # ---- Gamma: change in BS delta for 1% move ----
    F_up = F * 1.01
    F_dn = F * 0.99
    eu = F_up * 0.0005
    ed = F_dn * 0.0005
    delta_up = sign * size * (bs_fn(F_up + eu, K, T, sigma) - bs_fn(F_up - eu, K, T, sigma)) / (2.0 * eu)
    delta_dn = sign * size * (bs_fn(F_dn + ed, K, T, sigma) - bs_fn(F_dn - ed, K, T, sigma)) / (2.0 * ed)
    gamma_1pct = (delta_up - delta_dn) / 2.0

    # ---- Vega: USD P&L for 1 vol point ----
    vega_usd = sign * size * (bs_fn(F, K, T, sigma + 0.01) - v_usd_0)

    # ---- Theta: USD P&L per day ----
    dt = 1.0 / 365.25
    theta_usd = sign * size * (bs_fn(F, K, T - dt, sigma) - v_usd_0) if T > dt else 0.0

    return {
        "bs_delta": bs_delta,
        "smile_delta": smile_delta,
        "gamma_1pct": gamma_1pct,
        "vega_usd": vega_usd,
        "theta_usd": theta_usd,
    }


def compute_future_greeks(F, size, direction):
    sign = 1.0 if direction == "buy" else -1.0
    # BTC equivalent: sign * size / F
    return {
        "bs_delta": sign * size / F,
        "smile_delta": sign * size / F,
        "gamma_1pct": 0.0,
        "vega_usd": 0.0,
        "theta_usd": 0.0,
    }


# ====== DERIBIT HELPERS ======
def get_all_btc_futures():
    r = requests.get(f"{BASE_URL}/public/get_instruments",
                     params={"currency": "BTC", "kind": "future"}, timeout=10)
    instruments = r.json()["result"]
    now = datetime.now(timezone.utc)
    futures = []
    for inst in instruments:
        if inst.get("settlement_currency") != "BTC":
            continue
        name = inst["instrument_name"]
        if name == "BTC-PERPETUAL":
            continue
        exp_ts = inst["expiration_timestamp"] / 1000.0
        exp_dt = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
        T = (exp_dt - now).total_seconds() / (365.25 * 24 * 3600)
        if T > 0:
            futures.append({"name": name, "expiry_ts": exp_ts, "expiry_dt": exp_dt, "T": T})
    futures.sort(key=lambda x: x["T"])
    return futures


def get_all_expiries():
    if not BASE_URL:
        return []
    r = requests.get(f"{BASE_URL}/public/get_instruments",
                     params={"currency": "BTC", "kind": "option"}, timeout=10)
    instruments = r.json()["result"]
    expiries = set()
    for i in instruments:
        if i.get("settlement_currency") != "BTC":
            continue
        expiries.add(i["instrument_name"].split("-")[1])

    def parse_exp(s):
        try:
            return datetime.strptime(s, "%d%b%y")
        except ValueError:
            return datetime.max
    return sorted(expiries, key=parse_exp)


def get_option_instruments(expiry):
    r = requests.get(f"{BASE_URL}/public/get_instruments",
                     params={"currency": "BTC", "kind": "option"}, timeout=10)
    instruments = r.json()["result"]
    return [i for i in instruments
            if i.get("settlement_currency") == "BTC" and i["instrument_name"].split("-")[1] == expiry]


def get_expiry_datetime(expiry_str):
    dt = datetime.strptime(expiry_str, "%d%b%y")
    return dt.replace(hour=8, minute=0, second=0, tzinfo=timezone.utc)


# ====== SHORT RATE MODEL ======
class ShortRateModel:
    def __init__(self):
        self.spot = None
        self.futures_data = {}
        self.spline = None
        self.T_arr = None
        self.r_arr = None
        self.T_min = None
        self.T_max = None

    def update_spot(self, spot):
        self.spot = spot

    def update_future(self, name, T, mark_price):
        self.futures_data[name] = {"T": T, "mark_price": mark_price}

    def calibrate(self):
        if self.spot is None or self.spot <= 0 or len(self.futures_data) < 2:
            return False
        S = self.spot
        items = sorted(self.futures_data.values(), key=lambda x: x["T"])
        Ts, rs = [], []
        for item in items:
            T, F = item["T"], item["mark_price"]
            if T > 0 and F > 0:
                Ts.append(T)
                rs.append(math.log(F / S) / T)
        if len(Ts) < 2:
            return False
        self.T_arr = np.array(Ts)
        self.r_arr = np.array(rs)
        self.T_min, self.T_max = self.T_arr[0], self.T_arr[-1]
        if len(Ts) == 2:
            self.spline = None
        else:
            self.spline = CubicSpline(self.T_arr, self.r_arr, bc_type="natural")
        return True

    def get_rate(self, T):
        if self.T_arr is None:
            return 0.0
        if self.spline is not None:
            return float(self.spline(np.clip(T, self.T_min * 0.5, self.T_max * 1.5)))
        return float(np.interp(T, self.T_arr, self.r_arr))

    def get_forward(self, T):
        if self.spot is None:
            return None
        return self.spot * math.exp(self.get_rate(T) * T)

    def get_curve_data(self):
        if self.spot is None or self.T_arr is None:
            return None
        items = sorted(self.futures_data.values(), key=lambda x: x["T"])
        T_curve = np.linspace(0, self.T_max * 1.05, 100)
        return {
            "spot": self.spot,
            "futures_T": [it["T"] for it in items],
            "futures_F": [it["mark_price"] for it in items],
            "futures_names": sorted(self.futures_data.keys()),
            "curve_T": T_curve.tolist(),
            "curve_F": [self.spot * math.exp(self.get_rate(t) * t) for t in T_curve],
            "curve_r": [self.get_rate(t) * 100 for t in T_curve],
        }


# ====== LIVE DATA ENGINE ======
class SVIEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.expiry = None
        self.expiry_dt = None
        self.option_data = {}
        self.last_fit_time = 0
        self.ws = None
        self.ws_thread = None

        self.rate_model = ShortRateModel()
        self.futures_meta = []
        self.futures_ws = None
        self.futures_ws_thread = None
        self.spot_price = None

        self.latest = self._empty_latest()
        self.latest_futures = None

        # Risk
        self.risk_data = {"positions": [], "account": {}, "totals": {}, "bumps": None,
                          "spot": 0, "ts": None, "error": None}
        self.risk_svi_cache = {}

        # Hedge calculator cache
        self.hedge_cache = {"ts": 0, "data": None}

        # P&L history (in-memory ring buffer + CSV log)
        self.pnl_history = deque(maxlen=86400)  # ~24h at 1s intervals
        self.pnl_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pnl_history.csv")
        self._load_pnl_csv()

        # Authenticated WS for order execution
        self.order_ws = None
        self.order_ws_ready = threading.Event()
        self.order_ws_token = None
        self._order_pending = {}  # id -> threading.Event + result
        self._order_id_counter = 5000

    def _empty_latest(self):
        return {
            "expiry": None, "forward": None, "forward_source": None,
            "T": None, "params": None, "rmse": None, "atmf_iv": None,
            "atmf_call": None, "atmf_put": None,
            "strike_below": None, "iv_below_mkt": None, "iv_below_svi": None,
            "strike_above": None, "iv_above_mkt": None, "iv_above_svi": None,
            "market_strikes": [], "market_ivs": [],
            "svi_curve_k": [], "svi_curve_iv": [], "svi_curve_strikes": [],
            "n_points": 0, "ts": None,
        }

    # ---- Futures WebSocket ----
    def start_futures_ws(self):
        self.futures_meta = get_all_btc_futures()
        print(f"[Futures] Found {len(self.futures_meta)} inverse futures")
        self._seed_futures_from_rest()
        self.futures_ws_thread = threading.Thread(target=self._run_futures_ws, daemon=True)
        self.futures_ws_thread.start()

    def _seed_futures_from_rest(self):
        try:
            r = requests.get(f"{BASE_URL}/public/ticker",
                             params={"instrument_name": "BTC-PERPETUAL"}, timeout=10)
            idx = r.json().get("result", {}).get("index_price")
            if idx and idx > 0:
                self.spot_price = idx
                self.rate_model.update_spot(idx)
                print(f"[Futures] Spot: ${idx:,.2f}")
            now = datetime.now(timezone.utc)
            for fm in self.futures_meta:
                r = requests.get(f"{BASE_URL}/public/ticker",
                                 params={"instrument_name": fm["name"]}, timeout=10)
                mark = r.json().get("result", {}).get("mark_price")
                if mark and mark > 0:
                    T = (fm["expiry_dt"] - now).total_seconds() / (365.25 * 24 * 3600)
                    if T > 0:
                        self.rate_model.update_future(fm["name"], T, mark)
            if self.rate_model.calibrate():
                print(f"[Futures] Rate model seeded: {len(self.rate_model.futures_data)} futures")
        except Exception as e:
            print(f"[Futures] REST seed error: {e}")

    def _run_futures_ws(self):
        while True:
            try:
                self.futures_ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_futures_open,
                    on_message=self._on_futures_message,
                    on_error=lambda ws, e: print(f"[Futures WS] Error: {e}"),
                    on_close=lambda ws, c, m: print("[Futures WS] Closed, reconnecting..."),
                )
                self.futures_ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"[Futures WS] Exception: {e}")
            time.sleep(3)

    def _on_futures_open(self, ws):
        channels = [f"ticker.{fm['name']}.agg2" for fm in self.futures_meta]
        channels.append("ticker.BTC-PERPETUAL.agg2")
        ws.send(json.dumps({"jsonrpc": "2.0", "id": 999,
                             "method": "public/subscribe", "params": {"channels": channels}}))
        print(f"[Futures WS] Subscribed to {len(channels)} channels")

    def _on_futures_message(self, ws, message):
        data = json.loads(message)
        if data.get("method") != "subscription":
            return
        tick = data["params"]["data"]
        name = tick.get("instrument_name", "")
        if name == "BTC-PERPETUAL":
            idx = tick.get("index_price")
            if idx and idx > 0:
                with self.lock:
                    self.spot_price = idx
                    self.rate_model.update_spot(idx)
            return
        mark = tick.get("mark_price")
        if mark and mark > 0:
            for fm in self.futures_meta:
                if fm["name"] == name:
                    now = datetime.now(timezone.utc)
                    T = (fm["expiry_dt"] - now).total_seconds() / (365.25 * 24 * 3600)
                    if T > 0:
                        with self.lock:
                            self.rate_model.update_future(name, T, mark)
                            self.rate_model.calibrate()
                    break

    # ---- Options WebSocket ----
    def switch_expiry(self, expiry):
        if self.ws:
            self.ws.close()
            self.ws = None
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=3)

        with self.lock:
            self.expiry = expiry
            self.expiry_dt = get_expiry_datetime(expiry)
            self.option_data = {}
            self.last_fit_time = 0
            instruments = get_option_instruments(expiry)
            for inst in instruments:
                self.option_data[inst["instrument_name"]] = {
                    "strike": inst["strike"], "type": inst["option_type"], "mark_iv": None,
                }
            self.latest = self._empty_latest()
            self.latest["expiry"] = expiry

        self.ws_thread = threading.Thread(target=self._run_options_ws, daemon=True)
        self.ws_thread.start()

    def _run_options_ws(self):
        while True:
            try:
                self.ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_options_open,
                    on_message=self._on_options_message,
                    on_error=lambda ws, e: print(f"[Options WS] Error: {e}"),
                    on_close=lambda ws, c, m: print("[Options WS] Closed, reconnecting..."),
                )
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"[Options WS] Exception: {e}")
            time.sleep(3)

    def _on_options_open(self, ws):
        with self.lock:
            channels = [f"ticker.{name}.agg2" for name in self.option_data]
        for i in range(0, len(channels), 50):
            ws.send(json.dumps({"jsonrpc": "2.0", "id": i,
                                 "method": "public/subscribe",
                                 "params": {"channels": channels[i:i + 50]}}))
        print(f"[Options WS] Subscribed to {len(channels)} channels for {self.expiry}")

        def periodic_calibrate():
            while self.ws == ws:
                time.sleep(2.5)
                self._maybe_calibrate()
        threading.Thread(target=periodic_calibrate, daemon=True).start()

    def _on_options_message(self, ws, message):
        data = json.loads(message)
        if data.get("method") != "subscription":
            return
        tick = data["params"]["data"]
        name = tick.get("instrument_name", "")
        with self.lock:
            if name in self.option_data:
                iv = tick.get("mark_iv")
                if iv and iv > 0:
                    self.option_data[name]["mark_iv"] = iv
        self._maybe_calibrate()

    def _get_forward_for_expiry(self):
        if self.expiry is None or self.expiry_dt is None:
            return None, None
        now = datetime.now(timezone.utc)
        T = (self.expiry_dt - now).total_seconds() / (365.25 * 24 * 3600)
        if T <= 0:
            return None, None
        future_name = f"BTC-{self.expiry}"
        if future_name in self.rate_model.futures_data:
            return self.rate_model.futures_data[future_name]["mark_price"], f"future ({future_name})"
        F = self.rate_model.get_forward(T)
        if F and F > 0:
            return F, "rate model (interpolated)"
        return None, None

    def _maybe_calibrate(self):
        now = time.time()
        if now - self.last_fit_time < MIN_UPDATE_INTERVAL:
            return
        self.last_fit_time = now
        with self.lock:
            if self.expiry_dt is None:
                return
            now_dt = datetime.now(timezone.utc)
            T = (self.expiry_dt - now_dt).total_seconds() / (365.25 * 24 * 3600)
            if T <= 0:
                return
            F, f_source = self._get_forward_for_expiry()
            if F is None:
                return
            strikes, ivs = [], []
            for name, d in self.option_data.items():
                if d["mark_iv"] is None or d["mark_iv"] <= 0:
                    continue
                K, otype = d["strike"], d["type"]
                if (otype == "put" and K < F) or (otype == "call" and K >= F):
                    strikes.append(K)
                    ivs.append(d["mark_iv"] / 100.0)

        if len(strikes) < 6:
            return
        strikes = np.array(strikes)
        iv_arr = np.array(ivs)
        k_arr = np.log(strikes / F)
        order = np.argsort(k_arr)
        k_arr, iv_arr, strikes = k_arr[order], iv_arr[order], strikes[order]
        mask = np.abs(k_arr) < 0.25
        k_fit, iv_fit, strikes_fit = k_arr[mask], iv_arr[mask], strikes[mask]
        if len(k_fit) < 6:
            return
        params = calibrate_svi(k_fit, iv_fit, T)
        if params is None:
            return
        a, b, rho, m, sigma = params

        k_curve = np.linspace(k_fit.min() - 0.02, k_fit.max() + 0.02, 200)
        iv_curve = svi_implied_vol(k_curve, T, a, b, rho, m, sigma)
        strikes_curve = F * np.exp(k_curve)
        atmf_iv = float(svi_implied_vol(0.0, T, a, b, rho, m, sigma))
        call_prem = black76_call(F, F, T, atmf_iv) / F
        put_prem = black76_put(F, F, T, atmf_iv) / F
        iv_fitted = svi_implied_vol(k_fit, T, a, b, rho, m, sigma)
        rmse = float(np.sqrt(np.mean((iv_fitted - iv_fit) ** 2)) * 100)

        below_mask = strikes_fit < F
        above_mask = strikes_fit >= F
        s_below = float(strikes_fit[below_mask][-1]) if np.any(below_mask) else None
        s_above = float(strikes_fit[above_mask][0]) if np.any(above_mask) else None
        iv_b_mkt = float(iv_fit[below_mask][-1] * 100) if np.any(below_mask) else None
        iv_a_mkt = float(iv_fit[above_mask][0] * 100) if np.any(above_mask) else None
        iv_b_svi = float(svi_implied_vol(math.log(s_below / F), T, *params) * 100) if s_below else None
        iv_a_svi = float(svi_implied_vol(math.log(s_above / F), T, *params) * 100) if s_above else None
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")

        with self.lock:
            self.latest = {
                "expiry": self.expiry, "forward": round(F, 2), "forward_source": f_source,
                "T": round(T, 6),
                "params": {"a": round(a, 6), "b": round(b, 6), "rho": round(rho, 6),
                           "m": round(m, 6), "sigma": round(sigma, 6)},
                "rmse": round(rmse, 4), "atmf_iv": round(atmf_iv * 100, 2),
                "atmf_call": round(call_prem, 6), "atmf_put": round(put_prem, 6),
                "strike_below": s_below, "iv_below_mkt": iv_b_mkt,
                "iv_below_svi": round(iv_b_svi, 2) if iv_b_svi else None,
                "strike_above": s_above, "iv_above_mkt": iv_a_mkt,
                "iv_above_svi": round(iv_a_svi, 2) if iv_a_svi else None,
                "market_strikes": strikes_fit.tolist(), "market_ivs": (iv_fit * 100).tolist(),
                "svi_curve_strikes": strikes_curve.tolist(), "svi_curve_iv": (iv_curve * 100).tolist(),
                "n_points": len(k_fit), "ts": ts,
            }

    def get_futures_data(self):
        with self.lock:
            curve = self.rate_model.get_curve_data()
            if curve is None:
                return None
            table = []
            for fm in self.futures_meta:
                name = fm["name"]
                if name in self.rate_model.futures_data:
                    fd = self.rate_model.futures_data[name]
                    T, F = fd["T"], fd["mark_price"]
                    S = self.spot_price or curve["spot"]
                    r = math.log(F / S) / T if T > 0 and S > 0 else 0
                    table.append({"name": name, "T": round(T, 6), "mark": round(F, 2),
                                  "rate": round(r * 100, 4)})
            curve["table"] = table
            curve["spot"] = round(curve["spot"], 2)
            return curve

    # ---- Risk Poller ----
    def start_risk_poller(self):
        threading.Thread(target=self._risk_loop, daemon=True).start()

    def _risk_loop(self):
        while True:
            try:
                self._compute_risk()
            except Exception as e:
                traceback.print_exc()
                with self.lock:
                    self.risk_data["error"] = str(e)
                    self.risk_data["ts"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
            time.sleep(15)

    def _get_forward_for_expiry_str(self, expiry_str):
        exp_dt = get_expiry_datetime(expiry_str)
        now = datetime.now(timezone.utc)
        T = (exp_dt - now).total_seconds() / (365.25 * 24 * 3600)
        if T <= 0:
            return None, 0
        future_name = f"BTC-{expiry_str}"
        with self.lock:
            if future_name in self.rate_model.futures_data:
                return self.rate_model.futures_data[future_name]["mark_price"], T
            F = self.rate_model.get_forward(T)
        return (F, T) if F and F > 0 else (None, T)

    def _calibrate_svi_for_risk(self, expiry_str, F, T):
        cached = self.risk_svi_cache.get(expiry_str)
        if cached and (time.time() - cached["ts"]) < 30:
            return cached["params"], cached["T"]

        instruments = get_option_instruments(expiry_str)
        if not instruments:
            return None, T

        def fetch_ticker(inst):
            try:
                r = requests.get(f"{BASE_URL}/public/ticker",
                                 params={"instrument_name": inst["instrument_name"]}, timeout=5)
                return r.json().get("result", {})
            except Exception:
                return {}

        with ThreadPoolExecutor(max_workers=20) as pool:
            tickers = list(pool.map(fetch_ticker, instruments))

        strikes, ivs = [], []
        for inst, tick in zip(instruments, tickers):
            iv = tick.get("mark_iv")
            if not iv or iv <= 0:
                continue
            K, otype = inst["strike"], inst["option_type"]
            if (otype == "put" and K < F) or (otype == "call" and K >= F):
                strikes.append(K)
                ivs.append(iv / 100.0)

        if len(strikes) < 6:
            return None, T
        strikes = np.array(strikes)
        iv_arr = np.array(ivs)
        k_arr = np.log(strikes / F)
        order = np.argsort(k_arr)
        k_arr, iv_arr = k_arr[order], iv_arr[order]
        mask = np.abs(k_arr) < 0.25
        if np.sum(mask) < 6:
            return None, T
        params = calibrate_svi(k_arr[mask], iv_arr[mask], T)
        if params:
            self.risk_svi_cache[expiry_str] = {"params": params, "T": T, "ts": time.time()}
        return params, T

    def _compute_risk(self):
        token = authenticate()
        if not token:
            with self.lock:
                self.risk_data["error"] = "Auth failed"
            return

        positions = api_call("private/get_positions", {"currency": "BTC"}, token) or []
        positions = [p for p in positions if abs(p.get("size", 0)) > 0]  # filter closed
        account = api_call("private/get_account_summary", {"currency": "BTC"}, token) or {}

        with self.lock:
            spot = self.spot_price or 0

        # Calibrate SVI per expiry
        expiry_set = set()
        for pos in positions:
            if pos.get("kind") == "option":
                expiry_set.add(pos["instrument_name"].split("-")[1])

        svi_by_expiry = {}
        for exp in expiry_set:
            F, T = self._get_forward_for_expiry_str(exp)
            if F and T > 0:
                params, svi_T = self._calibrate_svi_for_risk(exp, F, T)
                if params:
                    svi_by_expiry[exp] = {"params": params, "F": F, "T": svi_T}

        # Process positions
        pos_data = []  # for bump analysis
        results = []   # for display
        totals = dict(ZERO_GREEKS)

        for pos in positions:
            name = pos["instrument_name"]
            kind = pos.get("kind", "")
            direction = pos.get("direction", "")
            size = abs(pos.get("size", 0))
            mark_price = pos.get("mark_price", 0)
            pnl = pos.get("floating_profit_loss", 0)
            sign = 1.0 if direction == "buy" else -1.0

            row = {"instrument": name, "kind": kind, "direction": direction,
                   "size": size, "mark_price": mark_price, "pnl": round(pnl, 8)}

            if kind == "future":
                F = mark_price if mark_price > 0 else spot
                greeks = compute_future_greeks(F, size, direction)
                row.update(greeks)
                pos_data.append({"kind": "future", "sign": sign, "size": size, "F": F})

            elif kind == "option":
                parts = name.split("-")
                expiry_str = parts[1]
                K = float(parts[2])
                option_type = "call" if parts[3] == "C" else "put"

                F, T = self._get_forward_for_expiry_str(expiry_str)
                if F is None or T <= 0:
                    F, T = spot, 0.001

                ticker = api_call("public/ticker", {"instrument_name": name})
                sigma = 0
                if ticker:
                    iv = ticker.get("mark_iv", 0)
                    if iv and iv > 0:
                        sigma = iv / 100.0

                svi_info = svi_by_expiry.get(expiry_str)
                svi_params = svi_info["params"] if svi_info else None
                svi_T = svi_info["T"] if svi_info else None

                if sigma > 0 and T > 0:
                    greeks = compute_option_greeks(F, K, T, sigma, option_type, size,
                                                   direction, svi_params, svi_T)
                else:
                    greeks = dict(ZERO_GREEKS)

                row["strike"] = K
                row["option_type"] = option_type
                row["iv"] = round(sigma * 100, 1) if sigma > 0 else 0
                row["T"] = round(T, 6)
                row.update(greeks)
                pos_data.append({"kind": "option", "sign": sign, "size": size,
                                 "F": F, "K": K, "T": T, "sigma": sigma,
                                 "option_type": option_type, "direction": direction,
                                 "svi_params": svi_params, "svi_T": svi_T})
            else:
                row.update(ZERO_GREEKS)

            for k in totals:
                totals[k] += row.get(k, 0)
            results.append(row)

        # Round
        for r in results:
            for k in ["bs_delta", "smile_delta", "gamma_1pct"]:
                r[k] = round(r.get(k, 0), 6)
            for k in ["vega_usd", "theta_usd"]:
                r[k] = round(r.get(k, 0), 2)
        for k in ["bs_delta", "smile_delta", "gamma_1pct"]:
            totals[k] = round(totals[k], 6)
        for k in ["vega_usd", "theta_usd"]:
            totals[k] = round(totals[k], 2)

        # Bump analysis
        bumps = self._compute_bumps(pos_data, spot)

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        with self.lock:
            self.risk_data = {
                "positions": results,
                "account": {
                    "equity": account.get("equity", 0),
                    "balance": account.get("balance", 0),
                    "available_funds": account.get("available_funds", 0),
                    "initial_margin": account.get("initial_margin", 0),
                },
                "totals": totals,
                "bumps": bumps,
                "spot": spot,
                "ts": ts,
                "error": None,
            }
        # Get perpetual mark price from positions data (avoids extra REST call)
        perp_mark = 0
        for pos in positions:
            if pos.get("instrument_name") == "BTC-PERPETUAL":
                perp_mark = pos.get("mark_price", 0)
                break
        if not perp_mark:
            # Fallback: fetch from ticker
            try:
                ticker = api_call("public/ticker", {"instrument_name": "BTC-PERPETUAL"})
                if ticker:
                    perp_mark = ticker.get("mark_price", 0) or 0
            except Exception:
                pass

        # Record P&L snapshot
        self._record_pnl(account, spot, perp_mark)

        print(f"[Risk] {ts}: {len(results)} positions, BS Δ={totals['bs_delta']:.4f}, "
              f"Smile Δ={totals['smile_delta']:.4f}")

    def _compute_bumps(self, pos_data, spot):
        bump_pcts = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
        rows = {"pnl": [], "bs_delta": [], "smile_delta": [],
                "gamma_1pct": [], "vega_usd": [], "theta_usd": []}
        spots = []

        for bp in bump_pcts:
            factor = 1 + bp / 100.0
            spots.append(round(spot * factor, 0))

            ttl = {k: 0.0 for k in rows}

            for pd in pos_data:
                sign, size = pd["sign"], pd["size"]

                if pd["kind"] == "future":
                    F0 = pd["F"]
                    F_b = F0 * factor
                    # PnL for inverse future
                    ttl["pnl"] += sign * size * (1.0 / F0 - 1.0 / F_b)
                    ttl["bs_delta"] += sign * size / F_b
                    ttl["smile_delta"] += sign * size / F_b

                elif pd["kind"] == "option":
                    F0 = pd["F"]
                    F_b = F0 * factor
                    K, T, sigma = pd["K"], pd["T"], pd["sigma"]
                    option_type = pd["option_type"]
                    direction = pd["direction"]
                    bs_fn = black76_call if option_type == "call" else black76_put

                    if sigma <= 0 or T <= 0 or F_b <= 0:
                        continue

                    # PnL (sticky strike, inverse: value in BTC)
                    v_btc_0 = bs_fn(F0, K, T, sigma) / F0
                    v_btc_b = bs_fn(F_b, K, T, sigma) / F_b
                    ttl["pnl"] += sign * size * (v_btc_b - v_btc_0)

                    # Greeks at bumped level
                    greeks = compute_option_greeks(
                        F_b, K, T, sigma, option_type, size, direction,
                        pd.get("svi_params"), pd.get("svi_T"))
                    for k in ["bs_delta", "smile_delta", "gamma_1pct", "vega_usd", "theta_usd"]:
                        ttl[k] += greeks[k]

            for k in rows:
                rows[k].append(round(ttl[k], 6))

        return {"bumps": bump_pcts, "spots": spots, **rows}


    # ---- P&L History ----
    def _load_pnl_csv(self):
        """Load previous session P&L data from CSV."""
        if not os.path.exists(self.pnl_csv):
            return
        try:
            with open(self.pnl_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.pnl_history.append({
                        "ts": row["ts"],
                        "equity": float(row["equity"]),
                        "balance": float(row["balance"]),
                        "spot": float(row["spot"]),
                        "equity_usd": float(row.get("equity_usd", 0)),
                        "unrealised_pnl": float(row.get("unrealised_pnl", 0)),
                        "perp_mark": float(row.get("perp_mark", row.get("spot", 0))),
                    })
            print(f"[P&L] Loaded {len(self.pnl_history)} historical points from CSV")
        except Exception as e:
            print(f"[P&L] Error loading CSV: {e}")

    def _record_pnl(self, account, spot, perp_mark=0):
        """Record a P&L snapshot."""
        if not account or not spot:
            return
        equity = account.get("equity", 0)
        balance = account.get("balance", 0)
        unrealised = equity - balance
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        point = {
            "ts": ts,
            "equity": round(equity, 8),
            "balance": round(balance, 8),
            "spot": round(spot, 2),
            "equity_usd": round(equity * spot, 2),
            "unrealised_pnl": round(unrealised, 8),
            "perp_mark": round(perp_mark, 2),
        }
        with self.lock:
            self.pnl_history.append(point)

        # Append to CSV
        try:
            write_header = not os.path.exists(self.pnl_csv) or os.path.getsize(self.pnl_csv) == 0
            with open(self.pnl_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["ts", "equity", "balance", "spot", "equity_usd", "unrealised_pnl", "perp_mark"])
                if write_header:
                    writer.writeheader()
                writer.writerow(point)
        except Exception as e:
            print(f"[P&L] CSV write error: {e}")

    def get_pnl_history(self, last_minutes=None):
        """Return P&L history for the chart. last_minutes filters by time."""
        with self.lock:
            data = list(self.pnl_history)
        if last_minutes and last_minutes > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=last_minutes)
            cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
            data = [p for p in data if p["ts"] >= cutoff_str]
        if not data:
            return None
        return {
            "timestamps": [p["ts"] for p in data],
            "equity_btc": [p["equity"] for p in data],
            "balance_btc": [p["balance"] for p in data],
            "equity_usd": [p["equity_usd"] for p in data],
            "spot": [p["spot"] for p in data],
            "perp_mark": [p.get("perp_mark", p["spot"]) for p in data],
            "unrealised_pnl": [p["unrealised_pnl"] for p in data],
            "count": len(data),
            "first_ts": data[0]["ts"],
            "last_ts": data[-1]["ts"],
        }

    # ---- Reconciliation ----
    def compute_reconciliation(self):
        """Fetch all trades, settlements/funding, and reconcile against API equity."""
        token = authenticate()
        if not token:
            return {"error": "Auth failed"}

        # 1. Account summary (current truth)
        account = api_call("private/get_account_summary", {"currency": "BTC"}, token)
        if not account:
            return {"error": "Could not fetch account summary"}

        api_equity = account.get("equity", 0)
        api_balance = account.get("balance", 0)

        # 2. Fetch ALL trades (paginate if needed)
        all_trades = []
        has_more = True
        start_seq = None
        while has_more:
            params = {"currency": "BTC", "count": 200, "sorting": "asc"}
            if start_seq:
                params["start_seq"] = start_seq
            result = api_call("private/get_user_trades_by_currency", params, token)
            if not result:
                break
            trades = result.get("trades", [])
            if not trades:
                break
            all_trades.extend(trades)
            has_more = result.get("has_more", False)
            if has_more and trades:
                start_seq = trades[-1].get("trade_seq", 0) + 1

        # 3. Fetch settlement/funding history (includes daily settlements & funding)
        all_settlements = []
        result = api_call("private/get_settlement_history_by_currency",
                          {"currency": "BTC", "count": 1000, "type": "settlement"}, token)
        if result and isinstance(result, dict):
            all_settlements = result.get("settlements", [])
        elif result and isinstance(result, list):
            all_settlements = result

        # Also get delivery settlements
        deliveries = api_call("private/get_settlement_history_by_currency",
                              {"currency": "BTC", "count": 1000, "type": "delivery"}, token)
        if deliveries:
            if isinstance(deliveries, dict):
                all_settlements.extend(deliveries.get("settlements", []))
            elif isinstance(deliveries, list):
                all_settlements.extend(deliveries)

        # 4. Fetch deposits/withdrawals
        transfers_result = api_call("private/get_transfers", {"currency": "BTC", "count": 1000}, token)
        all_transfers = []
        if transfers_result:
            all_transfers = transfers_result.get("data", []) if isinstance(transfers_result, dict) else transfers_result

        # 5. Process trades — compute realised PnL and premium flows
        trade_rows = []
        total_fees = 0.0
        total_realised_pnl = 0.0
        total_premium_received = 0.0  # premium from selling options
        total_premium_paid = 0.0      # premium from buying options

        for t in all_trades:
            ts = datetime.fromtimestamp(t["timestamp"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            instrument = t["instrument_name"]
            direction = t["direction"]
            amount = t.get("amount", 0)
            price = t.get("price", 0)
            fee = t.get("fee", 0)
            pnl = t.get("profit_loss", 0) or 0
            is_option = instrument.endswith("-C") or instrument.endswith("-P")

            total_fees += fee
            total_realised_pnl += pnl

            # For options: price is in BTC per contract, amount is number of contracts
            # Premium = price * amount (in BTC)
            if is_option:
                premium = price * amount
                if direction == "sell":
                    total_premium_received += premium
                else:
                    total_premium_paid += premium
            else:
                premium = 0

            trade_rows.append({
                "ts": ts,
                "instrument": instrument,
                "direction": direction,
                "amount": amount,
                "price": price,
                "fee": round(fee, 8),
                "pnl": round(pnl, 8),
                "premium": round(premium, 8),
                "is_option": is_option,
            })

        # 6. Process settlements/funding
        settlement_rows = []
        total_funding = 0.0
        total_session_pnl = 0.0

        for s in all_settlements:
            ts = datetime.fromtimestamp(s["timestamp"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            funding = s.get("funding", 0) or 0
            session_pl = s.get("session_profit_loss", 0) or 0
            total_funding += funding
            total_session_pnl += session_pl

            settlement_rows.append({
                "ts": ts,
                "type": s.get("type", "settlement"),
                "instrument": s.get("instrument_name", "BTC-PERPETUAL"),
                "funding": round(funding, 8),
                "session_pnl": round(session_pl, 8),
                "position": s.get("position", 0),
                "mark_price": s.get("mark_price", 0),
                "index_price": s.get("index_price", 0),
            })

        # 7. Process transfers (deposits/withdrawals)
        transfer_rows = []
        total_deposits = 0.0
        total_withdrawals = 0.0

        for tr in all_transfers:
            ts_val = tr.get("created_timestamp") or tr.get("updated_timestamp", 0)
            ts = datetime.fromtimestamp(ts_val / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if ts_val else "N/A"
            amount = tr.get("amount", 0)
            direction = tr.get("direction", "")
            state = tr.get("state", "")

            if state in ("completed", "confirmed") or not state:
                if direction == "payment":
                    total_deposits += amount
                elif direction == "withdrawal":
                    total_withdrawals += amount
                else:
                    total_deposits += amount  # assume deposit if unclear

            transfer_rows.append({
                "ts": ts,
                "type": direction or "deposit",
                "amount": round(amount, 8),
                "state": state,
                "tx_id": tr.get("transaction_id", ""),
            })

        # 8. Reconciliation
        # Deribit balance = initial_deposit - withdrawals + realised_pnl + funding - fees
        # So: initial_deposit = balance - realised_pnl - funding + fees + withdrawals
        unrealised = api_equity - api_balance

        if not all_transfers:
            # No explicit transfers — infer initial deposit from balance
            inferred_deposit = api_balance - total_realised_pnl - total_funding + total_fees + total_withdrawals
            total_deposits = inferred_deposit

        calculated_balance = (total_deposits - total_withdrawals
                              + total_realised_pnl
                              + total_funding
                              - total_fees)

        calculated_equity = calculated_balance + unrealised

        balance_diff = api_balance - calculated_balance
        equity_diff = api_equity - calculated_equity

        spot = self.spot_price or 0

        return {
            "api_equity": round(api_equity, 8),
            "api_balance": round(api_balance, 8),
            "calculated_balance": round(calculated_balance, 8),
            "calculated_equity": round(calculated_equity, 8),
            "balance_diff": round(balance_diff, 8),
            "equity_diff": round(equity_diff, 8),
            "unrealised": round(unrealised, 8),
            "total_deposits": round(total_deposits, 8),
            "total_withdrawals": round(total_withdrawals, 8),
            "total_realised_pnl": round(total_realised_pnl, 8),
            "total_fees": round(total_fees, 8),
            "total_funding": round(total_funding, 8),
            "total_premium_received": round(total_premium_received, 8),
            "total_premium_paid": round(total_premium_paid, 8),
            "net_premium": round(total_premium_received - total_premium_paid, 8),
            "spot": round(spot, 2),
            "trades": trade_rows,
            "settlements": settlement_rows,
            "transfers": transfer_rows,
            "trade_count": len(trade_rows),
            "settlement_count": len(settlement_rows),
            "transfer_count": len(transfer_rows),
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }

    # ---- Authenticated WebSocket for Orders ----
    def start_order_ws(self):
        """Start persistent authenticated WS connection for fast order execution."""
        threading.Thread(target=self._run_order_ws, daemon=True).start()

    def _run_order_ws(self):
        while True:
            try:
                self.order_ws_ready.clear()
                self.order_ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_order_ws_open,
                    on_message=self._on_order_ws_message,
                    on_error=lambda ws, e: print(f"[Order WS] Error: {e}"),
                    on_close=self._on_order_ws_close,
                )
                self.order_ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"[Order WS] Exception: {e}")
            self.order_ws_ready.clear()
            time.sleep(3)

    def _on_order_ws_open(self, ws):
        print("[Order WS] Connected, authenticating...")
        ws.send(json.dumps({
            "jsonrpc": "2.0", "id": 9999,
            "method": "public/auth",
            "params": {
                "grant_type": "client_credentials",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
            }
        }))

    def _on_order_ws_close(self, ws, code, msg):
        print(f"[Order WS] Closed (code={code}), reconnecting...")
        self.order_ws_ready.clear()

    def _on_order_ws_message(self, ws, message):
        data = json.loads(message)
        msg_id = data.get("id")

        # Auth response
        if msg_id == 9999:
            if "result" in data:
                self.order_ws_token = data["result"]["access_token"]
                self.order_ws_ready.set()
                print("[Order WS] Authenticated — ready for orders")
            else:
                print(f"[Order WS] Auth failed: {data.get('error')}")
            return

        # Order response
        if msg_id and msg_id in self._order_pending:
            self._order_pending[msg_id]["result"] = data
            self._order_pending[msg_id]["event"].set()

    def execute_order_ws(self, instrument, direction, size, timeout=10):
        """Execute a market order via the authenticated WS. Returns result dict."""
        if not self.order_ws_ready.wait(timeout=5):
            return {"ok": False, "error": "Order WS not ready (not authenticated)"}

        with self.lock:
            self._order_id_counter += 1
            req_id = self._order_id_counter

        event = threading.Event()
        self._order_pending[req_id] = {"event": event, "result": None}

        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": f"private/{direction}",
            "params": {
                "instrument_name": instrument,
                "amount": size,
                "type": "market",
            }
        }

        try:
            self.order_ws.send(json.dumps(msg))
        except Exception as e:
            del self._order_pending[req_id]
            return {"ok": False, "error": f"WS send failed: {e}"}

        # Wait for response
        if not event.wait(timeout=timeout):
            del self._order_pending[req_id]
            return {"ok": False, "error": "Order timeout — no response from Deribit"}

        data = self._order_pending.pop(req_id, {}).get("result", {})

        if "error" in data:
            err = data["error"]
            return {"ok": False, "error": err.get("message", str(err))}

        result = data.get("result", {})
        order = result.get("order", {})
        trades = result.get("trades", [])

        return {
            "ok": True,
            "order_id": order.get("order_id"),
            "state": order.get("order_state"),
            "filled_amount": order.get("filled_amount", 0),
            "average_price": order.get("average_price", 0),
            "direction": order.get("direction"),
            "instrument": instrument,
            "n_trades": len(trades),
            "method": "websocket",
        }

    # ---- Hedge Calculator ----
    def compute_hedge(self, hedge_delta=True, hedge_gamma=False, hedge_vega=False, target_expiry=None):
        """Compute optimal hedge trades to flatten selected greeks.

        Strategy:
          1. Delta only → trade perpetual
          2. Gamma + delta → pick best ATM straddle for gamma, then perp for residual delta
          3. Vega + delta → pick best ATM option for vega, then perp for residual delta
          4. Gamma + vega + delta → solve 2x2 system with two options, then perp for delta
        """
        with self.lock:
            risk = dict(self.risk_data)
            spot = self.spot_price or 0

        totals = risk.get("totals", {})
        if not totals or not spot:
            return {"error": "No risk data yet — wait for first risk refresh"}

        current = {
            "bs_delta": totals.get("bs_delta", 0),
            "gamma_1pct": totals.get("gamma_1pct", 0),
            "vega_usd": totals.get("vega_usd", 0),
            "theta_usd": totals.get("theta_usd", 0),
        }

        trades = []
        residual = dict(current)

        # Find candidate options for gamma/vega hedging
        candidates = []
        if hedge_gamma or hedge_vega:
            candidates = self._get_hedge_candidates(spot, target_expiry)

        # Step 1: If hedging gamma and/or vega, find option trades first
        if (hedge_gamma or hedge_vega) and candidates:
            if hedge_gamma and hedge_vega and len(candidates) >= 2:
                # Solve 2x2: pick two best candidates (different expiries or types if possible)
                trade = self._solve_gamma_vega_hedge(candidates, residual)
                if trade:
                    trades.extend(trade)
                    for t in trade:
                        residual["bs_delta"] += t["delta_impact"]
                        residual["gamma_1pct"] += t["gamma_impact"]
                        residual["vega_usd"] += t["vega_impact"]
                        residual["theta_usd"] += t["theta_impact"]

            elif hedge_gamma:
                trade = self._solve_single_greek_hedge(candidates, residual, "gamma_1pct")
                if trade:
                    trades.append(trade)
                    residual["bs_delta"] += trade["delta_impact"]
                    residual["gamma_1pct"] += trade["gamma_impact"]
                    residual["vega_usd"] += trade["vega_impact"]
                    residual["theta_usd"] += trade["theta_impact"]

            elif hedge_vega:
                trade = self._solve_single_greek_hedge(candidates, residual, "vega_usd")
                if trade:
                    trades.append(trade)
                    residual["bs_delta"] += trade["delta_impact"]
                    residual["gamma_1pct"] += trade["gamma_impact"]
                    residual["vega_usd"] += trade["vega_impact"]
                    residual["theta_usd"] += trade["theta_impact"]

        # Step 2: Delta hedge with perpetual
        if hedge_delta and abs(residual["bs_delta"]) > 1e-6:
            perp_usd = round(-residual["bs_delta"] * spot)
            # Deribit minimum is $10 increments
            perp_usd = round(perp_usd / 10) * 10
            if abs(perp_usd) >= 10:
                direction = "buy" if perp_usd > 0 else "sell"
                delta_impact = perp_usd / spot
                trades.append({
                    "instrument": "BTC-PERPETUAL",
                    "direction": direction,
                    "size": abs(perp_usd),
                    "size_label": f"${abs(perp_usd):,}",
                    "cost": "$0 (no premium)",
                    "delta_impact": delta_impact,
                    "gamma_impact": 0,
                    "vega_impact": 0,
                    "theta_impact": 0,
                    "rationale": f"Flatten delta: {perp_usd/spot:+.4f} BTC equiv",
                })
                residual["bs_delta"] += delta_impact

        after = {k: round(residual[k], 6) for k in current}

        return {
            "current": {k: round(v, 6) for k, v in current.items()},
            "trades": trades,
            "after": after,
            "spot": spot,
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "candidates_found": len(candidates),
        }

    def _get_hedge_candidates(self, spot, target_expiry=None):
        """Fetch near-ATM options and compute per-contract greeks."""
        # Get expiries that have options
        try:
            all_instruments = api_call("public/get_instruments",
                                       {"currency": "BTC", "kind": "option"})
            if not all_instruments:
                return []
        except Exception:
            return []

        # Filter to inverse, and optionally to a specific expiry
        inv_opts = [i for i in all_instruments if i.get("settlement_currency") == "BTC"]

        if target_expiry:
            inv_opts = [i for i in inv_opts if i["instrument_name"].split("-")[1] == target_expiry]
        else:
            # Use expiries that have positions, plus nearest expiry
            pos_expiries = set()
            for pos in self.risk_data.get("positions", []):
                if pos.get("kind") == "option":
                    pos_expiries.add(pos["instrument"].split("-")[1])
            if not pos_expiries:
                # Just use first 2 expiries
                all_exp = sorted(set(i["instrument_name"].split("-")[1] for i in inv_opts),
                                 key=lambda s: get_expiry_datetime(s))
                pos_expiries = set(all_exp[:2])
            inv_opts = [i for i in inv_opts if i["instrument_name"].split("-")[1] in pos_expiries]

        # Filter to near-ATM strikes (within 15% of spot)
        near_atm = []
        for i in inv_opts:
            K = i["strike"]
            if abs(K - spot) / spot < 0.15:
                near_atm.append(i)

        if not near_atm:
            return []

        # Fetch tickers in parallel
        def fetch_ticker(inst):
            try:
                r = requests.get(f"{BASE_URL}/public/ticker",
                                 params={"instrument_name": inst["instrument_name"]}, timeout=5)
                return r.json().get("result", {})
            except Exception:
                return {}

        with ThreadPoolExecutor(max_workers=20) as pool:
            tickers = list(pool.map(fetch_ticker, near_atm))

        candidates = []
        now = datetime.now(timezone.utc)
        for inst, tick in zip(near_atm, tickers):
            iv = tick.get("mark_iv")
            if not iv or iv <= 0:
                continue
            name = inst["instrument_name"]
            parts = name.split("-")
            expiry_str = parts[1]
            K = float(parts[2])
            option_type = "call" if parts[3] == "C" else "put"
            sigma = iv / 100.0

            F, T = self._get_forward_for_expiry_str(expiry_str)
            if not F or T <= 0:
                continue

            mark_btc = tick.get("mark_price", 0)

            # Per-contract greeks (for size=1, direction=buy)
            greeks = compute_option_greeks(F, K, T, sigma, option_type, 1.0, "buy")

            candidates.append({
                "instrument": name,
                "expiry": expiry_str,
                "K": K,
                "option_type": option_type,
                "F": F,
                "T": T,
                "sigma": sigma,
                "mark_btc": mark_btc,
                "bid": tick.get("best_bid_price", 0) or 0,
                "ask": tick.get("best_ask_price", 0) or 0,
                "delta_per": greeks["bs_delta"],
                "gamma_per": greeks["gamma_1pct"],
                "vega_per": greeks["vega_usd"],
                "theta_per": greeks["theta_usd"],
            })

        return candidates

    def _solve_single_greek_hedge(self, candidates, residual, greek_key):
        """Find the option that best hedges a single greek, compute size needed."""
        target = -residual[greek_key]  # we want to ADD this much to zero out

        if abs(target) < 1e-8:
            return None

        # Pick candidate with largest absolute per-contract value for this greek
        best = None
        best_abs = 0
        for c in candidates:
            per = c.get(greek_key.replace("gamma_1pct", "gamma_per").replace("vega_usd", "vega_per"), 0)
            if greek_key == "gamma_1pct":
                per = c["gamma_per"]
            elif greek_key == "vega_usd":
                per = c["vega_per"]
            if abs(per) > best_abs:
                best_abs = abs(per)
                best = c

        if not best or best_abs < 1e-10:
            return None

        per_key = "gamma_per" if greek_key == "gamma_1pct" else "vega_per"
        n_contracts = target / best[per_key]  # positive = buy, negative = sell

        direction = "buy" if n_contracts > 0 else "sell"
        size = abs(n_contracts)
        # Round to Deribit's min trade amount (0.1 contracts)
        size = round(size * 10) / 10
        if size < 0.1:
            size = 0.1

        sign = 1.0 if direction == "buy" else -1.0
        cost_btc = best["mark_btc"] * size

        return {
            "instrument": best["instrument"],
            "direction": direction,
            "size": size,
            "size_label": f"{size:.1f}",
            "cost": f"{cost_btc:.6f} BTC (≈${cost_btc * (self.spot_price or 0):,.0f})",
            "delta_impact": sign * size * best["delta_per"],
            "gamma_impact": sign * size * best["gamma_per"],
            "vega_impact": sign * size * best["vega_per"],
            "theta_impact": sign * size * best["theta_per"],
            "rationale": f"Best {greek_key.split('_')[0]} per $ premium near ATM",
        }

    def _solve_gamma_vega_hedge(self, candidates, residual):
        """Solve 2x2 system to hedge both gamma and vega with two options."""
        target_gamma = -residual["gamma_1pct"]
        target_vega = -residual["vega_usd"]

        if abs(target_gamma) < 1e-8 and abs(target_vega) < 1e-8:
            return None

        # Pick two candidates: one with high gamma/vega ratio, one with low
        # This ensures the 2x2 system is well-conditioned
        scored = []
        for c in candidates:
            g, v = c["gamma_per"], c["vega_per"]
            if abs(v) > 1e-6 and abs(g) > 1e-8:
                ratio = g / v
                scored.append((ratio, c))

        if len(scored) < 2:
            # Fall back to single greek hedge (gamma takes priority)
            trade = self._solve_single_greek_hedge(candidates, residual, "gamma_1pct")
            return [trade] if trade else None

        scored.sort(key=lambda x: x[0])
        # Pick most extreme ratio pair for best conditioning
        c1 = scored[0][1]   # lowest gamma/vega ratio (vega-heavy)
        c2 = scored[-1][1]  # highest gamma/vega ratio (gamma-heavy)

        g1, v1 = c1["gamma_per"], c1["vega_per"]
        g2, v2 = c2["gamma_per"], c2["vega_per"]

        # Solve: n1*g1 + n2*g2 = target_gamma
        #        n1*v1 + n2*v2 = target_vega
        det = g1 * v2 - g2 * v1
        if abs(det) < 1e-15:
            trade = self._solve_single_greek_hedge(candidates, residual, "gamma_1pct")
            return [trade] if trade else None

        n1 = (target_gamma * v2 - target_vega * g2) / det
        n2 = (target_vega * g1 - target_gamma * v1) / det

        trades = []
        for n, c in [(n1, c1), (n2, c2)]:
            direction = "buy" if n > 0 else "sell"
            size = abs(n)
            size = round(size * 10) / 10
            if size < 0.1:
                continue
            sign = 1.0 if direction == "buy" else -1.0
            cost_btc = c["mark_btc"] * size
            trades.append({
                "instrument": c["instrument"],
                "direction": direction,
                "size": size,
                "size_label": f"{size:.1f}",
                "cost": f"{cost_btc:.6f} BTC (≈${cost_btc * (self.spot_price or 0):,.0f})",
                "delta_impact": sign * size * c["delta_per"],
                "gamma_impact": sign * size * c["gamma_per"],
                "vega_impact": sign * size * c["vega_per"],
                "theta_impact": sign * size * c["theta_per"],
                "rationale": f"Gamma/vega hedge pair",
            })

        return trades if trades else None


# ====== AUTO HEDGER (Gamma-Aware Delta Hedging) ======
class AutoHedger:
    """Continuously monitors portfolio delta and rebalances via BTC-PERPETUAL.

    Gamma awareness: when |gamma| is large, use tighter delta bands and hedge more
    frequently, because delta changes rapidly with spot. When |gamma| is small,
    wider bands are acceptable.
    """

    def __init__(self, engine_ref):
        self.engine = engine_ref
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Configurable parameters
        self.base_delta_threshold = 0.02   # BTC — hedge when |delta| exceeds this
        self.gamma_scaling = True          # tighten threshold when gamma is large
        self.check_interval = 30           # seconds between checks
        self.min_trade_usd = 10            # Deribit minimum
        self.max_trade_usd = 100000        # safety cap per hedge

        # Log of actions (ring buffer)
        self.log = deque(maxlen=500)
        self.stats = {"hedges": 0, "total_traded_usd": 0, "last_hedge_ts": None}

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self._add_log("AUTO HEDGE STARTED", f"threshold={self.base_delta_threshold:.4f} BTC, "
                       f"interval={self.check_interval}s, gamma_scaling={'ON' if self.gamma_scaling else 'OFF'}")

    def stop(self):
        self.running = False
        self._add_log("AUTO HEDGE STOPPED", "")

    def update_params(self, threshold=None, interval=None, gamma_scaling=None, max_trade=None):
        with self.lock:
            if threshold is not None:
                self.base_delta_threshold = threshold
            if interval is not None:
                self.check_interval = max(5, int(interval))
            if gamma_scaling is not None:
                self.gamma_scaling = gamma_scaling
            if max_trade is not None:
                self.max_trade_usd = max_trade
        self._add_log("PARAMS UPDATED",
                       f"threshold={self.base_delta_threshold:.4f}, interval={self.check_interval}s, "
                       f"gamma={'ON' if self.gamma_scaling else 'OFF'}, max=${self.max_trade_usd:,.0f}")

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "threshold": self.base_delta_threshold,
                "interval": self.check_interval,
                "gamma_scaling": self.gamma_scaling,
                "max_trade_usd": self.max_trade_usd,
                "stats": dict(self.stats),
                "log": list(self.log),
            }

    def _add_log(self, action, detail):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = {"ts": ts, "action": action, "detail": detail}
        with self.lock:
            self.log.append(entry)
        print(f"[AutoHedge] {ts} {action}: {detail}")

    def _loop(self):
        while self.running:
            try:
                self._check_and_hedge()
            except Exception as e:
                self._add_log("ERROR", str(e))
                traceback.print_exc()
            # Sleep in small increments so stop() is responsive
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)

    def _get_effective_threshold(self, gamma_1pct):
        """Adjust delta threshold based on portfolio gamma.

        When gamma is large, delta changes quickly with spot, so we hedge
        more aggressively (tighter bands). The scaling is:
          effective = base_threshold / (1 + k * |gamma|)
        where k is tuned so that gamma of 0.01 BTC halves the threshold.
        """
        if not self.gamma_scaling or abs(gamma_1pct) < 1e-8:
            return self.base_delta_threshold

        # k=50: gamma of 0.01 → divisor = 1.5, gamma of 0.05 → divisor = 3.5
        k = 50.0
        divisor = 1.0 + k * abs(gamma_1pct)
        effective = self.base_delta_threshold / divisor
        # Floor at 20% of base to prevent infinite tightening
        return max(effective, self.base_delta_threshold * 0.2)

    def _check_and_hedge(self):
        with self.engine.lock:
            risk = dict(self.engine.risk_data)
            spot = self.engine.spot_price or 0

        totals = risk.get("totals", {})
        if not totals or not spot:
            self._add_log("SKIP", "No risk data or spot price available")
            return

        bs_delta = totals.get("bs_delta", 0)
        gamma_1pct = totals.get("gamma_1pct", 0)

        with self.lock:
            effective_threshold = self._get_effective_threshold(gamma_1pct)

        self._add_log("CHECK",
                       f"delta={bs_delta:+.6f} BTC, gamma={gamma_1pct:+.6f}, "
                       f"threshold={effective_threshold:.4f}")

        if abs(bs_delta) <= effective_threshold:
            return  # within tolerance

        # Compute hedge trade
        hedge_usd = round(-bs_delta * spot)
        hedge_usd = round(hedge_usd / 10) * 10  # Deribit $10 increments

        if abs(hedge_usd) < self.min_trade_usd:
            self._add_log("SKIP", f"Hedge too small: ${abs(hedge_usd)}")
            return

        with self.lock:
            cap = self.max_trade_usd
        if abs(hedge_usd) > cap:
            self._add_log("CAPPED", f"Hedge ${abs(hedge_usd):,.0f} capped to ${cap:,.0f}")
            hedge_usd = int(math.copysign(cap, hedge_usd))
            hedge_usd = round(hedge_usd / 10) * 10

        direction = "buy" if hedge_usd > 0 else "sell"
        size = abs(hedge_usd)

        self._add_log("HEDGING",
                       f"{direction.upper()} ${size:,} BTC-PERPETUAL "
                       f"(delta was {bs_delta:+.6f}, gamma={gamma_1pct:+.6f})")

        result = self.engine.execute_order_ws("BTC-PERPETUAL", direction, size)

        if result.get("ok"):
            with self.lock:
                self.stats["hedges"] += 1
                self.stats["total_traded_usd"] += size
                self.stats["last_hedge_ts"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
            self._add_log("FILLED",
                           f"${size:,} @ ${result.get('average_price', 0):,.2f} "
                           f"(filled {result.get('filled_amount', 0)})")
            # Trigger risk refresh so next check has updated delta
            try:
                self.engine._compute_risk()
            except Exception:
                pass
        else:
            self._add_log("FAILED", result.get("error", "Unknown error"))


# ====== BACKTEST ENGINE ======
class BacktestEngine:
    """Replay historical price data and simulate a delta-hedging strategy.

    Fetches OHLCV candles from Deribit, simulates an options portfolio,
    and delta-hedges at configurable intervals. Reports PnL, Sharpe, drawdown.
    """

    @staticmethod
    def fetch_candles(instrument, resolution, start_ts, end_ts):
        """Fetch historical candles from Deribit's public API.
        resolution: '1' (1min), '5', '15', '30', '60', '1D'
        timestamps in ms.
        """
        all_candles = []
        cursor = start_ts
        while cursor < end_ts:
            params = {
                "instrument_name": instrument,
                "resolution": resolution,
                "start_timestamp": int(cursor),
                "end_timestamp": int(end_ts),
                "count": 5000,
            }
            data = api_call("public/get_tradingview_chart_data", params)
            if not data or "ticks" not in data:
                break
            ticks = data["ticks"]
            closes = data["close"]
            highs = data["high"]
            lows = data["low"]
            opens = data["open"]
            volumes = data.get("volume", [0] * len(ticks))
            for i in range(len(ticks)):
                all_candles.append({
                    "ts": ticks[i],
                    "open": opens[i],
                    "high": highs[i],
                    "low": lows[i],
                    "close": closes[i],
                    "volume": volumes[i],
                })
            if len(ticks) < 2:
                break
            cursor = ticks[-1] + 1  # next candle after last
        return all_candles

    @staticmethod
    def run_delta_hedge_backtest(params):
        """Run a delta-hedging backtest.

        params:
          - instrument: e.g. "BTC-PERPETUAL"
          - days: lookback period
          - resolution: candle size ('60' = 1h, '1D' = daily)
          - option_type: 'call' or 'put'
          - strike_offset_pct: strike as % offset from initial spot (0 = ATM)
          - option_size: number of option contracts (in BTC)
          - option_direction: 'buy' or 'sell'
          - iv: assumed constant IV (decimal, e.g. 0.60)
          - tte: time to expiry at start (years, e.g. 0.0833 ≈ 30 days)
          - hedge_interval: how many candles between rehedges
          - cost_per_trade: transaction cost as fraction (e.g. 0.0005)
          - delta_threshold: only hedge when |delta change| exceeds this
        """
        instrument = params.get("instrument", "BTC-PERPETUAL")
        days = params.get("days", 30)
        resolution = params.get("resolution", "60")
        option_type = params.get("option_type", "call")
        strike_offset_pct = params.get("strike_offset_pct", 0)
        option_size = params.get("option_size", 1.0)
        option_direction = params.get("option_direction", "sell")
        iv = params.get("iv", 0.60)
        tte = params.get("tte", 30.0 / 365.25)
        hedge_interval = params.get("hedge_interval", 1)
        cost_per_trade = params.get("cost_per_trade", 0.0005)
        delta_threshold = params.get("delta_threshold", 0.001)

        # Fetch candles
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - int(days * 24 * 3600 * 1000)
        candles = BacktestEngine.fetch_candles(instrument, resolution, start_ms, now_ms)

        if len(candles) < 10:
            return {"error": f"Not enough data: only {len(candles)} candles"}

        # Determine time step per candle (in years)
        if resolution == "1D":
            dt_years = 1.0 / 365.25
        else:
            dt_years = int(resolution) / (365.25 * 24 * 60)

        # Setup
        prices = [c["close"] for c in candles]
        timestamps = [c["ts"] for c in candles]
        S0 = prices[0]
        K = S0 * (1.0 + strike_offset_pct / 100.0)
        sign = 1.0 if option_direction == "buy" else -1.0
        bs_fn = black76_call if option_type == "call" else black76_put

        # Initial option value (in USD)
        T_remaining = tte
        initial_option_usd = bs_fn(S0, K, T_remaining, iv)
        initial_premium_btc = sign * option_size * initial_option_usd / S0

        # Initial delta hedge
        eps = S0 * 0.0005
        v_up = bs_fn(S0 + eps, K, T_remaining, iv)
        v_dn = bs_fn(S0 - eps, K, T_remaining, iv)
        option_delta = sign * option_size * (v_up - v_dn) / (2.0 * eps)
        perp_position = -option_delta  # hedge: opposite delta
        # perp_position is in BTC-equivalent (delta units)

        # Track
        results = []
        total_costs = 0.0
        hedge_count = 0
        candle_since_hedge = 0
        prev_delta = option_delta

        for i, price in enumerate(prices):
            T_remaining = max(tte - i * dt_years, 1e-6)
            F = price  # for perpetual, F ≈ spot

            # Compute option greeks
            eps = F * 0.0005
            v_usd = bs_fn(F, K, T_remaining, iv)
            v_up = bs_fn(F + eps, K, T_remaining, iv)
            v_dn = bs_fn(F - eps, K, T_remaining, iv)
            current_delta = sign * option_size * (v_up - v_dn) / (2.0 * eps)

            # Gamma (change in delta for 1% move)
            F_u = F * 1.01
            F_d = F * 0.99
            eu = F_u * 0.0005
            ed = F_d * 0.0005
            d_up = sign * option_size * (bs_fn(F_u + eu, K, T_remaining, iv) - bs_fn(F_u - eu, K, T_remaining, iv)) / (2.0 * eu)
            d_dn = sign * option_size * (bs_fn(F_d + ed, K, T_remaining, iv) - bs_fn(F_d - ed, K, T_remaining, iv)) / (2.0 * ed)
            gamma_1pct = (d_up - d_dn) / 2.0

            # Portfolio delta = option delta + perp delta
            portfolio_delta = current_delta + perp_position

            # Check if we should rehedge
            candle_since_hedge += 1
            did_hedge = False
            trade_size_usd = 0

            if candle_since_hedge >= hedge_interval and abs(portfolio_delta) > delta_threshold:
                # Rehedge: trade perp to flatten delta
                trade_delta = -portfolio_delta
                trade_size_usd = abs(trade_delta * F)
                cost = trade_size_usd * cost_per_trade
                total_costs += cost
                perp_position += trade_delta
                portfolio_delta = current_delta + perp_position
                hedge_count += 1
                candle_since_hedge = 0
                did_hedge = True

            # Option PnL (in BTC): V_btc = V_usd / F
            option_btc = sign * option_size * v_usd / F
            # Perp PnL (cumulative, inverse): sum of position * (1/F_prev - 1/F_curr)
            # We track this incrementally
            if i == 0:
                perp_pnl_btc = 0.0
                prev_price = F
            else:
                # Perp PnL this step: position * (1/prev_price - 1/price)
                perp_pnl_step = perp_position * (1.0 / prev_price - 1.0 / price) * prev_price  # simplified
                # Actually for inverse perp: PnL in BTC = contracts_usd * (1/entry - 1/exit)
                # But we're tracking BTC-equivalent position, so:
                # PnL_btc = perp_delta * (price - prev_price) / price (approx for small moves)
                perp_pnl_step = perp_position * (price - prev_price) / price
                perp_pnl_btc += perp_pnl_step
                prev_price = price

            # Total portfolio BTC value
            total_btc = option_btc + perp_pnl_btc - total_costs / F + initial_premium_btc

            ts_str = datetime.fromtimestamp(timestamps[i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

            results.append({
                "ts": ts_str,
                "price": round(price, 2),
                "T_remaining": round(T_remaining, 6),
                "option_delta": round(current_delta, 6),
                "portfolio_delta": round(portfolio_delta, 6),
                "gamma_1pct": round(gamma_1pct, 6),
                "option_value_btc": round(option_btc, 8),
                "perp_pnl_btc": round(perp_pnl_btc, 8),
                "total_costs_btc": round(total_costs / F, 8),
                "total_btc": round(total_btc, 8),
                "hedged": did_hedge,
                "trade_size_usd": round(trade_size_usd, 0),
            })

            prev_delta = current_delta

        # Performance metrics
        if len(results) < 2:
            return {"error": "Not enough results to compute metrics"}

        pnl_series = [r["total_btc"] for r in results]
        returns = [(pnl_series[i] - pnl_series[i - 1]) for i in range(1, len(pnl_series))]
        final_pnl = pnl_series[-1] - pnl_series[0]

        # Sharpe (annualized)
        if len(returns) > 1:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            # Annualize: depends on resolution
            if resolution == "1D":
                periods_per_year = 365.25
            else:
                periods_per_year = 365.25 * 24 * 60 / int(resolution)
            sharpe = (mean_ret / std_ret * math.sqrt(periods_per_year)) if std_ret > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        peak = pnl_series[0]
        max_dd = 0
        for v in pnl_series:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd

        return {
            "results": results,
            "metrics": {
                "total_pnl_btc": round(final_pnl, 8),
                "sharpe": round(sharpe, 3),
                "max_drawdown_btc": round(max_dd, 8),
                "hedge_count": hedge_count,
                "total_costs_btc": round(total_costs / prices[-1], 8),
                "total_traded_usd": round(sum(r["trade_size_usd"] for r in results), 0),
                "n_candles": len(results),
                "initial_price": round(prices[0], 2),
                "final_price": round(prices[-1], 2),
                "strike": round(K, 2),
                "option_type": option_type,
                "option_direction": option_direction,
                "option_size": option_size,
                "iv": round(iv * 100, 1),
            },
            "timestamps": [r["ts"] for r in results],
            "prices": [r["price"] for r in results],
            "portfolio_delta": [r["portfolio_delta"] for r in results],
            "gamma": [r["gamma_1pct"] for r in results],
            "total_btc": [r["total_btc"] for r in results],
            "option_value": [r["option_value_btc"] for r in results],
            "perp_pnl": [r["perp_pnl_btc"] for r in results],
        }


# ====== FLASK APP ======
app = Flask(__name__)
engine = SVIEngine()
auto_hedger = AutoHedger(engine)

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SVI Vol Surface — Deribit</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: #0d1117; color: #e6edf3; padding: 16px 20px;
  }
  .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
  .header h1 { font-size: 20px; color: #58a6ff; }
  .header select {
    font-family: inherit; font-size: 14px; padding: 6px 12px;
    background: #161b22; color: #e6edf3; border: 1px solid #30363d;
    border-radius: 6px; cursor: pointer; outline: none;
  }
  .status-bar { font-size: 11px; color: #8b949e; margin-left: 14px; }
  .status-bar .live { color: #3fb950; }

  .tabs { display: flex; gap: 0; margin-bottom: 16px; border-bottom: 1px solid #30363d; }
  .tab {
    padding: 8px 20px; font-size: 13px; cursor: pointer;
    color: #8b949e; border-bottom: 2px solid transparent;
    background: none; border-top: none; border-left: none; border-right: none; font-family: inherit;
  }
  .tab:hover { color: #e6edf3; }
  .tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .grid { display: grid; grid-template-columns: 1fr 360px; gap: 16px; }
  .chart-container {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 8px; min-height: 460px;
  }
  .sidebar { display: flex; flex-direction: column; gap: 12px; }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px;
  }
  .card h2 {
    font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
    color: #8b949e; margin-bottom: 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px;
  }
  .param-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 14px; }
  .param-row { display: flex; justify-content: space-between; padding: 2px 0; }
  .param-label { color: #8b949e; font-size: 12px; }
  .param-value { color: #e6edf3; font-size: 12px; font-weight: 600; }
  .param-value.highlight { color: #58a6ff; }
  .param-value.green { color: #3fb950; }
  .param-value.yellow { color: #d29922; }
  .param-value.red { color: #f85149; }
  .param-value.dim { color: #8b949e; font-weight: normal; font-size: 10px; }

  .strike-table { width: 100%; border-collapse: collapse; font-size: 11px; }
  .strike-table th { text-align: left; color: #8b949e; padding: 3px 5px; border-bottom: 1px solid #21262d; font-weight: normal; }
  .strike-table td { padding: 4px 5px; }
  .strike-table tr.atmf td { color: #58a6ff; font-weight: 600; }

  .futures-grid { display: grid; grid-template-columns: 1fr 320px; gap: 16px; }
  .futures-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .futures-table th { text-align: left; color: #8b949e; padding: 6px 8px; border-bottom: 1px solid #21262d; font-weight: normal; font-size: 11px; }
  .futures-table td { padding: 6px 8px; border-bottom: 1px solid #161b22; }

  .waiting { text-align: center; color: #8b949e; padding: 30px; font-size: 13px; }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid #30363d; border-top: 2px solid #58a6ff; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Risk tab */
  .risk-header { display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }
  .risk-stat { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 18px; min-width: 150px; }
  .risk-stat .label { font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .risk-stat .value { font-size: 18px; font-weight: 700; color: #e6edf3; margin-top: 4px; }
  .risk-stat .value.blue { color: #58a6ff; }

  .risk-table-wrap { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; overflow-x: auto; }
  .risk-table { width: 100%; border-collapse: collapse; font-size: 12px; white-space: nowrap; }
  .risk-table th {
    text-align: right; color: #8b949e; padding: 8px 10px;
    border-bottom: 1px solid #30363d; font-weight: normal; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .risk-table th:first-child { text-align: left; }
  .risk-table th:nth-child(2) { text-align: center; }
  .risk-table td { text-align: right; padding: 7px 10px; border-bottom: 1px solid #21262d; }
  .risk-table td:first-child { text-align: left; color: #e6edf3; font-weight: 600; }
  .risk-table td:nth-child(2) { text-align: center; }
  .risk-table tr:hover { background: #1c2128; }
  .risk-table tr.total-row { border-top: 2px solid #30363d; background: #1c2128; }
  .risk-table tr.total-row td { font-weight: 700; color: #e6edf3; padding-top: 10px; }
  .risk-table .pos { color: #3fb950; }
  .risk-table .neg { color: #f85149; }
  .risk-table .dir-buy { color: #3fb950; }
  .risk-table .dir-sell { color: #f85149; }
  .risk-table .muted { color: #484f58; }

  /* Bump table */
  .bump-wrap { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; margin-top: 16px; overflow-x: auto; }
  .bump-wrap h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #8b949e; margin-bottom: 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
  .bump-table { width: 100%; border-collapse: collapse; font-size: 12px; white-space: nowrap; }
  .bump-table th {
    text-align: right; color: #8b949e; padding: 6px 10px;
    border-bottom: 1px solid #30363d; font-weight: normal; font-size: 11px;
  }
  .bump-table th:first-child { text-align: left; }
  .bump-table th.center-col { color: #58a6ff; font-weight: 600; }
  .bump-table td { text-align: right; padding: 6px 10px; border-bottom: 1px solid #21262d; }
  .bump-table td:first-child { text-align: left; color: #8b949e; font-weight: 600; }
  .bump-table td.center-col { background: rgba(88,166,255,0.05); }
  .bump-table .pos { color: #3fb950; }
  .bump-table .neg { color: #f85149; }
  .bump-table .muted { color: #484f58; }

  .risk-countdown { font-size: 11px; color: #484f58; margin-top: 8px; text-align: right; }

  /* Hedge tab */
  .hedge-layout { display: grid; grid-template-columns: 300px 1fr; gap: 16px; }
  .hedge-controls { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .hedge-controls h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #8b949e; margin-bottom: 12px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
  .hedge-check { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; cursor: pointer; font-size: 13px; }
  .hedge-check input[type="checkbox"] { accent-color: #58a6ff; width: 16px; height: 16px; cursor: pointer; }
  .hedge-check .sub { color: #8b949e; font-size: 11px; }
  .hedge-select { width: 100%; margin-top: 12px; font-family: inherit; font-size: 12px; padding: 6px 10px; background: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; outline: none; }
  .hedge-btn {
    width: 100%; margin-top: 16px; padding: 10px; font-family: inherit; font-size: 13px; font-weight: 600;
    background: #238636; color: #fff; border: none; border-radius: 6px; cursor: pointer; letter-spacing: 0.5px;
  }
  .hedge-btn:hover { background: #2ea043; }
  .hedge-btn:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
  .hedge-results { display: flex; flex-direction: column; gap: 16px; }
  .hedge-summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .hedge-greek-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }
  .hedge-greek-box .label { font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .hedge-greek-box .before { font-size: 14px; color: #f85149; margin-top: 4px; }
  .hedge-greek-box .arrow { font-size: 12px; color: #484f58; margin: 2px 0; }
  .hedge-greek-box .after { font-size: 14px; font-weight: 700; margin-top: 2px; }
  .hedge-greek-box .after.flat { color: #3fb950; }
  .hedge-greek-box .after.residual { color: #d29922; }
  .hedge-trades-wrap { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
  .hedge-trades-wrap h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #8b949e; margin-bottom: 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
  .hedge-trade-card {
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; margin-bottom: 8px;
    display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center;
  }
  .hedge-trade-card .inst { font-size: 14px; font-weight: 600; color: #e6edf3; }
  .hedge-trade-card .dir-buy { color: #3fb950; font-weight: 700; font-size: 12px; }
  .hedge-trade-card .dir-sell { color: #f85149; font-weight: 700; font-size: 12px; }
  .hedge-trade-card .detail { font-size: 11px; color: #8b949e; margin-top: 4px; }
  .hedge-trade-card .impacts { font-size: 11px; display: flex; gap: 16px; margin-top: 6px; }
  .hedge-trade-card .impacts span { color: #8b949e; }
  .hedge-trade-card .impacts .val { color: #e6edf3; font-weight: 600; }
  .hedge-no-trades { text-align: center; color: #8b949e; padding: 30px; font-size: 13px; }
  .close-pos-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; }
  .close-pos-row:last-child { border-bottom: none; }
  .close-pos-info { display: flex; flex-direction: column; gap: 2px; }
  .close-pos-name { font-size: 11px; color: #e6edf3; font-weight: 600; }
  .close-pos-detail { font-size: 10px; color: #8b949e; }
  .close-pos-btn { font-family: inherit; font-size: 10px; padding: 3px 8px; background: #21262d; color: #f85149; border: 1px solid #f8514930; border-radius: 4px; cursor: pointer; }
  .close-pos-btn:hover { background: #f8514920; }
  .close-pos-btn:disabled { color: #484f58; border-color: #30363d; cursor: not-allowed; }

  .exec-btn {
    padding: 6px 14px; font-family: inherit; font-size: 11px; font-weight: 600;
    border: 1px solid #238636; background: transparent; color: #3fb950;
    border-radius: 4px; cursor: pointer; letter-spacing: 0.3px; white-space: nowrap;
  }
  .exec-btn:hover { background: #238636; color: #fff; }
  .exec-btn:disabled { border-color: #21262d; color: #484f58; background: transparent; cursor: not-allowed; }
  .exec-btn.executing { border-color: #d29922; color: #d29922; }
  .exec-btn.filled { border-color: #3fb950; background: #238636; color: #fff; }
  .exec-btn.failed { border-color: #f85149; color: #f85149; }

  .exec-all-btn {
    width: 100%; margin-top: 12px; padding: 10px; font-family: inherit; font-size: 13px; font-weight: 700;
    background: #238636; color: #fff; border: none; border-radius: 6px; cursor: pointer;
    letter-spacing: 0.5px;
  }
  .exec-all-btn:hover { background: #2ea043; }
  .exec-all-btn:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }

  .fill-info { font-size: 10px; margin-top: 6px; padding: 6px 8px; background: #0d1117; border-radius: 4px; }
  .fill-info.success { color: #3fb950; border: 1px solid #238636; }
  .fill-info.error { color: #f85149; border: 1px solid #f85149; }

  /* Confirmation modal */
  .modal-overlay {
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7); z-index: 100; justify-content: center; align-items: center;
  }
  .modal-overlay.active { display: flex; }
  .modal-box {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 24px; max-width: 460px; width: 90%;
  }
  .modal-box h3 { font-size: 16px; color: #f0883e; margin-bottom: 12px; }
  .modal-box .modal-detail { font-size: 13px; color: #e6edf3; margin-bottom: 16px; line-height: 1.6; }
  .modal-box .modal-warn { font-size: 11px; color: #d29922; margin-bottom: 16px; padding: 8px 10px; background: rgba(210,169,34,0.1); border-radius: 4px; }
  .modal-btns { display: flex; gap: 10px; justify-content: flex-end; }
  .modal-btns button { padding: 8px 20px; font-family: inherit; font-size: 12px; font-weight: 600; border-radius: 6px; cursor: pointer; }
  .modal-cancel { background: #21262d; color: #e6edf3; border: 1px solid #30363d; }
  .modal-cancel:hover { background: #30363d; }
  .modal-confirm { background: #da3633; color: #fff; border: none; }
  .modal-confirm:hover { background: #f85149; }

  /* Recon sub-tabs */
  .recon-sub {
    padding: 6px 14px; font-family: inherit; font-size: 12px; font-weight: 600;
    background: #0d1117; color: #8b949e; border: 1px solid #30363d; border-radius: 6px;
    cursor: pointer; display: flex; align-items: center; gap: 6px;
  }
  .recon-sub:hover { background: #161b22; color: #e6edf3; }
  .recon-sub.active { background: #21262d; color: #58a6ff; border-color: #58a6ff; }
  .recon-sub .badge {
    font-size: 10px; background: #30363d; color: #8b949e; padding: 1px 6px;
    border-radius: 10px; font-weight: 400;
  }
  .recon-sub.active .badge { background: #1f6feb33; color: #58a6ff; }
  .recon-panel { display: none; }
  .recon-panel.active { display: block; }
</style>
</head>
<body>

<div class="header">
  <div style="display:flex; align-items:center;">
    <h1>SVI Vol Surface</h1>
    <span class="status-bar" id="status">
      <span class="live" id="dot">&#9679;</span>
      <span id="status-text">Connecting...</span>
    </span>
  </div>
  <div>
    <label style="font-size:12px; color:#8b949e; margin-right:6px;">Expiry:</label>
    <select id="expiry-select"></select>
  </div>
</div>

<div class="tabs">
  <button class="tab active" data-tab="vol-tab">Vol Smile</button>
  <button class="tab" data-tab="futures-tab">Futures Curve</button>
  <button class="tab" data-tab="risk-tab">Risk</button>
  <button class="tab" data-tab="hedge-tab">Hedge</button>
  <button class="tab" data-tab="pnl-tab">P&amp;L</button>
  <button class="tab" data-tab="recon-tab">Recon</button>
  <button class="tab" data-tab="autohedge-tab">Auto Hedge</button>
  <button class="tab" data-tab="backtest-tab">Backtest</button>
</div>

<!-- Vol Smile Tab -->
<div id="vol-tab" class="tab-content active">
  <div class="grid">
    <div class="chart-container"><div id="chart" style="width:100%;height:100%;"></div></div>
    <div class="sidebar">
      <div class="card">
        <h2>Market</h2>
        <div class="param-row"><span class="param-label">Forward</span><span class="param-value highlight" id="v-forward">&mdash;</span></div>
        <div class="param-row"><span class="param-label">Source</span><span class="param-value dim" id="v-fwd-source">&mdash;</span></div>
        <div class="param-row"><span class="param-label">Spot (Index)</span><span class="param-value" id="v-spot">&mdash;</span></div>
        <div class="param-row"><span class="param-label">Time to Expiry</span><span class="param-value" id="v-T">&mdash;</span></div>
        <div class="param-row"><span class="param-label">Points Used</span><span class="param-value" id="v-npoints">&mdash;</span></div>
        <div class="param-row"><span class="param-label">Last Update (UTC)</span><span class="param-value" id="v-ts">&mdash;</span></div>
      </div>
      <div class="card">
        <h2>SVI Parameters</h2>
        <div class="param-grid">
          <div class="param-row"><span class="param-label">a</span><span class="param-value" id="v-a">&mdash;</span></div>
          <div class="param-row"><span class="param-label">b</span><span class="param-value" id="v-b">&mdash;</span></div>
          <div class="param-row"><span class="param-label">&rho; (rho)</span><span class="param-value" id="v-rho">&mdash;</span></div>
          <div class="param-row"><span class="param-label">m</span><span class="param-value" id="v-m">&mdash;</span></div>
          <div class="param-row"><span class="param-label">&sigma; (sigma)</span><span class="param-value" id="v-sigma">&mdash;</span></div>
          <div class="param-row"><span class="param-label">RMSE</span><span class="param-value" id="v-rmse">&mdash;</span></div>
        </div>
      </div>
      <div class="card">
        <h2>ATMF Pricing</h2>
        <table class="strike-table">
          <thead><tr><th>Strike</th><th>Mkt IV</th><th>SVI IV</th></tr></thead>
          <tbody id="strike-body"><tr><td colspan="3" style="color:#8b949e">Waiting...</td></tr></tbody>
        </table>
        <div style="margin-top:10px;border-top:1px solid #21262d;padding-top:8px;">
          <div class="param-row"><span class="param-label">ATMF Call</span><span class="param-value green" id="v-call">&mdash;</span></div>
          <div class="param-row"><span class="param-label">ATMF Put</span><span class="param-value red" id="v-put">&mdash;</span></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Futures Curve Tab -->
<div id="futures-tab" class="tab-content">
  <div class="futures-grid">
    <div><div class="chart-container"><div id="futures-chart" style="width:100%;height:100%;"></div></div></div>
    <div class="sidebar">
      <div class="card">
        <h2>Spot &amp; Futures</h2>
        <div class="param-row"><span class="param-label">BTC Spot (Index)</span><span class="param-value highlight" id="vf-spot">&mdash;</span></div>
        <div style="margin-top:10px;">
          <table class="futures-table">
            <thead><tr><th>Contract</th><th>T (yr)</th><th>Mark ($)</th><th>Rate (%)</th></tr></thead>
            <tbody id="futures-body"><tr><td colspan="4" style="color:#8b949e">Waiting...</td></tr></tbody>
          </table>
        </div>
      </div>
      <div class="card"><h2>Rate Model</h2><div id="rate-chart" style="width:100%;height:200px;"></div></div>
    </div>
  </div>
</div>

<!-- Risk Tab -->
<div id="risk-tab" class="tab-content">
  <div id="risk-loading" class="waiting"><div class="spinner"></div><br>Loading portfolio data...</div>
  <div id="risk-content" style="display:none;">
    <div class="risk-header">
      <div class="risk-stat"><div class="label">Equity</div><div class="value blue" id="r-equity">&mdash;</div></div>
      <div class="risk-stat"><div class="label">Balance</div><div class="value" id="r-balance">&mdash;</div></div>
      <div class="risk-stat"><div class="label">Available</div><div class="value" id="r-available">&mdash;</div></div>
      <div class="risk-stat"><div class="label">Init Margin</div><div class="value" id="r-margin">&mdash;</div></div>
      <div class="risk-stat"><div class="label">BTC Spot</div><div class="value blue" id="r-spot">&mdash;</div></div>
      <div class="risk-stat"><div class="label">Last Update</div><div class="value" id="r-ts">&mdash;</div></div>
    </div>

    <div class="risk-table-wrap">
      <table class="risk-table">
        <thead><tr>
          <th style="text-align:left">Instrument</th>
          <th style="text-align:center">Dir</th>
          <th>Size</th><th>Mark</th><th>IV (%)</th><th>PnL (BTC)</th>
          <th>BS Delta</th><th>Smile Delta</th><th>Gamma (1%)</th><th>Vega ($)</th><th>Theta ($)</th>
        </tr></thead>
        <tbody id="risk-body"></tbody>
      </table>
    </div>

    <div class="bump-wrap">
      <h2>Scenario Analysis — Portfolio Greeks Across Spot Bumps</h2>
      <table class="bump-table">
        <thead id="bump-head"></thead>
        <tbody id="bump-body"></tbody>
      </table>
    </div>

    <div class="risk-countdown" id="risk-countdown"></div>
  </div>
</div>

<!-- P&L Tab -->
<div id="pnl-tab" class="tab-content">
  <div id="pnl-loading" class="waiting"><div class="spinner"></div><br>Collecting P&amp;L data... (updates every 15s with risk refresh)</div>
  <div id="pnl-content" style="display:none;">
    <div class="risk-header">
      <div class="risk-stat"><div class="label">Current Equity</div><div class="value blue" id="pnl-equity">—</div></div>
      <div class="risk-stat"><div class="label">Equity (USD)</div><div class="value green" id="pnl-equity-usd">—</div></div>
      <div class="risk-stat"><div class="label">Unrealised P&amp;L</div><div class="value" id="pnl-unrealised">—</div></div>
      <div class="risk-stat"><div class="label">Data Points</div><div class="value" id="pnl-count">—</div></div>
      <div class="risk-stat"><div class="label">Since</div><div class="value" id="pnl-since">—</div></div>
      <div class="risk-stat">
        <div class="label">View</div>
        <select id="pnl-range" style="font-family:inherit;font-size:12px;padding:4px 8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:4px;margin-top:4px;outline:none;">
          <option value="0">All</option>
          <option value="60">Last 1h</option>
          <option value="360">Last 6h</option>
          <option value="1440">Last 24h</option>
        </select>
      </div>
    </div>
    <div class="chart-container" style="min-height:360px;"><div id="pnl-chart-equity" style="width:100%;height:360px;"></div></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;">
      <div class="chart-container" style="min-height:280px;"><div id="pnl-chart-unrealised" style="width:100%;height:280px;"></div></div>
      <div class="chart-container" style="min-height:280px;position:relative;">
        <div style="position:absolute;top:8px;right:12px;z-index:10;">
          <select id="pnl-price-toggle" style="font-family:inherit;font-size:10px;padding:3px 6px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:4px;outline:none;cursor:pointer;">
            <option value="spot">BTC Spot (Index)</option>
            <option value="perp">BTC-PERPETUAL</option>
          </select>
        </div>
        <div id="pnl-chart-spot" style="width:100%;height:280px;"></div>
      </div>
    </div>
    <div style="margin-top:12px;text-align:right;">
      <span style="font-size:11px;color:#484f58;">History saved to pnl_history.csv · Persists across restarts</span>
    </div>
  </div>
</div>

<!-- Recon Tab -->
<div id="recon-tab" class="tab-content">
  <div id="recon-loading" class="waiting"><div class="spinner"></div><br>Click <strong>Load Reconciliation</strong> to fetch all transaction history</div>
  <div id="recon-content" style="display:none;">
    <!-- Summary cards -->
    <div class="risk-header" id="recon-summary"></div>

    <!-- Sub-tabs -->
    <div style="display:flex;gap:8px;margin:16px 0 12px;">
      <button class="recon-sub active" data-rsub="recon-trades-panel">Trades <span id="recon-trade-count" class="badge">0</span></button>
      <button class="recon-sub" data-rsub="recon-settlements-panel">Settlements &amp; Funding <span id="recon-settlement-count" class="badge">0</span></button>
      <button class="recon-sub" data-rsub="recon-transfers-panel">Transfers <span id="recon-transfer-count" class="badge">0</span></button>
      <button class="recon-sub" data-rsub="recon-reconcile-panel">Reconciliation</button>
    </div>

    <!-- Trades panel -->
    <div id="recon-trades-panel" class="recon-panel active">
      <div style="max-height:500px;overflow-y:auto;">
        <table class="risk-table" style="width:100%;">
          <thead><tr>
            <th>Time (UTC)</th><th>Instrument</th><th>Dir</th><th>Amount</th><th>Price</th><th>Premium (BTC)</th><th>Fee (BTC)</th><th>Realised P&amp;L (BTC)</th>
          </tr></thead>
          <tbody id="recon-trades-body"></tbody>
        </table>
      </div>
    </div>

    <!-- Settlements panel -->
    <div id="recon-settlements-panel" class="recon-panel">
      <div style="max-height:500px;overflow-y:auto;">
        <table class="risk-table" style="width:100%;">
          <thead><tr>
            <th>Time (UTC)</th><th>Instrument</th><th>Type</th><th>Funding (BTC)</th><th>Session P&amp;L (BTC)</th><th>Position</th><th>Mark Price</th><th>Index Price</th>
          </tr></thead>
          <tbody id="recon-settlements-body"></tbody>
        </table>
      </div>
    </div>

    <!-- Transfers panel -->
    <div id="recon-transfers-panel" class="recon-panel">
      <div style="max-height:500px;overflow-y:auto;">
        <table class="risk-table" style="width:100%;">
          <thead><tr>
            <th>Time (UTC)</th><th>Type</th><th>Amount (BTC)</th><th>State</th><th>Tx ID</th>
          </tr></thead>
          <tbody id="recon-transfers-body"></tbody>
        </table>
      </div>
      <div id="recon-no-transfers" style="padding:20px;text-align:center;color:#8b949e;display:none;">No transfers found (testnet accounts are pre-funded)</div>
    </div>

    <!-- Reconciliation panel -->
    <div id="recon-reconcile-panel" class="recon-panel">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;max-width:900px;">
        <div class="card">
          <h2>Calculated Breakdown</h2>
          <div class="param-row"><span class="param-label">Deposits</span><span class="param-value green" id="rc-deposits">—</span></div>
          <div class="param-row"><span class="param-label">Withdrawals</span><span class="param-value red" id="rc-withdrawals">—</span></div>
          <div class="param-row"><span class="param-label">Realised P&amp;L (trades)</span><span class="param-value" id="rc-realised">—</span></div>
          <div class="param-row"><span class="param-label">Funding Payments</span><span class="param-value" id="rc-funding">—</span></div>
          <div class="param-row"><span class="param-label">Total Fees</span><span class="param-value red" id="rc-fees">—</span></div>
          <div class="param-row"><span class="param-label">Premium Received</span><span class="param-value green" id="rc-prem-recv">—</span></div>
          <div class="param-row"><span class="param-label">Premium Paid</span><span class="param-value red" id="rc-prem-paid">—</span></div>
          <div class="param-row"><span class="param-label">Net Premium</span><span class="param-value" id="rc-prem-net">—</span></div>
          <div style="border-top:1px solid #21262d;margin:8px 0;"></div>
          <div class="param-row"><span class="param-label">Calculated Balance</span><span class="param-value blue" id="rc-calc-balance">—</span></div>
          <div class="param-row"><span class="param-label">Unrealised P&amp;L</span><span class="param-value" id="rc-unrealised">—</span></div>
          <div class="param-row"><span class="param-label"><strong>Calculated Equity</strong></span><span class="param-value blue" id="rc-calc-equity">—</span></div>
        </div>
        <div class="card">
          <h2>API (Deribit) Values</h2>
          <div class="param-row"><span class="param-label">API Balance</span><span class="param-value blue" id="rc-api-balance">—</span></div>
          <div class="param-row"><span class="param-label">API Equity</span><span class="param-value blue" id="rc-api-equity">—</span></div>
          <div style="border-top:1px solid #21262d;margin:8px 0;"></div>
          <h2 style="margin-top:16px;">Difference</h2>
          <div class="param-row"><span class="param-label">Balance Diff</span><span class="param-value" id="rc-diff-balance">—</span></div>
          <div class="param-row"><span class="param-label">Equity Diff</span><span class="param-value" id="rc-diff-equity">—</span></div>
          <div style="margin-top:16px;font-size:11px;color:#8b949e;">
            Diff = API value &minus; Calculated value<br>
            A small diff may arise from rounding, mark-to-market timing, or session PnL not captured in trade history.
          </div>
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:16px;text-align:center;">
    <button class="hedge-btn" id="recon-load-btn" onclick="loadRecon()" style="width:auto;padding:10px 32px;">Load Reconciliation</button>
    <div style="margin-top:8px;font-size:11px;color:#484f58;" id="recon-status"></div>
  </div>
</div>

<!-- Confirmation Modal -->
<div class="modal-overlay" id="exec-modal">
  <div class="modal-box">
    <h3>Confirm Trade Execution</h3>
    <div class="modal-detail" id="modal-detail"></div>
    <div class="modal-warn">This will place a MARKET order on Deribit testnet. It will execute immediately at best available price.</div>
    <div class="modal-btns">
      <button class="modal-cancel" onclick="closeModal()">Cancel</button>
      <button class="modal-confirm" id="modal-confirm-btn">Execute</button>
    </div>
  </div>
</div>

<!-- Hedge Tab -->
<div id="hedge-tab" class="tab-content">
  <div class="hedge-layout">
    <div>
      <div class="hedge-controls">
        <h2>Hedge Target</h2>
        <label class="hedge-check"><input type="checkbox" id="h-delta" checked><span>Delta</span><span class="sub">— via BTC-PERPETUAL</span></label>
        <label class="hedge-check"><input type="checkbox" id="h-gamma"><span>Gamma (1%)</span><span class="sub">— via options</span></label>
        <label class="hedge-check"><input type="checkbox" id="h-vega"><span>Vega ($)</span><span class="sub">— via options</span></label>
        <div style="margin-top:16px;">
          <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">HEDGE WITH EXPIRY</div>
          <select id="h-expiry" class="hedge-select">
            <option value="">Auto (same as positions)</option>
          </select>
        </div>
        <button class="hedge-btn" id="h-calc" onclick="computeHedge()">Calculate Hedge</button>
        <div style="margin-top:12px;font-size:11px;color:#484f58;" id="h-status"></div>
      </div>
      <div class="hedge-controls" style="margin-top:16px;">
        <h2>Close Positions</h2>
        <div id="close-positions-list" style="font-size:12px;color:#8b949e;">Loading...</div>
        <button class="hedge-btn" style="background:#f85149;margin-top:12px;" id="close-all-btn" onclick="closeAllPositions()">Close All Positions</button>
        <div style="margin-top:8px;font-size:11px;color:#484f58;" id="close-status"></div>
      </div>
    </div>
    <div class="hedge-results">
      <div class="hedge-summary" id="h-summary" style="display:none;"></div>
      <div class="hedge-trades-wrap" id="h-trades-wrap" style="display:none;">
        <h2>Suggested Trades</h2>
        <div id="h-trades"></div>
      </div>
      <div id="h-empty" class="hedge-no-trades">
        Select which greeks to flatten and click <strong>Calculate Hedge</strong>
      </div>
    </div>
  </div>
</div>

<!-- Auto Hedge Tab -->
<div id="autohedge-tab" class="tab-content">
  <div class="hedge-layout">
    <div>
      <div class="hedge-controls">
        <h2>Auto Delta Hedger</h2>
        <p style="font-size:11px;color:#8b949e;margin-bottom:16px;line-height:1.5;">
          Continuously monitors portfolio delta and hedges via BTC-PERPETUAL.
          When gamma is large, the hedger tightens its delta threshold automatically.
        </p>

        <div style="margin-bottom:12px;">
          <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">DELTA THRESHOLD (BTC)</div>
          <input type="number" id="ah-threshold" value="0.02" step="0.005" min="0.001" max="1"
            style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:13px;">
          <div style="font-size:10px;color:#484f58;margin-top:2px;">Hedge when |delta| exceeds this. Gamma scaling adjusts dynamically.</div>
        </div>

        <div style="margin-bottom:12px;">
          <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">CHECK INTERVAL (seconds)</div>
          <input type="number" id="ah-interval" value="30" step="5" min="5" max="600"
            style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:13px;">
        </div>

        <div style="margin-bottom:12px;">
          <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">MAX TRADE SIZE (USD)</div>
          <input type="number" id="ah-max-trade" value="100000" step="1000" min="10"
            style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:13px;">
        </div>

        <label class="hedge-check" style="margin-bottom:16px;">
          <input type="checkbox" id="ah-gamma-scaling" checked>
          <span>Gamma Scaling</span>
          <span class="sub">— tighten threshold when gamma is large</span>
        </label>

        <div style="display:flex;gap:8px;">
          <button class="hedge-btn" id="ah-start-btn" onclick="toggleAutoHedge()" style="flex:1;background:#238636;">Start Auto Hedge</button>
          <button class="hedge-btn" id="ah-update-btn" onclick="updateAutoHedgeParams()" style="flex:1;background:#30363d;">Update Params</button>
        </div>

        <div style="margin-top:12px;font-size:11px;color:#484f58;" id="ah-status-text"></div>
      </div>
    </div>

    <div style="flex:1;">
      <div class="card" style="margin-bottom:12px;">
        <h2>Status</h2>
        <div class="risk-header" id="ah-stats" style="margin-bottom:0;">
          <div class="risk-stat"><div class="label">State</div><div class="value" id="ah-state">STOPPED</div></div>
          <div class="risk-stat"><div class="label">Hedges</div><div class="value" id="ah-hedges">0</div></div>
          <div class="risk-stat"><div class="label">Total Traded</div><div class="value" id="ah-traded">$0</div></div>
          <div class="risk-stat"><div class="label">Last Hedge</div><div class="value" id="ah-last">&mdash;</div></div>
        </div>
      </div>

      <div class="card">
        <h2>Activity Log</h2>
        <div id="ah-log" style="max-height:400px;overflow-y:auto;font-size:11px;font-family:inherit;">
          <div style="color:#8b949e;padding:20px;text-align:center;">Start auto hedger to see activity</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Backtest Tab -->
<div id="backtest-tab" class="tab-content">
  <div class="hedge-layout">
    <div>
      <div class="hedge-controls">
        <h2>Backtest: Delta Hedging</h2>
        <p style="font-size:11px;color:#8b949e;margin-bottom:16px;line-height:1.5;">
          Simulate selling (or buying) an option and delta-hedging with the perpetual
          over historical data. See how hedging frequency and gamma affect P&amp;L.
        </p>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">OPTION TYPE</div>
            <select id="bt-option-type" style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
              <option value="call">Call</option>
              <option value="put">Put</option>
            </select>
          </div>
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">DIRECTION</div>
            <select id="bt-direction" style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
              <option value="sell">Sell (collect premium)</option>
              <option value="buy">Buy (pay premium)</option>
            </select>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">STRIKE OFFSET (%)</div>
            <input type="number" id="bt-strike-offset" value="0" step="1" min="-50" max="50"
              style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
            <div style="font-size:10px;color:#484f58;margin-top:2px;">0 = ATM, +5 = 5% OTM call</div>
          </div>
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">OPTION SIZE (BTC)</div>
            <input type="number" id="bt-size" value="1" step="0.1" min="0.1" max="100"
              style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">IMPLIED VOL (%)</div>
            <input type="number" id="bt-iv" value="60" step="5" min="5" max="300"
              style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
          </div>
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">TIME TO EXPIRY (days)</div>
            <input type="number" id="bt-tte" value="30" step="1" min="1" max="365"
              style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">LOOKBACK (days)</div>
            <input type="number" id="bt-days" value="30" step="1" min="1" max="365"
              style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
          </div>
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">CANDLE SIZE</div>
            <select id="bt-resolution" style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
              <option value="60">1 Hour</option>
              <option value="30">30 Min</option>
              <option value="15">15 Min</option>
              <option value="1D">Daily</option>
            </select>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">HEDGE EVERY N CANDLES</div>
            <input type="number" id="bt-hedge-interval" value="1" step="1" min="1" max="100"
              style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
          </div>
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">COST PER TRADE (%)</div>
            <input type="number" id="bt-cost" value="0.05" step="0.01" min="0" max="1"
              style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
          </div>
        </div>

        <div style="margin-bottom:12px;">
          <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">DELTA THRESHOLD</div>
          <input type="number" id="bt-delta-thresh" value="0.001" step="0.001" min="0" max="1"
            style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;font-family:inherit;font-size:12px;">
          <div style="font-size:10px;color:#484f58;margin-top:2px;">Only rehedge when |portfolio delta| exceeds this</div>
        </div>

        <button class="hedge-btn" id="bt-run-btn" onclick="runBacktest()" style="background:#238636;">Run Backtest</button>
        <div style="margin-top:8px;font-size:11px;color:#484f58;" id="bt-status"></div>
      </div>
    </div>

    <div style="flex:1;">
      <div id="bt-empty" style="text-align:center;color:#8b949e;padding:60px 20px;font-size:13px;">
        Configure parameters and click <strong>Run Backtest</strong> to simulate a delta-hedging strategy over historical data.
      </div>
      <div id="bt-results" style="display:none;">
        <div class="risk-header" id="bt-metrics" style="margin-bottom:16px;"></div>
        <div class="chart-container" style="margin-bottom:12px;"><div id="bt-chart-pnl" style="width:100%;height:300px;"></div></div>
        <div class="chart-container" style="margin-bottom:12px;"><div id="bt-chart-delta" style="width:100%;height:250px;"></div></div>
        <div class="chart-container"><div id="bt-chart-price" style="width:100%;height:250px;"></div></div>
      </div>
    </div>
  </div>
</div>

<script>
// ---- Tabs ----
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
    window.dispatchEvent(new Event('resize'));
  });
});

// ---- Charts ----
const chartDiv = document.getElementById('chart');
Plotly.newPlot(chartDiv, [
  { x: [], y: [], mode: 'markers', name: 'Market OTM IV', marker: { color: '#58a6ff', size: 6 } },
  { x: [], y: [], mode: 'lines', name: 'SVI Fit', line: { color: '#f0883e', width: 2.5 } },
  { x: [], y: [], mode: 'markers', name: 'ATMF', marker: { color: '#3fb950', size: 11, symbol: 'diamond' } },
], {
  paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
  font: { family: 'SF Mono, Fira Code, Consolas, monospace', color: '#8b949e', size: 10 },
  xaxis: { title: 'Strike ($)', gridcolor: '#21262d', zerolinecolor: '#30363d', tickformat: '$,.0f' },
  yaxis: { title: 'Implied Vol (%)', gridcolor: '#21262d', zerolinecolor: '#30363d', ticksuffix: '%' },
  legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)' },
  margin: { l: 55, r: 15, t: 15, b: 45 },
}, { responsive: true, displayModeBar: false });

const futChartDiv = document.getElementById('futures-chart');
Plotly.newPlot(futChartDiv, [
  { x: [], y: [], mode: 'markers', name: 'Traded Futures', marker: { color: '#58a6ff', size: 8 } },
  { x: [], y: [], mode: 'lines', name: 'Rate Model', line: { color: '#f0883e', width: 2 } },
  { x: [0], y: [0], mode: 'markers', name: 'Spot', marker: { color: '#3fb950', size: 10, symbol: 'diamond' } },
], {
  paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
  font: { family: 'SF Mono, Fira Code, Consolas, monospace', color: '#8b949e', size: 10 },
  xaxis: { title: 'Time to Expiry (years)', gridcolor: '#21262d', zerolinecolor: '#30363d' },
  yaxis: { title: 'Price ($)', gridcolor: '#21262d', zerolinecolor: '#30363d', tickformat: '$,.0f' },
  legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)' },
  margin: { l: 65, r: 15, t: 15, b: 45 },
}, { responsive: true, displayModeBar: false });

const rateChartDiv = document.getElementById('rate-chart');
Plotly.newPlot(rateChartDiv, [
  { x: [], y: [], mode: 'lines', name: 'Short Rate', line: { color: '#d2a8ff', width: 2 } },
  { x: [], y: [], mode: 'markers', name: 'Futures-implied', marker: { color: '#58a6ff', size: 6 } },
], {
  paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
  font: { family: 'SF Mono, Fira Code, Consolas, monospace', color: '#8b949e', size: 9 },
  xaxis: { title: 'T (yr)', gridcolor: '#21262d' },
  yaxis: { title: 'Rate (%)', gridcolor: '#21262d', ticksuffix: '%' },
  legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)', font: { size: 9 } },
  margin: { l: 45, r: 10, t: 5, b: 35 },
}, { responsive: true, displayModeBar: false });

// ---- Load expiries ----
const expirySelect = document.getElementById('expiry-select');
fetch('/api/expiries').then(r => r.json()).then(data => {
  data.expiries.forEach(exp => {
    const opt = document.createElement('option');
    opt.value = exp; opt.textContent = exp;
    expirySelect.appendChild(opt);
  });
  if (data.expiries.length > 0) expirySelect.value = data.current || data.expiries[0];
});
expirySelect.addEventListener('change', () => {
  document.getElementById('status-text').textContent = 'Switching...';
  fetch('/api/set_expiry?expiry=' + expirySelect.value).then(() => {
    document.getElementById('status-text').textContent = 'Waiting for data...';
  });
});

function fmt(v, dec) { return v !== null && v !== undefined ? v.toFixed(dec) : '\u2014'; }
function signedFmt(v, dec) {
  if (v === null || v === undefined) return '\u2014';
  const s = v.toFixed(dec);
  return v > 0 ? '+' + s : s;
}
function colorClass(v) {
  if (v > 1e-7) return 'pos';
  if (v < -1e-7) return 'neg';
  return 'muted';
}

// ---- Poll vol ----
function updateVol() {
  fetch('/api/data').then(r => r.json()).then(d => {
    if (!d.ts) { document.getElementById('status-text').textContent = 'Waiting for data...'; return; }
    document.getElementById('status-text').textContent = 'Live \u2014 ' + d.ts + ' UTC';
    document.getElementById('v-forward').textContent = '$' + (d.forward||0).toLocaleString();
    document.getElementById('v-fwd-source').textContent = d.forward_source || '\u2014';
    document.getElementById('v-T').textContent = fmt(d.T, 6) + 'y';
    document.getElementById('v-npoints').textContent = d.n_points;
    document.getElementById('v-ts').textContent = d.ts;
    if (d.params) {
      document.getElementById('v-a').textContent = fmt(d.params.a, 6);
      document.getElementById('v-b').textContent = fmt(d.params.b, 6);
      document.getElementById('v-rho').textContent = fmt(d.params.rho, 6);
      document.getElementById('v-m').textContent = fmt(d.params.m, 6);
      document.getElementById('v-sigma').textContent = fmt(d.params.sigma, 6);
    }
    const rmseEl = document.getElementById('v-rmse');
    rmseEl.textContent = fmt(d.rmse, 4) + '%';
    rmseEl.className = 'param-value ' + (d.rmse < 0.25 ? 'green' : d.rmse < 0.5 ? 'yellow' : 'red');
    document.getElementById('v-call').textContent = d.atmf_call !== null ? d.atmf_call.toFixed(6) + ' BTC' : '\u2014';
    document.getElementById('v-put').textContent = d.atmf_put !== null ? d.atmf_put.toFixed(6) + ' BTC' : '\u2014';
    let rows = '';
    if (d.strike_below) rows += '<tr><td>$' + d.strike_below.toLocaleString() + '</td><td>' + fmt(d.iv_below_mkt,2) + '%</td><td>' + fmt(d.iv_below_svi,2) + '%</td></tr>';
    rows += '<tr class="atmf"><td>$' + (d.forward||0).toLocaleString() + ' (F)</td><td>\u2014</td><td>' + fmt(d.atmf_iv,2) + '%</td></tr>';
    if (d.strike_above) rows += '<tr><td>$' + d.strike_above.toLocaleString() + '</td><td>' + fmt(d.iv_above_mkt,2) + '%</td><td>' + fmt(d.iv_above_svi,2) + '%</td></tr>';
    document.getElementById('strike-body').innerHTML = rows;
    Plotly.update(chartDiv, {
      x: [d.market_strikes, d.svi_curve_strikes, d.forward ? [d.forward] : []],
      y: [d.market_ivs, d.svi_curve_iv, d.atmf_iv !== null ? [d.atmf_iv] : []],
    }, {}, [0, 1, 2]);
  }).catch(() => {});
}

// ---- Poll futures ----
function updateFutures() {
  fetch('/api/futures').then(r => r.json()).then(d => {
    if (!d || !d.spot) return;
    document.getElementById('vf-spot').textContent = '$' + d.spot.toLocaleString();
    document.getElementById('v-spot').textContent = '$' + d.spot.toLocaleString();
    if (d.table && d.table.length > 0) {
      let rows = '';
      d.table.forEach(f => {
        rows += '<tr><td>' + f.name + '</td><td>' + f.T.toFixed(4) + '</td><td>$' + f.mark.toLocaleString() + '</td><td>' + f.rate.toFixed(3) + '%</td></tr>';
      });
      document.getElementById('futures-body').innerHTML = rows;
    }
    Plotly.update(futChartDiv, { x: [d.futures_T, d.curve_T, [0]], y: [d.futures_F, d.curve_F, [d.spot]] }, {}, [0, 1, 2]);
    if (d.table && d.table.length > 0) {
      Plotly.update(rateChartDiv, {
        x: [d.curve_T, d.table.map(f => f.T)], y: [d.curve_r, d.table.map(f => f.rate)]
      }, {}, [0, 1]);
    }
  }).catch(() => {});
}

// ---- Poll risk ----
let riskLastUpdate = 0;

function updateRisk() {
  fetch('/api/risk').then(r => r.json()).then(d => {
    if (d.error) {
      document.getElementById('risk-loading').innerHTML = '<div style="color:#f85149">Error: ' + d.error + '</div>';
      document.getElementById('risk-loading').style.display = 'block';
      document.getElementById('risk-content').style.display = 'none';
      return;
    }
    if (!d.ts) return;
    document.getElementById('risk-loading').style.display = 'none';
    document.getElementById('risk-content').style.display = 'block';

    const a = d.account || {};
    document.getElementById('r-equity').textContent = (a.equity||0).toFixed(4) + ' BTC';
    document.getElementById('r-balance').textContent = (a.balance||0).toFixed(4) + ' BTC';
    document.getElementById('r-available').textContent = (a.available_funds||0).toFixed(4) + ' BTC';
    document.getElementById('r-margin').textContent = (a.initial_margin||0).toFixed(4) + ' BTC';
    document.getElementById('r-spot').textContent = '$' + (d.spot||0).toLocaleString(undefined, {maximumFractionDigits:0});
    document.getElementById('r-ts').textContent = d.ts + ' UTC';
    riskLastUpdate = Date.now();

    // Positions table
    let rows = '';
    (d.positions || []).forEach(p => {
      const dc = p.direction === 'buy' ? 'dir-buy' : 'dir-sell';
      const dl = p.direction === 'buy' ? 'LONG' : 'SHORT';
      const mk = p.kind === 'future' ? '$' + (p.mark_price||0).toLocaleString(undefined,{maximumFractionDigits:0}) : (p.mark_price||0).toFixed(4) + ' BTC';
      const sz = p.kind === 'future' ? '$' + (p.size||0).toLocaleString() : (p.size||0).toFixed(1);
      const iv = p.iv !== undefined && p.iv > 0 ? p.iv.toFixed(1) + '%' : '\u2014';
      rows += '<tr>' +
        '<td>' + p.instrument + '</td>' +
        '<td class="' + dc + '">' + dl + '</td>' +
        '<td>' + sz + '</td><td>' + mk + '</td><td>' + iv + '</td>' +
        '<td class="' + colorClass(p.pnl) + '">' + signedFmt(p.pnl, 6) + '</td>' +
        '<td class="' + colorClass(p.bs_delta) + '">' + signedFmt(p.bs_delta, 4) + '</td>' +
        '<td class="' + colorClass(p.smile_delta) + '">' + signedFmt(p.smile_delta, 4) + '</td>' +
        '<td class="' + colorClass(p.gamma_1pct) + '">' + signedFmt(p.gamma_1pct, 4) + '</td>' +
        '<td class="' + colorClass(p.vega_usd) + '">' + signedFmt(p.vega_usd, 2) + '</td>' +
        '<td class="' + colorClass(p.theta_usd) + '">' + signedFmt(p.theta_usd, 2) + '</td></tr>';
    });
    const t = d.totals || {};
    rows += '<tr class="total-row"><td>TOTAL</td><td></td><td></td><td></td><td></td><td></td>' +
      '<td class="' + colorClass(t.bs_delta) + '">' + signedFmt(t.bs_delta, 4) + '</td>' +
      '<td class="' + colorClass(t.smile_delta) + '">' + signedFmt(t.smile_delta, 4) + '</td>' +
      '<td class="' + colorClass(t.gamma_1pct) + '">' + signedFmt(t.gamma_1pct, 4) + '</td>' +
      '<td class="' + colorClass(t.vega_usd) + '">' + signedFmt(t.vega_usd, 2) + '</td>' +
      '<td class="' + colorClass(t.theta_usd) + '">' + signedFmt(t.theta_usd, 2) + '</td></tr>';
    document.getElementById('risk-body').innerHTML = rows;

    // Bump / Scenario table
    const b = d.bumps;
    if (b && b.bumps) {
      const centerIdx = b.bumps.indexOf(0);
      // Header row
      let hdr = '<tr><th style="text-align:left"></th>';
      b.bumps.forEach((bp, i) => {
        const cls = i === centerIdx ? ' class="center-col"' : '';
        const label = bp === 0 ? 'Spot' : (bp > 0 ? '+' + bp + '%' : bp + '%');
        hdr += '<th' + cls + '>' + label + '</th>';
      });
      hdr += '</tr>';
      document.getElementById('bump-head').innerHTML = hdr;

      // Data rows
      const metrics = [
        { key: 'spots', label: 'Spot ($)', fmt: v => '$' + v.toLocaleString(undefined,{maximumFractionDigits:0}), noColor: true },
        { key: 'pnl', label: 'PnL (BTC)', fmt: v => signedFmt(v, 6) },
        { key: 'bs_delta', label: 'BS Delta', fmt: v => signedFmt(v, 4) },
        { key: 'smile_delta', label: 'Smile Delta', fmt: v => signedFmt(v, 4) },
        { key: 'gamma_1pct', label: 'Gamma (1%)', fmt: v => signedFmt(v, 4) },
        { key: 'vega_usd', label: 'Vega ($)', fmt: v => signedFmt(v, 2) },
        { key: 'theta_usd', label: 'Theta ($)', fmt: v => signedFmt(v, 2) },
      ];
      let body = '';
      metrics.forEach(m => {
        body += '<tr><td>' + m.label + '</td>';
        b[m.key].forEach((v, i) => {
          const cls = i === centerIdx ? ' center-col' : '';
          const cc = m.noColor ? '' : ' ' + colorClass(v);
          body += '<td class="' + cls + cc + '">' + m.fmt(v) + '</td>';
        });
        body += '</tr>';
      });
      document.getElementById('bump-body').innerHTML = body;
    }
  }).catch(() => {});
}

setInterval(() => {
  if (riskLastUpdate > 0) {
    const rem = Math.max(0, 15 - Math.floor((Date.now() - riskLastUpdate) / 1000));
    document.getElementById('risk-countdown').textContent = 'Refreshes in ' + rem + 's  (every 15s via REST)';
  }
}, 1000);

// ---- P&L Charts ----
const pnlChartEquity = document.getElementById('pnl-chart-equity');
const pnlChartUnrealised = document.getElementById('pnl-chart-unrealised');
const pnlChartSpot = document.getElementById('pnl-chart-spot');
const pnlDarkLayout = {
  paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
  font: { family: 'SF Mono, Fira Code, Consolas, monospace', color: '#8b949e', size: 10 },
  margin: { l: 60, r: 15, t: 30, b: 40 },
  legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)', font: { size: 9 } },
};
Plotly.newPlot(pnlChartEquity, [
  { x: [], y: [], mode: 'lines', name: 'Equity (BTC)', line: { color: '#58a6ff', width: 2 } },
  { x: [], y: [], mode: 'lines', name: 'Balance (BTC)', line: { color: '#8b949e', width: 1, dash: 'dot' } },
], { ...pnlDarkLayout, title: { text: 'Portfolio Equity (BTC)', font: { size: 12, color: '#58a6ff' } },
     xaxis: { gridcolor: '#21262d' }, yaxis: { title: 'BTC', gridcolor: '#21262d' } },
   { responsive: true, modeBarButtonsToInclude: ['resetScale2d', 'autoScale2d'], displayModeBar: true, displaylogo: false });

Plotly.newPlot(pnlChartUnrealised, [
  { x: [], y: [], mode: 'lines', name: 'Unrealised P&L', line: { color: '#f0883e', width: 2 },
    fill: 'tozeroy', fillcolor: 'rgba(240,136,62,0.1)' },
], { ...pnlDarkLayout, title: { text: 'Unrealised P&L (BTC)', font: { size: 12, color: '#f0883e' } },
     xaxis: { gridcolor: '#21262d' }, yaxis: { title: 'BTC', gridcolor: '#21262d', zeroline: true, zerolinecolor: '#30363d' } },
   { responsive: true, modeBarButtonsToInclude: ['resetScale2d', 'autoScale2d'], displayModeBar: true, displaylogo: false });

Plotly.newPlot(pnlChartSpot, [
  { x: [], y: [], mode: 'lines', name: 'BTC Spot', line: { color: '#3fb950', width: 2 } },
], { ...pnlDarkLayout, title: { text: 'BTC Spot Price', font: { size: 12, color: '#3fb950' } },
     xaxis: { gridcolor: '#21262d' }, yaxis: { title: '$', gridcolor: '#21262d', tickformat: '$,.0f' } },
   { responsive: true, modeBarButtonsToInclude: ['resetScale2d', 'autoScale2d'], displayModeBar: true, displaylogo: false });

function updatePnL() {
  const range = document.getElementById('pnl-range').value;
  const url = range !== '0' ? `/api/pnl?last=${range}` : '/api/pnl';
  fetch(url).then(r => r.json()).then(d => {
    if (!d || !d.count) return;
    document.getElementById('pnl-loading').style.display = 'none';
    document.getElementById('pnl-content').style.display = 'block';

    const last = d.count - 1;
    document.getElementById('pnl-equity').textContent = d.equity_btc[last].toFixed(4) + ' BTC';
    document.getElementById('pnl-equity-usd').textContent = '$' + d.equity_usd[last].toLocaleString(undefined, {maximumFractionDigits: 0});
    const uPnl = d.unrealised_pnl[last];
    const uEl = document.getElementById('pnl-unrealised');
    uEl.textContent = (uPnl >= 0 ? '+' : '') + uPnl.toFixed(6) + ' BTC';
    uEl.className = 'value ' + (uPnl >= 0 ? 'green' : 'red');
    document.getElementById('pnl-count').textContent = d.count;
    document.getElementById('pnl-since').textContent = d.first_ts;

    Plotly.update(pnlChartEquity, { x: [d.timestamps, d.timestamps], y: [d.equity_btc, d.balance_btc] }, {}, [0, 1]);
    Plotly.update(pnlChartUnrealised, { x: [d.timestamps], y: [d.unrealised_pnl] }, {}, [0]);

    // Bottom-right chart: spot or perpetual
    window._pnlData = d;
    updatePriceChart();
  }).catch(() => {});
}

function updatePriceChart() {
  const d = window._pnlData;
  if (!d) return;
  const mode = document.getElementById('pnl-price-toggle').value;
  const isPerp = mode === 'perp';
  const yData = isPerp ? d.perp_mark : d.spot;
  const name = isPerp ? 'BTC-PERPETUAL' : 'BTC Spot';
  const color = isPerp ? '#da8ee7' : '#3fb950';
  Plotly.update(pnlChartSpot,
    { x: [d.timestamps], y: [yData], name: [name], 'line.color': [color] },
    { title: { text: name + ' Price', font: { size: 12, color: color } } },
    [0]);
}

document.getElementById('pnl-price-toggle').addEventListener('change', updatePriceChart);
document.getElementById('pnl-range').addEventListener('change', updatePnL);

setInterval(updateVol, 1500);
setInterval(updateFutures, 2000);
setInterval(updateRisk, 5000);
setInterval(updatePnL, 5000);
updateVol(); updateFutures(); updateRisk(); updatePnL();

// ---- Hedge expiry dropdown ----
fetch('/api/expiries').then(r => r.json()).then(data => {
  const sel = document.getElementById('h-expiry');
  data.expiries.forEach(exp => {
    const opt = document.createElement('option');
    opt.value = exp; opt.textContent = exp;
    sel.appendChild(opt);
  });
});

// ---- Hedge calculator ----
function computeHedge() {
  const btn = document.getElementById('h-calc');
  const status = document.getElementById('h-status');
  btn.disabled = true;
  btn.textContent = 'Computing...';
  status.textContent = '';

  const delta = document.getElementById('h-delta').checked ? '1' : '0';
  const gamma = document.getElementById('h-gamma').checked ? '1' : '0';
  const vega = document.getElementById('h-vega').checked ? '1' : '0';
  const expiry = document.getElementById('h-expiry').value;

  let url = `/api/hedge?delta=${delta}&gamma=${gamma}&vega=${vega}`;
  if (expiry) url += '&expiry=' + expiry;

  fetch(url).then(r => r.json()).then(d => {
    btn.disabled = false;
    btn.textContent = 'Calculate Hedge';

    if (d.error) {
      status.textContent = d.error;
      status.style.color = '#f85149';
      return;
    }

    status.textContent = `Updated ${d.ts} UTC · ${d.candidates_found} candidates scanned`;
    status.style.color = '#484f58';

    document.getElementById('h-empty').style.display = 'none';

    // Summary boxes: before → after
    const greeks = [
      { key: 'bs_delta', label: 'Delta', dec: 4 },
      { key: 'gamma_1pct', label: 'Gamma (1%)', dec: 4 },
      { key: 'vega_usd', label: 'Vega ($)', dec: 2 },
      { key: 'theta_usd', label: 'Theta ($)', dec: 2 },
    ];
    let summaryHtml = '';
    greeks.forEach(g => {
      const before = d.current[g.key] || 0;
      const after = d.after[g.key] || 0;
      const isFlat = Math.abs(after) < (g.dec === 4 ? 0.001 : 1);
      const afterCls = isFlat ? 'flat' : 'residual';
      summaryHtml += `<div class="hedge-greek-box">
        <div class="label">${g.label}</div>
        <div class="before">${signedFmt(before, g.dec)}</div>
        <div class="arrow">\u2193</div>
        <div class="after ${afterCls}">${signedFmt(after, g.dec)}</div>
      </div>`;
    });
    const sumEl = document.getElementById('h-summary');
    sumEl.innerHTML = summaryHtml;
    sumEl.style.display = 'grid';

    // Trade cards with execute buttons
    const tradesEl = document.getElementById('h-trades');
    const tradesWrap = document.getElementById('h-trades-wrap');
    // Store trades globally for execution
    window._hedgeTrades = d.trades || [];

    if (d.trades && d.trades.length > 0) {
      tradesWrap.style.display = 'block';
      let html = '';
      d.trades.forEach((t, i) => {
        const dc = t.direction === 'buy' ? 'dir-buy' : 'dir-sell';
        html += `<div class="hedge-trade-card" id="trade-card-${i}">
          <div style="flex:1;">
            <div class="inst">${t.instrument}</div>
            <div class="detail">
              <span class="${dc}">${t.direction.toUpperCase()}</span> ${t.size_label} · Cost: ${t.cost}
            </div>
            <div class="impacts">
              <span>\u0394 <span class="val">${signedFmt(t.delta_impact, 4)}</span></span>
              <span>\u0393 <span class="val">${signedFmt(t.gamma_impact, 4)}</span></span>
              <span>\u03BD <span class="val">${signedFmt(t.vega_impact, 2)}</span></span>
              <span>\u0398 <span class="val">${signedFmt(t.theta_impact, 2)}</span></span>
            </div>
            <div class="fill-info" id="fill-${i}" style="display:none;"></div>
          </div>
          <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px;">
            <button class="exec-btn" id="exec-btn-${i}" onclick="confirmTrade(${i})">Execute</button>
            <div style="font-size:10px;color:#8b949e;text-align:right;max-width:160px;">${t.rationale}</div>
          </div>
        </div>`;
      });
      // Execute All button
      html += `<button class="exec-all-btn" id="exec-all-btn" onclick="confirmAllTrades()">Execute All Trades (${d.trades.length})</button>`;
      tradesEl.innerHTML = html;
    } else {
      tradesWrap.style.display = 'block';
      tradesEl.innerHTML = '<div class="hedge-no-trades">Portfolio already flat — no trades needed</div>';
    }
  }).catch(err => {
    btn.disabled = false;
    btn.textContent = 'Calculate Hedge';
    status.textContent = 'Error: ' + err;
    status.style.color = '#f85149';
  });
}

// ---- Trade execution ----
let _pendingExecCallback = null;

function closeModal() {
  document.getElementById('exec-modal').classList.remove('active');
  _pendingExecCallback = null;
}

function showModal(detailHtml, onConfirm) {
  document.getElementById('modal-detail').innerHTML = detailHtml;
  _pendingExecCallback = onConfirm;
  document.getElementById('modal-confirm-btn').onclick = () => {
    const cb = _pendingExecCallback;
    closeModal();
    if (cb) cb();
  };
  document.getElementById('exec-modal').classList.add('active');
}

function confirmTrade(idx) {
  const t = window._hedgeTrades[idx];
  if (!t) return;
  const dc = t.direction === 'buy' ? '<span style="color:#3fb950;font-weight:700">BUY</span>' : '<span style="color:#f85149;font-weight:700">SELL</span>';
  const detail = `${dc} <strong>${t.size_label}</strong> of <strong>${t.instrument}</strong><br>Order type: <strong>MARKET</strong>`;
  showModal(detail, () => executeTrade(idx));
}

function confirmAllTrades() {
  const trades = window._hedgeTrades || [];
  if (trades.length === 0) return;
  let lines = trades.map((t, i) => {
    const dc = t.direction === 'buy' ? '<span style="color:#3fb950">BUY</span>' : '<span style="color:#f85149">SELL</span>';
    return `${i+1}. ${dc} ${t.size_label} ${t.instrument}`;
  }).join('<br>');
  const detail = `Execute <strong>${trades.length} trades</strong> sequentially:<br><br>${lines}`;
  showModal(detail, () => executeAllTrades());
}

async function executeTrade(idx) {
  const t = window._hedgeTrades[idx];
  if (!t) return;

  const btn = document.getElementById('exec-btn-' + idx);
  const fillDiv = document.getElementById('fill-' + idx);
  btn.disabled = true;
  btn.textContent = 'Sending...';
  btn.className = 'exec-btn executing';

  try {
    const resp = await fetch('/api/execute_trade', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instrument: t.instrument,
        direction: t.direction,
        size: t.size,
      })
    });
    const d = await resp.json();

    fillDiv.style.display = 'block';
    if (d.ok) {
      btn.textContent = 'Filled';
      btn.className = 'exec-btn filled';
      const avgP = d.average_price ? '$' + d.average_price.toLocaleString(undefined,{maximumFractionDigits:2}) : 'N/A';
      fillDiv.className = 'fill-info success';
      fillDiv.innerHTML = `Filled ${d.filled_amount} @ ${avgP} · Order: ${d.order_id} · State: ${d.state}`;
      return true;
    } else {
      btn.textContent = 'Failed';
      btn.className = 'exec-btn failed';
      fillDiv.className = 'fill-info error';
      fillDiv.textContent = d.error || 'Unknown error';
      return false;
    }
  } catch (err) {
    btn.textContent = 'Error';
    btn.className = 'exec-btn failed';
    fillDiv.style.display = 'block';
    fillDiv.className = 'fill-info error';
    fillDiv.textContent = err.toString();
    return false;
  }
}

async function executeAllTrades() {
  const allBtn = document.getElementById('exec-all-btn');
  if (allBtn) { allBtn.disabled = true; allBtn.textContent = 'Executing...'; }

  const trades = window._hedgeTrades || [];
  let success = 0;
  for (let i = 0; i < trades.length; i++) {
    const btn = document.getElementById('exec-btn-' + i);
    if (btn && (btn.className.includes('filled'))) continue; // skip already filled
    const ok = await executeTrade(i);
    if (ok) success++;
    // Small delay between trades to avoid rate limiting
    if (i < trades.length - 1) await new Promise(r => setTimeout(r, 500));
  }

  if (allBtn) {
    allBtn.textContent = `Done — ${success}/${trades.length} filled`;
    if (success === trades.length) {
      allBtn.style.background = '#238636';
      allBtn.style.color = '#fff';
    }
  }
}

// ---- Close Positions ----
function refreshClosePanel(live) {
  const url = live ? '/api/positions?live=1' : '/api/positions';
  fetch(url).then(r => r.json()).then(positions => {
    positions = (positions || []).filter(p => Math.abs(p.size) > 0);
    const el = document.getElementById('close-positions-list');
    const btn = document.getElementById('close-all-btn');
    if (!positions || !positions.length) {
      el.innerHTML = '<div style="padding:8px 0;color:#3fb950;">No open positions</div>';
      btn.style.display = 'none';
      return;
    }
    btn.style.display = 'block';
    el.innerHTML = positions.map((p, i) => {
      const dir = p.direction === 'buy' ? 'LONG' : 'SHORT';
      const dirCls = p.direction === 'buy' ? 'color:#3fb950' : 'color:#f85149';
      const sizeLabel = p.kind === 'future' ? `$${p.size.toLocaleString()}` : p.size.toFixed(1);
      return `<div class="close-pos-row">
        <div class="close-pos-info">
          <div class="close-pos-name">${p.instrument}</div>
          <div class="close-pos-detail"><span style="${dirCls};font-weight:700;">${dir}</span> ${sizeLabel}</div>
        </div>
        <button class="close-pos-btn" id="close-btn-${i}" onclick="closeSinglePosition('${p.instrument}', this)">Close</button>
      </div>`;
    }).join('');
  }).catch(() => {});
}

function closeSinglePosition(instrument, btn) {
  const detail = `Close your entire <strong>${instrument}</strong> position at market?`;
  showModal(detail, async () => {
    btn.disabled = true;
    btn.textContent = '...';
    const statusEl = document.getElementById('close-status');
    statusEl.textContent = '';
    try {
      const resp = await fetch('/api/close_position', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({instrument})
      });
      const result = await resp.json();
      if (result.ok) {
        btn.textContent = 'Closed';
        btn.style.color = '#3fb950';
        btn.style.borderColor = '#3fb95030';
        statusEl.textContent = `Closed ${instrument} @ $${(result.average_price||0).toLocaleString()}`;
        statusEl.style.color = '#3fb950';
        setTimeout(() => refreshClosePanel(true), 1000);
      } else {
        btn.disabled = false;
        btn.textContent = 'Close';
        statusEl.textContent = result.error || 'Failed';
        statusEl.style.color = '#f85149';
      }
    } catch(e) {
      btn.disabled = false;
      btn.textContent = 'Close';
      statusEl.textContent = 'Network error';
      statusEl.style.color = '#f85149';
    }
  });
}

function closeAllPositions() {
  showModal('Close <strong>ALL</strong> open positions at market? This will send one close order per position.', async () => {
    const btn = document.getElementById('close-all-btn');
    const statusEl = document.getElementById('close-status');
    btn.disabled = true;
    btn.textContent = 'Closing...';
    statusEl.textContent = '';

    let resp = await fetch('/api/positions?live=1');
    const positions = await resp.json();
    if (!positions || !positions.length) {
      btn.textContent = 'No positions';
      return;
    }

    let closed = 0;
    for (const p of positions) {
      try {
        resp = await fetch('/api/close_position', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({instrument: p.instrument})
        });
        const result = await resp.json();
        if (result.ok) closed++;
      } catch(e) {}
      await new Promise(r => setTimeout(r, 500));
    }

    btn.textContent = `Closed ${closed}/${positions.length}`;
    btn.style.background = closed === positions.length ? '#238636' : '#d29922';
    statusEl.textContent = `${closed} positions closed`;
    statusEl.style.color = closed === positions.length ? '#3fb950' : '#d29922';
    setTimeout(() => {
      btn.disabled = false;
      btn.textContent = 'Close All Positions';
      btn.style.background = '#f85149';
      refreshClosePanel(true);
    }, 3000);
  });
}

// Refresh close panel when hedge tab is opened and periodically
document.querySelector('[data-tab="hedge-tab"]').addEventListener('click', refreshClosePanel);
refreshClosePanel();
setInterval(refreshClosePanel, 15000);

// ---- Recon sub-tabs ----
document.querySelectorAll('.recon-sub').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.recon-sub').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.recon-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.rsub).classList.add('active');
  });
});

// ---- Recon loader ----
function loadRecon() {
  const btn = document.getElementById('recon-load-btn');
  const status = document.getElementById('recon-status');
  btn.disabled = true;
  btn.textContent = 'Loading...';
  status.textContent = 'Fetching trades, settlements, and transfers from Deribit...';
  status.style.color = '#8b949e';

  fetch('/api/recon').then(r => r.json()).then(d => {
    btn.disabled = false;
    btn.textContent = 'Reload Reconciliation';

    if (d.error) {
      status.textContent = d.error;
      status.style.color = '#f85149';
      return;
    }

    status.textContent = `Loaded at ${d.ts} UTC — ${d.trade_count} trades, ${d.settlement_count} settlements, ${d.transfer_count} transfers`;
    status.style.color = '#3fb950';

    document.getElementById('recon-loading').style.display = 'none';
    document.getElementById('recon-content').style.display = 'block';

    // Counts
    document.getElementById('recon-trade-count').textContent = d.trade_count;
    document.getElementById('recon-settlement-count').textContent = d.settlement_count;
    document.getElementById('recon-transfer-count').textContent = d.transfer_count;

    // Summary header
    const summaryEl = document.getElementById('recon-summary');
    summaryEl.innerHTML = `
      <div class="risk-stat"><div class="label">API Equity</div><div class="value blue">${d.api_equity.toFixed(4)} BTC</div></div>
      <div class="risk-stat"><div class="label">Calculated Equity</div><div class="value green">${d.calculated_equity.toFixed(4)} BTC</div></div>
      <div class="risk-stat"><div class="label">Difference</div><div class="value ${Math.abs(d.equity_diff) < 0.0001 ? 'green' : 'red'}">${d.equity_diff >= 0 ? '+' : ''}${d.equity_diff.toFixed(8)} BTC</div></div>
      <div class="risk-stat"><div class="label">Total Trades</div><div class="value">${d.trade_count}</div></div>
      <div class="risk-stat"><div class="label">Realised P&amp;L</div><div class="value ${d.total_realised_pnl >= 0 ? 'green' : 'red'}">${d.total_realised_pnl >= 0 ? '+' : ''}${d.total_realised_pnl.toFixed(6)} BTC</div></div>
      <div class="risk-stat"><div class="label">Total Fees</div><div class="value red">-${d.total_fees.toFixed(6)} BTC</div></div>
    `;

    // Trades table
    const tbody = document.getElementById('recon-trades-body');
    tbody.innerHTML = d.trades.slice().reverse().map(t => {
      const dirCls = t.direction === 'buy' ? 'color:#3fb950' : 'color:#f85149';
      const pnlCls = t.pnl > 0 ? 'color:#3fb950' : t.pnl < 0 ? 'color:#f85149' : '';
      const priceStr = t.is_option ? t.price.toFixed(4) + ' BTC' : '$' + t.price.toLocaleString(undefined, {maximumFractionDigits: 2});
      const amtStr = t.is_option ? t.amount.toFixed(1) : '$' + t.amount.toLocaleString();
      return `<tr>
        <td>${t.ts}</td>
        <td>${t.instrument}</td>
        <td style="${dirCls};font-weight:700">${t.direction.toUpperCase()}</td>
        <td>${amtStr}</td>
        <td>${priceStr}</td>
        <td>${t.premium ? t.premium.toFixed(6) : '—'}</td>
        <td style="color:#f85149">${t.fee ? '-' + t.fee.toFixed(6) : '—'}</td>
        <td style="${pnlCls}">${t.pnl !== 0 ? (t.pnl > 0 ? '+' : '') + t.pnl.toFixed(8) : '—'}</td>
      </tr>`;
    }).join('');

    // Settlements table
    const stbody = document.getElementById('recon-settlements-body');
    if (d.settlements.length === 0) {
      stbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#8b949e;padding:20px;">No settlements found</td></tr>';
    } else {
      stbody.innerHTML = d.settlements.slice().reverse().map(s => {
        const fCls = s.funding > 0 ? 'color:#3fb950' : s.funding < 0 ? 'color:#f85149' : '';
        const pCls = s.session_pnl > 0 ? 'color:#3fb950' : s.session_pnl < 0 ? 'color:#f85149' : '';
        return `<tr>
          <td>${s.ts}</td>
          <td>${s.instrument}</td>
          <td>${s.type}</td>
          <td style="${fCls}">${s.funding !== 0 ? (s.funding > 0 ? '+' : '') + s.funding.toFixed(8) : '—'}</td>
          <td style="${pCls}">${s.session_pnl !== 0 ? (s.session_pnl > 0 ? '+' : '') + s.session_pnl.toFixed(8) : '—'}</td>
          <td>${s.position}</td>
          <td>$${s.mark_price.toLocaleString(undefined, {maximumFractionDigits: 2})}</td>
          <td>$${s.index_price.toLocaleString(undefined, {maximumFractionDigits: 2})}</td>
        </tr>`;
      }).join('');
    }

    // Transfers table
    const ttbody = document.getElementById('recon-transfers-body');
    if (d.transfers.length === 0) {
      document.getElementById('recon-no-transfers').style.display = 'block';
      ttbody.innerHTML = '';
    } else {
      document.getElementById('recon-no-transfers').style.display = 'none';
      ttbody.innerHTML = d.transfers.map(tr => {
        const tCls = tr.type === 'withdrawal' ? 'color:#f85149' : 'color:#3fb950';
        return `<tr>
          <td>${tr.ts}</td>
          <td style="${tCls};font-weight:700">${tr.type.toUpperCase()}</td>
          <td>${tr.amount.toFixed(8)}</td>
          <td>${tr.state || '—'}</td>
          <td style="font-size:10px;word-break:break-all;">${tr.tx_id || '—'}</td>
        </tr>`;
      }).join('');
    }

    // Reconciliation panel
    function rcVal(id, val, dec) {
      const el = document.getElementById(id);
      const s = (val >= 0 ? '+' : '') + val.toFixed(dec || 8) + ' BTC';
      el.textContent = s;
    }
    rcVal('rc-deposits', d.total_deposits, 4);
    rcVal('rc-withdrawals', -d.total_withdrawals, 4);
    rcVal('rc-realised', d.total_realised_pnl, 8);
    rcVal('rc-funding', d.total_funding, 8);
    document.getElementById('rc-fees').textContent = '-' + d.total_fees.toFixed(8) + ' BTC';
    rcVal('rc-prem-recv', d.total_premium_received, 6);
    document.getElementById('rc-prem-paid').textContent = '-' + d.total_premium_paid.toFixed(6) + ' BTC';
    rcVal('rc-prem-net', d.net_premium, 6);
    document.getElementById('rc-calc-balance').textContent = d.calculated_balance.toFixed(8) + ' BTC';
    rcVal('rc-unrealised', d.unrealised, 8);
    document.getElementById('rc-calc-equity').textContent = d.calculated_equity.toFixed(8) + ' BTC';
    document.getElementById('rc-api-balance').textContent = d.api_balance.toFixed(8) + ' BTC';
    document.getElementById('rc-api-equity').textContent = d.api_equity.toFixed(8) + ' BTC';

    const diffB = d.balance_diff;
    const diffE = d.equity_diff;
    const dbEl = document.getElementById('rc-diff-balance');
    const deEl = document.getElementById('rc-diff-equity');
    dbEl.textContent = (diffB >= 0 ? '+' : '') + diffB.toFixed(8) + ' BTC';
    deEl.textContent = (diffE >= 0 ? '+' : '') + diffE.toFixed(8) + ' BTC';
    dbEl.className = 'param-value ' + (Math.abs(diffB) < 0.0001 ? 'green' : 'red');
    deEl.className = 'param-value ' + (Math.abs(diffE) < 0.0001 ? 'green' : 'red');

  }).catch(err => {
    btn.disabled = false;
    btn.textContent = 'Load Reconciliation';
    status.textContent = 'Error: ' + err;
    status.style.color = '#f85149';
  });
}

// ---- Auto Hedge ----
let ahPolling = null;

function toggleAutoHedge() {
  const btn = document.getElementById('ah-start-btn');
  const stateEl = document.getElementById('ah-state');

  if (btn.textContent.includes('Start')) {
    // Collect params and start
    const params = {
      threshold: parseFloat(document.getElementById('ah-threshold').value) || 0.02,
      interval: parseInt(document.getElementById('ah-interval').value) || 30,
      gamma_scaling: document.getElementById('ah-gamma-scaling').checked,
      max_trade: parseFloat(document.getElementById('ah-max-trade').value) || 100000,
    };
    fetch('/api/autohedge/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params)
    }).then(r => r.json()).then(d => {
      if (d.ok) {
        btn.textContent = 'Stop Auto Hedge';
        btn.style.background = '#f85149';
        stateEl.textContent = 'RUNNING';
        stateEl.style.color = '#3fb950';
        startAhPolling();
      }
    });
  } else {
    fetch('/api/autohedge/stop', {method: 'POST'}).then(r => r.json()).then(d => {
      btn.textContent = 'Start Auto Hedge';
      btn.style.background = '#238636';
      stateEl.textContent = 'STOPPED';
      stateEl.style.color = '#e6edf3';
    });
  }
}

function updateAutoHedgeParams() {
  const params = {
    threshold: parseFloat(document.getElementById('ah-threshold').value) || 0.02,
    interval: parseInt(document.getElementById('ah-interval').value) || 30,
    gamma_scaling: document.getElementById('ah-gamma-scaling').checked,
    max_trade: parseFloat(document.getElementById('ah-max-trade').value) || 100000,
  };
  fetch('/api/autohedge/params', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params)
  }).then(r => r.json()).then(d => {
    document.getElementById('ah-status-text').textContent = d.ok ? 'Parameters updated' : 'Error';
    document.getElementById('ah-status-text').style.color = d.ok ? '#3fb950' : '#f85149';
    setTimeout(() => { document.getElementById('ah-status-text').textContent = ''; }, 3000);
  });
}

function startAhPolling() {
  if (ahPolling) return;
  ahPolling = setInterval(pollAutoHedge, 2000);
}

function pollAutoHedge() {
  fetch('/api/autohedge/status').then(r => r.json()).then(d => {
    const btn = document.getElementById('ah-start-btn');
    const stateEl = document.getElementById('ah-state');

    if (d.running) {
      btn.textContent = 'Stop Auto Hedge';
      btn.style.background = '#f85149';
      stateEl.textContent = 'RUNNING';
      stateEl.style.color = '#3fb950';
    } else {
      btn.textContent = 'Start Auto Hedge';
      btn.style.background = '#238636';
      stateEl.textContent = 'STOPPED';
      stateEl.style.color = '#e6edf3';
    }

    document.getElementById('ah-hedges').textContent = d.stats.hedges;
    document.getElementById('ah-traded').textContent = '$' + (d.stats.total_traded_usd || 0).toLocaleString();
    document.getElementById('ah-last').textContent = d.stats.last_hedge_ts || '\u2014';

    // Log
    const logEl = document.getElementById('ah-log');
    if (d.log && d.log.length > 0) {
      logEl.innerHTML = d.log.slice().reverse().map(l => {
        let color = '#8b949e';
        if (l.action === 'HEDGING' || l.action === 'FILLED') color = '#3fb950';
        else if (l.action === 'FAILED' || l.action === 'ERROR') color = '#f85149';
        else if (l.action === 'CAPPED') color = '#d29922';
        else if (l.action.includes('START')) color = '#58a6ff';
        else if (l.action.includes('STOP')) color = '#d29922';
        return `<div style="padding:3px 0;border-bottom:1px solid #21262d;">
          <span style="color:#484f58;">${l.ts}</span>
          <span style="color:${color};font-weight:600;margin:0 6px;">${l.action}</span>
          <span style="color:#8b949e;">${l.detail}</span>
        </div>`;
      }).join('');
    }
  }).catch(() => {});
}

// Poll auto hedge status when tab is opened
document.querySelector('[data-tab="autohedge-tab"]').addEventListener('click', () => {
  pollAutoHedge();
  startAhPolling();
});

// ---- Backtest ----
function runBacktest() {
  const btn = document.getElementById('bt-run-btn');
  const status = document.getElementById('bt-status');
  btn.disabled = true;
  btn.textContent = 'Running...';
  status.textContent = 'Fetching historical data and running simulation...';
  status.style.color = '#8b949e';

  const params = {
    option_type: document.getElementById('bt-option-type').value,
    option_direction: document.getElementById('bt-direction').value,
    strike_offset_pct: parseFloat(document.getElementById('bt-strike-offset').value) || 0,
    option_size: parseFloat(document.getElementById('bt-size').value) || 1,
    iv: (parseFloat(document.getElementById('bt-iv').value) || 60) / 100,
    tte: (parseFloat(document.getElementById('bt-tte').value) || 30) / 365.25,
    days: parseInt(document.getElementById('bt-days').value) || 30,
    resolution: document.getElementById('bt-resolution').value,
    hedge_interval: parseInt(document.getElementById('bt-hedge-interval').value) || 1,
    cost_per_trade: (parseFloat(document.getElementById('bt-cost').value) || 0.05) / 100,
    delta_threshold: parseFloat(document.getElementById('bt-delta-thresh').value) || 0.001,
  };

  fetch('/api/backtest', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params)
  }).then(r => r.json()).then(d => {
    btn.disabled = false;
    btn.textContent = 'Run Backtest';

    if (d.error) {
      status.textContent = d.error;
      status.style.color = '#f85149';
      return;
    }

    status.textContent = `Completed: ${d.metrics.n_candles} candles, ${d.metrics.hedge_count} hedges`;
    status.style.color = '#3fb950';

    document.getElementById('bt-empty').style.display = 'none';
    document.getElementById('bt-results').style.display = 'block';

    // Metrics
    const m = d.metrics;
    document.getElementById('bt-metrics').innerHTML = `
      <div class="risk-stat"><div class="label">Total P&amp;L</div><div class="value ${m.total_pnl_btc >= 0 ? 'green' : 'red'}">${m.total_pnl_btc >= 0 ? '+' : ''}${m.total_pnl_btc.toFixed(6)} BTC</div></div>
      <div class="risk-stat"><div class="label">Sharpe Ratio</div><div class="value blue">${m.sharpe.toFixed(3)}</div></div>
      <div class="risk-stat"><div class="label">Max Drawdown</div><div class="value red">${m.max_drawdown_btc.toFixed(6)} BTC</div></div>
      <div class="risk-stat"><div class="label">Hedges</div><div class="value">${m.hedge_count}</div></div>
      <div class="risk-stat"><div class="label">Costs</div><div class="value red">${m.total_costs_btc.toFixed(6)} BTC</div></div>
      <div class="risk-stat"><div class="label">Setup</div><div class="value" style="font-size:12px;">${m.option_direction} ${m.option_size} ${m.option_type} K=$${m.strike.toLocaleString()} IV=${m.iv}%</div></div>
    `;

    const plotStyle = {
      paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
      font: { family: 'SF Mono, Fira Code, Consolas, monospace', color: '#8b949e', size: 10 },
      margin: { l: 55, r: 15, t: 25, b: 35 },
    };

    // PnL chart
    Plotly.newPlot('bt-chart-pnl', [
      { x: d.timestamps, y: d.total_btc, name: 'Total Portfolio', line: { color: '#58a6ff', width: 2 } },
      { x: d.timestamps, y: d.option_value, name: 'Option Value', line: { color: '#f0883e', width: 1, dash: 'dot' } },
      { x: d.timestamps, y: d.perp_pnl, name: 'Perp Hedge P&L', line: { color: '#d2a8ff', width: 1, dash: 'dot' } },
    ], {
      ...plotStyle,
      title: { text: 'Portfolio P&L (BTC)', font: { size: 12, color: '#8b949e' } },
      xaxis: { gridcolor: '#21262d' },
      yaxis: { title: 'BTC', gridcolor: '#21262d' },
      legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)' },
    }, { responsive: true, displayModeBar: false });

    // Delta chart
    Plotly.newPlot('bt-chart-delta', [
      { x: d.timestamps, y: d.portfolio_delta, name: 'Portfolio Delta', line: { color: '#3fb950', width: 1.5 } },
      { x: d.timestamps, y: d.gamma, name: 'Gamma (1%)', line: { color: '#d29922', width: 1, dash: 'dash' }, yaxis: 'y2' },
    ], {
      ...plotStyle,
      title: { text: 'Delta & Gamma', font: { size: 12, color: '#8b949e' } },
      xaxis: { gridcolor: '#21262d' },
      yaxis: { title: 'Delta (BTC)', gridcolor: '#21262d' },
      yaxis2: { title: 'Gamma', overlaying: 'y', side: 'right', gridcolor: '#21262d20' },
      legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)' },
    }, { responsive: true, displayModeBar: false });

    // Price chart
    Plotly.newPlot('bt-chart-price', [
      { x: d.timestamps, y: d.prices, name: 'BTC Price', line: { color: '#e6edf3', width: 1.5 } },
      { x: d.timestamps, y: d.prices.map(() => m.strike), name: `Strike $${m.strike.toLocaleString()}`, line: { color: '#f85149', width: 1, dash: 'dash' } },
    ], {
      ...plotStyle,
      title: { text: 'Underlying Price', font: { size: 12, color: '#8b949e' } },
      xaxis: { gridcolor: '#21262d' },
      yaxis: { title: 'USD', gridcolor: '#21262d', tickformat: '$,.0f' },
      legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)' },
    }, { responsive: true, displayModeBar: false });

    window.dispatchEvent(new Event('resize'));

  }).catch(err => {
    btn.disabled = false;
    btn.textContent = 'Run Backtest';
    status.textContent = 'Error: ' + err;
    status.style.color = '#f85149';
  });
}
</script>
</body>
</html>
"""


LOGIN_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC Trading Dashboard — Login</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0d1117; color: #e6edf3;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh;
  }
  .login-box {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 40px; width: 440px; max-width: 90%;
  }
  h1 { font-size: 20px; color: #58a6ff; margin-bottom: 6px; }
  .subtitle { font-size: 12px; color: #8b949e; margin-bottom: 28px; }
  label { display: block; font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; margin-top: 16px; }
  input, select {
    width: 100%; padding: 10px 12px; font-family: inherit; font-size: 13px;
    background: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px;
    outline: none;
  }
  input:focus, select:focus { border-color: #58a6ff; }
  input::placeholder { color: #484f58; }
  .network-row { display: flex; gap: 10px; margin-top: 16px; }
  .net-btn {
    flex: 1; padding: 10px; font-family: inherit; font-size: 12px; font-weight: 600;
    background: #0d1117; color: #8b949e; border: 1px solid #30363d; border-radius: 6px;
    cursor: pointer; text-align: center;
  }
  .net-btn:hover { background: #161b22; color: #e6edf3; }
  .net-btn.active { border-color: #58a6ff; color: #58a6ff; background: #1f6feb15; }
  .net-btn .sub { display: block; font-size: 10px; font-weight: 400; color: #484f58; margin-top: 3px; }
  .connect-btn {
    width: 100%; margin-top: 24px; padding: 12px; font-family: inherit; font-size: 14px;
    font-weight: 700; background: #238636; color: #fff; border: none; border-radius: 8px;
    cursor: pointer; letter-spacing: 0.5px;
  }
  .connect-btn:hover { background: #2ea043; }
  .connect-btn:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
  .error { margin-top: 12px; padding: 10px; font-size: 11px; color: #f85149; background: rgba(248,81,73,0.1); border-radius: 6px; display: none; }
  .info { margin-top: 20px; font-size: 10px; color: #484f58; line-height: 1.6; text-align: center; }
  .info a { color: #58a6ff; text-decoration: none; }
</style>
</head>
<body>
<div class="login-box">
  <h1>BTC Trading Dashboard</h1>
  <div class="subtitle">SVI Vol Surface · Futures Curve · Portfolio Risk · P&amp;L</div>

  <label>Network</label>
  <div class="network-row">
    <div class="net-btn active" id="net-testnet" onclick="setNet('testnet')">
      Testnet<span class="sub">test.deribit.com</span>
    </div>
    <div class="net-btn" id="net-mainnet" onclick="setNet('mainnet')">
      Production<span class="sub">www.deribit.com</span>
    </div>
  </div>

  <label>API Client ID</label>
  <input type="text" id="client-id" placeholder="Your Client ID" autocomplete="off">

  <label>API Client Secret</label>
  <input type="password" id="client-secret" placeholder="Your API secret" autocomplete="off">

  <button class="connect-btn" id="connect-btn" onclick="connect()">Connect</button>
  <div class="error" id="error-msg"></div>

  <div class="info">
    Your credentials are sent directly to Deribit and are never stored.<br>
    Get API keys from <a href="https://test.deribit.com/account/BTC/api" target="_blank">Testnet</a> or
    <a href="https://www.deribit.com/account/BTC/api" target="_blank">Production</a>.
  </div>
</div>

<script>
let network = 'testnet';

function setNet(net) {
  network = net;
  document.getElementById('net-testnet').className = 'net-btn' + (net === 'testnet' ? ' active' : '');
  document.getElementById('net-mainnet').className = 'net-btn' + (net === 'mainnet' ? ' active' : '');
}

function connect() {
  const clientId = document.getElementById('client-id').value.trim();
  const clientSecret = document.getElementById('client-secret').value.trim();
  const errEl = document.getElementById('error-msg');
  const btn = document.getElementById('connect-btn');

  if (!clientId || !clientSecret) {
    errEl.textContent = 'Please enter both Client ID and Client Secret.';
    errEl.style.display = 'block';
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Connecting...';
  errEl.style.display = 'none';

  fetch('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ client_id: clientId, client_secret: clientSecret, network: network })
  }).then(r => r.json()).then(d => {
    if (d.ok) {
      window.location.href = '/dashboard';
    } else {
      errEl.textContent = d.error || 'Authentication failed. Check your credentials.';
      errEl.style.display = 'block';
      btn.disabled = false;
      btn.textContent = 'Connect';
    }
  }).catch(err => {
    errEl.textContent = 'Network error: ' + err;
    errEl.style.display = 'block';
    btn.disabled = false;
    btn.textContent = 'Connect';
  });
}

// Enter key to submit
document.getElementById('client-secret').addEventListener('keydown', e => {
  if (e.key === 'Enter') connect();
});
</script>
</body>
</html>
"""

_engine_started = False

@app.route("/")
def index():
    if CLIENT_ID and _engine_started:
        return render_template_string(HTML_PAGE)
    return render_template_string(LOGIN_PAGE)

@app.route("/dashboard")
def dashboard():
    if not CLIENT_ID or not _engine_started:
        return """<script>window.location.href='/';</script>"""
    return render_template_string(HTML_PAGE)

@app.route("/api/login", methods=["POST"])
def api_login():
    global CLIENT_ID, CLIENT_SECRET, BASE_URL, WS_URL, _engine_started

    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "error": "No data"}), 400

    cid = data.get("client_id", "").strip()
    csecret = data.get("client_secret", "").strip()
    net = data.get("network", "testnet")

    if not cid or not csecret:
        return jsonify({"ok": False, "error": "Client ID and Secret are required"}), 400

    if net not in NETWORKS:
        return jsonify({"ok": False, "error": "Invalid network"}), 400

    # Test authentication against Deribit
    net_urls = NETWORKS[net]
    try:
        r = requests.get(f"{net_urls['rest']}/public/auth", params={
            "client_id": cid, "client_secret": csecret,
            "grant_type": "client_credentials"
        }, timeout=10)
        result = r.json()
        if "result" not in result:
            err_msg = result.get("error", {}).get("message", "Authentication failed")
            return jsonify({"ok": False, "error": err_msg})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Connection error: {e}"})

    # Credentials valid — configure and start engine
    CLIENT_ID = cid
    CLIENT_SECRET = csecret
    BASE_URL = net_urls["rest"]
    WS_URL = net_urls["ws"]

    if not _engine_started:
        _engine_started = True
        # Start engine in background thread to not block the response
        def _start():
            expiries = get_all_expiries()
            if expiries:
                default = "24APR26" if "24APR26" in expiries else expiries[0]
                print(f"Starting with expiry: {default}")
                engine.start_futures_ws()
                engine.switch_expiry(default)
                engine.start_risk_poller()
                engine.start_order_ws()
        threading.Thread(target=_start, daemon=True).start()

    return jsonify({"ok": True, "network": net})

@app.route("/api/expiries")
def api_expiries():
    return jsonify({"expiries": get_all_expiries(), "current": engine.expiry})

@app.route("/api/set_expiry")
def api_set_expiry():
    expiry = request.args.get("expiry")
    if expiry:
        threading.Thread(target=engine.switch_expiry, args=(expiry,), daemon=True).start()
        return jsonify({"ok": True, "expiry": expiry})
    return jsonify({"ok": False}), 400

@app.route("/api/data")
def api_data():
    with engine.lock:
        return jsonify(engine.latest)

@app.route("/api/futures")
def api_futures():
    data = engine.get_futures_data()
    return jsonify(data if data else {})

@app.route("/api/risk")
def api_risk():
    with engine.lock:
        return jsonify(engine.risk_data)

@app.route("/api/hedge")
def api_hedge():
    hedge_delta = request.args.get("delta", "1") == "1"
    hedge_gamma = request.args.get("gamma", "0") == "1"
    hedge_vega = request.args.get("vega", "0") == "1"
    target_expiry = request.args.get("expiry", None)
    result = engine.compute_hedge(hedge_delta, hedge_gamma, hedge_vega, target_expiry)
    return jsonify(result)

@app.route("/api/execute_trade", methods=["POST"])
def api_execute_trade():
    """Execute a market order via authenticated WebSocket (fast path)."""
    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "error": "No data"}), 400

    instrument = data.get("instrument")
    direction = data.get("direction")
    size = data.get("size")

    if not instrument or not direction or not size:
        return jsonify({"ok": False, "error": "Missing instrument, direction, or size"}), 400

    result = engine.execute_order_ws(instrument, direction, size)
    return jsonify(result)

@app.route("/api/close_position", methods=["POST"])
def api_close_position():
    """Close a specific position via market order."""
    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "error": "No data"}), 400

    instrument = data.get("instrument")
    if not instrument:
        return jsonify({"ok": False, "error": "Missing instrument"}), 400

    # Get the current position to determine direction and size
    token = authenticate()
    if not token:
        return jsonify({"ok": False, "error": "Auth failed"})

    positions = api_call("private/get_positions", {"currency": "BTC"}, token) or []
    pos = next((p for p in positions if p["instrument_name"] == instrument), None)
    if not pos or pos.get("size", 0) == 0:
        return jsonify({"ok": False, "error": f"No open position for {instrument}"})

    size = abs(pos["size"])
    direction = pos.get("direction", "")
    # To close: sell if long, buy if short
    close_direction = "sell" if direction == "buy" else "buy"

    result = engine.execute_order_ws(instrument, close_direction, size)
    result["closed_instrument"] = instrument
    result["closed_size"] = size
    result["closed_direction"] = close_direction

    # Trigger a background risk refresh so positions update quickly
    threading.Thread(target=engine._compute_risk, daemon=True).start()

    return jsonify(result)


@app.route("/api/positions")
def api_positions():
    """Return current open positions. Use ?live=1 to fetch fresh from Deribit."""
    if request.args.get("live"):
        token = authenticate()
        if token:
            positions = api_call("private/get_positions", {"currency": "BTC"}, token) or []
            positions = [p for p in positions if abs(p.get("size", 0)) > 0]
            result = []
            for pos in positions:
                name = pos["instrument_name"]
                kind = pos.get("kind", "")
                direction = pos.get("direction", "")
                size = abs(pos.get("size", 0))
                result.append({"instrument": name, "kind": kind, "direction": direction, "size": size})
            return jsonify(result)
    with engine.lock:
        positions = engine.risk_data.get("positions", [])
    return jsonify([p for p in positions if abs(p.get("size", 0)) > 0])


@app.route("/api/pnl")
def api_pnl():
    """Return P&L history for charting."""
    last_minutes = request.args.get("last", None)
    last_minutes = int(last_minutes) if last_minutes else None
    data = engine.get_pnl_history(last_minutes)
    return jsonify(data if data else {})


@app.route("/api/recon")
def api_recon():
    """Fetch all trades, settlements, transfers and reconcile equity."""
    result = engine.compute_reconciliation()
    return jsonify(result)


# ---- Auto Hedge API ----
@app.route("/api/autohedge/start", methods=["POST"])
def api_autohedge_start():
    data = request.get_json() or {}
    auto_hedger.update_params(
        threshold=data.get("threshold"),
        interval=data.get("interval"),
        gamma_scaling=data.get("gamma_scaling"),
        max_trade=data.get("max_trade"),
    )
    auto_hedger.start()
    return jsonify({"ok": True})


@app.route("/api/autohedge/stop", methods=["POST"])
def api_autohedge_stop():
    auto_hedger.stop()
    return jsonify({"ok": True})


@app.route("/api/autohedge/params", methods=["POST"])
def api_autohedge_params():
    data = request.get_json() or {}
    auto_hedger.update_params(
        threshold=data.get("threshold"),
        interval=data.get("interval"),
        gamma_scaling=data.get("gamma_scaling"),
        max_trade=data.get("max_trade"),
    )
    return jsonify({"ok": True})


@app.route("/api/autohedge/status")
def api_autohedge_status():
    return jsonify(auto_hedger.get_status())


# ---- Backtest API ----
@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    data = request.get_json() or {}
    result = BacktestEngine.run_delta_hedge_backtest(data)
    return jsonify(result)


if __name__ == "__main__":
    import sys

    # Allow passing credentials via environment variables for convenience
    env_id = os.environ.get("DERIBIT_CLIENT_ID")
    env_secret = os.environ.get("DERIBIT_CLIENT_SECRET")
    env_net = os.environ.get("DERIBIT_NETWORK", "testnet")

    if env_id and env_secret:
        # Auto-start with env credentials (for local dev)
        CLIENT_ID = env_id
        CLIENT_SECRET = env_secret
        net = NETWORKS.get(env_net, NETWORKS["testnet"])
        BASE_URL = net["rest"]
        WS_URL = net["ws"]

        default_expiries = get_all_expiries()
        if not default_expiries:
            print("No option expiries found.")
            sys.exit(1)

        default = "24APR26" if "24APR26" in default_expiries else default_expiries[0]
        print(f"Starting with expiry: {default} ({env_net})")
        print(f"Available: {', '.join(default_expiries)}")

        engine.start_futures_ws()
        engine.switch_expiry(default)
        engine.start_risk_poller()
        engine.start_order_ws()
        _engine_started = True
    else:
        print("No credentials in environment. Starting with login page.")
        print("  (Set DERIBIT_CLIENT_ID and DERIBIT_CLIENT_SECRET env vars to skip login)")

    print("\n=== Open http://localhost:5050 ===\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
