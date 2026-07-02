#!/usr/bin/env python3
"""
Institutional-Grade Indian Stock Market Scanner
================================================
Scans NSE/BSE stocks via Yahoo Finance, combining technical + fundamental
analysis with fund-manager-style recommendation logic.

Usage:
    pip install yfinance pandas numpy requests
    python india_scanner.py                     # scan default NSE universe
    python india_scanner.py --universe my.txt   # custom symbol list (one per line, e.g. RELIANCE.NS)
    python india_scanner.py --limit 100         # cap number of stocks (for testing)
    python india_scanner.py --workers 8         # parallel download threads

Outputs (in ./output/):
    all_stocks.json, strong_buy.json, buy.json, hold.json, avoid.json,
    top_growth_stocks.json, top_value_stocks.json, top_breakout_stocks.json,
    sector_leaders.json, market_summary.json

DISCLAIMER: This is a screening/research tool, not investment advice.
Verify all data independently before making investment decisions.
"""

import argparse
import io
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("Install dependencies first: pip install yfinance pandas numpy requests")

try:
    import requests
except ImportError:
    requests = None

OUTPUT_DIR = "output"
MIN_AVG_VALUE_TRADED = 1e7   # ₹1 crore/day minimum liquidity
MIN_PRICE = 10.0             # skip penny stocks
HISTORY_PERIOD = "2y"

# ----------------------------------------------------------------------------
# Universe construction
# ----------------------------------------------------------------------------

NSE_INDEX_CSVS = {
    # NSE publishes index constituents as CSVs. Nifty 500 covers ~95% of
    # free-float market cap — a practical "whole market" institutional universe.
    "nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "midcap150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "smallcap250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}

FALLBACK_UNIVERSE = [
    # Nifty 100 core names as a fallback if NSE archives are unreachable.
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "BAJFINANCE", "NESTLEIND",
    "WIPRO", "M&M", "NTPC", "POWERGRID", "TATAMOTORS", "TATASTEEL", "HCLTECH",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "ONGC", "JSWSTEEL", "GRASIM",
    "TECHM", "HINDALCO", "DRREDDY", "CIPLA", "EICHERMOT", "BRITANNIA",
    "DIVISLAB", "APOLLOHOSP", "BAJAJFINSV", "HEROMOTOCO", "INDUSINDBK",
    "TATACONSUM", "BPCL", "SBILIFE", "HDFCLIFE", "LTIM", "BAJAJ-AUTO",
    "SHRIRAMFIN", "TRENT", "BEL", "ZOMATO", "DLF", "VBL", "PIDILITIND",
    "SIEMENS", "ABB", "HAL", "IRCTC", "PFC", "RECLTD", "TVSMOTOR",
    "GODREJCP", "DABUR", "HAVELLS", "AMBUJACEM", "SHREECEM", "TORNTPHARM",
    "MANKIND", "POLYCAB", "PERSISTENT", "COFORGE", "MPHASIS", "DIXON",
    "CGPOWER", "SUZLON", "BHEL", "NHPC", "IOC", "GAIL", "VEDL", "JINDALSTEL",
    "SAIL", "NMDC", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "IDFCFIRSTB",
    "FEDERALBNK", "AUBANK", "CHOLAFIN", "MUTHOOTFIN", "LICHSGFIN", "PAGEIND",
    "NAUKRI", "PAYTM", "POLICYBZR", "DMART", "BERGEPAINT", "MARICO", "COLPAL",
]


# ----------------------------------------------------------------------------
# US market universe
# ----------------------------------------------------------------------------

SP500_CSV = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
             "main/data/constituents.csv")

FALLBACK_US_UNIVERSE = [
    # S&P 100 core names as fallback
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO",
    "TSLA", "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK",
    "ABBV", "ORCL", "CVX", "CRM", "BAC", "KO", "NFLX", "AMD", "PEP", "TMO",
    "ADBE", "WMT", "LIN", "ACN", "MCD", "CSCO", "ABT", "INTU", "QCOM", "IBM",
    "GE", "CAT", "TXN", "DIS", "VZ", "AMGN", "PFE", "PM", "DHR", "NOW",
    "GS", "NEE", "UNP", "SPGI", "CMCSA", "RTX", "HON", "LOW", "T", "AXP",
    "UBER", "BKNG", "ISRG", "COP", "ELV", "SYK", "MS", "PLD", "BLK", "VRTX",
    "MDT", "SCHW", "LMT", "TJX", "C", "REGN", "ADP", "CB", "PGR", "MMC",
    "DE", "BSX", "ETN", "CI", "PANW", "MU", "SO", "BA", "FI", "MO",
    "KLAC", "DUK", "ICE", "SHW", "WM", "GD", "EMR", "APH", "PNC", "MSI",
]

MARKETS = {
    "IN": dict(currency="₹", min_value_traded=1e7, suffix=".NS"),
    "US": dict(currency="$", min_value_traded=5e6, suffix=""),
}


def load_us_universe():
    """S&P 500 constituents from the open datasets repo, with fallback."""
    if requests is not None:
        try:
            r = requests.get(SP500_CSV, timeout=15,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            syms = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
            print(f"  Loaded S&P 500: {len(syms)} symbols")
            return sorted(set(syms))
        except Exception as e:
            print(f"  Could not load S&P 500 list: {e}")
    print("  Falling back to built-in S&P 100 universe.")
    return sorted(FALLBACK_US_UNIVERSE)


def load_universe(custom_file=None):
    """Return list of Yahoo Finance tickers (SYMBOL.NS)."""
    if custom_file:
        with open(custom_file) as f:
            syms = [ln.strip() for ln in f if ln.strip()]
        return [s if "." in s else s + ".NS" for s in syms]

    symbols = set()
    if requests is not None:
        headers = {"User-Agent": "Mozilla/5.0"}
        for name, url in NSE_INDEX_CSVS.items():
            try:
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                col = "Symbol" if "Symbol" in df.columns else df.columns[2]
                symbols.update(df[col].astype(str).str.strip())
                print(f"  Loaded {name}: {len(df)} symbols")
            except Exception as e:
                print(f"  Could not load {name}: {e}")

    if not symbols:
        print("  Falling back to built-in Nifty 100 universe.")
        symbols = set(FALLBACK_UNIVERSE)

    return sorted(s + ".NS" for s in symbols if s and s != "nan")


# ----------------------------------------------------------------------------
# Technical indicators
# ----------------------------------------------------------------------------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close):
    line = ema(close, 12) - ema(close, 26)
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def stoch_rsi(close, n=14):
    r = rsi(close, n)
    lo, hi = r.rolling(n).min(), r.rolling(n).max()
    k = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    return k.rolling(3).mean()


def adx(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def accumulation_distribution(df):
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    return (mfm.fillna(0) * v).cumsum()


def detect_candles(df):
    """Detect notable candlestick patterns in last 3 sessions."""
    out = []
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    avg_body = body.rolling(20).mean()

    for i in range(max(len(df) - 3, 2), len(df)):
        b, r = body.iloc[i], rng.iloc[i]
        if pd.isna(r) or r == 0:
            continue
        up_wick = h.iloc[i] - max(o.iloc[i], c.iloc[i])
        dn_wick = min(o.iloc[i], c.iloc[i]) - l.iloc[i]
        bull = c.iloc[i] > o.iloc[i]

        if b / r < 0.1:
            out.append("Doji")
        elif b / r > 0.9:
            out.append("Bullish Marubozu" if bull else "Bearish Marubozu")
        if dn_wick > 2 * b and up_wick < 0.3 * b and b / r < 0.4:
            out.append("Hammer")
        if up_wick > 2 * b and dn_wick < 0.3 * b and b / r < 0.4:
            out.append("Shooting Star")
        if i >= 1:
            pb = c.iloc[i - 1] < o.iloc[i - 1]
            if pb and bull and c.iloc[i] > o.iloc[i - 1] and o.iloc[i] < c.iloc[i - 1]:
                out.append("Bullish Engulfing")
            if (not pb) and (not bull) and c.iloc[i] < o.iloc[i - 1] and o.iloc[i] > c.iloc[i - 1]:
                out.append("Bearish Engulfing")
        if i >= 2 and not pd.isna(avg_body.iloc[i]):
            big = avg_body.iloc[i]
            c1_bear = c.iloc[i - 2] < o.iloc[i - 2] and body.iloc[i - 2] > big
            c1_bull = c.iloc[i - 2] > o.iloc[i - 2] and body.iloc[i - 2] > big
            small_mid = body.iloc[i - 1] < 0.5 * big
            if c1_bear and small_mid and bull and c.iloc[i] > (o.iloc[i - 2] + c.iloc[i - 2]) / 2:
                out.append("Morning Star")
            if c1_bull and small_mid and not bull and c.iloc[i] < (o.iloc[i - 2] + c.iloc[i - 2]) / 2:
                out.append("Evening Star")
    return sorted(set(out))


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def safe(d, key, default=None):
    v = d.get(key, default)
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return default
    return v


def technical_analysis(df):
    """Return (tech_score, momentum_score, details dict)."""
    c = df["Close"]
    price = float(c.iloc[-1])
    e20, e50, e100, e200 = (float(ema(c, n).iloc[-1]) for n in (20, 50, 100, 200))
    r = float(rsi(c).iloc[-1])
    macd_line, macd_sig, macd_hist = macd(c)
    srsi = stoch_rsi(c)
    srsi_v = float(srsi.iloc[-1]) if not pd.isna(srsi.iloc[-1]) else 50.0
    adx_s, pdi, mdi = adx(df)
    adx_v = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 20.0
    atr_v = float(atr(df).iloc[-1])

    e50_series, e200_series = ema(c, 50), ema(c, 200)
    golden = bool((e50_series.iloc[-1] > e200_series.iloc[-1]) and
                  (e50_series.iloc[-60:] <= e200_series.iloc[-60:]).any())
    death = bool((e50_series.iloc[-1] < e200_series.iloc[-1]) and
                 (e50_series.iloc[-60:] >= e200_series.iloc[-60:]).any())

    vol = df["Volume"]
    avg_vol_50 = float(vol.rolling(50).mean().iloc[-1])
    vol_ratio = float(vol.iloc[-5:].mean() / avg_vol_50) if avg_vol_50 else 1.0
    ad = accumulation_distribution(df)
    ad_rising = float(ad.iloc[-1]) > float(ad.iloc[-20]) if len(ad) > 20 else False

    high_252 = float(df["High"].iloc[-252:].max())
    low_252 = float(df["Low"].iloc[-252:].min())
    resistance = float(df["High"].iloc[-60:-5].max()) if len(df) > 65 else high_252
    support = float(df["Low"].iloc[-60:-5].min()) if len(df) > 65 else low_252
    breakout = price > resistance and vol_ratio > 1.5
    dist_from_high = (high_252 - price) / high_252 * 100 if high_252 else 0

    lows = df["Low"].iloc[-60:]
    highs = df["High"].iloc[-60:]
    hh_hl = bool(lows.iloc[-20:].min() > lows.iloc[:20].min() and
                 highs.iloc[-20:].max() > highs.iloc[:20].max())

    rng20 = (df["High"].rolling(20).max() - df["Low"].rolling(20).min()) / c
    consolidating = float(rng20.iloc[-1]) < float(rng20.iloc[-60:].median()) * 0.7 if len(df) > 80 else False

    candles = detect_candles(df)

    # --- Trend score (0-100)
    trend = 0
    trend += 15 if price > e20 else 0
    trend += 15 if price > e50 else 0
    trend += 10 if price > e100 else 0
    trend += 20 if price > e200 else 0
    trend += 10 if e50 > e200 else 0
    trend += 10 if e20 > e50 else 0
    trend += min(20, adx_v * 0.6) if e50 > e200 else 0
    if golden:
        trend = min(100, trend + 5)
    if death:
        trend = max(0, trend - 15)

    # --- Price-action / volume additions folded into technical score
    tech = trend * 0.6
    tech += 12 if hh_hl else 0
    tech += 10 if breakout else (4 if consolidating else 0)
    tech += 8 if ad_rising else 0
    tech += min(10, max(0, (vol_ratio - 1) * 8))
    tech += 5 if dist_from_high < 10 else (2 if dist_from_high < 20 else 0)
    bull_candles = {"Bullish Engulfing", "Hammer", "Morning Star", "Bullish Marubozu"}
    bear_candles = {"Bearish Engulfing", "Shooting Star", "Evening Star", "Bearish Marubozu"}
    tech += 3 if bull_candles & set(candles) else 0
    tech -= 5 if bear_candles & set(candles) else 0
    tech = clamp(tech)

    # --- Momentum score
    mom = 0
    mom += 25 if 50 <= r <= 70 else (15 if 40 <= r < 50 or 70 < r <= 75 else (5 if r > 75 else 0))
    mom += 25 if float(macd_hist.iloc[-1]) > 0 else 0
    mom += 10 if float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-5]) else 0
    mom += 15 if float(macd_line.iloc[-1]) > 0 else 0
    ret_3m = (price / float(c.iloc[-63]) - 1) * 100 if len(c) > 63 else 0
    ret_6m = (price / float(c.iloc[-126]) - 1) * 100 if len(c) > 126 else 0
    ret_1m = (price / float(c.iloc[-21]) - 1) * 100 if len(c) > 21 else 0
    mom += clamp(ret_3m, 0, 15)
    mom += clamp(ret_6m / 2, 0, 10)
    mom = clamp(mom)

    trend_dir = ("Strong Uptrend" if price > e50 > e200 and adx_v > 25 else
                 "Uptrend" if price > e50 > e200 else
                 "Strong Downtrend" if price < e50 < e200 and adx_v > 25 else
                 "Downtrend" if price < e50 < e200 else "Sideways")

    details = dict(price=price, ema20=e20, ema50=e50, ema100=e100, ema200=e200,
                   rsi=r, stoch_rsi=srsi_v, adx=adx_v, atr=atr_v,
                   macd_hist=float(macd_hist.iloc[-1]), golden_cross=golden,
                   death_cross=death, vol_ratio=vol_ratio, ad_rising=ad_rising,
                   breakout=breakout, consolidating=consolidating, hh_hl=hh_hl,
                   support=support, resistance=resistance, high_252=high_252,
                   low_252=low_252, trend_dir=trend_dir, candles=candles,
                   ret_3m=ret_3m, ret_6m=ret_6m, ret_1m=ret_1m,
                   avg_value_traded=avg_vol_50 * price)
    return clamp(tech), mom, details


def fundamental_analysis(info):
    """Return (fund_score, quality_score, risk_score, details)."""
    pe = safe(info, "trailingPE")
    fpe = safe(info, "forwardPE")
    peg = safe(info, "pegRatio") or safe(info, "trailingPegRatio")
    pb = safe(info, "priceToBook")
    ev_ebitda = safe(info, "enterpriseToEbitda")
    rev_g = safe(info, "revenueGrowth")
    earn_g = safe(info, "earningsGrowth") or safe(info, "earningsQuarterlyGrowth")
    de = safe(info, "debtToEquity")            # yfinance reports in %
    cr = safe(info, "currentRatio")
    fcf = safe(info, "freeCashflow")
    roe = safe(info, "returnOnEquity")
    op_m = safe(info, "operatingMargins")
    np_m = safe(info, "profitMargins")
    inst = safe(info, "heldPercentInstitutions")
    insider = safe(info, "heldPercentInsiders")  # proxy for promoter holding
    beta = safe(info, "beta")
    mcap = safe(info, "marketCap", 0)

    fund = 50.0
    # Valuation (±20)
    if pe is not None:
        fund += 10 if 0 < pe < 25 else (5 if pe < 40 else (-8 if pe > 80 or pe <= 0 else -3))
    if peg is not None and peg > 0:
        fund += 8 if peg < 1.5 else (-4 if peg > 3 else 0)
    if ev_ebitda is not None and ev_ebitda > 0:
        fund += 4 if ev_ebitda < 15 else (-3 if ev_ebitda > 30 else 0)
    # Growth (±20)
    if rev_g is not None:
        fund += 10 if rev_g > 0.15 else (5 if rev_g > 0.08 else (-6 if rev_g < 0 else 0))
    if earn_g is not None:
        fund += 12 if earn_g > 0.20 else (6 if earn_g > 0.10 else (-8 if earn_g < 0 else 0))
    # Balance sheet (±15)
    if de is not None:
        fund += 8 if de < 50 else (3 if de < 100 else (-8 if de > 200 else -3))
    if cr is not None:
        fund += 3 if cr > 1.2 else (-3 if cr < 0.8 else 0)
    if fcf is not None:
        fund += 4 if fcf > 0 else -5
    fund = clamp(fund)

    quality = 50.0
    if roe is not None:
        quality += 20 if roe > 0.18 else (10 if roe > 0.12 else (-15 if roe < 0.05 else 0))
    if op_m is not None:
        quality += 10 if op_m > 0.18 else (5 if op_m > 0.10 else (-8 if op_m < 0.03 else 0))
    if np_m is not None:
        quality += 8 if np_m > 0.12 else (-8 if np_m < 0.02 else 0)
    if insider is not None:
        quality += 8 if insider > 0.40 else (4 if insider > 0.25 else 0)
    if inst is not None:
        quality += 5 if inst > 0.15 else 0
    quality = clamp(quality)

    # Risk score: 100 = low risk
    risk = 60.0
    if de is not None:
        risk += 12 if de < 50 else (-15 if de > 200 else 0)
    if beta is not None:
        risk += 8 if beta < 1.0 else (-8 if beta > 1.5 else 0)
    if mcap:
        risk += 12 if mcap > 5e11 else (6 if mcap > 1e11 else (-8 if mcap < 2e10 else 0))
    if fcf is not None and fcf < 0:
        risk -= 8
    risk = clamp(risk)

    details = dict(pe=pe, forward_pe=fpe, peg=peg, pb=pb, ev_ebitda=ev_ebitda,
                   revenue_growth=rev_g, earnings_growth=earn_g, debt_to_equity=de,
                   current_ratio=cr, fcf=fcf, roe=roe, operating_margin=op_m,
                   net_margin=np_m, institutional=inst, promoter_proxy=insider,
                   beta=beta, market_cap=mcap)
    return fund, quality, risk, details


def entry_timing(t, rec, target, stop):
    """Judge whether NOW is a good entry, or the move is already extended.

    Logic a trader actually uses:
    - Extension: how many ATRs is price above the 20/50 EMA? Buying >3 ATRs
      above the 20EMA is chasing.
    - Recent run-up: if the stock already rallied hard in 1 month, mean
      reversion risk is high.
    - Overbought: RSI > 75 argues for waiting for a pullback.
    - Risk-reward at *current* price: (target - price) / (price - stop).
      Below 1.5:1 the trade isn't worth taking here even if the stock is great.
    - Proximity to support/breakout level: entries near the 20/50 EMA or a
      just-broken resistance (now support) are the highest-quality entries.
    """
    price, atr_v = t["price"], max(t["atr"], 1e-9)
    ext20 = (price - t["ema20"]) / atr_v
    ext50 = (price - t["ema50"]) / atr_v
    ret_1m = t.get("ret_1m", 0)
    rr = (target - price) / max(price - stop, 1e-9) if target > price else 0

    score = 50.0
    score += 15 if ext20 < 1.5 else (-20 if ext20 > 3 else 0)
    score += 10 if ext50 < 3 else (-10 if ext50 > 5 else 0)
    score += 10 if t["rsi"] < 65 else (-15 if t["rsi"] > 75 else 0)
    score += 10 if ret_1m < 10 else (-15 if ret_1m > 25 else 0)
    score += 15 if rr >= 2 else (5 if rr >= 1.5 else -20)
    score += 10 if t["breakout"] else 0          # fresh breakout = timely
    score += 5 if t["consolidating"] else 0      # base = good entry area
    score = clamp(score)

    # Ideal entry zone: pullback toward 20EMA, floored by breakout/support
    zone_hi = round(max(t["ema20"], t["resistance"] * 1.005), 2)
    zone_lo = round(max(t["ema50"], stop * 1.02), 2)
    if zone_lo > zone_hi:
        zone_lo, zone_hi = zone_hi * 0.97, zone_hi

    if rec not in ("STRONG BUY", "BUY"):
        signal = "NO ENTRY"
    elif score >= 65 and rr >= 1.5:
        signal = "ENTER NOW"
    elif score >= 45:
        signal = "BUY ON DIPS"
    else:
        signal = "WAIT — EXTENDED"

    return dict(
        entry_signal=signal,
        entry_score=round(score, 1),
        entry_zone_low=zone_lo,
        entry_zone_high=zone_hi,
        risk_reward=round(rr, 2),
        extension_atr=round(ext20, 1),
        entry_note=(
            f"Price is {ext20:.1f} ATRs above 20EMA, RSI {t['rsi']:.0f}, "
            f"1-month move {ret_1m:+.0f}%, risk-reward {rr:.1f}:1 at current price. "
            + {"ENTER NOW": "Entry conditions favorable at current levels.",
               "BUY ON DIPS": f"Acceptable but better entry in the "
                              f"{zone_lo}–{zone_hi} pullback zone.",
               "WAIT — EXTENDED": f"Move already extended — wait for a pullback "
                                  f"toward {zone_lo}–{zone_hi} or a new base.",
               "NO ENTRY": "Not a buy candidate."}[signal]))


def recommend(tech, fund, mom, quality, risk, t, f):
    """Fund-manager overlay: both pillars must agree for STRONG BUY."""
    overall = 0.35 * fund + 0.35 * tech + 0.15 * mom + 0.10 * quality + 0.05 * risk

    strong_fund = (fund >= 70 and quality >= 60 and
                   (f["earnings_growth"] or 0) > 0.10 and
                   (f["debt_to_equity"] is None or f["debt_to_equity"] < 150))
    strong_tech = (tech >= 70 and t["price"] > t["ema50"] and
                   t["price"] > t["ema200"] and mom >= 55 and not t["death_cross"])

    if strong_fund and strong_tech and overall >= 72 and risk >= 45:
        rec = "STRONG BUY"
    elif fund >= 60 and tech >= 55 and overall >= 62:
        rec = "BUY"
    elif fund < 40 or (f["earnings_growth"] is not None and f["earnings_growth"] < -0.15) \
            or (f["debt_to_equity"] or 0) > 300 or (tech < 30 and mom < 30):
        rec = "AVOID"
    else:
        rec = "HOLD"

    # Expert overrides: one strong pillar alone caps at HOLD
    if tech >= 75 and fund < 50 and rec in ("BUY", "STRONG BUY"):
        rec = "HOLD"
    if fund >= 75 and tech < 45 and rec in ("BUY", "STRONG BUY"):
        rec = "HOLD"
    return rec, round(overall, 1)


def targets(t, rec):
    price, atr_v = t["price"], t["atr"]
    if rec in ("STRONG BUY", "BUY"):
        target = max(t["resistance"] * 1.02, price + 3 * atr_v)
        if t["high_252"] > price:
            target = max(target, t["high_252"])
        stop = max(t["support"], price - 2 * atr_v, t["ema50"] * 0.97)
        stop = min(stop, price * 0.97)
    else:
        target, stop = price, price * 0.92
    return round(target, 2), round(stop, 2)


# ----------------------------------------------------------------------------
# Per-stock pipeline
# ----------------------------------------------------------------------------

def analyze(symbol, min_value_traded=MIN_AVG_VALUE_TRADED, cutoff=None):
    """cutoff: optional 'YYYY-MM-DD' — truncate history there (for backtesting;
    note fundamentals are always current, so backtests are technical-only)."""
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period="5y" if cutoff else HISTORY_PERIOD, auto_adjust=True)
        if cutoff is not None and df is not None and not df.empty:
            df = df.loc[:cutoff]
        if df is None or len(df) < 220:
            return None
        df = df.dropna(subset=["Close", "Volume"])
        price = float(df["Close"].iloc[-1])
        if price < MIN_PRICE:
            return None

        tech, mom, t = technical_analysis(df)
        if t["avg_value_traded"] < min_value_traded:
            return None  # illiquid

        info = {}
        try:
            info = tk.info or {}
        except Exception:
            pass
        fund, quality, risk, f = fundamental_analysis(info)
        rec, overall = recommend(tech, fund, mom, quality, risk, t, f)
        target, stop = targets(t, rec)
        entry = entry_timing(t, rec, target, stop)

        strengths, risks = [], []
        if (f["roe"] or 0) > 0.15: strengths.append(f"High ROE ({f['roe']*100:.0f}%)")
        if (f["earnings_growth"] or 0) > 0.15: strengths.append(f"Strong earnings growth ({f['earnings_growth']*100:.0f}%)")
        if (f["debt_to_equity"] is not None and f["debt_to_equity"] < 50): strengths.append("Low debt")
        if t["trend_dir"].endswith("Uptrend"): strengths.append(t["trend_dir"])
        if t["breakout"]: strengths.append("Volume breakout above resistance")
        if t["golden_cross"]: strengths.append("Recent golden cross")
        if t["ad_rising"]: strengths.append("Accumulation pattern (A/D rising)")
        if (f["debt_to_equity"] or 0) > 150: risks.append("Elevated leverage")
        if (f["pe"] or 0) > 60: risks.append(f"Rich valuation (PE {f['pe']:.0f})")
        if t["rsi"] > 75: risks.append("Overbought (RSI > 75)")
        if t["death_cross"]: risks.append("Death cross")
        if (f["earnings_growth"] is not None and f["earnings_growth"] < 0): risks.append("Declining earnings")
        if t["trend_dir"].endswith("Downtrend"): risks.append("Bearish trend structure")
        if not strengths: strengths.append("None notable")
        if not risks: risks.append("Standard market risk")

        thesis = (f"{t['trend_dir']}; price {price:.1f} vs 50EMA {t['ema50']:.1f} / "
                  f"200EMA {t['ema200']:.1f}. Fundamentals score {fund:.0f}/100 "
                  f"(ROE {(f['roe'] or 0)*100:.0f}%, earnings growth "
                  f"{(f['earnings_growth'] or 0)*100:.0f}%). "
                  + ("Both pillars aligned — institutional-quality setup."
                     if rec == "STRONG BUY" else
                     "Constructive but awaiting fuller confirmation." if rec == "BUY" else
                     "Mixed signals; no edge at current levels." if rec == "HOLD" else
                     "Deteriorating fundamentals/technicals; capital better deployed elsewhere."))

        # Data quality: how many key fundamental fields Yahoo actually returned
        key_fields = ["pe", "peg", "revenue_growth", "earnings_growth",
                      "debt_to_equity", "current_ratio", "fcf", "roe",
                      "operating_margin", "net_margin", "beta", "market_cap"]
        dq = sum(1 for k in key_fields if f.get(k) is not None)
        if dq <= 4 and rec == "STRONG BUY":
            rec = "BUY"
            thesis += " Downgraded: insufficient fundamental data from Yahoo to justify highest conviction."

        # 52-week range position: 0 = at low, 100 = at high
        w52 = round(100 * (price - t["low_252"]) / max(t["high_252"] - t["low_252"], 1e-9), 1)

        # Score drivers: human-readable why-it-scored-this-way
        drivers = []
        drivers.append(f"{'✓' if t['price'] > t['ema200'] else '✗'} price vs 200EMA")
        drivers.append(f"{'✓' if t['price'] > t['ema50'] else '✗'} price vs 50EMA")
        drivers.append(f"{'✓' if t['ema50'] > t['ema200'] else '✗'} 50>200 EMA structure")
        if t["golden_cross"]: drivers.append("✓ golden cross (recent)")
        if t["death_cross"]: drivers.append("✗ death cross (recent)")
        drivers.append(f"{'✓' if t['hh_hl'] else '✗'} higher highs & lows")
        drivers.append(f"{'✓' if t['ad_rising'] else '✗'} accumulation (A/D)")
        if t["breakout"]: drivers.append("✓ volume breakout")
        drivers.append(f"{'✓' if 50 <= t['rsi'] <= 70 else '✗'} RSI healthy ({t['rsi']:.0f})")
        drivers.append(f"{'✓' if t['macd_hist'] > 0 else '✗'} MACD positive")
        if f["pe"] is not None:
            drivers.append(f"{'✓' if 0 < f['pe'] < 40 else '✗'} PE {f['pe']:.0f}")
        if f["earnings_growth"] is not None:
            drivers.append(f"{'✓' if f['earnings_growth'] > 0.10 else '✗'} "
                           f"earnings growth {f['earnings_growth']*100:.0f}%")
        if f["revenue_growth"] is not None:
            drivers.append(f"{'✓' if f['revenue_growth'] > 0.08 else '✗'} "
                           f"revenue growth {f['revenue_growth']*100:.0f}%")
        if f["debt_to_equity"] is not None:
            drivers.append(f"{'✓' if f['debt_to_equity'] < 100 else '✗'} "
                           f"D/E {f['debt_to_equity']:.0f}")
        if f["roe"] is not None:
            drivers.append(f"{'✓' if f['roe'] > 0.12 else '✗'} ROE {f['roe']*100:.0f}%")

        return {
            "symbol": symbol.replace(".NS", ""),
            "company_name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
            "current_price": round(price, 2),
            "technical_score": round(tech, 1),
            "fundamental_score": round(fund, 1),
            "momentum_score": round(mom, 1),
            "quality_score": round(quality, 1),
            "risk_score": round(risk, 1),
            "overall_score": overall,
            "recommendation": rec,
            "target_price": target,
            "upside_percent": round((target / price - 1) * 100, 1),
            "stop_loss": stop,
            **entry,
            "week52_position": w52,
            "data_quality": f"{dq}/{len(key_fields)}",
            "data_quality_pct": round(100 * dq / len(key_fields)),
            "score_drivers": drivers,
            "technical_summary": (f"{t['trend_dir']}, RSI {t['rsi']:.0f}, ADX {t['adx']:.0f}, "
                                  f"MACD hist {t['macd_hist']:+.2f}, vol {t['vol_ratio']:.1f}x avg"
                                  + (f", patterns: {', '.join(t['candles'])}" if t["candles"] else "")),
            "fundamental_summary": (f"PE {f['pe'] if f['pe'] else 'n/a'}, PEG {f['peg'] if f['peg'] else 'n/a'}, "
                                    f"ROE {(f['roe'] or 0)*100:.0f}%, D/E {f['debt_to_equity'] if f['debt_to_equity'] is not None else 'n/a'}, "
                                    f"rev growth {(f['revenue_growth'] or 0)*100:.0f}%"),
            "investment_thesis": thesis,
            "key_strengths": strengths[:5],
            "key_risks": risks[:5],
            "is_breakout": t["breakout"],
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "_growth": (f["earnings_growth"] or 0) + (f["revenue_growth"] or 0),
            "_value": (1 / f["pe"] if f["pe"] and f["pe"] > 0 else 0) + (quality / 200),
        }
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Sector strength + output
# ----------------------------------------------------------------------------

def sector_adjust(results):
    """Boost/penalize by sector relative strength (avg 3m return)."""
    df = pd.DataFrame(results)
    sec_strength = df.groupby("sector")["momentum_score"].mean().to_dict()
    market_avg = df["momentum_score"].mean()
    for r in results:
        rel = sec_strength.get(r["sector"], market_avg) - market_avg
        r["sector_relative_strength"] = round(rel, 1)
        if rel < -10 and r["recommendation"] == "STRONG BUY":
            r["recommendation"] = "BUY"  # weak sector caps conviction
            r["investment_thesis"] += " Downgraded: sector underperforming market."
    return results, sec_strength


def write(name, obj):
    with open(os.path.join(OUTPUT_DIR, name), "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"  wrote {name} ({len(obj) if isinstance(obj, list) else 'summary'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", help="file of symbols, one per line")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Building universe...")
    universe = load_universe(args.universe)
    if args.limit:
        universe = universe[: args.limit]
    print(f"Scanning {len(universe)} stocks...\n")

    results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(analyze, s): s for s in universe}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r:
                results.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(universe)} processed, {len(results)} passed filters "
                      f"({time.time()-start:.0f}s)")

    if not results:
        sys.exit("No stocks passed filters — check network access to Yahoo Finance.")

    results, sec_strength = sector_adjust(results)
    results.sort(key=lambda r: r["overall_score"], reverse=True)
    for r in results:
        r.pop("_growth_", None)

    by_rec = lambda rec: [r for r in results if r["recommendation"] == rec]
    strong_buy, buy = by_rec("STRONG BUY"), by_rec("BUY")
    hold, avoid = by_rec("HOLD"), by_rec("AVOID")

    growth = sorted([r for r in results if r["recommendation"] in ("STRONG BUY", "BUY")],
                    key=lambda r: r["_growth"], reverse=True)[:25]
    value = sorted([r for r in results if r["recommendation"] in ("STRONG BUY", "BUY", "HOLD")],
                   key=lambda r: r["_value"], reverse=True)[:25]
    breakouts = sorted([r for r in results if r["is_breakout"]],
                       key=lambda r: r["overall_score"], reverse=True)[:25]

    leaders = {}
    for r in results:
        leaders.setdefault(r["sector"], [])
        if len(leaders[r["sector"]]) < 3:
            leaders[r["sector"]].append(r)

    for r in results:
        r.pop("_growth", None); r.pop("_value", None)

    summary = {
        "scan_date": datetime.now(timezone.utc).isoformat(),
        "stocks_scanned": len(universe),
        "stocks_passing_liquidity_filter": len(results),
        "counts": {"STRONG_BUY": len(strong_buy), "BUY": len(buy),
                   "HOLD": len(hold), "AVOID": len(avoid)},
        "market_breadth_pct_above_200ema": round(
            100 * sum(1 for r in results if "Uptrend" in r["technical_summary"]
                      or r["technical_score"] > 55) / len(results), 1),
        "avg_overall_score": round(np.mean([r["overall_score"] for r in results]), 1),
        "strongest_sectors": sorted(sec_strength, key=sec_strength.get, reverse=True)[:5],
        "weakest_sectors": sorted(sec_strength, key=sec_strength.get)[:5],
        "top_10_overall": [{"symbol": r["symbol"], "score": r["overall_score"],
                            "rec": r["recommendation"]} for r in results[:10]],
        "disclaimer": "Screening output only. Not investment advice. Verify all data.",
    }

    print("\nWriting outputs to ./output/")
    write("all_stocks.json", results)
    write("strong_buy.json", strong_buy)
    write("buy.json", buy)
    write("hold.json", hold)
    write("avoid.json", avoid)
    write("top_growth_stocks.json", growth)
    write("top_value_stocks.json", value)
    write("top_breakout_stocks.json", breakouts)
    write("sector_leaders.json", leaders)
    write("market_summary.json", summary)

    print(f"\nDone in {time.time()-start:.0f}s. "
          f"STRONG BUY: {len(strong_buy)} | BUY: {len(buy)} | "
          f"HOLD: {len(hold)} | AVOID: {len(avoid)}")


if __name__ == "__main__":
    main()
