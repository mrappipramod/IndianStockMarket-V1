"""Crude technical-signal backtest: score stocks as of a past date, measure
forward returns. LIMITATION: yfinance only serves *current* fundamentals, so
fundamental/quality scores use today's data — treat results as a check of the
TECHNICAL/entry-timing rules only, not the full model.

Usage: python backtest.py --months-back 6 --limit 60
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
import scanner as sc

ap = argparse.ArgumentParser()
ap.add_argument("--months-back", type=int, default=6)
ap.add_argument("--limit", type=int, default=60)
ap.add_argument("--market", choices=["IN", "US"], default="IN")
args = ap.parse_args()

cutoff = (date.today() - timedelta(days=30 * args.months_back)).isoformat()
uni = (sorted(s + ".NS" for s in sc.FALLBACK_UNIVERSE) if args.market == "IN"
       else sorted(sc.FALLBACK_US_UNIVERSE))[: args.limit]
min_val = sc.MARKETS[args.market]["min_value_traded"]

print(f"Signal date: {cutoff} | {len(uni)} stocks | market {args.market}\n")

rows = []
def one(sym):
    r = sc.analyze(sym, min_val, cutoff=cutoff)
    if not r:
        return None
    h = yf.Ticker(sym).history(start=cutoff, auto_adjust=True)["Close"]
    if len(h) < 2:
        return None
    fwd = (float(h.iloc[-1]) / float(h.iloc[0]) - 1) * 100
    return dict(symbol=r["symbol"], rec=r["recommendation"],
                entry=r["entry_signal"], tech=r["technical_score"],
                fwd_return_pct=round(fwd, 1))

with ThreadPoolExecutor(max_workers=6) as ex:
    for f in as_completed({ex.submit(one, s): s for s in uni}):
        r = f.result()
        if r:
            rows.append(r)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No results — check network.")
print(df.groupby("rec")["fwd_return_pct"].agg(["count", "mean", "median"]).round(1))
print()
print(df.groupby("entry")["fwd_return_pct"].agg(["count", "mean", "median"]).round(1))
print("\nCaveat: survivorship bias (delisted stocks absent) and current-only "
      "fundamentals both flatter these numbers.")
