# Stock Market Scanner — India 🇮🇳 + US 🇺🇸

Institutional-style technical + fundamental screener via Yahoo Finance,
with entry timing, score transparency, nightly auto-scans, and backtesting.
**Screening tool only — not investment advice.**

## Features
- Multi-page: Indian market (Nifty 500 universe) and US market (S&P 500)
- 35/35/15/10/5 weighted scoring (fundamental/technical/momentum/quality/risk)
- Fund-manager overrides: STRONG BUY needs both pillars; sector & data-quality downgrades
- **Entry timing**: ENTER NOW / BUY ON DIPS / WAIT — EXTENDED, with ideal entry zone,
  risk-reward at current price, and ATR extension
- Price charts (1y + 20/50/200 EMA), score-driver breakdowns, 52-week range position
- Data-quality badge (flags stocks where Yahoo fundamentals are missing)
- Search + sector filters, watchlist comparison, JSON/CSV exports
- **Nightly auto-scans** via GitHub Actions committed to `data/` — the app loads them
  instantly so users don't wait for live scans
- `backtest.py` — sanity-check the technical signals on historical data

## Deploy
1. Push to a public GitHub repo (include `pages/`, `.github/`, `data/` folders)
2. share.streamlit.io → New app → repo, branch `main`, file `app.py`
3. In the repo: Actions tab → enable workflows. The nightly scan runs after
   NSE (18:00 IST) and US close, committing fresh JSON the app picks up.
   Trigger manually the first time via "Run workflow".

## Local use
```bash
pip install -r requirements.txt
streamlit run app.py                       # UI
python scanner.py --limit 50               # CLI scan → output/*.json
python backtest.py --months-back 6         # technical-signal backtest
```

## Known limitations (read this)
- Yahoo fundamentals for Indian small caps are often stale/missing — the
  data-quality badge flags this; verify against exchange filings.
- No ASM/GSM surveillance flags — cross-check NSE lists before trading.
- Backtests are technical-only (historical fundamentals unavailable) and
  carry survivorship bias.
- Scoring weights and entry thresholds are sensible heuristics, not
  validated alpha. Use as a research starting point.
