# India Stock Scanner (Streamlit)

Technical + fundamental screener for NSE stocks using Yahoo Finance.
Screening tool only — not investment advice.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public repo)
2. Go to https://share.streamlit.io → New app
3. Pick your repo, branch `main`, main file `app.py`
4. Deploy

## Files
- `app.py` — Streamlit UI
- `scanner.py` — analysis engine (also runnable via CLI: `python scanner.py --limit 50`)

## Notes
- Streamlit Cloud free tier has ~1 GB RAM and no persistent storage; keep scans ≤ ~200 stocks there. Results cache for 1 hour.
- Yahoo occasionally rate-limits cloud IPs; if scans return few results, lower workers or retry later.
