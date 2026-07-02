"""Shared scanner UI — parametrized by market so India and US pages stay identical."""
import json
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

import scanner as sc


def render_market(market: str):
    cfg = sc.MARKETS[market]
    cur, suffix = cfg["currency"], cfg["suffix"]
    flag = "🇮🇳" if market == "IN" else "🇺🇸"
    name = "Indian" if market == "IN" else "US"
    key = market  # session-state namespace so both pages keep separate results

    st.title(f"{flag} {name} Stock Market Scanner")
    st.caption("Technical + fundamental screening via Yahoo Finance. "
               "**Screening tool only — not investment advice.**")

    # ------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Scan settings")
        if market == "IN":
            options = ["Nifty 100 (built-in, fast)",
                       "Nifty 500 + Mid/Smallcap (full, slow)", "Custom list"]
        else:
            options = ["S&P 100 (built-in, fast)", "S&P 500 (full, slow)", "Custom list"]
        universe_choice = st.selectbox("Universe", options, key=f"{key}_uni")
        custom_syms = ""
        if universe_choice == "Custom list":
            default = "RELIANCE\nTCS\nHDFCBANK" if market == "IN" else "AAPL\nMSFT\nNVDA"
            custom_syms = st.text_area("Symbols (one per line)", default, key=f"{key}_syms")
        limit = st.slider("Max stocks to scan", 10, 900, 100, 10, key=f"{key}_limit",
                          help="Streamlit Cloud has limited resources; large scans take long.")
        workers = st.slider("Parallel workers", 1, 8, 4, key=f"{key}_workers")
        min_rec = st.multiselect("Show recommendations",
                                 ["STRONG BUY", "BUY", "HOLD", "AVOID"],
                                 default=["STRONG BUY", "BUY", "HOLD", "AVOID"],
                                 key=f"{key}_recs")
        run = st.button("🚀 Run scan", type="primary", use_container_width=True,
                        key=f"{key}_run")

    def build_universe():
        if universe_choice == "Custom list":
            syms = [s.strip().upper() for s in custom_syms.splitlines() if s.strip()]
            return [s if ("." in s or not suffix) else s + suffix for s in syms]
        if market == "IN":
            if universe_choice.startswith("Nifty 100"):
                return sorted(s + ".NS" for s in sc.FALLBACK_UNIVERSE)
            return sc.load_universe()
        if universe_choice.startswith("S&P 100"):
            return sorted(sc.FALLBACK_US_UNIVERSE)
        return sc.load_us_universe()

    @st.cache_data(ttl=3600, show_spinner=False)
    def run_scan(symbols, n_workers, market_key):
        results = []
        prog = st.progress(0.0, text="Scanning…")
        min_val = sc.MARKETS[market_key]["min_value_traded"]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(sc.analyze, s, min_val): s for s in symbols}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                if r:
                    results.append(r)
                prog.progress(i / len(symbols),
                              text=f"Scanning… {i}/{len(symbols)} "
                                   f"({len(results)} passed filters)")
        prog.empty()
        if results:
            results, _ = sc.sector_adjust(results)
            results.sort(key=lambda r: r["overall_score"], reverse=True)
            for r in results:
                r.pop("_growth", None)
                r.pop("_value", None)
        return results

    # ------------------------------------------------------------ main
    if run:
        universe = build_universe()[:limit]
        st.info(f"Scanning {len(universe)} stocks — this may take a few minutes.")
        st.session_state[f"{key}_results"] = run_scan(tuple(universe), workers, market)
        st.session_state[f"{key}_time"] = datetime.now(timezone.utc).isoformat()

    results = st.session_state.get(f"{key}_results")
    if not results:
        fast = "Nifty 100" if market == "IN" else "S&P 100"
        st.markdown(f"Configure a scan in the sidebar and hit **Run scan**. "
                    f"Start with {fast} to get results in ~2–5 minutes.")
        st.stop()

    df = pd.DataFrame(results)
    df = df[df["recommendation"].isin(min_rec)]

    c1, c2, c3, c4, c5 = st.columns(5)
    counts = pd.DataFrame(results)["recommendation"].value_counts()
    c1.metric("Scanned & passed", len(results))
    c2.metric("STRONG BUY", int(counts.get("STRONG BUY", 0)))
    c3.metric("BUY", int(counts.get("BUY", 0)))
    c4.metric("HOLD", int(counts.get("HOLD", 0)))
    c5.metric("AVOID", int(counts.get("AVOID", 0)))
    st.caption(f"Last scan: {st.session_state.get(f'{key}_time', '')} UTC · cached 1h")

    tab_table, tab_top, tab_sector, tab_detail, tab_export = st.tabs(
        ["📊 All results", "🏆 Top picks", "🏭 Sectors", "🔍 Stock detail", "⬇️ Export"])

    with tab_table:
        show_cols = ["symbol", "company_name", "sector", "current_price",
                     "recommendation", "overall_score", "technical_score",
                     "fundamental_score", "momentum_score", "quality_score",
                     "risk_score", "target_price", "upside_percent", "stop_loss"]
        score_col = lambda label: st.column_config.ProgressColumn(
            label, min_value=0, max_value=100, format="%.0f")
        st.dataframe(
            df[show_cols], use_container_width=True, height=600, hide_index=True,
            column_config={
                "overall_score": score_col("Overall"),
                "technical_score": score_col("Technical"),
                "fundamental_score": score_col("Fundamental"),
                "momentum_score": score_col("Momentum"),
                "quality_score": score_col("Quality"),
                "risk_score": score_col("Risk"),
                "current_price": st.column_config.NumberColumn("Price", format=f"{cur}%.2f"),
                "target_price": st.column_config.NumberColumn("Target", format=f"{cur}%.2f"),
                "stop_loss": st.column_config.NumberColumn("Stop", format=f"{cur}%.2f"),
                "upside_percent": st.column_config.NumberColumn("Upside", format="%.1f%%"),
            })

    with tab_top:
        left, right = st.columns(2)
        with left:
            st.subheader("🚀 Top growth")
            g = sorted([r for r in results if r["recommendation"] in ("STRONG BUY", "BUY")],
                       key=lambda r: r["fundamental_score"] + r["momentum_score"],
                       reverse=True)[:10]
            for r in g:
                st.markdown(f"**{r['symbol']}** · {r['recommendation']} · "
                            f"score {r['overall_score']} — {r['fundamental_summary']}")
        with right:
            st.subheader("💥 Breakouts")
            b = [r for r in results if r.get("is_breakout")]
            if not b:
                st.write("No volume breakouts detected in this scan.")
            for r in b[:10]:
                st.markdown(f"**{r['symbol']}** · {r['recommendation']} — "
                            f"{r['technical_summary']}")

    with tab_sector:
        sec = pd.DataFrame(results).groupby("sector").agg(
            stocks=("symbol", "count"),
            avg_score=("overall_score", "mean"),
            avg_momentum=("momentum_score", "mean"),
        ).round(1).sort_values("avg_score", ascending=False)
        st.bar_chart(sec["avg_score"])
        st.dataframe(sec, use_container_width=True)

    with tab_detail:
        pick = st.selectbox("Select stock", df["symbol"].tolist(), key=f"{key}_pick")
        r = next(x for x in results if x["symbol"] == pick)
        a, b = st.columns([1, 2])
        with a:
            st.metric("Price", f"{cur}{r['current_price']}")
            st.metric("Recommendation", r["recommendation"])
            st.metric("Overall score", r["overall_score"])
            st.metric("Target / Stop",
                      f"{cur}{r['target_price']} / {cur}{r['stop_loss']}",
                      f"{r['upside_percent']}% upside")
        with b:
            st.markdown(f"**Thesis:** {r['investment_thesis']}")
            st.markdown(f"**Technical:** {r['technical_summary']}")
            st.markdown(f"**Fundamental:** {r['fundamental_summary']}")
            st.markdown("**Strengths:** " + "; ".join(r["key_strengths"]))
            st.markdown("**Risks:** " + "; ".join(r["key_risks"]))
        scores = {k.replace("_score", ""): r[k] for k in
                  ["technical_score", "fundamental_score", "momentum_score",
                   "quality_score", "risk_score"]}
        st.bar_chart(pd.Series(scores))

    with tab_export:
        prefix = market.lower()
        st.download_button(f"{prefix}_all_stocks.json",
                           json.dumps(results, indent=2, default=str),
                           f"{prefix}_all_stocks.json", "application/json",
                           key=f"{key}_dl_all")
        for rec in ["STRONG BUY", "BUY", "HOLD", "AVOID"]:
            sub = [r for r in results if r["recommendation"] == rec]
            fn = f"{prefix}_{rec.lower().replace(' ', '_')}.json"
            st.download_button(f"{fn} ({len(sub)})",
                               json.dumps(sub, indent=2, default=str),
                               fn, "application/json", key=f"{key}_{rec}")
        st.download_button(f"{prefix}_results.csv", df.to_csv(index=False),
                           f"{prefix}_results.csv", "text/csv", key=f"{key}_csv")

    st.divider()
    st.caption("Data: Yahoo Finance (may be delayed). Automated screening only — "
               "not investment advice.")
