"""Nightly scan runner for GitHub Actions — writes data/{in,us}_all_stocks.json."""
import json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import scanner as sc

os.makedirs("data", exist_ok=True)

def scan(symbols, min_val, out):
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(sc.analyze, s, min_val): s for s in symbols}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                results.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(symbols)}")
    if results:
        results, _ = sc.sector_adjust(results)
        results.sort(key=lambda r: r["overall_score"], reverse=True)
        for r in results:
            r.pop("_growth", None); r.pop("_value", None)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print(f"Wrote {out}: {len(results)} stocks")

print("India scan…")
scan(sc.load_universe(), sc.MARKETS["IN"]["min_value_traded"], "data/in_all_stocks.json")
print("US scan…")
scan(sc.load_us_universe(), sc.MARKETS["US"]["min_value_traded"], "data/us_all_stocks.json")
