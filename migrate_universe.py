"""
One-shot migration: build universe_enriched.csv from current universe.csv +
dedup_map.json + yfinance metadata.

Output is written to `universe_enriched.csv` (NOT universe.csv) so the result
can be reviewed before swapping in.

Usage:
    python migrate_universe.py [--limit N] [--no-fetch]

Flags:
    --limit N    Only process first N symbols (for quick testing).
    --no-fetch   Skip yfinance enrichment (rules will run on metadata-less rows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from universe_rules import apply_rules, enrich_metadata, seed_classify

ROOT = Path(__file__).parent
SRC_CSV = ROOT / "universe.csv"
DEDUP_JSON = ROOT / "dedup_map.json"
OUT_CSV = ROOT / "universe_enriched.csv"


def progress_cb(i: int, total: int, sym: str) -> None:
    if i % 10 == 0 or i == total - 1:
        print(f"  [{i+1:>3}/{total}] {sym}", flush=True)


def main(limit: int | None, do_fetch: bool) -> int:
    if not SRC_CSV.exists():
        print(f"ERROR: {SRC_CSV} not found", file=sys.stderr)
        return 1

    src = pd.read_csv(SRC_CSV)
    print(f"Loaded {len(src)} rows from universe.csv")

    dedup = {}
    if DEDUP_JSON.exists():
        with open(DEDUP_JSON, "r") as f:
            dedup = json.load(f)
        print(f"Loaded {len(dedup)} dedup mappings")

    if limit:
        src = src.head(limit).copy()
        print(f"Limiting to first {limit} rows")

    # 1. Build base frame with seed classification
    rows = []
    for _, r in src.iterrows():
        sym = str(r["Symbol"]).strip().upper()
        name = str(r["Description"])
        focus = str(r["Focus"])
        ac, theme = seed_classify(sym, name, focus)
        rows.append({
            "symbol": sym,
            "name": name,
            "asset_class": ac.value,
            "theme": theme,
            "canonical_for": dedup.get(sym),  # if sym is in dedup_map, points to canonical
        })
    df = pd.DataFrame(rows)
    print(f"Seeded classification: {df['asset_class'].value_counts().to_dict()}")

    # 2. Enrich via yfinance
    if do_fetch:
        print(f"\nFetching yfinance metadata for {len(df)} symbols (this may take a few minutes)...")
        enriched = enrich_metadata(df["symbol"].tolist(), progress=progress_cb)
        df = df.merge(enriched, left_on="symbol", right_index=True, how="left")
        n_aum = df["aum"].notna().sum()
        n_inc = df["inception_date"].notna().sum()
        print(f"  Got AUM for {n_aum}/{len(df)} symbols")
        print(f"  Got inception for {n_inc}/{len(df)} symbols")
    else:
        for col in ["aum", "avg_volume_30d", "inception_date", "expense_ratio",
                    "is_leveraged", "is_single_stock"]:
            df[col] = pd.NA
        df["is_leveraged"] = False
        df["is_single_stock"] = False
        print("Skipping yfinance fetch (--no-fetch)")

    # 3. Manual override column (none initially)
    df["override"] = False

    # 4. Apply rules → populate included / exclude_reason
    df = apply_rules(df, preserve_overrides=False)

    # 5. Reorder columns for readability
    cols = [
        "symbol", "name", "asset_class", "theme",
        "included", "exclude_reason",
        "canonical_for",
        "aum", "avg_volume_30d", "inception_date", "expense_ratio",
        "is_leveraged", "is_single_stock",
        "override",
    ]
    df = df[cols]

    # 6. Write
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")

    # 7. Summary
    print("\n-- Summary --")
    print(f"Included:  {int(df['included'].sum())}")
    print(f"Excluded:  {int((~df['included']).sum())}")
    if (~df["included"]).any():
        print("\nExclusion reasons:")
        for reason, count in df.loc[~df["included"], "exclude_reason"].value_counts().items():
            print(f"  {reason:>20} : {count}")
    print("\nIncluded by asset_class:")
    inc = df[df["included"]]
    for ac, count in inc["asset_class"].value_counts().items():
        print(f"  {ac:>20} : {count}")
    print("\nIncluded by theme:")
    for th, count in inc["theme"].value_counts().items():
        print(f"  {th:>20} : {count}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()
    sys.exit(main(limit=args.limit, do_fetch=not args.no_fetch))
