"""Quick audit — shape, date range, null %, memory for every table.

Run:  .venv/Scripts/python.exe scripts/audit_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data_loader import ALL_LOADERS


def main() -> None:
    rows = []
    for name, loader in ALL_LOADERS.items():
        df = loader()

        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        dmin = dmax = None
        if date_cols:
            all_dates = pd.concat([df[c] for c in date_cols])
            dmin, dmax = all_dates.min(), all_dates.max()

        null_pct = (df.isna().sum() / len(df) * 100).round(2)
        high_null = null_pct[null_pct > 0].to_dict()

        rows.append(
            {
                "table": name,
                "rows": len(df),
                "cols": len(df.columns),
                "mem_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
                "date_cols": ", ".join(date_cols) if date_cols else "-",
                "date_min": dmin.date().isoformat() if dmin is not None else "-",
                "date_max": dmax.date().isoformat() if dmax is not None else "-",
                "nulls": high_null,
            }
        )

    summary = pd.DataFrame(rows).set_index("table")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(summary[["rows", "cols", "mem_mb", "date_cols", "date_min", "date_max"]])
    print("\nColumns with nulls (pct):")
    for r in rows:
        if r["nulls"]:
            print(f"  {r['table']:<18} {r['nulls']}")

    print("\n=== Key question: which tables cover the test period 2023-01 to 2024-07? ===")
    test_start = pd.Timestamp("2023-01-01")
    test_end = pd.Timestamp("2024-07-01")
    for r in rows:
        if r["date_max"] == "-":
            continue
        dmax = pd.Timestamp(r["date_max"])
        covers = dmax >= test_start
        fully_covers = dmax >= test_end
        flag = "FULLY" if fully_covers else ("PARTIAL" if covers else "NO")
        print(f"  {r['table']:<18} date_max={r['date_max']}  -> covers test period: {flag}")


if __name__ == "__main__":
    main()
