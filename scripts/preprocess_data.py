"""Preprocessing script — mirrors datathon-data-preprocessing.ipynb.

Reads raw CSVs from data/, writes clean_*.csv to preprocessed/.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT  = ROOT / "preprocessed"
OUT.mkdir(exist_ok=True)


def main():
    print("Loading all 14 datasets...")
    sales       = pd.read_csv(DATA / "sales.csv",       parse_dates=["Date"])
    orders      = pd.read_csv(DATA / "orders.csv",      parse_dates=["order_date"])
    order_items = pd.read_csv(DATA / "order_items.csv", low_memory=False)
    products    = pd.read_csv(DATA / "products.csv")
    customers   = pd.read_csv(DATA / "customers.csv",   parse_dates=["signup_date"])
    payments    = pd.read_csv(DATA / "payments.csv")
    shipments   = pd.read_csv(DATA / "shipments.csv",   parse_dates=["ship_date", "delivery_date"])
    returns     = pd.read_csv(DATA / "returns.csv",     parse_dates=["return_date"])
    promotions  = pd.read_csv(DATA / "promotions.csv",  parse_dates=["start_date", "end_date"])
    geography   = pd.read_csv(DATA / "geography.csv")
    inventory   = pd.read_csv(DATA / "inventory.csv",   parse_dates=["snapshot_date"])
    web_traffic = pd.read_csv(DATA / "web_traffic.csv", parse_dates=["date"])
    reviews     = pd.read_csv(DATA / "reviews.csv",     parse_dates=["review_date"])
    print("Loaded successfully.\n")

    # ── 1. Promotions ──────────────────────────────────────────────────────
    promotions["applicable_category"] = promotions["applicable_category"].fillna("All_Categories")
    mask = promotions["end_date"] < promotions["start_date"]
    if mask.sum() > 0:
        promotions.loc[mask, ["start_date", "end_date"]] = (
            promotions.loc[mask, ["end_date", "start_date"]].values
        )
        print(f"[Promotions] Swapped {mask.sum()} inverted date rows.")

    # ── 2. Order Items ─────────────────────────────────────────────────────
    order_items["promo_id"] = order_items["promo_id"].fillna("NO_PROMO")
    order_items["has_stacked_promo"] = order_items["promo_id_2"].notna().astype(int)
    order_items = order_items.drop(columns=["promo_id_2"])
    print("[Order Items] Processed promo_id NaN and converted promo_id_2 to flag.")

    # ── 3. Sales ───────────────────────────────────────────────────────────
    sales["Revenue"] = sales["Revenue"].abs()
    sales["COGS"]    = sales["COGS"].abs()
    q1r, q3r = sales["Revenue"].quantile(0.01), sales["Revenue"].quantile(0.99)
    q1c, q3c = sales["COGS"].quantile(0.01),    sales["COGS"].quantile(0.99)
    sales["Revenue"] = sales["Revenue"].clip(lower=q1r, upper=q3r)
    sales["COGS"]    = sales["COGS"].clip(lower=q1c, upper=q3c)
    print("[Sales] Processed negatives and clipped 1%-99% IQR outliers.")

    # ── 4. Web Traffic ─────────────────────────────────────────────────────
    web_traffic["bounce_rate"] = web_traffic["bounce_rate"].clip(lower=0, upper=1)
    print("[Web Traffic] Clipped bounce_rate to [0, 1].")

    # ── Integrity checks ───────────────────────────────────────────────────
    assert order_items["promo_id"].isnull().sum() == 0
    assert promotions["applicable_category"].isnull().sum() == 0
    assert len(sales[(sales["Revenue"] < 0) | (sales["COGS"] < 0)]) == 0
    assert len(promotions[promotions["end_date"] < promotions["start_date"]]) == 0
    assert web_traffic["bounce_rate"].max() <= 1.0
    print("\nAll integrity checks passed.")

    # ── Export ─────────────────────────────────────────────────────────────
    tables = {
        "sales": sales, "order_items": order_items, "products": products,
        "shipments": shipments, "promotions": promotions, "inventory": inventory,
        "web_traffic": web_traffic, "orders": orders, "customers": customers,
        "payments": payments, "returns": returns, "geography": geography,
        "reviews": reviews,
    }
    print("\nExporting clean datasets...")
    for name, df in tables.items():
        path = OUT / f"clean_{name}.csv"
        df.to_csv(path, index=False)
        print(f"  clean_{name}.csv  shape={df.shape}")

    print(f"\nAll {len(tables)} tables exported to {OUT}/")


if __name__ == "__main__":
    main()
