"""Single source of truth for loading the DATATHON 2026 data.

Each function returns a pandas DataFrame with dtypes already parsed.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _read(file: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / file, parse_dates=parse_dates, low_memory=False)


def load_products() -> pd.DataFrame:
    return _read("products.csv")


def load_customers() -> pd.DataFrame:
    return _read("customers.csv", parse_dates=["signup_date"])


def load_promotions() -> pd.DataFrame:
    return _read("promotions.csv", parse_dates=["start_date", "end_date"])


def load_geography() -> pd.DataFrame:
    return _read("geography.csv")


def load_orders() -> pd.DataFrame:
    return _read("orders.csv", parse_dates=["order_date"])


def load_order_items() -> pd.DataFrame:
    return _read("order_items.csv")


def load_payments() -> pd.DataFrame:
    return _read("payments.csv")


def load_shipments() -> pd.DataFrame:
    return _read("shipments.csv", parse_dates=["ship_date", "delivery_date"])


def load_returns() -> pd.DataFrame:
    return _read("returns.csv", parse_dates=["return_date"])


def load_reviews() -> pd.DataFrame:
    return _read("reviews.csv", parse_dates=["review_date"])


def load_sales() -> pd.DataFrame:
    return _read("sales.csv", parse_dates=["Date"])


def load_sample_submission() -> pd.DataFrame:
    return _read("sample_submission.csv", parse_dates=["Date"])


def load_inventory() -> pd.DataFrame:
    return _read("inventory.csv", parse_dates=["snapshot_date"])


def load_web_traffic() -> pd.DataFrame:
    return _read("web_traffic.csv", parse_dates=["date"])


ALL_LOADERS = {
    "products": load_products,
    "customers": load_customers,
    "promotions": load_promotions,
    "geography": load_geography,
    "orders": load_orders,
    "order_items": load_order_items,
    "payments": load_payments,
    "shipments": load_shipments,
    "returns": load_returns,
    "reviews": load_reviews,
    "sales": load_sales,
    "sample_submission": load_sample_submission,
    "inventory": load_inventory,
    "web_traffic": load_web_traffic,
}
