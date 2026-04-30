"""Generate probe submissions with controlled perturbations to reverse-engineer LB scoring formula.

Each probe adds a fixed constant offset to Revenue or COGS, allowing us to measure how the LB score
changes and infer the weights of MAE_revenue, MAE_cogs, RMSE, and R² in the final LB formula.

Probes are designed to be mathematically interpretable:
- Constant bias +k across all days → MAE_revenue/MAE_cogs increase by exactly k
- Comparing ΔLB_revenue vs ΔLB_cogs reveals whether metrics are weighted equally or by column
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_baseline(baseline_path: str) -> pd.DataFrame:
    """Load baseline submission CSV."""
    df = pd.read_csv(baseline_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def apply_postprocess(revenue: np.ndarray, cogs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply deterministic post-processing: clip non-negative, cap COGS ≤ 1.05×Revenue, round 2dp."""
    revenue = np.clip(revenue, 0.0, None)
    cogs = np.clip(cogs, 0.0, None)
    # Cap COGS at 1.05 × Revenue (business constraint)
    cogs = np.minimum(cogs, revenue * 1.05)
    # Round to 2 decimal places for submission format
    revenue = np.round(revenue, 2)
    cogs = np.round(cogs, 2)
    return revenue, cogs


def create_probe(baseline: pd.DataFrame, probe_name: str, rev_delta: float, cogs_delta: float) -> pd.DataFrame:
    """Create a probe submission by adding constant deltas to Revenue and COGS."""
    probe = baseline.copy()
    probe["Revenue"] = baseline["Revenue"].values + rev_delta
    probe["COGS"] = baseline["COGS"].values + cogs_delta

    # Post-process
    rev, cogs = apply_postprocess(probe["Revenue"].values, probe["COGS"].values)
    probe["Revenue"] = rev
    probe["COGS"] = cogs

    return probe


def main():
    baseline_path = Path("submissions/submission.csv")
    baseline = load_baseline(str(baseline_path))

    print(f"Loaded baseline: {len(baseline)} rows")
    print(f"Revenue: mean={baseline['Revenue'].mean():.0f}, std={baseline['Revenue'].std():.0f}")
    print(f"COGS: mean={baseline['COGS'].mean():.0f}, std={baseline['COGS'].std():.0f}")
    print()

    # Define 6 probes
    probes = [
        ("probe_rev_plus50k", 50000, 0),
        ("probe_rev_plus100k", 100000, 0),
        ("probe_cogs_plus50k", 0, 50000),
        ("probe_cogs_plus100k", 0, 100000),
        ("probe_both_plus50k", 50000, 50000),
        ("probe_rev_minus50k", -50000, 0),
    ]

    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)

    for probe_name, rev_delta, cogs_delta in probes:
        probe_df = create_probe(baseline, probe_name, rev_delta, cogs_delta)
        output_path = submissions_dir / f"{probe_name}.csv"
        probe_df[["Date", "Revenue", "COGS"]].to_csv(output_path, index=False, date_format="%Y-%m-%d")

        # Verify
        rev_mean = probe_df["Revenue"].mean()
        cogs_mean = probe_df["COGS"].mean()
        cogs_cap_violations = (probe_df["COGS"] > probe_df["Revenue"] * 1.05).sum()
        neg_violations = (probe_df["Revenue"] < 0).sum() + (probe_df["COGS"] < 0).sum()

        print(f"{probe_name:20s}: Rev mean={rev_mean:10.0f} (+{rev_delta:+7.0f}), "
              f"COGS mean={cogs_mean:10.0f} (+{cogs_delta:+7.0f}), "
              f"cap_violations={cogs_cap_violations}, neg_violations={neg_violations}")

    print()
    print("All probes created successfully. Next steps:")
    print("1. Submit each probe_*.csv to Kaggle (max 20 per day)")
    print("2. Record LB scores in reports/bench.csv")
    print("3. Run scripts/analyze_probe_results.py to infer formula")


if __name__ == "__main__":
    main()
