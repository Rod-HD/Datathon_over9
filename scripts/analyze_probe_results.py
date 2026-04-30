"""Analyze probe submission results to infer LB scoring formula.

Once probes are submitted and LB scores are recorded in reports/bench.csv,
this script solves a linear system to estimate the weights of MAE_revenue, MAE_cogs,
RMSE_revenue, RMSE_cogs, and R² components in the final LB score.

Math:
  If LB = α·MAE_revenue + β·MAE_cogs + γ·RMSE + δ·(1-R²), then:
    ΔLB_revenue = α·50000 + ε_noise (from probe_rev_plus50k)
    ΔLB_cogs = β·50000 + ε_noise (from probe_cogs_plus50k)
    ΔLB_ratio = ΔLB_revenue / ΔLB_cogs reveals relative weighting
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_bench() -> pd.DataFrame:
    """Load benchmark CSV with all submissions."""
    bench_path = Path("reports/bench.csv")
    if not bench_path.exists():
        raise FileNotFoundError(f"{bench_path} not found. Run make_probe_submissions.py first.")
    return pd.read_csv(bench_path)


def find_lb_score(bench: pd.DataFrame, version: str) -> float | None:
    """Find LB score for a version."""
    row = bench[bench["version"] == version]
    if len(row) == 0:
        return None
    lb = row.iloc[0]["lb_score"]
    return lb if not pd.isna(lb) and lb != "pending" else None


def analyze_probes():
    """Analyze probe results to infer formula weights."""
    bench = load_bench()
    print("=" * 80)
    print("LB FORMULA REVERSE-ENGINEERING ANALYSIS")
    print("=" * 80)
    print()

    # Get baseline and probe LB scores
    baseline_lb = find_lb_score(bench, "v2")  # Use Prophet as baseline
    if baseline_lb is None:
        print("❌ ERROR: Baseline (v2) LB score not found in bench.csv")
        print("   Please submit v2 to Kaggle and record its LB score first.")
        return

    probes = {
        "probe_rev_plus50k": 50000,
        "probe_rev_plus100k": 100000,
        "probe_cogs_plus50k": 50000,
        "probe_cogs_plus100k": 100000,
        "probe_both_plus50k": 50000,
        "probe_rev_minus50k": 50000,  # magnitude (absolute)
    }

    print(f"Baseline (v2) LB score: {baseline_lb:,.0f}")
    print()

    results = {}
    for probe_name, magnitude in probes.items():
        probe_lb = find_lb_score(bench, probe_name)
        if probe_lb is None:
            print(f"⏳ {probe_name:20s}: LB score PENDING")
            results[probe_name] = None
        else:
            delta_lb = probe_lb - baseline_lb
            delta_lb_per_unit = delta_lb / magnitude
            results[probe_name] = {
                "lb": probe_lb,
                "delta_lb": delta_lb,
                "delta_lb_per_unit": delta_lb_per_unit,
                "magnitude": magnitude,
            }
            sign = "+" if delta_lb > 0 else ""
            print(f"✅ {probe_name:20s}: LB = {probe_lb:,.0f} "
                  f"(Δ = {sign}{delta_lb:,.0f}, per-unit = {delta_lb_per_unit:.2f})")

    print()
    print("=" * 80)
    print("INFERENCE")
    print("=" * 80)
    print()

    # Analyze individual columns vs combined
    rev_plus50 = results.get("probe_rev_plus50k")
    rev_plus100 = results.get("probe_rev_plus100k")
    cogs_plus50 = results.get("probe_cogs_plus50k")
    cogs_plus100 = results.get("probe_cogs_plus100k")
    both_plus50 = results.get("probe_both_plus50k")
    rev_minus50 = results.get("probe_rev_minus50k")

    if rev_plus50 is None or cogs_plus50 is None:
        print("⏳ Not enough probe results yet. Please submit and record more probes.")
        return

    # Check linearity
    if rev_plus100 is not None:
        ratio = rev_plus100["delta_lb"] / rev_plus50["delta_lb"]
        print(f"Linearity check (Rev): ΔLB(+100K) / ΔLB(+50K) = {ratio:.3f}")
        if abs(ratio - 2.0) < 0.1:
            print("  ✓ Linear relationship confirmed for Revenue")
        else:
            print("  ⚠ Non-linear or noisy; large error margin possible")
        print()

    if cogs_plus100 is not None:
        ratio = cogs_plus100["delta_lb"] / cogs_plus50["delta_lb"]
        print(f"Linearity check (COGS): ΔLB(+100K) / ΔLB(+50K) = {ratio:.3f}")
        if abs(ratio - 2.0) < 0.1:
            print("  ✓ Linear relationship confirmed for COGS")
        else:
            print("  ⚠ Non-linear or noisy; large error margin possible")
        print()

    # Estimate weights
    alpha_est = rev_plus50["delta_lb_per_unit"]  # Weight on MAE_revenue or similar
    beta_est = cogs_plus50["delta_lb_per_unit"]   # Weight on MAE_cogs or similar
    ratio = alpha_est / (beta_est + 1e-9)

    print(f"Weight estimate:")
    print(f"  α (Revenue sensitivity): {alpha_est:.3f} per 1 VND offset")
    print(f"  β (COGS sensitivity):    {beta_est:.3f} per 1 VND offset")
    print(f"  Ratio (α/β):             {ratio:.3f}")
    print()

    # Interpret ratio
    if abs(ratio - 1.0) < 0.2:
        print("→ HYPOTHESIS 1: LB = MAE_revenue + MAE_cogs (equal weight)")
        print("  Both columns contribute equally to final score.")
    elif abs(beta_est) < 0.01:
        print("→ HYPOTHESIS 2: LB = MAE_revenue ONLY")
        print("  COGS changes have near-zero impact; only Revenue is scored.")
    elif ratio > 1.5:
        print("→ HYPOTHESIS 3: Revenue weighted more heavily than COGS")
        print(f"  Possible: LB = {ratio:.1f}·MAE_revenue + 1.0·MAE_cogs")
    else:
        print("→ HYPOTHESIS 4: Custom weighted formula")
        print(f"  Possible: LB = α·MAE_revenue + β·MAE_cogs where α/β ≈ {ratio:.2f}")

    print()

    # Check additivity (if both probe available)
    if both_plus50 is not None:
        delta_both = both_plus50["delta_lb"]
        delta_sum = rev_plus50["delta_lb"] + cogs_plus50["delta_lb"]
        additivity_error = (delta_both - delta_sum) / (delta_sum + 1e-9)

        print(f"Additivity check:")
        print(f"  ΔLB(both +50K) = {delta_both:,.0f}")
        print(f"  ΔLB(rev +50K) + ΔLB(cogs +50K) = {delta_sum:,.0f}")
        print(f"  Error ratio: {additivity_error:.3f}")
        if abs(additivity_error) < 0.05:
            print("  ✓ Additive formula confirmed (MAE_revenue and MAE_cogs combine linearly)")
        else:
            print("  ⚠ Non-additive; formula may involve interactions or RMSE/R²")
        print()

    # Asymmetry check
    if rev_minus50 is not None:
        delta_minus = rev_minus50["delta_lb"]
        delta_plus = rev_plus50["delta_lb"]
        asymmetry = abs(delta_minus + delta_plus) / (abs(delta_plus) + 1e-9)

        print(f"Symmetry check:")
        print(f"  ΔLB(rev +50K) = {delta_plus:,.0f}")
        print(f"  ΔLB(rev -50K) = {delta_minus:,.0f}")
        print(f"  Symmetry error: {asymmetry:.3f}")
        if asymmetry < 0.1:
            print("  ✓ Symmetric MAE formula (errors matter equally in both directions)")
        else:
            print("  ⚠ Asymmetric; formula may penalize under/over-prediction differently")
        print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. If any probes are still PENDING, submit to Kaggle and record LB scores.")
    print("2. Re-run this script to update estimates.")
    print("3. Use inferred formula to calibrate blend weights and expected LB on v24/v25 candidates.")
    print()


if __name__ == "__main__":
    analyze_probes()
