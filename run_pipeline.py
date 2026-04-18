"""End-to-end reproduction entry point.

Running this from a clean checkout (with data/ restored) regenerates every
submission and explainability artifact:

    python -m venv .venv
    .venv\\Scripts\\activate           # Windows
    pip install -r requirements.txt
    python run_pipeline.py

Steps:
    1. Data audit           -> stdout table
    2. Baseline submission  -> submissions/submission_v0_baseline.csv
    3. Detrended LGBM       -> submissions/submission_v3_detrended_lgbm.csv
    4. Prophet              -> submissions/submission_v2_prophet.csv
    5. Ensemble + CV        -> submissions/submission_v4_ensemble.csv
    6. Evaluate ensemble CV -> stdout
    7. SHAP + Prophet comps -> reports/figures/
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STEPS = [
    ("Data audit", "scripts/audit_data.py"),
    ("Baseline submission", "scripts/make_baseline_submission.py"),
    ("Detrended LightGBM submission", "scripts/make_detrended_lgbm_submission.py"),
    ("Prophet submission", "scripts/make_prophet_submission.py"),
    ("Ensemble submission", "scripts/make_ensemble_submission.py"),
    ("Ensemble CV evaluation", "scripts/evaluate_ensemble_cv.py"),
    ("SHAP + Prophet components", "scripts/explain_shap.py"),
]


def main() -> None:
    for label, path in STEPS:
        print(f"\n{'=' * 60}\n>>> {label}\n{'=' * 60}")
        runpy.run_path(str(ROOT / path), run_name="__main__")


if __name__ == "__main__":
    main()
