# -*- coding: utf-8 -*-
"""Build the scheme-aware numerical-diffusion risk atlas.

The script reads the scheme-specific ``Expansion_Study_*.csv`` tables,
classifies positive-diffusion, compressive, and invalid cases, estimates the
scheme-specific 95th-percentile envelopes, and constructs the recommendation
atlas. Positive-diffusion envelopes are fitted only to cases that pass the
screening criteria. Reliability is evaluated before error and implementation
complexity are used for scheme ranking.

Use ``--base-dir`` to select the calibration tables and ``--out-dir`` to keep
generated results separate from the archived inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    px = None
    go = None
    make_subplots = None


# =====================================================================

# =====================================================================


SCHEME_FILES: Dict[str, str] = {
    "upwind1": "Expansion_Study_HLLC_Linear_upwind1.csv",
    "upwind2": "Expansion_Study_HLLC_Linear_upwind2.csv",
    "central": "Expansion_Study_HLLC_Linear_central.csv",
    "minmod": "Expansion_Study_HLLC_Linear_minmod.csv",
    "vanleer": "Expansion_Study_HLLC_Linear_vanleer.csv",
    "mc": "Expansion_Study_HLLC_Linear_mc.csv",
    "superbee": "Expansion_Study_HLLC_Linear_superbee.csv",

    "unlimited_central": "Expansion_Study_HLLC_Linear_unlimited_central.csv",
}


RECOMMENDATION_SCHEMES = [
    "upwind1", "upwind2", "central", "minmod", "vanleer", "mc", "superbee"
]

SCHEME_DISPLAY = {
    "upwind1": "1st upwind",
    "upwind2": "2nd upwind",
    "central": "central",
    "minmod": "minmod",
    "vanleer": "van Leer",
    "mc": "MC",
    "superbee": "superbee",
    "unlimited_central": "unlimited central",
}

SCHEME_ORDER = [
    "upwind1", "minmod", "vanleer", "mc", "upwind2", "superbee", "central", "unlimited_central"
]


SCHEME_COMPLEXITY = {
    "central": 1.0,
    "upwind1": 1.0,
    "upwind2": 2.0,
    "minmod": 3.0,
    "vanleer": 3.0,
    "mc": 3.0,
    "superbee": 3.0,
    "unlimited_central": 3.0,
}


MASS_ERR_LIMIT_PCT = 0.5
L2_REL_LIMIT = 1.0
PEAK_RATIO_MIN = 0.0
PEAK_RATIO_MAX = 1.5
SHIFT_LIMIT_GRID = 2.0
BOUNDARY_SIGMA = 4.0


LEFT_MARGIN_SIGMA0 = 10.0
RIGHT_MARGIN_SIGMA_FINAL = 10.0


DEFAULT_INVALID_TOL = 0.10       # Unstable or invalid fraction within a region.
DEFAULT_COMP_TOL = 0.10          # Clearly compressive fraction within a region.
DEFAULT_COMP_EPS = 0.05          # Clear compression: Cmax*/Cana > 1 + 0.05.
DEFAULT_EP95_LIMIT = 0.10        # Upper limit for q95(|shift|/Nw).
DEFAULT_L2_Q95_LIMIT = 1.0       # Retain the overall profile-shape constraint.


WEIGHT_RND = 1.0
WEIGHT_EP = 0.25
WEIGHT_L2 = 0.10
WEIGHT_COMPLEXITY = 0.02


TIE_REL_TOL = 0.05

DPI = 300


# =====================================================================

# =====================================================================

@dataclass
class ThresholdSet:
    invalid_tol: float = DEFAULT_INVALID_TOL
    comp_tol: float = DEFAULT_COMP_TOL
    comp_eps: float = DEFAULT_COMP_EPS
    ep95_limit: float = DEFAULT_EP95_LIMIT
    l2_q95_limit: float = DEFAULT_L2_Q95_LIMIT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scheme-aware risk screening and recommendation atlas.")
    parser.add_argument("--base-dir", type=str, default=None, help="Directory containing input CSV files; defaults to the script directory.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory; defaults to base-dir/scheme_aware_risk_atlas_outputs.")
    parser.add_argument("--invalid-tol", type=float, default=DEFAULT_INVALID_TOL, help="Maximum invalid-case fraction for recommendation (default: 0.10).")
    parser.add_argument("--comp-tol", type=float, default=DEFAULT_COMP_TOL, help="Maximum pronounced-compression fraction (default: 0.10).")
    parser.add_argument("--comp-eps", type=float, default=DEFAULT_COMP_EPS, help="Pronounced-compression threshold Cmax* > 1 + eps (default: 0.05).")
    parser.add_argument("--ep95-limit", type=float, default=DEFAULT_EP95_LIMIT, help="95th-percentile Ep limit (default: 0.10).")
    parser.add_argument("--l2-q95-limit", type=float, default=DEFAULT_L2_Q95_LIMIT, help="95th-percentile relative L2 limit (default: 1.0).")
    parser.add_argument("--skip-interactive", action="store_true", help="Skip interactive Plotly atlas output.")
    parser.add_argument("--full-sensitivity", action="store_true", help="Run all 27 threshold-sensitivity settings instead of the nine-setting quick check.")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip recommendation-threshold sensitivity analysis.")
    return parser.parse_args()


def resolve_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.base_dir:
        base_dir = Path(args.base_dir).expanduser().resolve()
    else:
        base_dir = Path(__file__).resolve().parent
    if not base_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base_dir}")
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = base_dir / "scheme_aware_risk_atlas_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, out_dir


def find_file(base_dir: Path, expected_name: str) -> Path:
    """Return the canonical input file, or the newest numbered duplicate."""
    p = base_dir / expected_name
    if p.exists():
        return p
    stem = Path(expected_name).stem
    suffix = Path(expected_name).suffix
    candidates = list(base_dir.glob(stem + "*" + suffix))
    if candidates:
        import re
        def candidate_rank(path: Path):
            m = re.search(r"\((\d+)\)$", path.stem)
            suffix_no = int(m.group(1)) if m else 0
            return (suffix_no, path.stat().st_mtime)
        return sorted(candidates, key=candidate_rank, reverse=True)[0]
    raise FileNotFoundError(f"Neither {expected_name} nor {stem}*{suffix} was found")


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_quantile(series: pd.Series, q: float) -> float:
    s = safe_num(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.quantile(q))


def finite_mask(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for c in cols:
        if c in df.columns:
            mask &= np.isfinite(safe_num(df[c]))
        else:
            mask &= False
    return mask


def sort_by_scheme_order(df: pd.DataFrame, scheme_col: str = "scheme") -> pd.DataFrame:
    if scheme_col not in df.columns:
        return df
    out = df.copy()
    out["_scheme_order"] = out[scheme_col].map({s: i for i, s in enumerate(SCHEME_ORDER)}).fillna(999)
    sort_cols = ["_scheme_order"]
    for c in ["Nw", "Pe", "CFL", "tau"]:
        if c in out.columns:
            sort_cols.append(c)
    out = out.sort_values(sort_cols).drop(columns="_scheme_order")
    return out


# =====================================================================

# =====================================================================

REQUIRED_COLUMNS = [
    "Nw", "CFL", "Pe", "tau",
    "Max_C_Ratio", "D_num_peak", "D_num_peak_over_D_phy", "D_num_peak_star",
    "L2_rel", "Corrected_Shift", "Final_Mass_Error(%)",
]

OPTIONAL_COLUMNS = [
    "D_phy", "D_num_variance", "R_squared", "Kurtosis", "Shape_Error",
    "Diss_Ratio(%)", "Disp_Ratio(%)", "L2_abs", "Shift_sigma",
]


def load_all_scheme_files(base_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    frames: List[pd.DataFrame] = []
    notes: List[str] = []
    for scheme, fname in SCHEME_FILES.items():
        try:
            path = find_file(base_dir, fname)
        except FileNotFoundError as exc:
            notes.append(f"SKIP {scheme}: {exc}")
            continue
        df = pd.read_csv(path)
        df["scheme"] = scheme
        df["scheme_display"] = SCHEME_DISPLAY.get(scheme, scheme)
        df["source_file"] = path.name
        frames.append(df)
        notes.append(f"Loaded {scheme}: {path.name}, n={len(df)}")

    if not frames:
        raise RuntimeError("No input CSV file could be read.")
    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged, notes


def derive_boundary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the four-sigma boundary-clearance check from Nw, Pe, and tau.
    sigma_ana/sigma0 = sqrt(1 + 2*tau/(Nw*Pe))
    """
    out = df.copy()
    Nw = safe_num(out["Nw"])
    Pe = safe_num(out["Pe"])
    tau = safe_num(out["tau"])
    sigma_ratio = np.sqrt(1.0 + 2.0 * tau / (Nw * Pe))
    upstream_margin_sigma = (LEFT_MARGIN_SIGMA0 + tau) / sigma_ratio
    downstream_margin_sigma = pd.Series(RIGHT_MARGIN_SIGMA_FINAL, index=out.index, dtype=float)
    out["sigma_ana_over_sigma0_reconstructed"] = sigma_ratio
    out["upstream_margin_sigma"] = upstream_margin_sigma
    out["downstream_margin_sigma"] = downstream_margin_sigma
    out["min_boundary_margin_sigma"] = np.minimum(upstream_margin_sigma, downstream_margin_sigma)
    for s in [3, 4, 5]:
        out[f"Boundary_Safe_{s}sigma"] = out["min_boundary_margin_sigma"] >= float(s)
    return out


def add_dimensionless_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if c in out.columns:
            out[c] = safe_num(out[c])


    out["eta_D"] = out["D_num_peak_over_D_phy"]


    kappa_from_star = out["D_num_peak_star"] if "D_num_peak_star" in out.columns else np.nan
    kappa_from_eta = out["eta_D"] / out["Pe"]
    out["kappa_s"] = kappa_from_star
    out.loc[~np.isfinite(out["kappa_s"]), "kappa_s"] = kappa_from_eta.loc[~np.isfinite(out["kappa_s"])]


    out["kappa_s_from_eta_over_Pe"] = kappa_from_eta
    out["kappa_consistency_error"] = out["kappa_s"] - out["kappa_s_from_eta_over_Pe"]

    out["Ep"] = out["Corrected_Shift"].abs() / out["Nw"]
    out["Abs_Corrected_Shift"] = out["Corrected_Shift"].abs()
    out["Peak_Overshoot"] = out["Max_C_Ratio"] - 1.0
    return out


def classify_cases(df: pd.DataFrame, comp_eps: float = DEFAULT_COMP_EPS) -> pd.DataFrame:
    out = df.copy()
    out = derive_boundary_columns(out)
    out = add_dimensionless_metrics(out)

    needed = [
        "Nw", "Pe", "CFL", "tau", "Max_C_Ratio", "D_num_peak",
        "eta_D", "kappa_s", "L2_rel", "Corrected_Shift", "Final_Mass_Error(%)",
    ]
    finite = finite_mask(out, needed)
    mass_ok = out["Final_Mass_Error(%)"].abs() <= MASS_ERR_LIMIT_PCT
    l2_ok = out["L2_rel"] <= L2_REL_LIMIT
    peak_ok = (out["Max_C_Ratio"] > PEAK_RATIO_MIN) & (out["Max_C_Ratio"] <= PEAK_RATIO_MAX)
    shift_ok = out["Corrected_Shift"].abs() <= SHIFT_LIMIT_GRID
    boundary_ok = out["min_boundary_margin_sigma"] >= BOUNDARY_SIGMA

    valid = finite & mass_ok.fillna(False) & l2_ok.fillna(False) & peak_ok.fillna(False) & shift_ok.fillna(False) & boundary_ok.fillna(False)
    positive = valid & (out["Max_C_Ratio"] < 1.0) & (out["D_num_peak"] > 0.0) & (out["eta_D"] > 0.0) & (out["kappa_s"] > 0.0)
    compressive = valid & ((out["Max_C_Ratio"] > 1.0) | (out["D_num_peak"] < 0.0) | (out["eta_D"] < 0.0))
    obvious_compressive = valid & ((out["Max_C_Ratio"] > 1.0 + comp_eps) | (out["D_num_peak"] < 0.0) | (out["eta_D"] < 0.0))

    out["Valid_For_Screening"] = valid.astype(bool)
    out["Positive_Diffusion_For_Envelope"] = positive.astype(bool)
    out["Compression_or_Antidiffusion"] = compressive.astype(bool)
    out["Obvious_Compression_or_Antidiffusion"] = obvious_compressive.astype(bool)

    out["Risk_Category"] = "valid positive diffusion"
    out.loc[~out["Valid_For_Screening"], "Risk_Category"] = "unstable or invalid"
    out.loc[out["Valid_For_Screening"] & out["Compression_or_Antidiffusion"], "Risk_Category"] = "compressive or anti-diffusive"
    out.loc[out["Valid_For_Screening"] & out["Positive_Diffusion_For_Envelope"], "Risk_Category"] = "valid positive diffusion"
    return out


# =====================================================================

# =====================================================================

def risk_composition_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cat_order = ["valid positive diffusion", "compressive or anti-diffusive", "unstable or invalid"]
    counts = df.groupby(["scheme", "Risk_Category"]).size().rename("n").reset_index()
    totals = df.groupby("scheme").size().rename("total").reset_index()
    comp = counts.merge(totals, on="scheme", how="left")
    comp["percent"] = comp["n"] / comp["total"] * 100.0

    schemes = [s for s in SCHEME_ORDER if s in df["scheme"].unique()] + [s for s in df["scheme"].unique() if s not in SCHEME_ORDER]
    full_index = pd.MultiIndex.from_product([schemes, cat_order], names=["scheme", "Risk_Category"])
    comp = comp.set_index(["scheme", "Risk_Category"]).reindex(full_index).reset_index()
    comp["n"] = comp["n"].fillna(0).astype(int)
    comp["total"] = comp.groupby("scheme")["n"].transform("sum")
    comp["percent"] = np.where(comp["total"] > 0, comp["n"] / comp["total"] * 100.0, np.nan)
    comp["scheme_display"] = comp["scheme"].map(SCHEME_DISPLAY).fillna(comp["scheme"])

    wide = comp.pivot(index="scheme", columns="Risk_Category", values="percent").reset_index()
    wide["scheme_display"] = wide["scheme"].map(SCHEME_DISPLAY).fillna(wide["scheme"])
    for c in cat_order:
        if c not in wide.columns:
            wide[c] = 0.0
    wide = sort_by_scheme_order(wide, "scheme")
    return comp, wide


def fit_rational_envelope(nw: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, str]:
    """Fit y = A/(Nw+B); return A, B, R2, RMSE, and fitting method."""
    x = np.asarray(nw, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(yy) & (yy > 0)
    x, yy = x[ok], yy[ok]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan, "not_enough_points"

    def calc_metrics(pred):
        ss_res = np.sum((yy - pred) ** 2)
        ss_tot = np.sum((yy - np.mean(yy)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else np.nan
        rmse = math.sqrt(np.mean((yy - pred) ** 2))
        return r2, rmse

    if curve_fit is not None:
        try:
            def f(xx, A, B):
                return A / (xx + B)
            popt, _ = curve_fit(f, x, yy, p0=[0.5, 2.0], bounds=([1e-12, 0.0], [np.inf, np.inf]), maxfev=50000)
            A, B = float(popt[0]), float(popt[1])
            pred = f(x, A, B)
            r2, rmse = calc_metrics(pred)
            return A, B, r2, rmse, "rational_curve_fit"
        except Exception as exc:
            warnings.warn(f"curve_fit failed, fallback grid search: {exc}")

    # fallback grid search
    best = (np.inf, np.nan, np.nan)
    for B in np.linspace(0.0, 50.0, 5001):
        z = 1.0 / (x + B)
        A = np.sum(yy * z) / np.sum(z * z)
        pred = A * z
        sse = np.sum((yy - pred) ** 2)
        if sse < best[0]:
            best = (sse, float(A), float(B))
    A, B = best[1], best[2]
    pred = A / (x + B)
    r2, rmse = calc_metrics(pred)
    return A, B, r2, rmse, "rational_grid_search"


def fit_constant_envelope(y: np.ndarray) -> Tuple[float, float]:
    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy) & (yy > 0)]
    if len(yy) == 0:
        return np.nan, np.nan
    c = float(np.mean(yy))
    rmse = float(math.sqrt(np.mean((yy - c) ** 2)))
    return c, rmse


def scheme_kappa95_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pos = df[df["Positive_Diffusion_For_Envelope"]].copy()
    rows = []
    raw_rows = []
    for (scheme, nw), g in pos.groupby(["scheme", "Nw"], dropna=False):
        k = safe_num(g["kappa_s"]).replace([np.inf, -np.inf], np.nan).dropna()
        eta = safe_num(g["eta_D"]).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "scheme": scheme,
            "scheme_display": SCHEME_DISPLAY.get(scheme, scheme),
            "Nw": float(nw),
            "n_positive": int(len(k)),
            "kappa_mean": float(k.mean()) if len(k) else np.nan,
            "kappa_median": float(k.median()) if len(k) else np.nan,
            "kappa_q75": float(k.quantile(0.75)) if len(k) else np.nan,
            "kappa_q90": float(k.quantile(0.90)) if len(k) else np.nan,
            "kappa_q95": float(k.quantile(0.95)) if len(k) else np.nan,
            "kappa_max": float(k.max()) if len(k) else np.nan,
            "eta_median": float(eta.median()) if len(eta) else np.nan,
            "eta_q95": float(eta.quantile(0.95)) if len(eta) else np.nan,
        })
        for _, r in g.iterrows():
            raw_rows.append({
                "scheme": scheme,
                "scheme_display": SCHEME_DISPLAY.get(scheme, scheme),
                "Nw": r["Nw"],
                "Pe": r["Pe"],
                "CFL": r["CFL"],
                "tau": r["tau"],
                "kappa_s": r["kappa_s"],
                "eta_D": r["eta_D"],
                "Max_C_Ratio": r["Max_C_Ratio"],
                "Ep": r["Ep"],
                "L2_rel": r["L2_rel"],
            })

    q95_df = pd.DataFrame(rows)
    if len(q95_df):
        q95_df = sort_by_scheme_order(q95_df, "scheme")

    raw_df = pd.DataFrame(raw_rows)
    if len(raw_df):
        raw_df = sort_by_scheme_order(raw_df, "scheme")

    fit_rows = []
    for scheme, sub in q95_df.groupby("scheme", dropna=False):
        sub = sub.dropna(subset=["Nw", "kappa_q95"]).copy()
        n_points = len(sub)
        n_positive_total = int(sub["n_positive"].sum()) if n_points else 0
        A, B, r2, rmse, method = fit_rational_envelope(sub["Nw"].values, sub["kappa_q95"].values)
        c, rmse_const = fit_constant_envelope(sub["kappa_q95"].values)


        if scheme == "upwind1":
            recommended = "constant"
        elif n_points < 3 or n_positive_total < 30:
            recommended = "not_recommended_few_positive_cases"
        elif scheme in ["central", "unlimited_central"]:
            recommended = "not_recommended_compressive_or_unconstrained"
        else:
            recommended = "rational_A_over_Nw_plus_B"

        fit_rows.append({
            "scheme": scheme,
            "scheme_display": SCHEME_DISPLAY.get(scheme, scheme),
            "n_Nw_points": n_points,
            "n_positive_total": n_positive_total,
            "rational_A": A,
            "rational_B": B,
            "rational_R2": r2,
            "rational_RMSE": rmse,
            "constant_kappa95": c,
            "constant_RMSE": rmse_const,
            "fit_method": method,
            "recommended_envelope_model": recommended,
        })

    fit_df = pd.DataFrame(fit_rows)
    if len(fit_df):
        fit_df = sort_by_scheme_order(fit_df, "scheme")
    return q95_df, fit_df, raw_df


def predict_kappa95_for_scheme(nw: float, scheme: str, fit_df: pd.DataFrame) -> float:
    row = fit_df.loc[fit_df["scheme"] == scheme]
    if len(row) == 0:
        return np.nan
    r = row.iloc[0]
    model = r["recommended_envelope_model"]
    if model == "constant":
        return float(r["constant_kappa95"])
    if model == "rational_A_over_Nw_plus_B" or np.isfinite(r.get("rational_A", np.nan)):
        A = float(r["rational_A"])
        B = float(r["rational_B"])
        if np.isfinite(A) and np.isfinite(B):
            return A / (float(nw) + B)
    return np.nan


# =====================================================================

# =====================================================================

def reliability_by_region(df: pd.DataFrame, comp_eps: float = DEFAULT_COMP_EPS) -> pd.DataFrame:
    """
    Aggregate by scheme, Nw, and Pe across the sampled CFL and tau values.
    The resulting Pe-Nw map therefore summarizes their sampled effects.

    Vectorized aggregation supports repeated sensitivity analyses.
    """
    d = df.copy()
    d["_valid"] = d["Valid_For_Screening"].astype(bool)
    d["_positive"] = d["Positive_Diffusion_For_Envelope"].astype(bool)
    d["_compression"] = d["Valid_For_Screening"].astype(bool) & d["Compression_or_Antidiffusion"].astype(bool)
    d["_obvious_compression"] = (
        d["Valid_For_Screening"].astype(bool)
        & ((d["Max_C_Ratio"] > 1.0 + comp_eps) | (d["D_num_peak"] < 0.0) | (d["eta_D"] < 0.0))
    )
    d["_invalid"] = ~d["_valid"]

    group_cols = ["scheme", "Nw", "Pe"]
    base = d.groupby(group_cols, dropna=False).agg(
        total_cases=("scheme", "size"),
        valid_cases=("_valid", "sum"),
        positive_cases=("_positive", "sum"),
        compression_cases=("_compression", "sum"),
        obvious_compression_cases=("_obvious_compression", "sum"),
        invalid_cases=("_invalid", "sum"),
    ).reset_index()

    for c in ["valid_cases", "positive_cases", "compression_cases", "obvious_compression_cases", "invalid_cases"]:
        base[c] = base[c].astype(int)
    base["valid_ratio"] = base["valid_cases"] / base["total_cases"]
    base["positive_ratio"] = base["positive_cases"] / base["total_cases"]
    base["compression_ratio"] = base["compression_cases"] / base["total_cases"]
    base["obvious_compression_ratio"] = base["obvious_compression_cases"] / base["total_cases"]
    base["invalid_ratio"] = base["invalid_cases"] / base["total_cases"]

    valid = d[d["_valid"]].copy()
    pos = d[d["_positive"]].copy()

    def q(v, p):
        return v.quantile(p)

    if len(valid):
        valid_q = valid.groupby(group_cols, dropna=False).agg(
            Max_C_Ratio_median_valid=("Max_C_Ratio", "median"),
            Max_C_Ratio_q95_valid=("Max_C_Ratio", lambda x: q(x, 0.95)),
            Ep_q95_valid=("Ep", lambda x: q(x, 0.95)),
            L2_rel_q95_valid=("L2_rel", lambda x: q(x, 0.95)),
            Abs_Shift_q95_valid=("Abs_Corrected_Shift", lambda x: q(x, 0.95)),
        ).reset_index()
        base = base.merge(valid_q, on=group_cols, how="left")
    else:
        for col in ["Max_C_Ratio_median_valid", "Max_C_Ratio_q95_valid", "Ep_q95_valid", "L2_rel_q95_valid", "Abs_Shift_q95_valid"]:
            base[col] = np.nan

    if len(pos):
        pos_q = pos.groupby(group_cols, dropna=False).agg(
            eta_D_median_positive=("eta_D", "median"),
            eta_D_q95_positive=("eta_D", lambda x: q(x, 0.95)),
            kappa_s_median_positive=("kappa_s", "median"),
            kappa_s_q95_positive=("kappa_s", lambda x: q(x, 0.95)),
        ).reset_index()
        base = base.merge(pos_q, on=group_cols, how="left")
    else:
        for col in ["eta_D_median_positive", "eta_D_q95_positive", "kappa_s_median_positive", "kappa_s_q95_positive"]:
            base[col] = np.nan

    base["scheme_display"] = base["scheme"].map(SCHEME_DISPLAY).fillna(base["scheme"])
    base = sort_by_scheme_order(base, "scheme")
    return base


def make_recommendation_atlas(region_df: pd.DataFrame, fit_df: pd.DataFrame, thresholds: ThresholdSet) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Determine eligible and recommended schemes in each Pe-Nw region."""
    rows = []
    candidate_rows = []
    for (nw, pe), g0 in region_df.groupby(["Nw", "Pe"], dropna=False):
        g = g0[g0["scheme"].isin(RECOMMENDATION_SCHEMES)].copy()
        cand = []
        reject_reasons: Dict[str, str] = {}
        for _, r in g.iterrows():
            scheme = r["scheme"]
            reasons = []
            if r["invalid_ratio"] > thresholds.invalid_tol:
                reasons.append(f"invalid>{thresholds.invalid_tol:.0%}")
            if r["obvious_compression_ratio"] > thresholds.comp_tol:
                reasons.append(f"obvious_comp>{thresholds.comp_tol:.0%}")
            if np.isfinite(r["Ep_q95_valid"]) and r["Ep_q95_valid"] > thresholds.ep95_limit:
                reasons.append(f"Ep95>{thresholds.ep95_limit:g}")
            if np.isfinite(r["L2_rel_q95_valid"]) and r["L2_rel_q95_valid"] > thresholds.l2_q95_limit:
                reasons.append(f"L2q95>{thresholds.l2_q95_limit:g}")
            kappa_env = predict_kappa95_for_scheme(float(nw), scheme, fit_df)
            if not np.isfinite(kappa_env):
                reasons.append("no_positive_diffusion_envelope")
            if reasons:
                reject_reasons[scheme] = ";".join(reasons)
                continue



            rnd = float(kappa_env) * float(pe)
            ep = float(r["Ep_q95_valid"]) if np.isfinite(r["Ep_q95_valid"]) else 0.0
            l2 = float(r["L2_rel_q95_valid"]) if np.isfinite(r["L2_rel_q95_valid"]) else 0.0
            complexity = SCHEME_COMPLEXITY.get(scheme, 3.0)
            score = WEIGHT_RND * rnd + WEIGHT_EP * ep + WEIGHT_L2 * l2 + WEIGHT_COMPLEXITY * complexity
            item = {
                "scheme": scheme,
                "scheme_display": SCHEME_DISPLAY.get(scheme, scheme),
                "Nw": float(nw),
                "Pe": float(pe),
                "RND_scheme": rnd,
                "kappa95_scheme": kappa_env,
                "invalid_ratio": r["invalid_ratio"],
                "obvious_compression_ratio": r["obvious_compression_ratio"],
                "compression_ratio": r["compression_ratio"],
                "Ep_q95_valid": r["Ep_q95_valid"],
                "L2_rel_q95_valid": r["L2_rel_q95_valid"],
                "complexity": complexity,
                "score": score,
            }
            cand.append(item)
            candidate_rows.append(item)

        if cand:
            cand_df = pd.DataFrame(cand)
            min_score = cand_df["score"].min()
            near = cand_df[cand_df["score"] <= min_score * (1.0 + TIE_REL_TOL)].copy()
            if len(near) > 1:
                chosen = near.sort_values(["complexity", "score"]).iloc[0]
                tie_note = "tie_by_score_choose_lower_complexity"
            else:
                chosen = cand_df.sort_values("score").iloc[0]
                tie_note = "unique_min_score"
            candidate_list = ";".join(cand_df.sort_values("score")["scheme"].tolist())
            candidate_display = ";".join(cand_df.sort_values("score")["scheme_display"].tolist())
            recommended = chosen["scheme"]
            recommended_display = chosen["scheme_display"]
            best_rnd = chosen["RND_scheme"]
            best_score = chosen["score"]
            status = "recommended"
        else:
            candidate_list = ""
            candidate_display = ""
            recommended = "no_reliable_scheme"
            recommended_display = "no reliable scheme"
            best_rnd = np.nan
            best_score = np.nan
            tie_note = "no_candidate"
            status = "no_candidate"

        rejected = "; ".join([f"{k}:{v}" for k, v in sorted(reject_reasons.items())])
        rows.append({
            "Nw": float(nw),
            "Pe": float(pe),
            "threshold_invalid_tol": thresholds.invalid_tol,
            "threshold_comp_tol": thresholds.comp_tol,
            "threshold_comp_eps": thresholds.comp_eps,
            "threshold_ep95_limit": thresholds.ep95_limit,
            "candidate_schemes": candidate_list,
            "candidate_schemes_display": candidate_display,
            "recommended_scheme": recommended,
            "recommended_scheme_display": recommended_display,
            "recommended_RND": best_rnd,
            "recommended_score": best_score,
            "recommendation_status": status,
            "tie_note": tie_note,
            "rejected_reasons": rejected,
        })
    atlas = pd.DataFrame(rows).sort_values(["Nw", "Pe"]).reset_index(drop=True)
    candidates = pd.DataFrame(candidate_rows)
    if len(candidates):
        candidates = sort_by_scheme_order(candidates, "scheme")
    return atlas, candidates


def make_full_4d_atlas(df: pd.DataFrame, fit_df: pd.DataFrame, thresholds: ThresholdSet) -> pd.DataFrame:
    """
    Recommendation table at individual Nw-Pe-CFL-tau nodes.
    Each scheme has one result per node, so compression and invalidity are
    binary indicators rather than fractions. This table supports lookup.
    """
    rows = []
    sub_df = df[df["scheme"].isin(RECOMMENDATION_SCHEMES)].copy()
    for keys, g in sub_df.groupby(["Nw", "Pe", "CFL", "tau"], dropna=False):
        nw, pe, cfl, tau = map(float, keys)
        cand = []
        rejects = {}
        for _, r in g.iterrows():
            scheme = r["scheme"]
            reasons = []
            if not bool(r["Valid_For_Screening"]):
                reasons.append("invalid")
            if bool(r["Obvious_Compression_or_Antidiffusion"]):
                reasons.append("obvious_compression")
            if np.isfinite(r["Ep"]) and r["Ep"] > thresholds.ep95_limit:
                reasons.append("Ep_high")
            if np.isfinite(r["L2_rel"]) and r["L2_rel"] > thresholds.l2_q95_limit:
                reasons.append("L2_high")
            kappa_env = predict_kappa95_for_scheme(nw, scheme, fit_df)
            if not np.isfinite(kappa_env):
                reasons.append("no_positive_diffusion_envelope")
            if reasons:
                rejects[scheme] = ";".join(reasons)
                continue
            rnd = kappa_env * pe
            score = WEIGHT_RND * rnd + WEIGHT_EP * float(r["Ep"]) + WEIGHT_L2 * float(r["L2_rel"]) + WEIGHT_COMPLEXITY * SCHEME_COMPLEXITY.get(scheme, 3.0)
            cand.append((scheme, score, rnd, SCHEME_COMPLEXITY.get(scheme, 3.0)))
        if cand:
            cand_sorted = sorted(cand, key=lambda x: (x[1], x[3]))
            min_score = cand_sorted[0][1]
            near = [x for x in cand_sorted if x[1] <= min_score * (1.0 + TIE_REL_TOL)]
            chosen = sorted(near, key=lambda x: (x[3], x[1]))[0]
            recommended = chosen[0]
            status = "recommended"
            candidate_schemes = ";".join([x[0] for x in cand_sorted])
            score = chosen[1]
            rnd = chosen[2]
        else:
            recommended = "no_reliable_scheme"
            status = "no_candidate"
            candidate_schemes = ""
            score = np.nan
            rnd = np.nan
        rows.append({
            "Nw": nw, "Pe": pe, "CFL": cfl, "tau": tau,
            "recommended_scheme": recommended,
            "recommended_scheme_display": SCHEME_DISPLAY.get(recommended, recommended),
            "candidate_schemes": candidate_schemes,
            "recommendation_status": status,
            "recommended_score": score,
            "recommended_RND": rnd,
            "rejected_reasons": "; ".join([f"{k}:{v}" for k, v in sorted(rejects.items())]),
        })
    return pd.DataFrame(rows).sort_values(["Nw", "Pe", "CFL", "tau"]).reset_index(drop=True)


def threshold_sensitivity(region_df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = []
    for invalid_tol in [0.05, 0.10, 0.20]:
        for comp_tol in [0.05, 0.10, 0.20]:
            for comp_eps in [0.01, 0.05, 0.10]:
                scenarios.append(ThresholdSet(invalid_tol=invalid_tol, comp_tol=comp_tol, comp_eps=comp_eps,
                                              ep95_limit=DEFAULT_EP95_LIMIT, l2_q95_limit=DEFAULT_L2_Q95_LIMIT))
    for th in scenarios:



        atlas, _ = make_recommendation_atlas(region_df, fit_df, th)
        total = len(atlas)
        no_candidate = int((atlas["recommended_scheme"] == "no_reliable_scheme").sum())
        rec_counts = atlas["recommended_scheme"].value_counts(dropna=False).to_dict()
        row = {
            "invalid_tol": th.invalid_tol,
            "comp_tol": th.comp_tol,
            "comp_eps": th.comp_eps,
            "ep95_limit": th.ep95_limit,
            "total_regions": total,
            "no_candidate_regions": no_candidate,
            "no_candidate_percent": no_candidate / total * 100.0 if total else np.nan,
        }
        for scheme in RECOMMENDATION_SCHEMES + ["no_reliable_scheme"]:
            row[f"recommend_{scheme}_count"] = int(rec_counts.get(scheme, 0))
            row[f"recommend_{scheme}_percent"] = int(rec_counts.get(scheme, 0)) / total * 100.0 if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================

# =====================================================================

def save_plot(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_risk_composition(comp_long: pd.DataFrame, out_png: Path) -> None:
    cat_order = ["valid positive diffusion", "compressive or anti-diffusive", "unstable or invalid"]
    schemes = [s for s in SCHEME_ORDER if s in comp_long["scheme"].unique()]
    pivot = comp_long.pivot(index="scheme", columns="Risk_Category", values="percent").reindex(schemes)
    for c in cat_order:
        if c not in pivot.columns:
            pivot[c] = 0.0
    pivot = pivot[cat_order].fillna(0.0)
    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bottom = np.zeros(len(pivot))
    labels = {
        "valid positive diffusion": "valid positive diffusion",
        "compressive or anti-diffusive": "compressive / anti-diffusive",
        "unstable or invalid": "unstable / invalid",
    }
    for c in cat_order:
        ax.bar(x, pivot[c].values, bottom=bottom, label=labels[c], edgecolor="black", linewidth=0.4)
        bottom += pivot[c].values
    ax.set_xticks(x)
    ax.set_xticklabels([SCHEME_DISPLAY.get(s, s) for s in pivot.index], rotation=25, ha="right")
    ax.set_ylabel("Percentage of cases (%)")
    ax.set_xlabel("scheme")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_plot(fig, out_png)


def plot_kappa95_envelopes(q95_df: pd.DataFrame, fit_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    schemes = [s for s in SCHEME_ORDER if s in q95_df["scheme"].unique()]
    x_line = np.linspace(0.5, 6.0, 300)
    for scheme in schemes:
        sub = q95_df[q95_df["scheme"] == scheme].sort_values("Nw")
        if len(sub) == 0:
            continue

        label = SCHEME_DISPLAY.get(scheme, scheme)
        ax.scatter(sub["Nw"], sub["kappa_q95"], s=28, alpha=0.85, label=f"{label} q95")
        row = fit_df[fit_df["scheme"] == scheme]
        if len(row):
            model = row.iloc[0]["recommended_envelope_model"]
            if model == "constant":
                c = float(row.iloc[0]["constant_kappa95"])
                ax.plot(x_line, np.full_like(x_line, c), linewidth=1.6, alpha=0.8)
            elif model == "rational_A_over_Nw_plus_B":
                A = float(row.iloc[0]["rational_A"])
                B = float(row.iloc[0]["rational_B"])
                if np.isfinite(A) and np.isfinite(B):
                    ax.plot(x_line, A / (x_line + B), linewidth=1.6, alpha=0.8)
    ax.set_xlabel(r"$N_w$")
    ax.set_ylabel(r"$\kappa_{s,95}$")
    ax.set_title(r"Scheme-specific positive-diffusion envelopes $\kappa_{95}^{s}(N_w)$")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_plot(fig, out_png)


def plot_region_recommendation_heatmap(atlas: pd.DataFrame, out_png: Path) -> None:

    scheme_list = [s for s in RECOMMENDATION_SCHEMES if s in atlas["recommended_scheme"].unique()]
    if "no_reliable_scheme" in atlas["recommended_scheme"].unique():
        scheme_list.append("no_reliable_scheme")
    code_map = {s: i for i, s in enumerate(scheme_list)}
    data = atlas.copy()
    data["code"] = data["recommended_scheme"].map(code_map)
    pivot = data.pivot(index="Nw", columns="Pe", values="code").sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{int(x)}" if float(x).is_integer() else f"{x:g}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{x:g}" for x in pivot.index])
    ax.set_xlabel("Pe")
    ax.set_ylabel(r"$N_w$")
    ax.set_title("Scheme recommendation atlas aggregated over CFL and tau")
    cbar = fig.colorbar(im, ax=ax, ticks=list(code_map.values()))
    cbar.ax.set_yticklabels([SCHEME_DISPLAY.get(s, s) for s in code_map.keys()])
    save_plot(fig, out_png)


def plot_invalid_compression_maps(region_df: pd.DataFrame, out_png: Path) -> None:
    schemes = [s for s in SCHEME_ORDER if s in region_df["scheme"].unique() and s in RECOMMENDATION_SCHEMES]
    n = len(schemes)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows), squeeze=False)
    for ax, scheme in zip(axes.flat, schemes):
        sub = region_df[region_df["scheme"] == scheme]
        pivot = sub.pivot(index="Nw", columns="Pe", values="invalid_ratio").sort_index(ascending=True)
        im = ax.imshow(pivot.values * 100.0, aspect="auto", origin="lower", vmin=0, vmax=100)
        ax.set_title(SCHEME_DISPLAY.get(scheme, scheme))
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{int(x)}" if float(x).is_integer() else f"{x:g}" for x in pivot.columns], fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{x:g}" for x in pivot.index], fontsize=8)
        ax.set_xlabel("Pe")
        ax.set_ylabel(r"$N_w$")
    for ax in axes.flat[n:]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    cbar.set_label("unstable/invalid ratio (%)")
    fig.suptitle("Unstable/invalid ratio by scheme in Pe-Nw regions", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_interactive_html(atlas: pd.DataFrame, full4d: pd.DataFrame, out_html: Path) -> None:
    if px is None:
        out_html.write_text("Plotly is not installed; interactive atlas was skipped.", encoding="utf-8")
        return
    atlas = atlas.copy()
    atlas["Pe_str"] = atlas["Pe"].astype(str)
    atlas["Nw_str"] = atlas["Nw"].astype(str)
    fig1 = px.scatter(
        atlas,
        x="Pe", y="Nw", color="recommended_scheme_display",
        hover_data=["candidate_schemes_display", "recommended_RND", "recommended_score", "rejected_reasons"],
        title="Aggregated scheme recommendation atlas over CFL and tau",
        labels={"Pe": "Pe", "Nw": "Nw", "recommended_scheme_display": "recommended scheme"},
    )
    fig1.update_traces(marker=dict(size=14, line=dict(width=0.6, color="black")))
    fig1.update_xaxes(type="log")

    html_parts = [fig1.to_html(full_html=False, include_plotlyjs="cdn")]

    if len(full4d):
        full = full4d.copy()
        fig2 = px.scatter(
            full,
            x="Pe", y="Nw", color="recommended_scheme_display",
            animation_frame="CFL", animation_group="tau",
            facet_col="tau", facet_col_wrap=2,
            hover_data=["CFL", "tau", "candidate_schemes", "recommended_RND", "rejected_reasons"],
            title="Full 4D recommendation query: Pe-Nw with CFL animation and tau facets",
            labels={"recommended_scheme_display": "recommended scheme"},
        )
        fig2.update_traces(marker=dict(size=11, line=dict(width=0.5, color="black")))
        fig2.update_xaxes(type="log")
        html_parts.append(fig2.to_html(full_html=False, include_plotlyjs=False))

    html = """
    <html><head><meta charset='utf-8'><title>Scheme-aware risk atlas</title></head>
    <body><h1>Scheme-aware prior risk atlas</h1>
    <p>Reliability-first recommendation. Hover points to inspect candidates and rejected reasons.</p>
    {figs}
    </body></html>
    """.format(figs="\n<hr>\n".join(html_parts))
    out_html.write_text(html, encoding="utf-8")


# =====================================================================

# =====================================================================

def main() -> None:
    args = parse_args()
    base_dir, out_dir = resolve_dirs(args)
    th = ThresholdSet(
        invalid_tol=args.invalid_tol,
        comp_tol=args.comp_tol,
        comp_eps=args.comp_eps,
        ep95_limit=args.ep95_limit,
        l2_q95_limit=args.l2_q95_limit,
    )


    raw, notes = load_all_scheme_files(base_dir)
    classified = classify_cases(raw, comp_eps=th.comp_eps)
    classified = sort_by_scheme_order(classified, "scheme")
    classified.to_csv(out_dir / "01_all_schemes_classified_cases.csv", index=False, encoding="utf-8-sig")


    consistency = classified.groupby("scheme").agg(
        n_cases=("scheme", "size"),
        max_abs_kappa_consistency_error=("kappa_consistency_error", lambda x: float(np.nanmax(np.abs(x)))),
        median_abs_kappa_consistency_error=("kappa_consistency_error", lambda x: float(np.nanmedian(np.abs(x)))),
    ).reset_index()
    consistency = sort_by_scheme_order(consistency, "scheme")
    consistency.to_csv(out_dir / "01b_kappa_dimensionless_consistency_check.csv", index=False, encoding="utf-8-sig")


    comp_long, comp_wide = risk_composition_table(classified)
    comp_long.to_csv(out_dir / "02_scheme_risk_composition_long.csv", index=False, encoding="utf-8-sig")
    comp_wide.to_csv(out_dir / "02_scheme_risk_composition_wide.csv", index=False, encoding="utf-8-sig")
    plot_risk_composition(comp_long, out_dir / "02_scheme_risk_composition.png")


    q95_df, fit_df, raw_pos_df = scheme_kappa95_tables(classified)
    q95_df.to_csv(out_dir / "03_scheme_kappa95_by_Nw.csv", index=False, encoding="utf-8-sig")
    fit_df.to_csv(out_dir / "03_scheme_kappa95_fit.csv", index=False, encoding="utf-8-sig")
    raw_pos_df.to_csv(out_dir / "03_scheme_positive_diffusion_raw_points.csv", index=False, encoding="utf-8-sig")
    if len(q95_df):
        plot_kappa95_envelopes(q95_df, fit_df, out_dir / "03_scheme_kappa95_envelopes.png")


    region = reliability_by_region(classified, comp_eps=th.comp_eps)
    region.to_csv(out_dir / "04_scheme_reliability_by_Nw_Pe_region.csv", index=False, encoding="utf-8-sig")
    if len(region):
        plot_invalid_compression_maps(region, out_dir / "04_invalid_ratio_maps_by_scheme.png")


    atlas, candidates = make_recommendation_atlas(region, fit_df, th)
    atlas.to_csv(out_dir / "05_scheme_recommendation_atlas_by_Nw_Pe.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(out_dir / "05_candidate_scheme_scores_by_Nw_Pe.csv", index=False, encoding="utf-8-sig")
    if len(atlas):
        plot_region_recommendation_heatmap(atlas, out_dir / "05_scheme_recommendation_atlas_by_Nw_Pe.png")

    full4d = make_full_4d_atlas(classified, fit_df, th)
    full4d.to_csv(out_dir / "06_scheme_recommendation_atlas_full_Nw_Pe_CFL_tau.csv", index=False, encoding="utf-8-sig")


    sens_rows = []
    if args.skip_sensitivity:
        sensitivity = pd.DataFrame()
        sensitivity.to_csv(out_dir / "07_recommendation_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    else:


        comp_eps_values = [0.01, 0.05, 0.10] if args.full_sensitivity else [th.comp_eps]
        region_cache = {ce: reliability_by_region(classified, comp_eps=ce) for ce in comp_eps_values}
        for invalid_tol in [0.05, 0.10, 0.20]:
            for comp_tol in [0.05, 0.10, 0.20]:
                for comp_eps in comp_eps_values:
                    region2 = region_cache[comp_eps]
                    th2 = ThresholdSet(invalid_tol=invalid_tol, comp_tol=comp_tol, comp_eps=comp_eps,
                                       ep95_limit=th.ep95_limit, l2_q95_limit=th.l2_q95_limit)
                    atlas2, _ = make_recommendation_atlas(region2, fit_df, th2)
                    total = len(atlas2)
                    counts = atlas2["recommended_scheme"].value_counts(dropna=False).to_dict()
                    row = {
                        "invalid_tol": invalid_tol,
                        "comp_tol": comp_tol,
                        "comp_eps": comp_eps,
                        "ep95_limit": th.ep95_limit,
                        "l2_q95_limit": th.l2_q95_limit,
                        "total_regions": total,
                        "no_candidate_regions": int(counts.get("no_reliable_scheme", 0)),
                        "no_candidate_percent": counts.get("no_reliable_scheme", 0) / total * 100 if total else np.nan,
                    }
                    for scheme in RECOMMENDATION_SCHEMES + ["no_reliable_scheme"]:
                        row[f"recommend_{scheme}_count"] = int(counts.get(scheme, 0))
                        row[f"recommend_{scheme}_percent"] = counts.get(scheme, 0) / total * 100 if total else np.nan
                    sens_rows.append(row)
        sensitivity = pd.DataFrame(sens_rows)
        sensitivity.to_csv(out_dir / "07_recommendation_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")


    if args.skip_interactive:
        (out_dir / "08_interactive_scheme_recommendation_atlas.html").write_text(
            "Interactive atlas skipped by --skip-interactive.", encoding="utf-8"
        )
    else:
        make_interactive_html(atlas, full4d, out_dir / "08_interactive_scheme_recommendation_atlas.html")


    readme_lines = []
    readme_lines.append("Scheme-aware risk atlas outputs")
    readme_lines.append("=" * 70)
    readme_lines.append(f"Base directory: {base_dir}")
    readme_lines.append(f"Output directory: {out_dir}")
    readme_lines.append("")
    readme_lines.append("Loaded files:")
    for n in notes:
        readme_lines.append(f"- {n}")
    readme_lines.append("")
    readme_lines.append("Default recommendation thresholds:")
    readme_lines.append(json.dumps(th.__dict__, indent=2, ensure_ascii=False))
    readme_lines.append("")
    readme_lines.append("Risk composition (%):")
    if len(comp_wide):
        show_cols = ["scheme_display", "valid positive diffusion", "compressive or anti-diffusive", "unstable or invalid"]
        show_cols = [c for c in show_cols if c in comp_wide.columns]
        readme_lines.append(comp_wide[show_cols].to_string(index=False))
    readme_lines.append("")
    readme_lines.append("Kappa envelope fit:")
    if len(fit_df):
        show_cols = ["scheme_display", "n_positive_total", "recommended_envelope_model", "rational_A", "rational_B", "rational_R2", "constant_kappa95"]
        readme_lines.append(fit_df[show_cols].to_string(index=False))
    readme_lines.append("")
    readme_lines.append("Recommendation counts under default thresholds:")
    if len(atlas):
        readme_lines.append(atlas["recommended_scheme_display"].value_counts().to_string())
    readme_lines.append("")
    readme_lines.append("Generated key files:")
    for p in sorted(out_dir.glob("*")):
        readme_lines.append(f"- {p.name}")
    (out_dir / "README_scheme_aware_risk_atlas.txt").write_text("\n".join(readme_lines), encoding="utf-8")

    print("Done. Outputs saved to:", out_dir)
    print("\n".join(readme_lines[:80]))


if __name__ == "__main__":
    main()
