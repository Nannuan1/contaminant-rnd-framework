# -*- coding: utf-8 -*-
"""Evaluate envelope coverage and leave-one-limiter-out robustness.

The analysis reads the four MUSCL/TVD limiter tables directly, checks coverage
of the pooled 95% envelope, and repeats the fit while omitting one limiter at a
time. The unlimited central reconstruction is retained as a high-risk control.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================

# =========================

USER_BASE_DIR = Path(__file__).resolve().parent.parent / "examples" / "calibration_smoke"


BASE_DIR = USER_BASE_DIR if USER_BASE_DIR.exists() else Path(__file__).resolve().parent

RAW_LIMITER_FILES: Dict[str, str] = {
    "minmod": "Expansion_Study_HLLC_Linear_minmod.csv",
    "vanleer": "Expansion_Study_HLLC_Linear_vanleer.csv",
    "mc": "Expansion_Study_HLLC_Linear_mc.csv",
    "superbee": "Expansion_Study_HLLC_Linear_superbee.csv",
}
UNLIMITED_FILE_NAME = "Expansion_Study_HLLC_Linear_unlimited_central.csv"

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "leave_one_limiter_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


KAPPA95_A = 0.6952
KAPPA95_B = 4.2167


MASS_ERR_LIMIT_PCT = 0.5
L2_REL_LIMIT = 1.0
PEAK_RATIO_MIN = 0.0
PEAK_RATIO_MAX = 1.5
SHIFT_LIMIT_GRID = 2.0
BOUNDARY_SIGMA = 4.0



LEFT_MARGIN_SIGMA0 = 10.0
RIGHT_MARGIN_SIGMA_FINAL = 10.0

DPI = 300


# =========================

# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Risk framework checks from raw limiter CSV files.")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Directory containing input CSV files; defaults to the bundled example data.")
    return parser.parse_args()


def resolve_base_dir(args: argparse.Namespace) -> Path:
    if args.base_dir:
        p = Path(args.base_dir)
    else:
        p = BASE_DIR
    if not p.exists():
        raise FileNotFoundError(f"Data directory does not exist: {p}")
    return p


def find_file(base_dir: Path, file_name: str) -> Path:
    """Locate the exact filename or a matching suffixed copy."""
    p = base_dir / file_name
    if p.exists():
        return p
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    candidates = sorted(base_dir.glob(stem + "*" + suffix))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Neither {p} nor {stem}*{suffix} was found")


def safe_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def safe_quantile(x: pd.Series, q: float) -> float:
    x = safe_num(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.quantile(q))


def kappa95_formula(nw, A: float = KAPPA95_A, B: float = KAPPA95_B):
    return A / (np.asarray(nw, dtype=float) + B)


def normalize_limiter_name(name: str) -> str:
    mp = {
        "van Leer": "vanleer", "Van Leer": "vanleer", "van_leer": "vanleer",
        "MC": "mc", "Minmod": "minmod", "Superbee": "superbee",
    }
    return mp.get(str(name).strip(), str(name).strip())


def load_raw_limiter_files(base_dir: Path, include_unlimited: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    frames = []
    for limiter, fname in RAW_LIMITER_FILES.items():
        f = find_file(base_dir, fname)
        df = pd.read_csv(f)
        df["limiter"] = limiter
        df["source_file"] = f.name
        frames.append(df)
        notes.append(f"Loaded {limiter}: {f.name}, n={len(df)}")

    if include_unlimited:
        try:
            f = find_file(base_dir, UNLIMITED_FILE_NAME)
            df = pd.read_csv(f)
            df["limiter"] = "unlimited_central"
            df["source_file"] = f.name
            frames.append(df)
            notes.append(f"Loaded unlimited_central: {f.name}, n={len(df)}")
        except FileNotFoundError as e:
            notes.append(f"Unlimited file not found and skipped: {e}")

    out = pd.concat(frames, ignore_index=True, sort=False)
    return out, notes


def derive_boundary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct boundary clearance from Nw, Pe, and tau.

    Analytical width: sigma_ana/sigma0 = sqrt(1 + 2*tau/(Nw*Pe)).
    The initial centroid is 10*sigma0 from the left boundary and moves by
    tau*sigma0; its final left-boundary distance is (10+tau)*sigma0.
    Thus upstream_margin_sigma = (10+tau)/sigma_ratio, while the right
    boundary is placed 10*sigma_ana downstream of the final centroid.
    """
    df = df.copy()
    Nw = safe_num(df["Nw"])
    Pe = safe_num(df["Pe"])
    tau = safe_num(df["tau"])

    sigma_ratio = np.sqrt(1.0 + 2.0 * tau / (Nw * Pe))
    upstream = (LEFT_MARGIN_SIGMA0 + tau) / sigma_ratio
    downstream = pd.Series(RIGHT_MARGIN_SIGMA_FINAL, index=df.index, dtype="float64")

    df["sigma_ana_over_sigma0_reconstructed"] = sigma_ratio
    df["upstream_margin_sigma"] = upstream
    df["downstream_margin_sigma"] = downstream
    df["min_boundary_margin_sigma"] = np.minimum(upstream, downstream)
    df["Boundary_Safe_3sigma"] = df["min_boundary_margin_sigma"] >= 3.0
    df["Boundary_Safe_4sigma"] = df["min_boundary_margin_sigma"] >= 4.0
    df["Boundary_Safe_5sigma"] = df["min_boundary_margin_sigma"] >= 5.0
    return df


def add_basic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Nw", "Pe", "CFL", "tau", "Max_C_Ratio", "D_num_peak", "D_num_peak_over_D_phy", "L2_rel", "Corrected_Shift", "Final_Mass_Error(%)"]:
        if col in df.columns:
            df[col] = safe_num(df[col])

    df["eta_D"] = df["D_num_peak_over_D_phy"] if "D_num_peak_over_D_phy" in df.columns else np.nan
    df["kappa_s"] = df["eta_D"] / df["Pe"]
    df["kappa95_current"] = kappa95_formula(df["Nw"])
    df["RND_current"] = df["kappa95_current"] * df["Pe"]
    df["covered_by_current_kappa95"] = df["kappa_s"] <= df["kappa95_current"]
    return df


def classify_cases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["limiter"] = df["limiter"].map(normalize_limiter_name)
    df = derive_boundary_columns(df)
    df = add_basic_metrics(df)

    finite_cols = [
        "Nw", "Pe", "CFL", "tau", "Max_C_Ratio", "D_num_peak",
        "D_num_peak_over_D_phy", "eta_D", "kappa_s", "L2_rel",
        "Corrected_Shift", "Final_Mass_Error(%)",
    ]
    finite = pd.Series(True, index=df.index)
    for c in finite_cols:
        if c in df.columns:
            finite &= np.isfinite(safe_num(df[c]))

    mass_ok = df["Final_Mass_Error(%)"].abs() <= MASS_ERR_LIMIT_PCT
    l2_ok = df["L2_rel"] <= L2_REL_LIMIT
    peak_ok = (df["Max_C_Ratio"] > PEAK_RATIO_MIN) & (df["Max_C_Ratio"] <= PEAK_RATIO_MAX)
    shift_ok = df["Corrected_Shift"].abs() <= SHIFT_LIMIT_GRID
    boundary_ok = df["min_boundary_margin_sigma"] >= BOUNDARY_SIGMA

    valid = finite & mass_ok.fillna(False) & l2_ok.fillna(False) & peak_ok.fillna(False) & shift_ok.fillna(False) & boundary_ok.fillna(False)
    positive = valid & (df["Max_C_Ratio"] < 1.0) & (df["D_num_peak"] > 0.0) & (df["eta_D"] > 0.0) & (df["kappa_s"] > 0.0)
    compression = valid & ((df["Max_C_Ratio"] > 1.0) | (df["D_num_peak"] < 0.0) | (df["eta_D"] < 0.0))

    df["Valid_For_Screening"] = valid.astype(bool)
    df["Positive_Diffusion_For_Envelope"] = positive.astype(bool)
    df["Compression_or_Antidiffusion"] = compression.astype(bool)

    df["Risk_Category"] = "valid non-compressive"
    df.loc[~df["Valid_For_Screening"], "Risk_Category"] = "unstable or invalid"
    df.loc[df["Valid_For_Screening"] & df["Compression_or_Antidiffusion"], "Risk_Category"] = "compressive or anti-diffusive"

    return df


def fit_kappa95_curve(nw_values: np.ndarray, kappa_q95: np.ndarray) -> Tuple[float, float, str]:
    x = np.asarray(nw_values, dtype=float)
    y = np.asarray(kappa_q95, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan, np.nan, "not_enough_points"
    try:
        from scipy.optimize import curve_fit

        def f(xx, A, B):
            return A / (xx + B)

        popt, _ = curve_fit(
            f, x, y, p0=[0.7, 4.0], bounds=([1e-12, 0.0], [np.inf, np.inf]), maxfev=20000
        )
        return float(popt[0]), float(popt[1]), "scipy_curve_fit"
    except Exception as e:
        B_grid = np.linspace(0.0, 30.0, 3001)
        best = (np.inf, np.nan, np.nan)
        for B in B_grid:
            z = 1.0 / (x + B)
            A = np.sum(y * z) / np.sum(z * z)
            sse = np.sum((y - A * z) ** 2)
            if sse < best[0]:
                best = (sse, float(A), float(B))
        return best[1], best[2], f"grid_search_fallback:{type(e).__name__}"


def coverage_summary(df_pos: pd.DataFrame, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if not group_cols:
        d = df_pos.copy()
        return pd.DataFrame({
            "group": ["ALL"],
            "n_positive": [len(d)],
            "n_covered": [int(d["covered_by_current_kappa95"].sum())],
            "coverage": [float(d["covered_by_current_kappa95"].mean()) if len(d) else np.nan],
            "kappa_median": [safe_quantile(d["kappa_s"], 0.50)],
            "kappa_q95": [safe_quantile(d["kappa_s"], 0.95)],
            "kappa_max": [safe_quantile(d["kappa_s"], 1.00)],
            "eta_median": [safe_quantile(d["eta_D"], 0.50)],
            "eta_q95": [safe_quantile(d["eta_D"], 0.95)],
            "eta_max": [safe_quantile(d["eta_D"], 1.00)],
        })

    rows = []
    for keys, d in df_pos.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            "n_positive": len(d),
            "n_covered": int(d["covered_by_current_kappa95"].sum()),
            "coverage": float(d["covered_by_current_kappa95"].mean()) if len(d) else np.nan,
            "kappa_median": safe_quantile(d["kappa_s"], 0.50),
            "kappa_q95": safe_quantile(d["kappa_s"], 0.95),
            "kappa_max": safe_quantile(d["kappa_s"], 1.00),
            "eta_median": safe_quantile(d["eta_D"], 0.50),
            "eta_q95": safe_quantile(d["eta_D"], 0.95),
            "eta_max": safe_quantile(d["eta_D"], 1.00),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def risk_composition(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["limiter", "Risk_Category"]).size().rename("n").reset_index()
    total = df.groupby("limiter").size().rename("total").reset_index()
    comp = counts.merge(total, on="limiter", how="left")
    comp["percent"] = comp["n"] / comp["total"] * 100.0

    order = ["valid non-compressive", "compressive or anti-diffusive", "unstable or invalid"]
    all_lim = comp["limiter"].drop_duplicates().tolist()
    full_index = pd.MultiIndex.from_product([all_lim, order], names=["limiter", "Risk_Category"])
    comp = comp.set_index(["limiter", "Risk_Category"]).reindex(full_index).reset_index()
    comp["n"] = comp["n"].fillna(0).astype(int)
    comp["total"] = comp.groupby("limiter")["n"].transform("sum")
    comp["percent"] = np.where(comp["total"] > 0, comp["n"] / comp["total"] * 100.0, np.nan)
    return comp


def plot_risk_composition(comp: pd.DataFrame, out_png: Path) -> None:
    order_lim = ["minmod", "vanleer", "mc", "superbee", "unlimited_central"]
    existing = comp["limiter"].drop_duplicates().tolist()
    ordered = [x for x in order_lim if x in existing] + [x for x in existing if x not in order_lim]
    cat_order = ["valid non-compressive", "compressive or anti-diffusive", "unstable or invalid"]
    pivot = comp.pivot(index="limiter", columns="Risk_Category", values="percent").reindex(ordered)
    for c in cat_order:
        if c not in pivot.columns:
            pivot[c] = 0.0
    pivot = pivot[cat_order].fillna(0.0)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    bottom = np.zeros(len(pivot))
    colors = {
        "valid non-compressive": "#FDBA2D",
        "compressive or anti-diffusive": "#CC4778",
        "unstable or invalid": "#5E03A3",
    }
    labels = {
        "valid non-compressive": "valid non-compressive",
        "compressive or anti-diffusive": "compressive/anti-diffusive",
        "unstable or invalid": "unstable/invalid",
    }
    x = np.arange(len(pivot))
    for c in cat_order:
        ax.bar(x, pivot[c].values, bottom=bottom, label=labels[c], color=colors[c], edgecolor="black", linewidth=0.4)
        bottom += pivot[c].values
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=20, ha="right")
    ax.set_ylabel("Percentage of cases (%)")
    ax.set_xlabel("scheme / limiter type")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_leave_one_coverage(loocv: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(loocv))
    ax.bar(x, loocv["test_coverage"].values * 100.0, edgecolor="black", linewidth=0.5)
    ax.axhline(95.0, linestyle="--", linewidth=1.2, color="black", label="target 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(loocv["left_out_limiter"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Coverage of left-out positive cases (%)")
    ax.set_xlabel("left-out limiter")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_current_envelope(df_pos: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(df_pos["Nw"], df_pos["kappa_s"], s=12, alpha=0.35, label="positive cases")
    x_line = np.linspace(df_pos["Nw"].min(), df_pos["Nw"].max(), 200)
    ax.plot(x_line, kappa95_formula(x_line), color="black", linewidth=2, label="current kappa95 envelope")
    ax.set_xlabel(r"$N_w$")
    ax.set_ylabel(r"$\kappa_s = D_{num}/(u\Delta x)$")
    ax.set_title("Coverage check for current kappa95 envelope")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args)
    global OUT_DIR
    OUT_DIR = base_dir / "risk_framework_three_checks_outputs_raw_input"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


    main_raw, notes = load_raw_limiter_files(base_dir, include_unlimited=False)
    main_df = classify_cases(main_raw)
    main_df.to_csv(OUT_DIR / "combined_from_raw_limiters_classified.csv", index=False, encoding="utf-8-sig")

    pos_df = main_df[main_df["Positive_Diffusion_For_Envelope"]].copy()
    pos_df = pos_df[np.isfinite(pos_df["kappa_s"]) & np.isfinite(pos_df["Nw"]) & np.isfinite(pos_df["Pe"])]


    coverage_all = coverage_summary(pos_df)
    coverage_by_limiter = coverage_summary(pos_df, ["limiter"])
    coverage_by_nw = coverage_summary(pos_df, ["Nw"])
    coverage_by_pe = coverage_summary(pos_df, ["Pe"])
    coverage_by_cfl = coverage_summary(pos_df, ["CFL"])
    coverage_by_tau = coverage_summary(pos_df, ["tau"])
    coverage_by_limiter_nw = coverage_summary(pos_df, ["limiter", "Nw"])
    coverage_by_limiter_pe = coverage_summary(pos_df, ["limiter", "Pe"])

    coverage_all.to_csv(OUT_DIR / "01_coverage_overall.csv", index=False, encoding="utf-8-sig")
    coverage_by_limiter.to_csv(OUT_DIR / "01_coverage_by_limiter.csv", index=False, encoding="utf-8-sig")
    coverage_by_nw.to_csv(OUT_DIR / "01_coverage_by_Nw.csv", index=False, encoding="utf-8-sig")
    coverage_by_pe.to_csv(OUT_DIR / "01_coverage_by_Pe.csv", index=False, encoding="utf-8-sig")
    coverage_by_cfl.to_csv(OUT_DIR / "01_coverage_by_CFL.csv", index=False, encoding="utf-8-sig")
    coverage_by_tau.to_csv(OUT_DIR / "01_coverage_by_tau.csv", index=False, encoding="utf-8-sig")
    coverage_by_limiter_nw.to_csv(OUT_DIR / "01_coverage_by_limiter_Nw.csv", index=False, encoding="utf-8-sig")
    coverage_by_limiter_pe.to_csv(OUT_DIR / "01_coverage_by_limiter_Pe.csv", index=False, encoding="utf-8-sig")
    plot_current_envelope(pos_df, OUT_DIR / "01_current_kappa95_coverage_scatter.png")

    # 2) leave-one-limiter-out
    limiters = sorted(pos_df["limiter"].dropna().unique().tolist())
    loocv_rows = []
    loocv_q95_rows = []
    for left in limiters:
        train = pos_df[pos_df["limiter"] != left].copy()
        test = pos_df[pos_df["limiter"] == left].copy()
        q95 = train.groupby("Nw", as_index=False)["kappa_s"].quantile(0.95).rename(columns={"kappa_s": "kappa_q95_train"}).sort_values("Nw")
        A, B, fit_method = fit_kappa95_curve(q95["Nw"].to_numpy(), q95["kappa_q95_train"].to_numpy())
        for _, r in q95.iterrows():
            loocv_q95_rows.append({
                "left_out_limiter": left,
                "Nw": r["Nw"],
                "kappa_q95_train": r["kappa_q95_train"],
                "A_fit": A,
                "B_fit": B,
                "fit_method": fit_method,
            })
        if np.isfinite(A) and np.isfinite(B) and len(test) > 0:
            test_pred = kappa95_formula(test["Nw"], A=A, B=B)
            train_pred = kappa95_formula(train["Nw"], A=A, B=B)
            test_cov = float((test["kappa_s"].to_numpy() <= test_pred).mean())
            train_cov = float((train["kappa_s"].to_numpy() <= train_pred).mean())
        else:
            test_cov = np.nan
            train_cov = np.nan
        loocv_rows.append({
            "left_out_limiter": left,
            "n_train_positive": len(train),
            "n_test_positive": len(test),
            "A_fit": A,
            "B_fit": B,
            "fit_method": fit_method,
            "train_coverage": train_cov,
            "test_coverage": test_cov,
            "target_coverage": 0.95,
        })

    loocv = pd.DataFrame(loocv_rows)
    loocv_q95 = pd.DataFrame(loocv_q95_rows)
    loocv.to_csv(OUT_DIR / "02_leave_one_limiter_validation.csv", index=False, encoding="utf-8-sig")
    loocv_q95.to_csv(OUT_DIR / "02_leave_one_limiter_q95_points.csv", index=False, encoding="utf-8-sig")
    if len(loocv):
        plot_leave_one_coverage(loocv, OUT_DIR / "02_leave_one_limiter_coverage.png")


    all_raw, notes_all = load_raw_limiter_files(base_dir, include_unlimited=True)
    all_df = classify_cases(all_raw)
    all_df.to_csv(OUT_DIR / "all_raw_limiters_plus_unlimited_classified.csv", index=False, encoding="utf-8-sig")
    comp = risk_composition(all_df)
    comp.to_csv(OUT_DIR / "03_risk_composition_with_unlimited.csv", index=False, encoding="utf-8-sig")
    comp_wide = comp.pivot(index="limiter", columns="Risk_Category", values="percent").reset_index()
    comp_wide.to_csv(OUT_DIR / "03_risk_composition_percent_wide.csv", index=False, encoding="utf-8-sig")
    plot_risk_composition(comp, OUT_DIR / "03_risk_composition_with_unlimited.png")


    unlimited_df = all_df[all_df["limiter"] == "unlimited_central"].copy()
    unlimited_pos = unlimited_df[unlimited_df["Positive_Diffusion_For_Envelope"]].copy()
    if len(unlimited_pos):
        unlimited_pos["kappa_over_current_envelope"] = unlimited_pos["kappa_s"] / unlimited_pos["kappa95_current"]
        unlimited_pos.to_csv(OUT_DIR / "04_unlimited_positive_cases_envelope_check.csv", index=False, encoding="utf-8-sig")
        unlimited_summary = coverage_summary(unlimited_pos)
        unlimited_summary.to_csv(OUT_DIR / "04_unlimited_positive_cases_envelope_summary.csv", index=False, encoding="utf-8-sig")

    readme = []
    readme.append("Risk framework three checks from raw input CSV files")
    readme.append("=" * 60)
    readme.append(f"Base directory: {base_dir}")
    readme.append("")
    readme.append("Loaded files:")
    for n in notes_all:
        readme.append(f"- {n}")
    readme.append("")
    readme.append("Task 1: current kappa95 coverage")
    readme.append(coverage_all.to_string(index=False))
    readme.append("")
    readme.append("Task 2: leave-one-limiter-out validation")
    readme.append(loocv.to_string(index=False) if len(loocv) else "No LOOCV result.")
    readme.append("")
    readme.append("Task 3: risk composition with unlimited central")
    readme.append(comp_wide.to_string(index=False))
    readme.append("")
    readme.append("Generated files:")
    for p in sorted(OUT_DIR.glob("*")):
        readme.append(f"- {p.name}")
    (OUT_DIR / "README_summary.txt").write_text("\n".join(readme), encoding="utf-8")

    print("Done. Outputs saved to:", OUT_DIR)
    print("\n".join(readme[:45]))


if __name__ == "__main__":
    main()
