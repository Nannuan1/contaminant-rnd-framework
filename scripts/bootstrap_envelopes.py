# -*- coding: utf-8 -*-
"""Run a stratified bootstrap check for each scheme-specific envelope.

Valid positive-diffusion cases are resampled with replacement within each
``Nw`` stratum. The 95th-percentile values and envelope parameters are then
re-estimated independently for each scheme. First-order upwind uses a constant
envelope; the other estimable schemes use ``A / (Nw + B)``. Schemes with too
few valid positive-diffusion cases are reported as not estimable.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.sf": "Times New Roman",
    "mathtext.tt": "Times New Roman",
    "mathtext.cal": "Times New Roman:italic",
    "mathtext.fallback": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
})
import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
except Exception as exc:  # pragma: no cover - startup guard
    raise RuntimeError("This script requires scipy.optimize.curve_fit.") from exc


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

SCHEME_ORDER = [
    "upwind1",
    "upwind2",
    "central",
    "minmod",
    "vanleer",
    "mc",
    "superbee",
    "unlimited_central",
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

# Pre-specified model family. Do not select a different model after looking at
# each Bootstrap realization; doing so would mix model-selection uncertainty
# with sampling uncertainty.
MODEL_BY_SCHEME = {scheme: "reciprocal" for scheme in SCHEME_ORDER}
MODEL_BY_SCHEME["upwind1"] = "constant"

BASE_THRESHOLDS = {
    "mass_limit_pct": 0.5,
    "l2_rel_limit": 1.0,
    "peak_upper": 1.5,
    "shift_limit_grid": 2.0,
    "boundary_sigma": 4.0,
}

LEFT_MARGIN_SIGMA0 = 10.0
RIGHT_MARGIN_SIGMA_FINAL = 10.0
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_SEED = 20260610
DPI = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-scheme stratified Bootstrap check for kappa95(Nw) envelopes."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Directory containing Expansion_Study_HLLC_Linear_*.csv; default: examples/calibration_smoke.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory; default: outputs/bootstrap_envelopes.",
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        default=None,
        help="Subset of schemes; default: all eight schemes.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parent.parent
    base_dir = (
        Path(args.base_dir).expanduser().resolve()
        if args.base_dir
        else repo_root / "examples" / "calibration_smoke"
    )
    if not base_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base_dir}")
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else repo_root / "outputs" / "bootstrap_envelopes"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, out_dir


def choose_schemes(args: argparse.Namespace) -> List[str]:
    schemes = list(args.schemes) if args.schemes else list(SCHEME_ORDER)
    unknown = [s for s in schemes if s not in SCHEME_FILES]
    if unknown:
        raise ValueError(f"Unknown schemes: {unknown}; available: {list(SCHEME_FILES)}")
    return schemes


def find_input_file(base_dir: Path, scheme: str) -> Path:
    exact = base_dir / SCHEME_FILES[scheme]
    if exact.exists():
        return exact
    raise FileNotFoundError(
        f"Missing exact input for {scheme}: {exact}. "
        "For reproducibility, the script does not substitute similarly named CSV files."
    )


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing(df: pd.DataFrame, names: Iterable[str], required: bool = True) -> pd.Series:
    for name in names:
        if name in df.columns:
            return safe_numeric(df[name])
    if required:
        raise ValueError(f"Missing required columns; tried: {list(names)}")
    return pd.Series(np.nan, index=df.index, dtype=float)


def load_and_standardize(path: Path, scheme: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    d = pd.DataFrame(index=raw.index)
    d["scheme"] = scheme
    d["source_file"] = path.name
    for col in ["Nw", "Pe", "CFL", "tau"]:
        if col not in raw.columns:
            raise ValueError(f"{path.name} is missing required column: {col}")
        d[col] = safe_numeric(raw[col])

    d["Cmax_star"] = first_existing(raw, ["Max_C_Ratio", "Cmax_star"])
    d["L2_rel"] = first_existing(raw, ["L2_rel"], required=False)
    d["mass_error_pct"] = first_existing(
        raw, ["Final_Mass_Error(%)", "mass_error_pct"], required=False
    ).abs()
    d["shift_grid"] = first_existing(
        raw, ["Corrected_Shift", "Normalized_Shift"], required=False
    ).abs()
    d["D_num_peak"] = first_existing(raw, ["D_num_peak"], required=False)
    d["eta_D"] = first_existing(
        raw, ["D_num_peak_over_D_phy", "D_num_peak_star"], required=False
    )
    if d["eta_D"].isna().all() and {"D_num_peak", "D_phy"}.issubset(raw.columns):
        d["eta_D"] = safe_numeric(raw["D_num_peak"]) / safe_numeric(raw["D_phy"])
    d["kappa_s"] = d["eta_D"] / d["Pe"]

    # Same analytically reconstructed boundary margin used by the existing
    # four-limiter sensitivity/Bootstrap script.
    sigma_ratio = np.sqrt(1.0 + 2.0 * d["tau"] / (d["Nw"] * d["Pe"]))
    upstream_margin_sigma = (LEFT_MARGIN_SIGMA0 + d["tau"]) / sigma_ratio
    downstream_margin_sigma = pd.Series(RIGHT_MARGIN_SIGMA_FINAL, index=d.index, dtype=float)
    d["min_boundary_margin_sigma"] = np.minimum(upstream_margin_sigma, downstream_margin_sigma)
    return d


def classify_baseline(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    required = [
        "Nw",
        "Pe",
        "CFL",
        "tau",
        "Cmax_star",
        "L2_rel",
        "mass_error_pct",
        "shift_grid",
        "D_num_peak",
        "eta_D",
        "kappa_s",
        "min_boundary_margin_sigma",
    ]
    finite = np.ones(len(out), dtype=bool)
    for col in required:
        finite &= np.isfinite(out[col].to_numpy(dtype=float))

    valid = (
        finite
        & (out["mass_error_pct"] <= BASE_THRESHOLDS["mass_limit_pct"])
        & (out["L2_rel"] <= BASE_THRESHOLDS["l2_rel_limit"])
        & (out["Cmax_star"] > 0.0)
        & (out["Cmax_star"] <= BASE_THRESHOLDS["peak_upper"])
        & (out["shift_grid"] <= BASE_THRESHOLDS["shift_limit_grid"])
        & (out["min_boundary_margin_sigma"] >= BASE_THRESHOLDS["boundary_sigma"])
    )
    positive = (
        valid
        & (out["Cmax_star"] < 1.0)
        & (out["D_num_peak"] > 0.0)
        & (out["eta_D"] > 0.0)
        & (out["kappa_s"] > 0.0)
    )
    out["valid"] = valid
    out["positive"] = positive
    return out


def q95_points(positive: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for nw, group in positive.groupby("Nw", dropna=False):
        values = group["kappa_s"].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if len(values) == 0:
            continue
        rows.append(
            {
                "Nw": float(nw),
                "n_positive": int(len(values)),
                "kappa_median": float(np.quantile(values, 0.50)),
                "kappa_q95": float(np.quantile(values, 0.95)),
                "kappa_max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows).sort_values("Nw").reset_index(drop=True) if rows else pd.DataFrame()


def reciprocal(x: np.ndarray, A: float, B: float) -> np.ndarray:
    return A / (x + B)


def fit_stats(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    residual = y - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
    }


def fit_envelope(points: pd.DataFrame, model: str) -> Dict[str, float | str | bool]:
    result: Dict[str, float | str | bool] = {
        "success": False,
        "model": model,
        "A_fit": np.nan,
        "B_fit": np.nan,
        "C_fit": np.nan,
        "R2": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "reason": "not_enough_q95_points",
    }
    if points.empty:
        return result
    x = points["Nw"].to_numpy(dtype=float)
    y = points["kappa_q95"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x, y = x[mask], y[mask]

    if model == "constant":
        if len(y) < 2:
            return result
        C = float(np.mean(y))
        pred = np.full_like(y, C)
        result.update({"success": True, "C_fit": C, "reason": "ok"})
        result.update(fit_stats(y, pred))
        return result

    if len(y) < 3:
        return result
    try:
        popt, _ = curve_fit(
            reciprocal,
            x,
            y,
            p0=[max(y) * (min(x) + 1.0), 1.0],
            bounds=([1e-12, 0.0], [np.inf, 1e6]),
            maxfev=50000,
        )
        A, B = float(popt[0]), float(popt[1])
        pred = reciprocal(x, A, B)
        result.update({"success": True, "A_fit": A, "B_fit": B, "reason": "ok"})
        result.update(fit_stats(y, pred))
    except Exception as exc:
        result["reason"] = f"fit_failed:{type(exc).__name__}"
    return result


def predict(x: np.ndarray, fit: Dict[str, float | str | bool]) -> np.ndarray:
    if not bool(fit.get("success", False)):
        return np.full_like(x, np.nan, dtype=float)
    if fit["model"] == "constant":
        return np.full_like(x, float(fit["C_fit"]), dtype=float)
    return reciprocal(x, float(fit["A_fit"]), float(fit["B_fit"]))


def sample_coverage(positive: pd.DataFrame, fit: Dict[str, float | str | bool]) -> float:
    if positive.empty or not bool(fit.get("success", False)):
        return np.nan
    envelope = predict(positive["Nw"].to_numpy(dtype=float), fit)
    observed = positive["kappa_s"].to_numpy(dtype=float)
    ok = np.isfinite(envelope) & np.isfinite(observed)
    return float(np.mean(observed[ok] <= envelope[ok])) if np.any(ok) else np.nan


def bootstrap_one_scheme(
    positive: pd.DataFrame,
    model: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if positive.empty:
        return pd.DataFrame(), pd.DataFrame()

    groups = {
        float(nw): group["kappa_s"].to_numpy(dtype=float)
        for nw, group in positive.groupby("Nw", dropna=False)
    }
    groups = {
        nw: values[np.isfinite(values) & (values > 0.0)]
        for nw, values in groups.items()
        if np.isfinite(nw)
    }
    nw_values = np.array(sorted(groups), dtype=float)
    fit_rows: List[Dict[str, float | str | bool | int]] = []
    curve_rows: List[Dict[str, float | int]] = []

    for bootstrap_id in range(n_bootstrap):
        points = []
        for nw in nw_values:
            values = groups[float(nw)]
            if len(values) == 0:
                continue
            resampled = rng.choice(values, size=len(values), replace=True)
            points.append({"Nw": nw, "kappa_q95": float(np.quantile(resampled, 0.95))})
        points_df = pd.DataFrame(points)
        fit = fit_envelope(points_df, model)
        fit_rows.append({"bootstrap_id": bootstrap_id, **fit})
        if bool(fit["success"]):
            values = predict(nw_values, fit)
            for nw, value in zip(nw_values, values):
                curve_rows.append(
                    {
                        "bootstrap_id": bootstrap_id,
                        "Nw": float(nw),
                        "kappa95_bootstrap": float(value),
                    }
                )

    return pd.DataFrame(fit_rows), pd.DataFrame(curve_rows)


def percentile_summary(values: pd.Series, prefix: str) -> Dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {
            f"{prefix}_median": np.nan,
            f"{prefix}_low95": np.nan,
            f"{prefix}_high95": np.nan,
            f"{prefix}_low90": np.nan,
            f"{prefix}_high90": np.nan,
        }
    return {
        f"{prefix}_median": float(x.quantile(0.50)),
        f"{prefix}_low95": float(x.quantile(0.025)),
        f"{prefix}_high95": float(x.quantile(0.975)),
        f"{prefix}_low90": float(x.quantile(0.05)),
        f"{prefix}_high90": float(x.quantile(0.95)),
    }


def summarize_bootstrap_parameters(scheme: str, fit_df: pd.DataFrame, n_requested: int) -> Dict[str, float | str | int]:
    if fit_df.empty:
        return {
            "scheme": scheme,
            "scheme_display": SCHEME_DISPLAY[scheme],
            "model": MODEL_BY_SCHEME[scheme],
            "n_bootstrap_requested": n_requested,
            "n_bootstrap_success": 0,
            "fit_success_rate": 0.0,
            "A_median": np.nan,
            "A_low95": np.nan,
            "A_high95": np.nan,
            "B_median": np.nan,
            "B_low95": np.nan,
            "B_high95": np.nan,
            "C_median": np.nan,
            "C_low95": np.nan,
            "C_high95": np.nan,
        }
    success = fit_df[fit_df["success"].astype(bool)].copy()
    row: Dict[str, float | str | int] = {
        "scheme": scheme,
        "scheme_display": SCHEME_DISPLAY[scheme],
        "model": MODEL_BY_SCHEME[scheme],
        "n_bootstrap_requested": n_requested,
        "n_bootstrap_success": int(len(success)),
        "fit_success_rate": float(len(success) / n_requested) if n_requested else np.nan,
    }
    row.update(percentile_summary(success.get("A_fit", pd.Series(dtype=float)), "A"))
    row.update(percentile_summary(success.get("B_fit", pd.Series(dtype=float)), "B"))
    row.update(percentile_summary(success.get("C_fit", pd.Series(dtype=float)), "C"))
    return row


def summarize_curve(scheme: str, curve_df: pd.DataFrame) -> pd.DataFrame:
    if curve_df.empty:
        return pd.DataFrame()
    summary = (
        curve_df.groupby("Nw")["kappa95_bootstrap"]
        .agg(
            kappa95_boot_median="median",
            kappa95_boot_low95=lambda x: x.quantile(0.025),
            kappa95_boot_high95=lambda x: x.quantile(0.975),
            kappa95_boot_low90=lambda x: x.quantile(0.05),
            kappa95_boot_high90=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    summary.insert(0, "scheme_display", SCHEME_DISPLAY[scheme])
    summary.insert(0, "scheme", scheme)
    return summary


def curve_ci_on_dense_grid(fit_df: pd.DataFrame, model: str, x_dense: np.ndarray) -> pd.DataFrame:
    if fit_df.empty:
        return pd.DataFrame()
    successful = fit_df[fit_df["success"].astype(bool)]
    curves = []
    for _, row in successful.iterrows():
        fit = row.to_dict()
        y = predict(x_dense, fit)
        if np.isfinite(y).all():
            curves.append(y)
    if not curves:
        return pd.DataFrame()
    matrix = np.vstack(curves)
    return pd.DataFrame(
        {
            "Nw": x_dense,
            "median": np.quantile(matrix, 0.50, axis=0),
            "low95": np.quantile(matrix, 0.025, axis=0),
            "high95": np.quantile(matrix, 0.975, axis=0),
        }
    )


def plot_scheme(
    scheme: str,
    points: pd.DataFrame,
    baseline_fit: Dict[str, float | str | bool],
    fit_df: pd.DataFrame,
    n_positive: int,
    out_dir: Path,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    standalone = ax is None

    if standalone:
        _, ax = plt.subplots(figsize=(8.5, 6.5))


    ax.set_title(
        f"{SCHEME_DISPLAY[scheme]}  (positive n={n_positive})",
        fontsize=22,
        pad=10,
    )
    ax.set_xlabel(r"$N_w$", fontsize=20)
    ax.set_ylabel(r"$\kappa_{s,95}$", fontsize=20)
    ax.tick_params(axis="both", labelsize=15)

    if points.empty or not bool(baseline_fit.get("success", False)):
        ax.text(
            0.5,
            0.52,
            "No estimable positive-diffusion envelope\nunder baseline screening",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        x_min, x_max = float(points["Nw"].min()), float(points["Nw"].max())
        x_dense = np.linspace(x_min, x_max, 240)
        ci = curve_ci_on_dense_grid(fit_df, str(baseline_fit["model"]), x_dense)
        if not ci.empty:
            ax.fill_between(
                ci["Nw"].to_numpy(),
                ci["low95"].to_numpy(),
                ci["high95"].to_numpy(),
                color="#7EA6D8",
                alpha=0.30,
                label="Bootstrap 95% CI",
                zorder=1,
            )
            ax.plot(ci["Nw"], ci["median"], color="#1F4E79", linewidth=1.6, label="Bootstrap median")
        ax.plot(
            x_dense,
            predict(x_dense, baseline_fit),
            color="#C00000",
            linestyle="--",
            linewidth=1.4,
            label="Original fit",
            zorder=3,
        )
        ax.scatter(
            points["Nw"],
            points["kappa_q95"],
            s=32,
            color="black",
            edgecolor="white",
            linewidth=0.5,
            label="Observed group q95",
            zorder=4,
        )
        if str(baseline_fit["model"]) == "constant":
            fit_title = rf"Fit: $\kappa_{{s,95}}={float(baseline_fit['C_fit']):.4f}$"
        else:
            fit_title = (
                rf"Fit: $\kappa_{{s,95}}={float(baseline_fit['A_fit']):.4f}"
                rf"/(N_w+{float(baseline_fit['B_fit']):.4f})$"
            )
        r2 = float(baseline_fit.get("R2", np.nan))
        if np.isfinite(r2) and str(baseline_fit["model"]) != "constant":
            fit_title += "\n" + rf"$R^2={r2:.3f}$"
        ax.legend(
            frameon=False,
            fontsize=14,
            title=fit_title,
            title_fontsize=12,
            loc="best",
        )

    if standalone:
        plt.tight_layout()
        for suffix in ["png", "svg"]:
            plt.savefig(out_dir / f"bootstrap_{scheme}.{suffix}", dpi=DPI, bbox_inches="tight")
        plt.close()
    return ax


def main() -> None:
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")
    base_dir, out_dir = resolve_paths(args)
    schemes = choose_schemes(args)
    rng = np.random.default_rng(args.seed)

    baseline_rows = []
    all_points = []
    all_bootstrap_fits = []
    parameter_rows = []
    all_curve_summary = []
    plot_payload = {}

    for scheme in schemes:
        path = find_input_file(base_dir, scheme)
        standardized = load_and_standardize(path, scheme)
        classified = classify_baseline(standardized)
        positive = classified[classified["positive"]].copy()
        positive = positive[
            np.isfinite(positive["Nw"])
            & np.isfinite(positive["kappa_s"])
            & (positive["kappa_s"] > 0.0)
        ]
        points = q95_points(positive)
        if not points.empty:
            points.insert(0, "scheme_display", SCHEME_DISPLAY[scheme])
            points.insert(0, "scheme", scheme)
            all_points.append(points)

        model = MODEL_BY_SCHEME[scheme]
        baseline_fit = fit_envelope(points, model)
        coverage = sample_coverage(positive, baseline_fit)
        baseline_rows.append(
            {
                "scheme": scheme,
                "scheme_display": SCHEME_DISPLAY[scheme],
                "source_file": path.name,
                "model": model,
                "n_total": int(len(classified)),
                "n_valid": int(classified["valid"].sum()),
                "n_positive": int(len(positive)),
                "n_Nw_groups": int(len(points)),
                "A_fit": baseline_fit["A_fit"],
                "B_fit": baseline_fit["B_fit"],
                "C_fit": baseline_fit["C_fit"],
                "R2": baseline_fit["R2"],
                "RMSE": baseline_fit["RMSE"],
                "MAE": baseline_fit["MAE"],
                "positive_sample_coverage": coverage,
                "fit_success": baseline_fit["success"],
                "fit_reason": baseline_fit["reason"],
            }
        )

        fit_df, curve_df = bootstrap_one_scheme(positive, model, args.n_bootstrap, rng)
        if not fit_df.empty:
            fit_df.insert(0, "scheme_display", SCHEME_DISPLAY[scheme])
            fit_df.insert(0, "scheme", scheme)
            all_bootstrap_fits.append(fit_df)
        parameter_rows.append(summarize_bootstrap_parameters(scheme, fit_df, args.n_bootstrap))
        curve_summary = summarize_curve(scheme, curve_df)
        if not curve_summary.empty:
            all_curve_summary.append(curve_summary)

        plot_payload[scheme] = (points, baseline_fit, fit_df, len(positive))
        plot_scheme(scheme, points, baseline_fit, fit_df, len(positive), out_dir)
        print(
            f"{scheme:20s} total={len(classified):4d} valid={int(classified['valid'].sum()):4d} "
            f"positive={len(positive):4d} groups={len(points):2d} "
            f"bootstrap_success={int(fit_df['success'].sum()) if not fit_df.empty else 0}/{args.n_bootstrap}"
        )

    baseline_df = pd.DataFrame(baseline_rows)
    points_df = pd.concat(all_points, ignore_index=True) if all_points else pd.DataFrame()
    fits_df = pd.concat(all_bootstrap_fits, ignore_index=True) if all_bootstrap_fits else pd.DataFrame()
    params_df = pd.DataFrame(parameter_rows)
    curves_df = pd.concat(all_curve_summary, ignore_index=True) if all_curve_summary else pd.DataFrame()

    baseline_df.to_csv(out_dir / "00_baseline_fit_and_coverage_by_scheme.csv", index=False, encoding="utf-8-sig")
    points_df.to_csv(out_dir / "01_baseline_q95_points_by_scheme.csv", index=False, encoding="utf-8-sig")
    fits_df.to_csv(out_dir / "02_bootstrap_fit_parameters_all_schemes.csv", index=False, encoding="utf-8-sig")
    params_df.to_csv(out_dir / "03_bootstrap_parameter_CI_by_scheme.csv", index=False, encoding="utf-8-sig")
    curves_df.to_csv(out_dir / "04_bootstrap_envelope_CI_by_scheme_Nw.csv", index=False, encoding="utf-8-sig")



    combined_schemes = [scheme for scheme in schemes if scheme != "central"]
    n_cols = 2
    n_rows = max(1, math.ceil(len(combined_schemes) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(17, 6.5 * n_rows),
        squeeze=False,
    )
    for panel_index, (ax, scheme) in enumerate(zip(axes.ravel(), combined_schemes)):
        points, baseline_fit, fit_df, n_positive = plot_payload[scheme]
        plot_scheme(scheme, points, baseline_fit, fit_df, n_positive, out_dir, ax=ax)

        ax.text(
            -0.10, 1.035,
            f"({chr(ord('a') + panel_index)})",
            transform=ax.transAxes,
            fontsize=22,
            fontfamily="Times New Roman",
            ha="left", va="bottom",
            clip_on=False,
        )
    for ax in axes.ravel()[len(combined_schemes):]:
        ax.axis("off")
    fig.suptitle(r"Per-scheme Bootstrap uncertainty of $\kappa_{s,95}(N_w)$ envelopes", fontsize=22, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.5, w_pad=2.0)
    for suffix in ["png", "svg"]:
        fig.savefig(out_dir / f"bootstrap_all_schemes.{suffix}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    readme_lines = [
        "Per-scheme stratified Bootstrap envelope check",
        "=" * 56,
        f"Base directory: {base_dir}",
        f"Bootstrap replicates per estimable scheme: {args.n_bootstrap}",
        f"Random seed: {args.seed}",
        "Resampling strata: Nw within each scheme",
        "Models: upwind1 = constant C; other schemes = A/(Nw+B)",
        "Baseline screening includes reconstructed boundary margin >= 4 sigma.",
        "",
        "Scientific interpretation:",
        "- Bootstrap CI quantifies finite-sample uncertainty for each scheme separately.",
        "- A scheme with no/too few positive-diffusion groups is reported as not estimable.",
        "- Bootstrap does not replace independent parameter-combination or external validation.",
        "",
        "Generated files:",
    ]
    readme_lines.extend(f"- {p.name}" for p in sorted(out_dir.iterdir()) if p.is_file())
    (out_dir / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"\nOutput directory: {out_dir}")
    print(baseline_df[["scheme", "model", "n_positive", "positive_sample_coverage", "fit_success"]].to_string(index=False))


if __name__ == "__main__":
    main()
