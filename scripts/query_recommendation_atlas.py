#!/usr/bin/env python3
"""Query the frozen Gaussian-calibrated scheme recommendation atlas.

The script never reads non-Gaussian simulation results.  It accepts one point
or a CSV of points defined by (Nw, Pe, CFL, tau), returns a preregistered scheme
decision, and writes per-scheme diagnostic quantities.

Decision rule
-------------
1. At an exact atlas node, use the stored recommendation directly.
2. Between atlas nodes, perform multilinear voting over the surrounding atlas
   cells.  Pe is interpolated in log space, matching the atlas representation;
   Nw, CFL, and tau are interpolated linearly.
3. If the leading vote is below ``--min-consensus`` (default 0.50), retain the
   leading scheme for stress testing but mark the decision
   ``ambiguous_boundary``.  It must not be described as a unique recommendation.
4. A scheme without a calibrated positive-diffusion envelope has RND=NA.  It is
   never assigned RND=0.

The local Ep/L2/reliability interpolation written to the diagnostic CSV is an
audit aid only.  The categorical atlas vote is the frozen decision, so the
non-Gaussian outcomes cannot feed back into the recommendation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ATLAS_DIR = SCRIPT_DIR.parent / "reference"

PARAMETERS = ("Nw", "Pe", "CFL", "tau")
RECOMMENDATION_SCHEMES = (
    "upwind1",
    "upwind2",
    "central",
    "minmod",
    "vanleer",
    "mc",
    "superbee",
)
SCHEME_DISPLAY = {
    "upwind1": "1st upwind",
    "upwind2": "2nd upwind",
    "central": "central",
    "minmod": "minmod",
    "vanleer": "van Leer",
    "mc": "MC",
    "superbee": "superbee",
    "no_reliable_scheme": "no reliable scheme",
}
SCHEME_COMPLEXITY = {
    "upwind1": 1.0,
    "central": 1.0,
    "upwind2": 2.0,
    "minmod": 3.0,
    "vanleer": 3.0,
    "mc": 3.0,
    "superbee": 3.0,
}

WEIGHT_RND = 1.0
WEIGHT_EP = 0.25
WEIGHT_L2 = 0.10
WEIGHT_COMPLEXITY = 0.02

INVALID_TOL = 0.10
COMPRESSION_TOL = 0.10
EP_LIMIT = 0.10
L2_LIMIT = 1.0

DECISION_RULE_VERSION = "gaussian_atlas_multilinear_vote_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query and freeze Gaussian-calibrated scheme recommendations."
    )
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, default=None)
    parser.add_argument("--nw", type=float, default=None)
    parser.add_argument("--pe", type=float, default=None)
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--case-id", type=str, default="query_case")
    parser.add_argument("--profile-shape", type=str, default="unspecified")
    parser.add_argument(
        "--min-consensus",
        type=float,
        default=0.50,
        help="Minimum leading atlas vote for a unique interpolated recommendation.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def close_mask(series: pd.Series, value: float) -> pd.Series:
    return np.isclose(pd.to_numeric(series, errors="coerce"), float(value), rtol=1e-10, atol=1e-12)


def bracket_with_weights(values: Sequence[float], target: float, log_space: bool = False) -> List[Tuple[float, float]]:
    grid = np.sort(np.unique(np.asarray(values, dtype=float)))
    if len(grid) == 0:
        raise ValueError("Empty atlas axis.")
    tol = 1e-10 * max(1.0, abs(float(target)))
    if target < grid[0] - tol or target > grid[-1] + tol:
        raise ValueError(
            f"value {target:g} is outside calibrated range [{grid[0]:g}, {grid[-1]:g}]"
        )
    exact = grid[np.isclose(grid, target, rtol=1e-10, atol=1e-12)]
    if len(exact):
        return [(float(exact[0]), 1.0)]
    lo = float(grid[grid < target].max())
    hi = float(grid[grid > target].min())
    if log_space:
        if lo <= 0 or hi <= 0 or target <= 0:
            raise ValueError("Log-space interpolation requires positive values.")
        fraction = (np.log(target) - np.log(lo)) / (np.log(hi) - np.log(lo))
    else:
        fraction = (target - lo) / (hi - lo)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return [(lo, 1.0 - fraction), (hi, fraction)]


def build_corners(atlas: pd.DataFrame, point: Dict[str, float]) -> List[Tuple[Dict[str, float], float]]:
    axes = {
        name: bracket_with_weights(
            pd.to_numeric(atlas[name], errors="coerce").dropna().unique(),
            point[name],
            log_space=(name == "Pe"),
        )
        for name in PARAMETERS
    }
    corners: List[Tuple[Dict[str, float], float]] = []
    for choices in itertools.product(*(axes[name] for name in PARAMETERS)):
        coordinates = {name: float(choice[0]) for name, choice in zip(PARAMETERS, choices)}
        weight = float(np.prod([choice[1] for choice in choices]))
        if weight > 0:
            corners.append((coordinates, weight))
    total = sum(weight for _, weight in corners)
    if total <= 0:
        raise RuntimeError("Interpolation corner weights sum to zero.")
    return [(coordinates, weight / total) for coordinates, weight in corners]


def row_at_coordinates(df: pd.DataFrame, coordinates: Dict[str, float]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for name, value in coordinates.items():
        mask &= close_mask(df[name], value)
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one atlas row at {coordinates}, found {len(rows)}."
        )
    return rows.iloc[0]


def predict_kappa95(nw: float, scheme: str, fit_df: pd.DataFrame) -> float:
    rows = fit_df.loc[fit_df["scheme"].astype(str) == scheme]
    if len(rows) != 1:
        return np.nan
    row = rows.iloc[0]
    model = str(row.get("recommended_envelope_model", ""))
    if model == "constant":
        value = pd.to_numeric(pd.Series([row.get("constant_kappa95")]), errors="coerce").iloc[0]
        return float(value) if np.isfinite(value) else np.nan
    if model == "rational_A_over_Nw_plus_B":
        a = pd.to_numeric(pd.Series([row.get("rational_A")]), errors="coerce").iloc[0]
        b = pd.to_numeric(pd.Series([row.get("rational_B")]), errors="coerce").iloc[0]
        if np.isfinite(a) and np.isfinite(b):
            return float(a) / (float(nw) + float(b))
    return np.nan


def atlas_vote(
    atlas: pd.DataFrame,
    point: Dict[str, float],
    min_consensus: float,
) -> Tuple[Dict[str, object], List[Tuple[Dict[str, float], float]]]:
    exact_mask = pd.Series(True, index=atlas.index)
    for name in PARAMETERS:
        exact_mask &= close_mask(atlas[name], point[name])
    exact_rows = atlas.loc[exact_mask]
    if len(exact_rows) == 1:
        row = exact_rows.iloc[0]
        scheme = str(row["recommended_scheme"])
        result = {
            "recommended_scheme": scheme,
            "runner_up_scheme": "",
            "recommendation_status": str(row.get("recommendation_status", "exact_atlas_node")),
            "atlas_consensus": 1.0,
            "atlas_vote_distribution": json.dumps({scheme: 1.0}, ensure_ascii=False, sort_keys=True),
            "decision_source": "exact_atlas_node",
            "corner_count": 1,
        }
        return result, [({name: point[name] for name in PARAMETERS}, 1.0)]

    corners = build_corners(atlas, point)
    votes: Dict[str, float] = {}
    for coordinates, weight in corners:
        row = row_at_coordinates(atlas, coordinates)
        scheme = str(row["recommended_scheme"])
        votes[scheme] = votes.get(scheme, 0.0) + weight
    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    leading_scheme, consensus = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else ""
    if leading_scheme == "no_reliable_scheme":
        status = "no_reliable_scheme"
    elif consensus + 1e-12 >= min_consensus:
        status = "recommended_interpolated"
    else:
        status = "ambiguous_boundary"
    result = {
        # For an ambiguous boundary this is the leading candidate, not a unique optimum.
        "recommended_scheme": leading_scheme,
        "runner_up_scheme": runner_up,
        "recommendation_status": status,
        "atlas_consensus": float(consensus),
        "atlas_vote_distribution": json.dumps(
            {scheme: float(weight) for scheme, weight in ranked},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "decision_source": "multilinear_atlas_vote_logPe",
        "corner_count": len(corners),
    }
    return result, corners


def local_scheme_diagnostics(
    classified: pd.DataFrame,
    fit_df: pd.DataFrame,
    point: Dict[str, float],
    corners: List[Tuple[Dict[str, float], float]],
    case_id: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scheme in RECOMMENDATION_SCHEMES:
        scheme_df = classified.loc[classified["scheme"].astype(str) == scheme]
        invalid_weight = 0.0
        compression_weight = 0.0
        ep_weighted = 0.0
        l2_weighted = 0.0
        valid_metric_weight = 0.0
        for coordinates, weight in corners:
            mask = pd.Series(True, index=scheme_df.index)
            for name, value in coordinates.items():
                mask &= close_mask(scheme_df[name], value)
            corner_rows = scheme_df.loc[mask]
            if len(corner_rows) != 1:
                raise ValueError(
                    f"Expected one classified row for {scheme} at {coordinates}, found {len(corner_rows)}."
                )
            row = corner_rows.iloc[0]
            valid = as_bool(row["Valid_For_Screening"])
            obvious = as_bool(row["Obvious_Compression_or_Antidiffusion"])
            invalid_weight += weight * (not valid)
            compression_weight += weight * obvious
            ep = pd.to_numeric(pd.Series([row.get("Ep")]), errors="coerce").iloc[0]
            l2 = pd.to_numeric(pd.Series([row.get("L2_rel")]), errors="coerce").iloc[0]
            if valid and np.isfinite(ep) and np.isfinite(l2):
                ep_weighted += weight * float(ep)
                l2_weighted += weight * float(l2)
                valid_metric_weight += weight

        ep_interp = ep_weighted / valid_metric_weight if valid_metric_weight > 0 else np.nan
        l2_interp = l2_weighted / valid_metric_weight if valid_metric_weight > 0 else np.nan
        kappa95 = predict_kappa95(point["Nw"], scheme, fit_df)
        rnd = kappa95 * point["Pe"] if np.isfinite(kappa95) else np.nan
        reasons: List[str] = []
        if not np.isfinite(kappa95):
            reasons.append("no_positive_diffusion_envelope")
        if invalid_weight > INVALID_TOL:
            reasons.append(f"local_invalid_weight>{INVALID_TOL:g}")
        if compression_weight > COMPRESSION_TOL:
            reasons.append(f"local_compression_weight>{COMPRESSION_TOL:g}")
        if np.isfinite(ep_interp) and ep_interp > EP_LIMIT:
            reasons.append(f"local_Ep>{EP_LIMIT:g}")
        if np.isfinite(l2_interp) and l2_interp > L2_LIMIT:
            reasons.append(f"local_L2>{L2_LIMIT:g}")
        diagnostic_admissible = len(reasons) == 0
        score = np.nan
        if diagnostic_admissible:
            score = (
                WEIGHT_RND * rnd
                + WEIGHT_EP * ep_interp
                + WEIGHT_L2 * l2_interp
                + WEIGHT_COMPLEXITY * SCHEME_COMPLEXITY[scheme]
            )
        rows.append(
            {
                "case_id": case_id,
                **point,
                "scheme": scheme,
                "scheme_display": SCHEME_DISPLAY[scheme],
                "kappa95_scheme": kappa95,
                "RND_scheme": rnd,
                "local_invalid_weight": invalid_weight,
                "local_obvious_compression_weight": compression_weight,
                "local_Ep_interpolated": ep_interp,
                "local_L2_interpolated": l2_interp,
                "complexity": SCHEME_COMPLEXITY[scheme],
                "diagnostic_admissible": int(diagnostic_admissible),
                "diagnostic_score": score,
                "diagnostic_rejection_reasons": ";".join(reasons),
                "diagnostic_note": "audit_only_not_used_for_frozen_atlas_vote",
            }
        )
    return rows


def load_queries(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv is not None:
        queries = pd.read_csv(args.input_csv)
        require_columns(queries, ["case_id", *PARAMETERS], args.input_csv)
        if "profile_shape" not in queries.columns:
            queries["profile_shape"] = "unspecified"
        return queries
    values = [args.nw, args.pe, args.cfl, args.tau]
    if any(value is None for value in values):
        raise ValueError(
            "Provide --input-csv, or provide all of --nw, --pe, --cfl, and --tau."
        )
    return pd.DataFrame(
        [
            {
                "case_id": args.case_id,
                "profile_shape": args.profile_shape,
                "Nw": args.nw,
                "Pe": args.pe,
                "CFL": args.cfl,
                "tau": args.tau,
            }
        ]
    )


def main() -> None:
    args = parse_args()
    atlas_dir = args.atlas_dir.expanduser().resolve()
    full_atlas_path = atlas_dir / "06_scheme_recommendation_atlas_full_Nw_Pe_CFL_tau.csv"
    classified_path = atlas_dir / "01_all_schemes_classified_cases.csv"
    fit_path = atlas_dir / "03_scheme_kappa95_fit.csv"
    for path in (full_atlas_path, fit_path):
        if not path.exists():
            raise FileNotFoundError(path)

    atlas = pd.read_csv(full_atlas_path)
    classified = pd.read_csv(classified_path) if classified_path.exists() else None
    fit_df = pd.read_csv(fit_path)
    require_columns(atlas, [*PARAMETERS, "recommended_scheme"], full_atlas_path)
    if classified is not None:
        require_columns(
            classified,
            [
                "scheme",
                *PARAMETERS,
                "Valid_For_Screening",
                "Obvious_Compression_or_Antidiffusion",
                "Ep",
                "L2_rel",
            ],
            classified_path,
        )
    require_columns(
        fit_df,
        ["scheme", "recommended_envelope_model"],
        fit_path,
    )

    queries = load_queries(args)
    frozen_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    created_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    atlas_hash = sha256_file(full_atlas_path)
    fit_hash = sha256_file(fit_path)

    for _, query in queries.iterrows():
        case_id = str(query["case_id"])
        point = {name: float(query[name]) for name in PARAMETERS}
        decision, corners = atlas_vote(atlas, point, args.min_consensus)
        recommended = str(decision["recommended_scheme"])
        runner_up = str(decision["runner_up_scheme"])
        kappa95 = predict_kappa95(point["Nw"], recommended, fit_df)
        recommended_rnd = kappa95 * point["Pe"] if np.isfinite(kappa95) else np.nan
        frozen_rows.append(
            {
                "case_id": case_id,
                "profile_shape": str(query.get("profile_shape", "unspecified")),
                **point,
                "recommended_scheme": recommended,
                "recommended_scheme_display": SCHEME_DISPLAY.get(recommended, recommended),
                "runner_up_scheme": runner_up,
                "runner_up_scheme_display": SCHEME_DISPLAY.get(runner_up, runner_up),
                "recommendation_status": decision["recommendation_status"],
                "is_unique_recommendation": int(
                    decision["recommendation_status"]
                    in {"recommended", "recommended_interpolated", "exact_atlas_node"}
                ),
                "atlas_consensus": decision["atlas_consensus"],
                "atlas_vote_distribution": decision["atlas_vote_distribution"],
                "decision_source": decision["decision_source"],
                "corner_count": decision["corner_count"],
                "recommended_kappa95": kappa95,
                "recommended_RND": recommended_rnd,
                "decision_rule_version": DECISION_RULE_VERSION,
                "min_consensus": args.min_consensus,
                "atlas_sha256": atlas_hash,
                "fit_sha256": fit_hash,
                "created_utc": created_utc,
            }
        )
        if classified is not None:
            diagnostic_rows.extend(
                local_scheme_diagnostics(classified, fit_df, point, corners, case_id)
            )

    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frozen_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    details_csv = None
    if diagnostic_rows:
        details_csv = (
            args.details_csv.expanduser().resolve()
            if args.details_csv is not None
            else output_csv.with_name(output_csv.stem + "_scheme_diagnostics.csv")
        )
        details_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostic_rows).to_csv(details_csv, index=False, encoding="utf-8-sig")

    show = pd.DataFrame(frozen_rows)[
        [
            "case_id",
            "profile_shape",
            "Nw",
            "Pe",
            "CFL",
            "tau",
            "recommended_scheme_display",
            "recommendation_status",
            "atlas_consensus",
            "runner_up_scheme_display",
        ]
    ]
    print(show.to_string(index=False))
    print(f"\nFrozen recommendations: {output_csv}")
    if details_csv is not None:
        print(f"Per-scheme diagnostics: {details_csv}")
    else:
        print("Per-scheme diagnostics were skipped because the full classified-case table is not included.")


if __name__ == "__main__":
    main()
