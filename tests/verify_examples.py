#!/usr/bin/env python3
"""Check the compact examples and manuscript reference values."""

from __future__ import annotations

import csv
import io
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> int:
    smoke_files = sorted((ROOT / "examples" / "calibration_smoke").glob("Expansion_Study_*.csv"))
    require(len(smoke_files) == 8, "Eight scheme-specific smoke tables are present")
    require(all(len(rows(path)) == 16 for path in smoke_files), "Each smoke table contains 16 records")

    gaussian = ROOT / "examples" / "gaussian_mc_case"
    for name in ("reach_C_profile.csv", "reach_analytical_profile.csv", "mass_conservation.csv"):
        require((gaussian / name).exists(), f"Gaussian example contains {name}")

    non_gaussian = ROOT / "examples" / "non_gaussian_v9_cosine_bell"
    for scheme in ("vanleer", "mc"):
        for name in ("reach_C_profile.csv", "reach_analytical_profile.csv", "mass_conservation.csv"):
            require((non_gaussian / scheme / name).exists(), f"V9 {scheme} example contains {name}")

    atlas = rows(ROOT / "reference" / "06_scheme_recommendation_atlas_full_Nw_Pe_CFL_tau.csv")
    require(len(atlas) == 672, "Frozen recommendation atlas contains 672 nodes")
    counts = Counter(row["recommended_scheme"] for row in atlas)
    require(counts["no_reliable_scheme"] == 22, "Atlas contains 22 no-reliable-scheme nodes")

    risk = rows(ROOT / "reference" / "02_scheme_risk_composition_wide.csv")
    require(len(risk) == 8, "Risk-composition table contains eight schemes")

    prior = {row["case_id"]: row for row in rows(ROOT / "reference" / "non_gaussian_preregistered_recommendations.csv")}
    v9 = prior["NG_V9_high_risk_cosine_bell"]
    require(v9["recommended_scheme"] == "vanleer", "V9 cosine-bell prior choice is van Leer")

    summary = {row["case_id"]: row for row in rows(ROOT / "reference" / "non_gaussian_transferability_summary.csv")}
    v9_result = summary["NG_V9_high_risk_cosine_bell"]
    require(int(float(v9_result["decision_validation_rank"])) == 2, "V9 van Leer choice ranks second after simulation")
    require(v9_result["actual_best_scheme"] == "mc", "V9 post-simulation minimum-score scheme is MC")

    top1 = sum(int(float(row["top1_hit"])) for row in summary.values())
    top2 = sum(int(float(row["top2_hit"])) for row in summary.values())
    require(top1 == 4 and top2 == 6, "Six validation cases reproduce the reported Top-1 and Top-2 counts")

    han = re.compile(r"[\u4e00-\u9fff]")
    comment_hits = []
    for path in (ROOT / "scripts").glob("*.py"):
        text = path.read_text(encoding="utf-8-sig")
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT and han.search(token.string):
                comment_hits.append(f"{path.name}:{token.start[0]}")
    require(not comment_hits, "Source-code comments are written in English")

    public_docs = [ROOT / "README.md", ROOT / "SOFTWARE_AND_DATA_AVAILABILITY.md"]
    public_docs.extend((ROOT / "docs").glob("*.md"))
    non_english_docs = [path.name for path in public_docs if han.search(path.read_text(encoding="utf-8"))]
    require(not non_english_docs, "Public documentation is written in English")

    print("\nAll compact-repository checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

