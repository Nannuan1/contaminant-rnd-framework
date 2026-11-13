# Gaussian-calibrated numerical-diffusion risk framework

This repository provides the numerical code and compact test cases associated
with the manuscript **“A Gaussian-Calibrated Framework for Numerical Diffusion
Assessment in Modelling Contaminant Transport in River Channels.”**

The study addresses a practical question that arises before a contaminant
transport simulation is run: for a specified plume width, grid spacing,
physical diffusion coefficient, and time step, is the numerical diffusion of a
candidate scheme likely to interfere with the physical diffusion? Gaussian
benchmark calculations are used to calibrate a scheme-specific risk index,
`RND_s`, and a recommendation atlas for eight finite-volume schemes.

## Contents

```text
scripts/      Simulation, calibration, robustness, and atlas-query code
examples/     Compact Gaussian and non-Gaussian test cases
reference/    Envelope coefficients, risk composition, atlas, and validation summary
tests/        Dependency-free integrity and regression checks
docs/         Method and data-field documentation
```

The repository is intentionally compact. It does not contain manuscript files,
duplicate development scripts, all intermediate figures, or the complete set
of grid-level outputs from every benchmark run.

The reference tables are study outputs. The included examples test data access
and the published decision workflow; the full calibration database is needed
to reproduce all fitted coefficients independently.

## Methods represented in the repository

1. `scripts/run_gaussian_scheme.py` evaluates the four MUSCL/TVD schemes and
   three baseline advection schemes. `run_gaussian_unlimited_central.py`
   evaluates the unlimited central reconstruction used as a high-risk control.
2. `scripts/build_rnd_atlas.py` screens invalid, compressive, and
   positive-diffusion cases; fits scheme-specific 95% engineering envelopes;
   and constructs the recommendation atlas.
3. `scripts/query_recommendation_atlas.py` queries the frozen Gaussian atlas at
   a prescribed `(Nw, Pe, time-step factor, tau)` point.
4. `scripts/run_non_gaussian_validation.py` evaluates top-hat and cosine-bell
   plumes against convolution reference solutions without refitting the
   Gaussian envelopes.

The score used for scheme comparison is

```text
Score_s = RND_s + 0.25 E_p + 0.10 L2 + 0.02 Complexity.
```

Non-Gaussian results are used only for external validation. They are not used
to recalibrate `RND_s` or modify the recommendation atlas.

## Included examples

### Gaussian example

`examples/gaussian_mc_case/` contains the analytical and numerical profiles for
the MC scheme at

```text
Nw = 2.0, Pe = 10.0, time-step factor = 0.3, tau = 30.
```

The folder includes the final concentration profile, analytical profile,
water-level profile, and mass-conservation record.

### Calibration smoke data

`examples/calibration_smoke/` contains 16 records for each of the eight schemes.
These records are a small subset of the calibration database and are intended
to check file handling and analysis execution. They are not intended to replace
the complete 5,376-case calibration database or to reproduce the final fitted
coefficients.

### Non-Gaussian example

`examples/non_gaussian_v9_cosine_bell/` contains numerical and convolution
reference profiles for van Leer and MC at

```text
Nw = 1.25, Pe = 8.0, time-step factor = 0.4, tau = 75.
```

The Gaussian-calibrated atlas selected van Leer before the non-Gaussian
simulation. In the subsequent validation ranking, van Leer ranked second and MC
ranked first. This case is included because it demonstrates how the framework
is evaluated when the prior choice is competitive but not the post-simulation
minimum-score scheme.

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The source code uses NumPy, pandas, SciPy, Matplotlib, and Plotly. The integrity
test below uses only the Python standard library.

## Quick checks

Verify the repository contents and the reported reference values:

```bash
python tests/verify_examples.py
```

Run the compact calibration subset:

```bash
python scripts/build_rnd_atlas.py \
  --base-dir examples/calibration_smoke \
  --out-dir outputs/smoke_atlas \
  --skip-interactive \
  --skip-sensitivity
```

The smoke run checks execution only. Final manuscript values are stored in
`reference/` and were obtained from the complete calibration database.

Query the frozen atlas for the V9 example:

```bash
python scripts/query_recommendation_atlas.py \
  --atlas-dir reference \
  --nw 1.25 --pe 8.0 --cfl 0.4 --tau 75 \
  --case-id NG_V9_high_risk_cosine_bell \
  --profile-shape cosine_bell \
  --output-csv outputs/v9_prior_decision.csv
```

## Reference results

The compact reference tables allow the main numerical statements to be checked
without downloading all intermediate solver output:

- `02_scheme_risk_composition_wide.csv`: risk composition of the eight schemes;
- `03_scheme_kappa95_fit.csv`: fitted scheme-specific envelope coefficients;
- `06_scheme_recommendation_atlas_full_Nw_Pe_CFL_tau.csv`: 672-node frozen atlas;
- `non_gaussian_preregistered_recommendations.csv`: prior choices for the six
  external-validation cases; and
- `non_gaussian_transferability_summary.csv`: post-simulation ranks and score
  gaps.

The full study contains 5,376 Gaussian calibration records. The complete
calibration and grid-level output archive should be deposited separately in a
versioned data repository and linked to the journal article. GitHub is used for
the maintained source code, examples, and compact reference tables.

## Interpretation and scope

The coefficients and recommendations apply to the governing equation,
finite-volume implementation, parameter range, and schemes calibrated in the
manuscript. A new numerical scheme must be calibrated using the same benchmark,
screening, envelope-fitting, and validation procedure before it is added to the
atlas.

The input field `CFL` denotes the time-step factor applied to the combined
explicit stability limit. The realized advective Courant number `u*dt/dx` is
reported separately as `CFL_adv_actual`.

## Citation and licences

Citation metadata are provided in `CITATION.cff`. Code is licensed under the
MIT License. Example data and reference tables are licensed under CC BY 4.0.

