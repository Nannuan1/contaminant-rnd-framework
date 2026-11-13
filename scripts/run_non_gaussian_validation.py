import csv
import numpy as np
from scipy.stats import linregress
import math
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

scheme_name = "HLLC_Linear"

RUN_AMPLITUDE_TEST = False


# =====================================================================

# =====================================================================

#   1) VALIDATION_STAGE = "MUSCL_RND"


#
#   2) VALIDATION_STAGE = "BASELINE_EXTRA"


#
#   3) RUN_ATLAS_VALIDATION = False

# =====================================================================
RUN_ATLAS_VALIDATION = True
VALIDATION_STAGE = "NON_GAUSSIAN"     # Options: MUSCL_RND, BASELINE_EXTRA, or NON_GAUSSIAN.

if RUN_ATLAS_VALIDATION:

    if VALIDATION_STAGE == "MUSCL_RND":
        base_save_path = r'./outputs/atlas_validation_muscl'




        case_list = [
            {"case_id": "V1_low_risk_interp",      "Nw": 5.0,  "Pe": 1.5,  "CFL": 0.2, "tau": 20.0},
            {"case_id": "V2_warning_zone",         "Nw": 5.0,  "Pe": 4.0,  "CFL": 0.2, "tau": 45.0},
            {"case_id": "V3_significant_zone",     "Nw": 5.0,  "Pe": 8.0,  "CFL": 0.4, "tau": 45.0},
            {"case_id": "V4_high_risk",            "Nw": 5.0,  "Pe": 14.0, "CFL": 0.4, "tau": 75.0},

            {"case_id": "V5_medium_Nw_low_Pe",     "Nw": 2.5,  "Pe": 1.5,  "CFL": 0.2, "tau": 20.0},
            {"case_id": "V6_medium_Nw_threshold",  "Nw": 2.5,  "Pe": 4.8,  "CFL": 0.2, "tau": 45.0},
            {"case_id": "V7_medium_Nw_high_risk",  "Nw": 2.5,  "Pe": 9.7,  "CFL": 0.4, "tau": 75.0},

            {"case_id": "V8_low_Nw_medium_risk",   "Nw": 1.25, "Pe": 4.0,  "CFL": 0.2, "tau": 45.0},
            {"case_id": "V9_low_Nw_high_risk",     "Nw": 1.25, "Pe": 8.0,  "CFL": 0.4, "tau": 75.0},
            {"case_id": "V10_very_low_Nw_risk",    "Nw": 0.65, "Pe": 7.0,  "CFL": 0.4, "tau": 75.0},
        ]

    elif VALIDATION_STAGE == "BASELINE_EXTRA":
        base_save_path = r'./outputs/atlas_validation_baseline'



        case_list = [
            {"case_id": "V1_low_risk_interp",      "Nw": 5.0,  "Pe": 1.5, "CFL": 0.2, "tau": 20.0},
            {"case_id": "V6_medium_Nw_threshold",  "Nw": 2.5,  "Pe": 4.8, "CFL": 0.2, "tau": 45.0},
            {"case_id": "V9_low_Nw_high_risk",     "Nw": 1.25, "Pe": 8.0, "CFL": 0.4, "tau": 75.0},
            {"case_id": "V10_very_low_Nw_risk",    "Nw": 0.65, "Pe": 7.0, "CFL": 0.4, "tau": 75.0},
        ]

    elif VALIDATION_STAGE == "NON_GAUSSIAN":





        base_save_path = r'./outputs/non_gaussian_validation'



        preregistration_file = os.path.abspath(os.path.join(
            SCRIPT_DIR, '..', 'reference',
            'non_gaussian_preregistered_recommendations.csv'
        ))
        if not os.path.exists(preregistration_file):
            raise FileNotFoundError(
                'Missing preregistered non-Gaussian recommendations: '
                + preregistration_file
                + '\nRun scripts/query_recommendation_atlas.py or restore the reference file.'
            )

        required_prereg_columns = {
            'case_id', 'profile_shape', 'Nw', 'Pe', 'CFL', 'tau',
            'recommended_scheme', 'recommendation_status', 'atlas_consensus',
            'decision_source', 'decision_rule_version'
        }
        case_list = []
        with open(preregistration_file, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            missing = required_prereg_columns.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    'Missing columns in prespecified recommendations: ' + ', '.join(sorted(missing))
                )
            for row in reader:
                case_list.append({
                    'case_id': row['case_id'],
                    'profile_shape': row['profile_shape'],
                    'Nw': float(row['Nw']),
                    'Pe': float(row['Pe']),
                    'CFL': float(row['CFL']),
                    'tau': float(row['tau']),

                    'expected_recommended_scheme': row['recommended_scheme'],
                    'recommendation_status': row['recommendation_status'],
                    'atlas_consensus': float(row['atlas_consensus']),
                    'decision_source': row['decision_source'],
                    'decision_rule_version': row['decision_rule_version'],
                })
        if not case_list:
            raise ValueError('Prespecified recommendation table is empty: ' + preregistration_file)
        case_ids = [c['case_id'] for c in case_list]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError('Duplicate case_id in prespecified recommendations.')

    else:
        raise ValueError('VALIDATION_STAGE must be MUSCL_RND, BASELINE_EXTRA, or NON_GAUSSIAN')

else:
    base_save_path = r'./outputs/gaussian_main_cases'

    Nw_list = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    CFL_list = [0.1, 0.3, 0.5]
    Pe_list = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    tau_list = [10.0, 30.0, 60.0, 90.0]

    case_list = []
    for Nw in Nw_list:
        for CFL in CFL_list:
            for Pe in Pe_list:
                for tau in tau_list:
                    case_list.append({
                        "case_id": f"Nw_{Nw}_CFL_{CFL}_Pe_{Pe}_tau_{tau}",
                        "Nw": Nw,
                        "Pe": Pe,
                        "CFL": CFL,
                        "tau": tau
                    })


u_fixed = 1.0            # Mean flow velocity (m/s).
B = 10.0                 # Channel width (m).
ini_H = 2.0              # Initial water depth (m).
x0_gauss_factor = 10.0   # Initial centroid factor; left_margin = 10*sigma0.
k_decay = 0.0
Manningn = 0.0
NZ = 20
HOL = 0.01


# =====================================================================

# =====================================================================
if RUN_AMPLITUDE_TEST:
    M_total_list = [1e6, 1e8, 1e10]
else:
    M_total_list = [1e9]


INCLUDE_AMPLITUDE_TAG_IN_PATH = len(M_total_list) > 1

# =====================================================================

# =====================================================================





def minmod(a, b, eps=1e-12):
    """Minmod slope limiter: return the limited slope from adjacent slopes."""
    if a * b <= 0.0:
        return 0.0
    return math.copysign(min(abs(a), abs(b)), a)


def vanleer(a, b, eps=1e-12):
    """Van Leer slope limiter: return the limited slope from adjacent slopes."""
    if a * b <= 0.0:
        return 0.0
    denom = a + b
    if abs(denom) < eps:
        return 0.0
    return 2.0 * a * b / denom


def superbee(a, b, eps=1e-12):
    """Superbee slope limiter: return the limited slope from adjacent slopes."""
    if a * b <= 0.0:
        return 0.0
    s = math.copysign(1.0, a)
    return s * max(
        min(2.0 * abs(a), abs(b)),
        min(abs(a), 2.0 * abs(b))
    )


def mc(a, b, eps=1e-12):
    """Monotonized Central (MC) slope limiter: return the limited slope from adjacent slopes."""
    if a * b <= 0.0:
        return 0.0
    s = math.copysign(1.0, a)
    return s * min(
        0.5 * abs(a + b),
        2.0 * abs(a),
        2.0 * abs(b)
    )

# =====================================================================

# =====================================================================










# =====================================================================

# =====================================================================
if RUN_ATLAS_VALIDATION and VALIDATION_STAGE == "MUSCL_RND":


    scheme_dict = {
        'minmod':   {'type': 'muscl_limiter', 'limiter_func': minmod},
        'vanleer':  {'type': 'muscl_limiter', 'limiter_func': vanleer},
        'mc':       {'type': 'muscl_limiter', 'limiter_func': mc},
        'superbee': {'type': 'muscl_limiter', 'limiter_func': superbee},
    }

elif RUN_ATLAS_VALIDATION and VALIDATION_STAGE == "BASELINE_EXTRA":


    scheme_dict = {
        'upwind1':  {'type': 'upwind1', 'limiter_func': None},
        'upwind2':  {'type': 'upwind2', 'limiter_func': None},
        'central':  {'type': 'central', 'limiter_func': None},
    }

elif RUN_ATLAS_VALIDATION and VALIDATION_STAGE == "NON_GAUSSIAN":



    scheme_dict = {
        'upwind1':  {'type': 'upwind1', 'limiter_func': None},
        'upwind2':  {'type': 'upwind2', 'limiter_func': None},
        'central':  {'type': 'central', 'limiter_func': None},
        'minmod':   {'type': 'muscl_limiter', 'limiter_func': minmod},
        'vanleer':  {'type': 'muscl_limiter', 'limiter_func': vanleer},
        'mc':       {'type': 'muscl_limiter', 'limiter_func': mc},
        'superbee': {'type': 'muscl_limiter', 'limiter_func': superbee},
    }

else:


    scheme_dict = {
        'minmod':   {'type': 'muscl_limiter', 'limiter_func': minmod},
        'vanleer':  {'type': 'muscl_limiter', 'limiter_func': vanleer},
        'mc':       {'type': 'muscl_limiter', 'limiter_func': mc},
        'superbee': {'type': 'muscl_limiter', 'limiter_func': superbee},
        'upwind1':  {'type': 'upwind1', 'limiter_func': None},
        'upwind2':  {'type': 'upwind2', 'limiter_func': None},
        'central':  {'type': 'central', 'limiter_func': None},
    }





KAPPA95_FIT_FILE = os.path.normpath(os.path.join(
    SCRIPT_DIR,
    '..',
    'reference',
    '03_scheme_kappa95_fit.csv',
))


def load_frozen_kappa95_fits(path):
    if not os.path.exists(path):
        raise FileNotFoundError('Missing fitted envelope table: ' + path)
    fits = {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {
            'scheme', 'recommended_envelope_model',
            'rational_A', 'rational_B', 'constant_kappa95'
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing columns in fitted envelope table: ' + ', '.join(sorted(missing)))
        for row in reader:
            scheme = row['scheme'].strip()
            model = row['recommended_envelope_model'].strip()
            if model == 'constant':
                fits[scheme] = {'form': 'const', 'C': float(row['constant_kappa95'])}
            elif model == 'rational_A_over_Nw_plus_B':
                fits[scheme] = {
                    'form': 'inv',
                    'A': float(row['rational_A']),
                    'B': float(row['rational_B']),
                }

    return fits


KAPPA95_FITS = load_frozen_kappa95_fits(KAPPA95_FIT_FILE)

SCHEME_COMPLEXITY = {
    'upwind1': 1,
    'central': 1,
    'upwind2': 2,
    'minmod': 3,
    'vanleer': 3,
    'mc': 3,
    'superbee': 3,
}


def predict_kappa95(scheme, Nw):
    fit = KAPPA95_FITS.get(scheme)
    if fit is None:
        return np.nan
    if fit['form'] == 'const':
        return fit['C']
    return fit['A'] / (fit['B'] + Nw)


def classify_rnd(rnd):
    if not np.isfinite(rnd):
        return 'NA'
    if rnd < 0.1:
        return 'low'
    if rnd < 0.5:
        return 'warning'
    if rnd < 1.0:
        return 'significant'
    return 'high'

# =====================================================================

# =====================================================================
def calculate_spatial_variance(concentrations, x_coords, dx, A=1.0):
    C_array = np.array(concentrations)
    x_array = np.array(x_coords)
    M0 = np.sum(C_array * dx * A)
    if M0 <= 1e-12:
        return 0.0, 0.0, M0, 3.0
    xc = np.sum(x_array * C_array * dx * A) / M0
    variance = np.sum(((x_array - xc) ** 2) * C_array * dx * A) / M0
    m4 = np.sum(((x_array - xc) ** 4) * C_array * dx * A) / M0
    kurtosis = m4 / (variance ** 2 + 1e-12) if variance > 1e-12 else 3.0
    return variance, xc, M0, kurtosis

def extract_global_diffusion(times, variances, start_ratio=0.5):
    start_idx = int(len(times) * start_ratio)
    fit_times = np.array(times[start_idx:])
    fit_variances = np.array(variances[start_idx:])
    if len(fit_times) < 2:
        return 0.0, 0.0
    slope, intercept, r_value, p_value, std_err = linregress(fit_times, fit_variances)
    return slope / 2.0, r_value ** 2

def calculate_takacs_errors(analytical_array, numerical_array):
    u = np.array(analytical_array)
    v = np.array(numerical_array)
    mean_u, mean_v = np.mean(u), np.mean(v)
    std_u, std_v = np.std(u), np.std(v)
    rho = np.corrcoef(u, v)[0, 1] if (std_u > 1e-12 and std_v > 1e-12) else 1.0
    dissipation_error = (std_u - std_v) ** 2 + (mean_u - mean_v) ** 2
    dispersion_error = 2 * (1 - rho) * std_u * std_v
    total_mse = dissipation_error + dispersion_error
    if total_mse > 1e-12:
        diss_ratio = (dissipation_error / total_mse) * 100
        disp_ratio = (dispersion_error / total_mse) * 100
    else:
        diss_ratio = 0.0
        disp_ratio = 0.0
    return total_mse, dissipation_error, dispersion_error, diss_ratio, disp_ratio


def format_amplitude_tag(M_total):
    """Convert total mass to a directory-safe label."""
    return f"M_{M_total:.0e}".replace("+", "")


def calculate_peak_based_numerical_diffusion(c_ratio, sigma_ref_phys, D_phy, t_final):
    """
    Infer an equivalent numerical diffusion coefficient from peak retention.

    For a one-dimensional Gaussian plume, peak concentration is inversely proportional to width:
        Cmax ~ 1 / sigma
    With R_p = Cmax_num / Cmax_ana:
        sigma_eff^2 = sigma_ana^2 / R_p^2
        D_num_peak = (sigma_eff^2 - sigma_ana^2) / (2 t)

    When c_ratio > 1, D_num_peak may be negative, indicating peak amplification or anti-diffusive behavior.
    """
    if t_final <= 1e-12 or c_ratio <= 1e-12:
        return np.nan
    sigma_ana_sq = sigma_ref_phys ** 2 + 2.0 * D_phy * t_final
    sigma_eff_sq = sigma_ana_sq / (c_ratio ** 2)
    return (sigma_eff_sq - sigma_ana_sq) / (2.0 * t_final)



# =====================================================================

# =====================================================================





_REFERENCE_PROFILE_CACHE = {}


def get_profile_support_half_width(profile_shape, sigma_eq):
    """Return the half-width of a finite-support plume with equivalent standard deviation sigma_eq."""
    if profile_shape == 'top_hat':

        return math.sqrt(3.0) * sigma_eq
    if profile_shape == 'triangular':

        return math.sqrt(6.0) * sigma_eq
    if profile_shape == 'cosine_bell':

        return sigma_eq / math.sqrt(1.0 / 3.0 - 2.0 / (math.pi ** 2))
    raise ValueError(f"Unknown non-Gaussian profile_shape={profile_shape}")


def initial_shape_unnormalized(x, center, sigma_eq, profile_shape):
    """Evaluate the unnormalized initial non-Gaussian profile."""
    x = np.asarray(x, dtype=float)
    r = x - center

    if profile_shape == 'top_hat':
        a = get_profile_support_half_width(profile_shape, sigma_eq)
        return (np.abs(r) <= a).astype(float)

    if profile_shape == 'triangular':
        a = get_profile_support_half_width(profile_shape, sigma_eq)
        return np.maximum(1.0 - np.abs(r) / a, 0.0)

    if profile_shape == 'cosine_bell':
        a = get_profile_support_half_width(profile_shape, sigma_eq)
        y = r / a
        out = np.zeros_like(y)
        mask = np.abs(y) <= 1.0
        out[mask] = 0.5 * (1.0 + np.cos(math.pi * y[mask]))
        return out

    raise ValueError(f"Unknown non-Gaussian profile_shape={profile_shape}")


def build_initial_profile_cell_averages(x_edges, center, sigma_eq, profile_shape, M_total, A_fixed, nq=8):
    """Construct cell-mean initial concentrations normalized to M_total."""
    x_edges = np.asarray(x_edges, dtype=float)
    dx_cells = np.diff(x_edges)

    if profile_shape == 'gaussian':
        denom = math.sqrt(2.0) * sigma_eq
        concs = []
        for x_left, x_right, dx_cell in zip(x_edges[:-1], x_edges[1:], dx_cells):
            erf_right = math.erf((x_right - center) / denom)
            erf_left = math.erf((x_left - center) / denom)
            mass_per_area = (M_total / A_fixed) * 0.5 * (erf_right - erf_left)
            concs.append(mass_per_area / max(dx_cell, 1e-30))
        concs = np.asarray(concs, dtype=float)
    else:
        cell_integrals = []
        for x_left, x_right, dx_cell in zip(x_edges[:-1], x_edges[1:], dx_cells):
            q = x_left + (np.arange(nq, dtype=float) + 0.5) / nq * dx_cell
            vals = initial_shape_unnormalized(q, center, sigma_eq, profile_shape)
            cell_integrals.append(float(np.mean(vals) * dx_cell))
        cell_integrals = np.asarray(cell_integrals, dtype=float)
        total_integral = float(np.sum(cell_integrals))
        if total_integral <= 1e-30:
            raise ValueError(f"Initial plume integral is zero; check profile_shape={profile_shape}  and the domain.")
        mass_per_area_cells = (M_total / A_fixed) * cell_integrals / total_integral
        concs = mass_per_area_cells / np.maximum(dx_cells, 1e-30)


    mass_now = float(np.sum(concs * dx_cells * A_fixed))
    if mass_now > 1e-30:
        concs *= M_total / mass_now
    return concs.tolist()


def build_reference_profile_cell_averages(x_edges, center0, sigma_eq, profile_shape, M_total, A_fixed, D_phy, t, u_fixed,
                                          nq_cell=3, n_source_min=800, n_source_max=2400):
    """Construct cell-mean reference concentrations at time t.

    Gaussian: use the analytical cell-mean solution;
    Non-Gaussian: integrate the initial profile against the advection-diffusion Green function.
    """
    x_edges = np.asarray(x_edges, dtype=float)
    dx_cells = np.diff(x_edges)

    key = (
        profile_shape, round(center0, 12), round(sigma_eq, 12), round(M_total, 6), round(A_fixed, 12),
        round(D_phy, 12), round(t, 10), round(u_fixed, 12), len(x_edges),
        round(float(x_edges[0]), 12), round(float(x_edges[-1]), 12)
    )
    cached = _REFERENCE_PROFILE_CACHE.get(key)
    if cached is not None:
        return cached.copy().tolist()

    if t <= 1e-14:
        concs = np.asarray(build_initial_profile_cell_averages(x_edges, center0, sigma_eq, profile_shape, M_total, A_fixed), dtype=float)
        _REFERENCE_PROFILE_CACHE[key] = concs.copy()
        return concs.tolist()

    if profile_shape == 'gaussian':
        sigma_t = math.sqrt(sigma_eq ** 2 + 2.0 * D_phy * t)
        denom = math.sqrt(2.0) * sigma_t
        center_t = center0 + u_fixed * t
        concs = []
        for x_left, x_right, dx_cell in zip(x_edges[:-1], x_edges[1:], dx_cells):
            erf_right = math.erf((x_right - center_t) / denom)
            erf_left = math.erf((x_left - center_t) / denom)
            mass_per_area = (M_total / A_fixed) * 0.5 * (erf_right - erf_left)
            concs.append(mass_per_area / max(dx_cell, 1e-30))
        concs = np.asarray(concs, dtype=float)
    else:
        a = get_profile_support_half_width(profile_shape, sigma_eq)
        src_left, src_right = center0 - a, center0 + a
        min_dx = float(np.min(dx_cells))
        n_source = int(max(n_source_min, min(n_source_max, math.ceil((src_right - src_left) / max(min_dx / 6.0, 1e-6)))))
        xi_edges = np.linspace(src_left, src_right, n_source + 1)
        dxi = np.diff(xi_edges)
        xi = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        f = initial_shape_unnormalized(xi, center0, sigma_eq, profile_shape)
        integral_f = float(np.sum(f * dxi))
        if integral_f <= 1e-30:
            raise ValueError(f"Reference source integral is zero; check profile_shape={profile_shape}。")
        source_density = (M_total / A_fixed) * f / integral_f  # Discrete convolution weights.

        coef = 1.0 / math.sqrt(4.0 * math.pi * D_phy * t)
        denom = 4.0 * D_phy * t
        concs = []
        for x_left, x_right, dx_cell in zip(x_edges[:-1], x_edges[1:], dx_cells):
            xq = x_left + (np.arange(nq_cell, dtype=float) + 0.5) / nq_cell * dx_cell
            # C(x,t)=∫ C0(xi) G(x-xi-u*t,t) dxi
            y = xq[:, None] - u_fixed * t - xi[None, :]
            G = coef * np.exp(-(y ** 2) / denom)
            cq = np.sum(source_density[None, :] * G * dxi[None, :], axis=1)
            concs.append(float(np.mean(cq)))
        concs = np.asarray(concs, dtype=float)


    mass_ref = float(np.sum(concs * dx_cells * A_fixed))
    if mass_ref > 1e-30:
        concs *= M_total / mass_ref
    _REFERENCE_PROFILE_CACHE[key] = concs.copy()
    return concs.tolist()


def reference_type_name(profile_shape):
    if profile_shape == 'gaussian':
        return 'analytical_gaussian_cell_average'
    return 'green_function_convolution_reference'


def relative_range(values, eps=1e-30):
    """Measure relative metric variation across concentration amplitudes."""
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mean_val = float(np.mean(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    abs_range = max_val - min_val
    rel_range = abs_range / (abs(mean_val) + eps)
    return mean_val, min_val, max_val, abs_range, rel_range


def write_amplitude_invariance_summary(records, output_file):
    """
    Summarize normalized metrics across amplitudes at fixed limiter, Nw, CFL, Pe, and tau.
    Near-zero variation with M_total supports amplitude invariance of relative errors.
    """
    if not records:
        return

    key_fields = ['case_id', 'limiter', 'Nw', 'CFL', 'Pe', 'tau']
    metric_fields = [
        'Max_C_Ratio',
        'Peak_Dissipation_Index',
        'D_num_peak',
        'D_num_variance',
        'L2_rel',
        'Normalized_Shift',
        'Corrected_Shift',
        'Shift_sigma',
        'Diss_Ratio(%)',
        'Disp_Ratio(%)',
        'Final_Mass_Error(%)'
    ]

    grouped = {}
    for rec in records:
        key = tuple(rec[k] for k in key_fields)
        grouped.setdefault(key, []).append(rec)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = key_fields + ['Num_Amplitudes', 'M_total_min', 'M_total_max']
        for metric in metric_fields:
            header += [
                f'{metric}_mean',
                f'{metric}_min',
                f'{metric}_max',
                f'{metric}_abs_range',
                f'{metric}_rel_range'
            ]
        writer.writerow(header)

        for key, group in sorted(grouped.items()):
            masses = [g['M_total'] for g in group]
            row = list(key) + [len(group), min(masses), max(masses)]
            for metric in metric_fields:
                vals = [g[metric] for g in group]
                row += list(relative_range(vals))
            writer.writerow(row)

class Element:
    def __init__(self):
        self.zb = 0.0
        self.zs = 0.0
        self.H = 0.0
        self.u = 0.0
        self.Q = 0.0
        self.S = 0.0
        self.C = 0.0
        self.S_c = 0.0
        self.grad_C = 0.0
        self.grad_zs = 0.0
        self.grad_Q = 0.0
        self.dx = 0.0
        self.slope = 0.0
        self.F1 = 0
        self.F2 = 0
        self.NB = 0
        self.type = 0

class Face:
    def __init__(self):
        self.f1 = 0.0
        self.f2 = 0.0
        self.gc = 0.0
        self.Dc = 0.0
        self.DL = 0.0
        self.zmin = 0.0
        self.zmax = 0.0
        self.gradz = 0.0
        self.cc = [0, 0]
        self.zz = [0] * NZ
        self.Az = [0] * NZ
        self.Rz = [0] * NZ
        self.Fz = [0] * NZ
        self.Hz = [0] * NZ
        self.Bz = [0] * NZ
        self.E1 = 1
        self.E2 = 1

    def build_cross_section_trap(self, B0, rSlope, z0, z1):
        self.zmin = z0
        self.zmax = z1
        self.rdz = NZ / (z1 - z0)
        self.zz[0] = z0
        self.Bz[0] = B0
        self.Az[0] = 0.0
        self.Hz[0] = 0.0
        self.Rz[0] = 0.0
        self.Fz[0] = 0.0
        for i in range(1, NZ):
            self.zz[i] = z0 + i * (z1 - z0) / NZ
            self.Bz[i] = B0 + 2.0 * rSlope * i * (z1 - z0) / NZ
            self.Az[i] = 0.5 * (B0 + self.Bz[i]) * i * (z1 - z0) / NZ
            self.Hz[i] = i * (z1 - z0) / NZ
            self.Rz[i] = self.Az[i] / (B0 + 2.0 * self.Hz[i] / np.sin(np.arctan(1.0 / (rSlope + 0.001))))
            self.Fz[i] = 9.81 * Manningn * Manningn / (pow(self.Rz[i], 4.0 / 3.0) * self.Az[i])

    def getB(self, zs):
        if zs < self.zmin:
            return 0.0
        elif zs >= self.zmax:
            return self.Bz[NZ - 1]
        else:
            for i in range(1, NZ):
                if zs < self.zz[i]:
                    return self.Bz[i - 1] + (zs - self.zz[i - 1]) * (self.Bz[i] - self.Bz[i - 1]) * self.rdz
            return self.Bz[NZ - 1]

    def getA(self, zs):
        if zs < self.zmin:
            return 0.0
        elif zs >= self.zmax:
            return self.Az[NZ - 1]
        else:
            for i in range(1, NZ):
                if zs < self.zz[i]:
                    return self.Az[i - 1] + (zs - self.zz[i - 1]) * (self.Az[i] - self.Az[i - 1]) * self.rdz
            return self.Az[NZ - 1]

    def getFric(self, zs):
        if zs < self.zmin:
            return 0.0
        elif zs >= self.zmax:
            return self.Fz[NZ - 1]
        else:
            for i in range(1, NZ):
                if zs < self.zz[i]:
                    return self.Fz[i - 1] + (zs - self.zz[i - 1]) * (self.Fz[i] - self.Fz[i - 1]) * self.rdz
            return self.Fz[NZ - 1]

    def compute_fluxes(self, Q1, z1, zb1, Q2, z2, zb2, C1, C2):
        self.s1 = 0.0
        self.s2 = 0.0
        self.sm = 0.0
        h1 = max(z1 - zb1, HOL)
        h2 = max(z2 - zb2, HOL)
        z1_corr = zb1 + h1
        z2_corr = zb2 + h2
        A1 = max(self.getA(z1_corr), 1e-6)
        A2 = max(self.getA(z2_corr), 1e-6)
        B_loc = self.getB(z1_corr)
        B_loc = max(B_loc, 1e-6)
        u1 = Q1 / A1 if A1 > 1e-6 else 0.0
        u2 = Q2 / A2 if A2 > 1e-6 else 0.0
        g = 9.81
        p1 = 0.5 * g * A1 * h1
        p2 = 0.5 * g * A2 * h2
        F_mom_L = Q1 * Q1 / A1 + p1
        F_mom_R = Q2 * Q2 / A2 + p2
        us = 0.5 * (u1 + u2) + np.sqrt(9.81 * h1) - np.sqrt(9.81 * h2)
        cs = 0.5 * (np.sqrt(9.81 * h1) + np.sqrt(9.81 * h2)) + 0.25 * (u1 - u2)
        cs = max(cs, 1e-12)
        s1 = min(u1 - np.sqrt(9.81 * h1), us - cs)
        s2 = max(u2 + np.sqrt(9.81 * h2), us + cs)
        if s1 >= 0:
            f1 = Q1
            f2 = F_mom_L
        elif s2 <= 0:
            f1 = Q2
            f2 = F_mom_R
        else:
            f1 = (s2 * Q1 - s1 * Q2 + s1 * s2 * (A2 - A1)) / (s2 - s1)
            f2 = (s2 * F_mom_L - s1 * F_mom_R + s1 * s2 * (Q2 - Q1)) / (s2 - s1)
        self.f1 = f1
        self.f2 = f2
        numerator = s1 * h2 * (u2 - s2) - s2 * h1 * (u1 - s1)
        denominator = h2 * (u2 - s2) - h1 * (u1 - s1)
        if abs(denominator) < 1e-6:
            s_m = 0.5 * (u1 + u2)
        else:
            s_m = numerator / denominator
        F_L = Q1 * C1
        F_R = Q2 * C2
        if s1 >= 0:
            self.gc = F_L
        elif s2 <= 0:
            self.gc = F_R
        elif s1 <= 0 and s_m >= 0:
            U_L = A1 * C1
            factor = (s1 - u1) / (s1 - s_m) if abs(s1 - s_m) > 1e-6 else 1.0
            U_star_L = U_L * factor
            self.gc = F_L + s1 * (U_star_L - U_L)
        elif s_m <= 0 and s2 >= 0:
            U_R = A2 * C2
            factor = (s2 - u2) / (s2 - s_m) if abs(s2 - s_m) > 1e-6 else 1.0
            U_star_R = U_R * factor
            self.gc = F_R + s2 * (U_star_R - U_R)
        self.s1 = s1
        self.s2 = s2
        self.sm = s_m

    def compute_diffusive_flux_improved(self, EL, ER, dx_face):
        dx_face = max(dx_face, 1e-6)
        dCdx = (ER.C - EL.C) / dx_face
        zs_face = 0.5 * (EL.zs + ER.zs)
        A_face = max(self.getA(zs_face), 1e-6)
        self.Dc = -A_face * self.DL * dCdx

class Junction:
    def __init__(self):
        self.reachInOut = [1, 1, 2, 2]
        self.reachId = [0, 1, 2, 3]
        self.Bcoef = 1.0
        self.zb = 0.0
        self.zs = 0.0
        self.type = 0
        self.cc = [0, 0]
        self.Q = 0.0
        self.bc = 0.0
        self.C = 0.0
        self.pump_flow = 0.0
        self.gate_opening = 0.0
        self.discharge_coeff = 0.6
        self.gate_width = 10.0

class Reach:
    def __init__(self):
        self.J1 = 1
        self.J2 = 1
        self.Length = 100
        self.NE = 0
        self.E1 = 0
        self.E2 = 0
        self.F1 = 0
        self.F2 = 0
        self.Q = 0
        self.Dispersion = 0.0

def update_junction_concentrations(Js, Rs, Es, Fs, NE):
    for i, J in enumerate(Js):
        if J.type == 0:
            total_flow_in = 0.0
            total_mass_in = 0.0
            for j in range(4):
                if J.reachId[j] == 0:
                    continue
                reach_id = J.reachId[j] - 1
                R = Rs[reach_id]
                if R.J1 == i:
                    Q_face = Fs[R.F1].f1
                    if Q_face < 0:
                        total_flow_in += abs(Q_face)
                        total_mass_in += abs(Q_face) * Es[R.E1].C
                elif R.J2 == i:
                    Q_face = Fs[R.F2].f1
                    if Q_face > 0:
                        total_flow_in += Q_face
                        total_mass_in += Q_face * Es[R.E2].C
            if total_flow_in > 1e-6:
                J.C = total_mass_in / total_flow_in

def compute_gradients_MUSCL(Es, Fs, NE, limiter_func):
    for i in range(NE):
        E = Es[i]
        left_id = Fs[E.F1].E1
        right_id = Fs[E.F2].E2
        if left_id < 0 or right_id < 0:
            E.grad_C = 0.0
            E.grad_zs = 0.0
            E.grad_Q = 0.0
            continue
        slope_L_C = (E.C - Es[left_id].C) / E.dx
        slope_R_C = (Es[right_id].C - E.C) / E.dx
        E.grad_C = limiter_func(slope_L_C, slope_R_C)
        slope_L_zs = (E.zs - Es[left_id].zs) / E.dx
        slope_R_zs = (Es[right_id].zs - E.zs) / E.dx
        E.grad_zs = limiter_func(slope_L_zs, slope_R_zs)
        slope_L_Q = (E.Q - Es[left_id].Q) / E.dx
        slope_R_Q = (Es[right_id].Q - E.Q) / E.dx
        E.grad_Q = limiter_func(slope_L_Q, slope_R_Q)



def compute_gradients_by_scheme(Es, Fs, NE, scheme_type, limiter_func=None):
    """
    Compute reconstructed slopes for the selected advection scheme.

    These slopes only reconstruct HLLC states. For central/upwind1/upwind2,
    the scalar advective flux is replaced by the corresponding scheme in
    compute_scalar_advective_flux_by_scheme(); grad_C is set to zero here.

    For MUSCL/TVD, compute_gradients_MUSCL() retains the limited slope; limiter_func
    takes adjacent slopes [C]/[L] and returns a limited slope [C]/[L].
    """
    if scheme_type == 'muscl_limiter':
        if limiter_func is None:
            raise ValueError("scheme_type='muscl_limiter' requires limiter_func")
        compute_gradients_MUSCL(Es, Fs, NE, limiter_func)
        return




    for i in range(NE):
        E = Es[i]
        E.grad_C = 0.0
        E.grad_zs = 0.0
        E.grad_Q = 0.0


def _valid_element_index(idx, Es):
    return isinstance(idx, int) and 0 <= idx < len(Es)


def _left_neighbor_id_of_element(E, Fs):
    """Return the left neighbor of cell E, or -1 if none exists."""
    try:
        return Fs[E.F1].E1
    except Exception:
        return -1


def _right_neighbor_id_of_element(E, Fs):
    """Return the right neighbor of cell E, or -1 if none exists."""
    try:
        return Fs[E.F2].E2
    except Exception:
        return -1


def compute_scalar_advective_flux_by_scheme(F, EL, ER, Es, Fs, scheme_type, nonnegative_reconstruction=True):
    """
    Set scalar advective flux F.gc for the selected baseline scheme.

    This function applies to the baseline advection schemes:
        central : C_face = 0.5*(C_L + C_R)
        upwind1 : C_face = C_upwind
        upwind2 : C_face = 1.5*C_upwind - 0.5*C_next_upwind

    Units:
        Q_face = F.f1                     [m^3/s]
        C_face                            [mass/m^3]
        F.gc = Q_face * C_face            [mass/s]

    Notes:
        - MUSCL/TVD schemes retain their HLLC scalar flux;
        - upwind2 reverts to first-order upwind where a second upstream cell is unavailable;
        - interface concentration is clipped at zero, consistent with reconstructed states.
    """
    Q_face = F.f1

    if scheme_type == 'central':
        C_face = 0.5 * (EL.C + ER.C)

    elif scheme_type == 'upwind1':
        C_face = EL.C if Q_face >= 0.0 else ER.C

    elif scheme_type == 'upwind2':
        if Q_face >= 0.0:

            # C_face = C_i + 0.5*dx*(C_i - C_{i-1})/dx = 1.5*C_i - 0.5*C_{i-1}
            left2_id = _left_neighbor_id_of_element(EL, Fs)
            if _valid_element_index(left2_id, Es) and left2_id != F.E1:
                slope_up = (EL.C - Es[left2_id].C) / max(EL.dx, 1e-12)
                C_face = EL.C + 0.5 * EL.dx * slope_up
            else:
                C_face = EL.C
        else:

            # C_face = C_{i+1} - 0.5*dx*(C_{i+2} - C_{i+1})/dx = 1.5*C_{i+1} - 0.5*C_{i+2}
            right2_id = _right_neighbor_id_of_element(ER, Fs)
            if _valid_element_index(right2_id, Es) and right2_id != F.E2:
                slope_up = (Es[right2_id].C - ER.C) / max(ER.dx, 1e-12)
                C_face = ER.C - 0.5 * ER.dx * slope_up
            else:
                C_face = ER.C
    else:

        return

    if nonnegative_reconstruction:
        C_face = max(0.0, C_face)

    F.gc = Q_face * C_face

def compute_boundary_flow_from_waterlevel(is_left, E, F, J, Fs):
    g = 9.81
    h_in = E.H
    A_in = (Fs[E.F1].getA(E.zs) + Fs[E.F2].getA(E.zs)) * 0.5
    u_in = E.Q / max(A_in, 1e-6)
    h_b = J.zs - J.zb
    A_b = F.getA(J.zs)
    c_in = np.sqrt(g * h_in)
    c_b = np.sqrt(g * h_b)
    if is_left:
        u_b = u_in - 2.0 * (c_in - c_b)
    else:
        u_b = u_in + 2.0 * (c_in - c_b)
    Q_b = A_b * u_b
    return Q_b

# =====================================================================

# =====================================================================
for limiter_name, scheme_cfg in scheme_dict.items():
    scheme_type = scheme_cfg['type']
    limiter_func = scheme_cfg.get('limiter_func', None)


    scheme_dir = os.path.join(base_save_path, f'{scheme_name}_{limiter_name}')
    if not os.path.exists(scheme_dir):
        os.makedirs(scheme_dir)

    amplitude_records = []


    summary_file = os.path.join(scheme_dir, f'Expansion_Study_{scheme_name}_{limiter_name}.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'validation_stage', 'case_id', 'scheme', 'profile_shape', 'expected_recommended_scheme',
            'recommendation_status', 'atlas_consensus', 'decision_source', 'decision_rule_version',
            'reference_type', 'sigma_eq',
            'Nw', 'CFL', 'Pe', 'tau', 'M_total', 'C0_equiv', 't_final',
            'sigma_scale_type', 'sigma_ref_phys', 'dx_phys', 'Nw_check', 'Pe_check', 'tau_check',
            'dt_actual', 'CFL_adv_actual', 'CFL_wave_actual', 'Fo_diff_actual',
            'D_phy', 'D_global', 'D_num_variance',
            'D_num_peak', 'D_num_peak_star', 'D_num_peak_over_D_phy', 'R_squared',
            'Kurtosis', 'Shape_Error',
            'Max_C_Ratio', 'Peak_Dissipation_Index', 'Peak_Error_Abs', 'Overshoot_Ratio',
            'Total_MSE', 'Diss_Error', 'Disp_Error', 'Diss_Ratio(%)', 'Disp_Ratio(%)',
            'L2_abs', 'L2_rel',

            'u_num_regression', 'delta_L_regression', 'Regression_Shift',

            'xc_num_final', 'xc_ana_final', 'delta_L_corrected', 'Corrected_Shift',

            'Normalized_Shift', 'Shift_sigma',
            'Final_Mass_Error(%)',
            'Valid_Baseline', 'Positive_Diffusion', 'Compression_or_Antidiffusion', 'Obvious_Compression',
            'kappa95_pred', 'RND_pred', 'RND_level', 'Complexity', 'Score_actual_like'
        ])

    for case in case_list:
        case_id = case['case_id']
        Nw = float(case['Nw'])
        CFL = float(case['CFL'])
        Pe = float(case['Pe'])
        tau = float(case['tau'])
        profile_shape = case.get('profile_shape', 'gaussian')
        expected_recommended_scheme = case.get('expected_recommended_scheme', '')
        recommendation_status = case.get('recommendation_status', '')
        atlas_consensus = case.get('atlas_consensus', np.nan)
        decision_source = case.get('decision_source', '')
        decision_rule_version = case.get('decision_rule_version', '')
        ref_type = reference_type_name(profile_shape)

        for M_total in M_total_list:
            print("\n" + "=" * 65)
            print(f"Starting case: {case_id} | shape={profile_shape} | {limiter_name} ({scheme_type}) | M_total={M_total:.1e}, Nw={Nw}, CFL={CFL}, Pe={Pe}, tau={tau}")
            if VALIDATION_STAGE == 'NON_GAUSSIAN':
                print(
                    f"Prespecified Gaussian-atlas choice: {expected_recommended_scheme} | "
                    f"status={recommendation_status} | consensus={atlas_consensus:.3f}"
                )
            print("=" * 65)


            # -----------------------------------------------------------------





            # -----------------------------------------------------------------
            sigma0_phys = 1.0
            sigma_ref_phys = sigma0_phys
            sigma_scale_type = 'sigma0_gaussian' if profile_shape == 'gaussian' else 'sigma_eq_non_gaussian'

            dx = sigma_ref_phys / Nw
            D_phy = u_fixed * dx / Pe
            c_wave = math.sqrt(9.81 * ini_H)
            max_celerity = u_fixed + c_wave
            dt_max = dx ** 2 / (max_celerity * dx + 2.0 * D_phy)
            dt_target = CFL * dt_max
            dt_target = math.floor(dt_target * 10000) / 10000.0
            t_total = tau * sigma_ref_phys / u_fixed






            if dt_target <= 0.0:
                raise ValueError("dt_target <= 0; check CFL, dx, and D_phy.")
            NT = int(math.ceil(t_total / dt_target)) + 1
            dt = t_total / (NT - 1)



            CFL_adv_actual = u_fixed * dt / dx
            CFL_wave_actual = max_celerity * dt / dx
            Fo_diff_actual = D_phy * dt / (dx ** 2)


            Nw_check = sigma_ref_phys / dx
            Pe_check = u_fixed * dx / D_phy
            tau_check = u_fixed * t_total / sigma_ref_phys
            if abs(Nw_check - Nw) > 1e-10 or abs(Pe_check - Pe) > 1e-10 or abs(tau_check - tau) > 1e-10:
                raise RuntimeError(
                    f"Dimensionless input check failed: Nw={Nw}({Nw_check}), Pe={Pe}({Pe_check}), tau={tau}({tau_check})"
                )

            sigma_final = math.sqrt(sigma_ref_phys ** 2 + 2.0 * D_phy * t_total)
            safe_factor = 10.0
            left_margin = safe_factor * sigma_ref_phys
            right_margin = u_fixed * t_total + safe_factor * sigma_final
            L_total = left_margin + right_margin
            x0_profile_phys = left_margin
            x0_gauss_phys = x0_profile_phys  # Legacy name for the initial plume centroid.
            L_total = max(L_total, 200.0)
            NE = int(L_total / dx) + 1
            L_total = NE * dx

            print(f"dx={dx:.3f} m, D_phy={D_phy:.4f} m²/s, dt={dt:.5f} s, NT={NT}, L_total={L_total:.1f} m")
            print(f"Realized stability metrics: CFL_adv={CFL_adv_actual:.4f}, CFL_wave={CFL_wave_actual:.4f}, Fo_diff={Fo_diff_actual:.4f}")
            print(f"sigma_ref={sigma_ref_phys} m ({sigma_scale_type}), x0_profile={x0_profile_phys:.1f} m, t_total={t_total:.1f} s")

            DL_const = D_phy
            A_fixed = B * ini_H

            if profile_shape == 'gaussian':
                C0_equiv = M_total / (A_fixed * math.sqrt(2.0 * math.pi) * sigma_ref_phys)
            else:
                C0_equiv = M_total / (A_fixed * sigma_ref_phys)


            amp_tag = format_amplitude_tag(M_total)
            case_tag = str(case_id).replace(' ', '_').replace('/', '_')
            if INCLUDE_AMPLITUDE_TAG_IN_PATH:
                save_path = os.path.join(
                    scheme_dir,
                    f'{case_tag}_{amp_tag}_Nw_{Nw}_CFL_{CFL}_Pe_{Pe}_tau_{tau}'
                )
            else:
                save_path = os.path.join(
                    scheme_dir,
                    f'{case_tag}_Nw_{Nw}_CFL_{CFL}_Pe_{Pe}_tau_{tau}'
                )
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            C_Profile_file = os.path.join(save_path, 'reach_C_profile.csv')
            WL_Profile_file = os.path.join(save_path, 'reach_waterlevel_profile.csv')
            Analytical_Profile_file = os.path.join(save_path, 'reach_analytical_profile.csv')
            mass_file_path = os.path.join(save_path, 'mass_conservation.csv')


            NE = 0
            NF = 0
            NR = 1
            NJ = 2

            Rs = []
            Es = []
            Fs = []
            Js = []

            record_times = []
            record_variances = []
            record_mass = []
            record_kurtosis = []

            history_time = []
            history_xc_num = []

            print('Starting one-dimensional transport simulation')
            ReachLengths = [L_total]
            ReachBedwidth = [B]
            ReachrSlope = [0.0]
            ReachBedslope = [0.0]
            ReachJunction = [[1, 2]]
            JunctionReachId = [[1, 0, 0, 0], [1, 0, 0, 0]]
            JunctionReachInOut = [[2, 0, 0, 0], [1, 0, 0, 0]]
            JunctionBcType = [1, 2]
            JunctionBcValue = [u_fixed * B * ini_H, ini_H]
            JunctionZb = [0.0, 0.0]
            JunctionBcConcValue = [0.0, 0.0]

            for i in range(NJ):
                J = Junction()
                J.bc = JunctionBcValue[i]
                J.reachId = JunctionReachId[i]
                J.reachInOut = JunctionReachInOut[i]
                J.type = JunctionBcType[i]
                J.zb = JunctionZb[i]
                J.C = JunctionBcConcValue[i]
                if i == 0:
                    J.zs = J.zb + ini_H
                if i == 1:
                    J.zs = J.zb + ini_H
                if J.type == 1:
                    J.Q = J.bc
                if J.type == 2:
                    J.zs = J.bc
                Js.append(J)

            for i in range(NR):
                R = Reach()
                R.Length = ReachLengths[i]
                R.J1 = ReachJunction[i][0] - 1
                R.J2 = ReachJunction[i][1] - 1
                R.E1 = NE
                R.NE = int(R.Length / dx + 0.1)
                R.E2 = NE + R.NE - 1
                tmpd = R.Length / R.NE

                x_edges_init = np.array([jj * tmpd for jj in range(int(R.NE) + 1)], dtype=float)
                initial_concs = build_initial_profile_cell_averages(
                    x_edges_init, x0_profile_phys, sigma_ref_phys, profile_shape, M_total, A_fixed
                )
                for j in range(int(R.NE)):
                    E = Element()
                    E.Q = u_fixed * B * ini_H
                    E.dx = tmpd
                    E.zs = E.zb + ini_H
                    E.C = max(float(initial_concs[j]), 0.0)
                    E.slope = ReachBedslope[i]
                    E.zb = Js[R.J1].zb + (Js[R.J2].zb - Js[R.J1].zb) * (j + 0.5) / R.NE
                    E.H = E.zs - E.zb
                    E.F1 = NF + j
                    E.F2 = NF + j + 1
                    Es.append(E)
                    NE += 1
                R.F1 = NF
                for j in range(R.NE + 1):
                    F = Face()
                    fz = Js[R.J1].zb + (Js[R.J2].zb - Js[R.J1].zb) * (j) / R.NE
                    F.build_cross_section_trap(ReachBedwidth[i], ReachrSlope[i], fz, fz + 20.0)
                    F.DL = DL_const
                    if j == 0:
                        F.E1 = -1
                        F.E2 = R.E1
                    elif j == R.NE:
                        F.E1 = R.E2
                        F.E2 = -1
                    else:
                        F.E1 = R.E1 + j - 1
                        F.E2 = R.E1 + j
                    Fs.append(F)
                    NF += 1
                R.F2 = NF - 1
                Rs.append(R)


            header_C_Profile = ['itime', 't', 'Pump_Flow', 'Gate_Opening']
            for i in range(NE):
                distance = (i + 0.5) * dx
                header_C_Profile.append(f'C_Dist_{distance:.1f}m')
            with open(C_Profile_file, 'w', newline='') as f:
                csv.writer(f).writerow(header_C_Profile)

            header_WL_Profile = ['itime', 't']
            for i in range(NE):
                distance = (i + 0.5) * dx
                header_WL_Profile.append(f'WL_Dist_{distance:.0f}m')
            with open(WL_Profile_file, 'w', newline='') as f:
                csv.writer(f).writerow(header_WL_Profile)

            header_Analytical = ['itime', 't'] + [f'Analytical_Dist_{(i + 0.5) * dx:.0f}m' for i in range(NE)]
            with open(Analytical_Profile_file, 'w', newline='') as f:
                csv.writer(f).writerow(header_Analytical)

            # Ghost cells
            GhostCellStart = NE
            for i in range(NR):
                R = Rs[i]
                E_left = Element()
                E_left.NB = R.E1
                E_left.type = Js[R.J1].type
                E_left.Q = Js[R.J1].Q
                E_left.C = 0.0
                E_left.zs = Js[R.J1].zs
                E_left.zb = Js[R.J1].zb
                E_left.dx = Es[R.E1].dx
                E_left_index = len(Es)
                Fs[int(R.F1)].E1 = E_left_index
                E_left.F1 = R.F1
                E_left.F2 = R.F1
                Es.append(E_left)

                E_right = Element()
                E_right.NB = R.E2
                E_right.type = Js[R.J2].type
                E_right.Q = Js[R.J2].Q
                E_right.C = 0.0
                E_right.zs = Js[R.J2].zs
                E_right.zb = Js[R.J2].zb
                E_right.dx = Es[R.E2].dx
                E_right_index = len(Es)
                Fs[R.F2].E2 = E_right_index
                E_right.F1 = R.F2
                E_right.F2 = R.F2
                Es.append(E_right)

            for i in range(NJ):
                J = Js[i]
                J.Bcoef = 1.0 / (2.0 * 5.0 * B * np.sqrt(9.81 * 1.0))

            x_coords_list = [(i + 0.5) * dx for i in range(NE)]
            x_edges_list = [i * dx for i in range(NE + 1)]

            update_junction_concentrations(Js, Rs, Es, Fs, NE)


            for itime in range(0, NT):
                current_time = itime * dt
                update_junction_concentrations(Js, Rs, Es, Fs, NE)

                for i in range(0, NR):
                    R = Rs[i]
                    J = Js[R.J1]
                    E = Es[R.E1]
                    EE = Es[NE + 2 * i]
                    if J.type == 1 or J.type == 3:
                        EE.C = J.C
                    elif J.type == 0:
                        EE.C = J.C
                    elif J.type == 2 or J.type == 4:
                        EE.C = E.C
                    elif J.type == 5:
                        EE.C = E.C

                    J = Js[R.J2]
                    E = Es[R.E2]
                    EE = Es[NE + 2 * i + 1]
                    if J.type == 1:
                        EE.C = J.C
                    elif J.type == 0:
                        EE.C = J.C
                    elif J.type == 2 or J.type == 3 or J.type == 4:
                        EE.C = E.C
                    elif J.type == 5:
                        EE.C = E.C

                compute_gradients_by_scheme(Es, Fs, NE, scheme_type, limiter_func)

                for i in range(0, NF):
                    F = Fs[i]
                    EL = Es[F.E1]
                    ER = Es[F.E2]
                    zs_L_rec = EL.zs + 0.5 * EL.dx * EL.grad_zs
                    Q_L_rec = EL.Q + 0.5 * EL.dx * EL.grad_Q
                    C_L_rec = EL.C + 0.5 * EL.dx * EL.grad_C
                    zs_R_rec = ER.zs - 0.5 * ER.dx * ER.grad_zs
                    Q_R_rec = ER.Q - 0.5 * ER.dx * ER.grad_Q
                    C_R_rec = ER.C - 0.5 * ER.dx * ER.grad_C
                    C_L_rec = max(0.0, C_L_rec)
                    C_R_rec = max(0.0, C_R_rec)
                    zs_L_rec = max(zs_L_rec, EL.zb + HOL)
                    zs_R_rec = max(zs_R_rec, ER.zb + HOL)
                    F.compute_fluxes(Q_L_rec, zs_L_rec, EL.zb, Q_R_rec, zs_R_rec, ER.zb, C_L_rec, C_R_rec)




                    compute_scalar_advective_flux_by_scheme(F, EL, ER, Es, Fs, scheme_type)

                for i in range(NF):
                    F = Fs[i]
                    EL = Es[F.E1]
                    ER = Es[F.E2]
                    dx_face = (EL.dx + ER.dx) * 0.5
                    F.compute_diffusive_flux_improved(EL, ER, dx_face)

                for i in range(NJ):
                    J = Js[i]
                    if J.type == 0:
                        tmpd = 0
                        for j in range(4):
                            if J.reachId[j] == 0:
                                continue
                            R = Rs[J.reachId[j] - 1]
                            if J.reachInOut[j] == 1:
                                tmpd += Fs[R.F2].f1
                            elif J.reachInOut[j] == 2:
                                tmpd -= Fs[R.F1].f1
                        tmpd = tmpd * J.Bcoef / np.sqrt(J.zs - J.zb)
                        J.zs += tmpd

                for i in range(0, NR):
                    R = Rs[i]
                    J = Js[R.J1]
                    E = Es[R.E1]
                    EE = Es[NE + 2 * i]
                    F = Fs[R.F1]
                    if J.type == 1:
                        E.Q = J.Q
                        EE.Q = J.Q
                        EE.zs = E.zs
                        F.f1 = J.Q
                        J.zs = E.zs
                    elif J.type == 2:
                        Q_b = compute_boundary_flow_from_waterlevel(is_left=True, E=E, F=F, J=J, Fs=Fs)
                        EE.Q = Q_b
                        EE.zs = J.zs
                        J.Q = Q_b
                    if J.type == 5:
                        E.Q = J.Q
                        EE.Q = 0
                        EE.zs = E.zs
                        F.f1 = 0
                        J.zs = E.zs
                    elif J.type == 0:
                        EE.Q = E.Q
                        EE.zs = J.zs

                    J = Js[R.J2]
                    E = Es[R.E2]
                    EE = Es[NE + 2 * i + 1]
                    F = Fs[R.F2]
                    if J.type == 1:
                        E.Q = J.Q
                        EE.Q = J.Q
                        EE.zs = E.zs
                        F.f1 = J.Q
                        J.zs = E.zs
                    elif J.type == 2:
                        Q_b = compute_boundary_flow_from_waterlevel(is_left=False, E=E, F=F, J=J, Fs=Fs)
                        EE.Q = Q_b
                        EE.zs = J.zs
                        J.Q = Q_b
                    elif J.type == 0:
                        EE.Q = E.Q
                        EE.zs = J.zs
                    if J.type == 5:
                        E.Q = J.Q
                        EE.Q = 0
                        EE.zs = E.zs
                        F.f1 = 0
                        J.zs = E.zs

                for i in range(0, NR):
                    R = Rs[i]
                    J_left = Js[R.J1]
                    F_left = Fs[R.F1]
                    if J_left.type == 1 and abs(J_left.Q) < 1e-12:
                        F_left.gc = 0.0
                    J_right = Js[R.J2]
                    F_right = Fs[R.F2]
                    if J_right.type == 1 and abs(J_right.Q) < 1e-12:
                        F_right.gc = 0.0

                for i in range(0, NE):
                    E = Es[i]
                    FL = Fs[E.F1]
                    FR = Fs[E.F2]
                    A = (FL.getA(E.zs) + FR.getA(E.zs)) * 0.5
                    try:
                        E.S = 9.81 * A * 0.5 * (Es[FL.E1].zb - Es[FR.E2].zb) / E.dx
                    except IndexError:
                        E.S = 0.0
                    Fr = (FL.getFric(E.zs) + FR.getFric(E.zs)) * 0.5
                    E.S += -Fr * abs(E.Q) * E.Q
                    E.S_c = -k_decay * E.C

                for i in range(0, NE):
                    E = Es[i]
                    FL = Fs[E.F1]
                    FR = Fs[E.F2]
                    A_old = (FL.getA(E.zs) + FR.getA(E.zs)) * 0.5
                    Vol_old = A_old * E.dx
                    Mass_old = Vol_old * E.C
                    E.zs_old = E.zs
                    BW = (FL.getB(E.zs) + FR.getB(E.zs)) * 0.5
                    BW = max(BW, 1e-6)
                    dz = (-dt / (E.dx * BW)) * (FR.f1 - FL.f1)
                    E.zs += dz
                    dQ = (-dt / E.dx) * (FR.f2 - FL.f2) + dt * E.S
                    E.Q += dQ
                    E.H = max(E.zs - E.zb, HOL)
                    A_new = (FL.getA(E.zs) + FR.getA(E.zs)) * 0.5
                    Vol_new = A_new * E.dx
                    E.u = E.Q / A_new if A_new > 1e-6 else 0.0
                    total_flux_R = FR.gc + FR.Dc
                    total_flux_L = FL.gc + FL.Dc
                    net_flux_out = total_flux_R - total_flux_L
                    mass_source_term = Vol_old * (-k_decay * E.C)
                    Mass_new = Mass_old - dt * net_flux_out + dt * mass_source_term
                    if Vol_new > 1e-6:
                        E.C = Mass_new / Vol_new
                    else:
                        E.C = 0.0
                    E.C = max(E.C, 0.0)

                current_concs = [E.C for E in Es[:NE]]
                variance, xc, current_mass, kurtosis = calculate_spatial_variance(
                    current_concs, x_coords_list, dx, A_fixed
                )
                record_times.append(current_time)
                record_variances.append(variance)
                record_mass.append(current_mass)
                record_kurtosis.append(kurtosis)
                history_time.append(current_time)
                history_xc_num.append(xc)

                print_interval = max(1, NT // 10)
                if itime % print_interval == 0 or itime == NT - 1:
                    progress = (itime / NT) * 100
                    mass_loss_ratio = (1.0 - current_mass / M_total) * 100
                    if itime > 0:
                        print(f"  [progress {progress:.0f}%] t={current_time:.1f}s | variance={variance:.2f} | mass change={mass_loss_ratio:.4f}%")

                    with open(C_Profile_file, 'a', newline='') as f:
                        csv.writer(f).writerow([itime, current_time, 0.0, 0.0] + current_concs)
                    with open(WL_Profile_file, 'a', newline='') as f:
                        current_zs = [E.zs for E in Es[:NE]]
                        csv.writer(f).writerow([itime, current_time] + current_zs)
                    total_water = sum([max(Fs[Es[i].F1].getA(Es[i].zs), 1e-6) * Es[i].dx for i in range(NE)])
                    with open(mass_file_path, 'a', newline='') as f:
                        csv.writer(f).writerow([itime, current_time, current_mass, total_water])

                    current_reference = build_reference_profile_cell_averages(
                        x_edges_list, x0_profile_phys, sigma_ref_phys, profile_shape,
                        M_total, A_fixed, D_phy, current_time, u_fixed
                    )
                    with open(Analytical_Profile_file, 'a', newline='') as f:
                        csv.writer(f).writerow([itime, current_time] + current_reference)


            # ------------------------------------------------------------




            #



            #


            # ------------------------------------------------------------


            relaxation_ratio = 0.20
            t_start = relaxation_ratio * t_total
            valid_times, valid_xc = [], []
            for t_val, xc_val in zip(history_time, history_xc_num):
                if t_val >= t_start:
                    valid_times.append(t_val)
                    valid_xc.append(xc_val)
            if len(valid_times) > 1:
                slope_xc, _, _, _, _ = linregress(valid_times, valid_xc)
                u_num_regression = slope_xc
            else:
                u_num_regression = u_fixed
            delta_L_regression = (u_num_regression - u_fixed) * t_total
            regression_shift = delta_L_regression / dx


            D_global, r_sq = extract_global_diffusion(record_times, record_variances, start_ratio=0.5)
            D_numerical = D_global - D_phy


            t_final = record_times[-1] if record_times else t_total
            numerical_concs = [E.C for E in Es[:NE]]



            analytical_concs = build_reference_profile_cell_averages(
                x_edges_list, x0_profile_phys, sigma_ref_phys, profile_shape,
                M_total, A_fixed, D_phy, t_final, u_fixed
            )

            total_mse, diss_err, disp_err, diss_ratio, disp_ratio = calculate_takacs_errors(analytical_concs, numerical_concs)
            max_c_sim = max(numerical_concs) if numerical_concs else 0.0
            max_c_ana = max(analytical_concs) if analytical_concs else 1e-12
            c_ratio = max_c_sim / max_c_ana
            peak_diss_index = 1.0 - c_ratio
            peak_error_abs = abs(c_ratio - 1.0)
            overshoot_ratio = max(c_ratio - 1.0, 0.0)
            final_kurt = record_kurtosis[-1] if record_kurtosis else 3.0
            shape_error = abs(final_kurt - 3.0)

            l2_abs = np.sqrt(total_mse)
            l2_ana = np.sqrt(np.mean(np.array(analytical_concs) ** 2))
            l2_rel = l2_abs / l2_ana if l2_ana > 1e-12 else 0.0


            sigma_ana_sq = sigma_ref_phys ** 2 + 2.0 * D_phy * t_final
            sigma_ana = math.sqrt(max(sigma_ana_sq, 1e-30))
            if profile_shape == 'gaussian':
                D_num_peak = calculate_peak_based_numerical_diffusion(c_ratio, sigma_ref_phys, D_phy, t_final)
                D_num_peak_star = D_num_peak / (u_fixed * dx) if np.isfinite(D_num_peak) and abs(u_fixed * dx) > 1e-30 else np.nan
                D_num_peak_over_D_phy = D_num_peak / D_phy if np.isfinite(D_num_peak) and abs(D_phy) > 1e-30 else np.nan
            else:

                D_num_peak = np.nan
                D_num_peak_star = np.nan
                D_num_peak_over_D_phy = np.nan


            #    xc_num_final = sum(x*C*dx*A)/sum(C*dx*A)
            #    xc_ana_final = x0 + u*t_total
            #    Corrected_Shift = (xc_num_final - xc_ana_final)/dx
            final_variance, xc_num_final, final_mass, _ = calculate_spatial_variance(
                numerical_concs, x_coords_list, dx, A_fixed
            )
            xc_ana_final = x0_gauss_phys + u_fixed * t_total
            delta_L_corrected = xc_num_final - xc_ana_final
            corrected_shift = delta_L_corrected / dx
            normalized_shift = corrected_shift  # Legacy alias for the centroid-corrected shift.
            shift_sigma = abs(delta_L_corrected) / sigma_ana

            final_mass_error_pct = (final_mass / M_total - 1.0) * 100.0

            if profile_shape == 'gaussian':
                finite_core = all(np.isfinite(v) for v in [
                    c_ratio, D_num_peak, D_num_peak_over_D_phy, l2_rel,
                    corrected_shift, final_mass_error_pct
                ])
            else:
                finite_core = all(np.isfinite(v) for v in [
                    c_ratio, l2_rel, corrected_shift, final_mass_error_pct
                ])

            valid_baseline = bool(
                finite_core
                and abs(final_mass_error_pct) <= 0.5
                and l2_rel <= 1.0
                and (0.0 < c_ratio <= 1.5)
                and abs(corrected_shift) <= 2.0
            )

            if profile_shape == 'gaussian':
                positive_diffusion = bool(
                    valid_baseline
                    and c_ratio < 1.0
                    and np.isfinite(D_num_peak)
                    and D_num_peak > 0.0
                    and D_num_peak_over_D_phy > 0.0
                    and D_num_peak_star > 0.0
                )
                compression_or_antidiffusion = bool(
                    valid_baseline
                    and (c_ratio > 1.0 or D_num_peak < 0.0 or D_num_peak_over_D_phy < 0.0)
                )
                obvious_compression = bool(
                    valid_baseline
                    and (c_ratio > 1.05 or D_num_peak < 0.0 or D_num_peak_over_D_phy < 0.0)
                )
            else:

                positive_diffusion = False  # Gaussian peak-inferred diffusion is not defined here.
                compression_or_antidiffusion = bool(valid_baseline and c_ratio > 1.0)
                obvious_compression = bool(valid_baseline and c_ratio > 1.05)

            kappa95_pred = predict_kappa95(limiter_name, Nw)
            rnd_pred = kappa95_pred * Pe if np.isfinite(kappa95_pred) else np.nan
            rnd_level = classify_rnd(rnd_pred)
            complexity = SCHEME_COMPLEXITY.get(limiter_name, 3)



            if valid_baseline and (not obvious_compression):
                if profile_shape == 'gaussian':
                    eta_pos = max(D_num_peak_over_D_phy, 0.0) if np.isfinite(D_num_peak_over_D_phy) else 1e6
                    score_actual_like = eta_pos + 0.25 * abs(shift_sigma) + 0.10 * l2_rel + 0.02 * complexity
                else:
                    score_actual_like = peak_error_abs + 0.25 * abs(shift_sigma) + 0.10 * l2_rel + 0.02 * complexity
            else:
                score_actual_like = np.nan

            print(f"=== Case complete ===")
            print(f"D_num_variance = {D_numerical:.4f} | R² = {r_sq:.4f} | kurtosis = {final_kurt:.2f} | shape error = {shape_error:.4f}")
            print(f"peak retention = {c_ratio * 100:.1f}% | peak attenuation index = {peak_diss_index:.4f} | D_num_peak = {D_num_peak:.4f} | D_num_peak* = {D_num_peak_star:.4f}")
            print(f"Error components: dissipative share {diss_ratio:.1f}%, dispersive share {disp_ratio:.1f}%")
            print(f"absolute L2 error = {l2_abs:.4e}, relative error = {l2_rel:.4f}")
            print(f"regression displacement = {regression_shift:.4f} cells | centroid displacement = {corrected_shift:.4f} cells | width-normalized displacement = {shift_sigma:.4f} | mass error = {final_mass_error_pct:.4e}%")


            with open(summary_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    VALIDATION_STAGE if RUN_ATLAS_VALIDATION else 'MAIN_SCAN', case_id, limiter_name,
                    profile_shape, expected_recommended_scheme,
                    recommendation_status, atlas_consensus, decision_source, decision_rule_version,
                    ref_type, sigma_ref_phys,
                    Nw, CFL, Pe, tau, M_total, C0_equiv, t_final,
                    sigma_scale_type, sigma_ref_phys, dx, Nw_check, Pe_check, tau_check,
                    dt, CFL_adv_actual, CFL_wave_actual, Fo_diff_actual,
                    D_phy, D_global, D_numerical,
                    D_num_peak, D_num_peak_star, D_num_peak_over_D_phy, r_sq,
                    final_kurt, shape_error,
                    c_ratio, peak_diss_index, peak_error_abs, overshoot_ratio,
                    total_mse, diss_err, disp_err, diss_ratio, disp_ratio,
                    l2_abs, l2_rel,
                    u_num_regression, delta_L_regression, regression_shift,
                    xc_num_final, xc_ana_final, delta_L_corrected, corrected_shift,
                    normalized_shift, shift_sigma,
                    final_mass_error_pct,
                    int(valid_baseline), int(positive_diffusion),
                    int(compression_or_antidiffusion), int(obvious_compression),
                    kappa95_pred, rnd_pred, rnd_level, complexity, score_actual_like
                ])

            amplitude_records.append({
                'validation_stage': VALIDATION_STAGE if RUN_ATLAS_VALIDATION else 'MAIN_SCAN',
                'case_id': case_id,
                'profile_shape': profile_shape,
                'expected_recommended_scheme': expected_recommended_scheme,
                'limiter': limiter_name,
                'Nw': Nw,
                'CFL': CFL,
                'Pe': Pe,
                'tau': tau,
                'M_total': M_total,
                'Max_C_Ratio': c_ratio,
                'Peak_Dissipation_Index': peak_diss_index,
                'D_num_peak': D_num_peak,
                'D_num_variance': D_numerical,
                'L2_rel': l2_rel,
                'Normalized_Shift': normalized_shift,
                'Corrected_Shift': corrected_shift,
                'Shift_sigma': shift_sigma,
                'Diss_Ratio(%)': diss_ratio,
                'Disp_Ratio(%)': disp_ratio,
                'Final_Mass_Error(%)': final_mass_error_pct
            })


    if RUN_AMPLITUDE_TEST and len(M_total_list) > 1:
        amplitude_summary_file = os.path.join(scheme_dir, f'Amplitude_Invariance_{scheme_name}_{limiter_name}.csv')
        write_amplitude_invariance_summary(amplitude_records, amplitude_summary_file)
        print(f"📌 Amplitude-invariance table written: {amplitude_summary_file}")


def merge_scheme_summary_files(base_path, output_name='ALL_Validation_Results.csv'):
    """Combine scheme-level summary files for comparison and plotting."""
    merged_rows = []
    merged_header = None

    if not os.path.exists(base_path):
        return None

    for root, _, files in os.walk(base_path):
        for fn in files:
            if not (fn.startswith('Expansion_Study_') and fn.endswith('.csv')):
                continue
            fp = os.path.join(root, fn)
            with open(fp, 'r', newline='') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                if merged_header is None:
                    merged_header = header + ['source_file']
                for row in reader:
                    merged_rows.append(row + [fp])

    if merged_header is None:
        return None

    out_file = os.path.join(base_path, output_name)
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(merged_header)
        writer.writerows(merged_rows)
    return out_file


merged_file = merge_scheme_summary_files(base_save_path)
if merged_file:
    print(f"📌 Combined validation table written: {merged_file}")

print(f"\n🎉 All selected schemes and cases completed.")
print(f"Summary tables: {base_save_path}/{{limiter}}/Expansion_Study_*.csv")
print("Normalized_Shift records centroid displacement. Non-Gaussian cases use the convolution reference; D_num_peak is undefined.")
