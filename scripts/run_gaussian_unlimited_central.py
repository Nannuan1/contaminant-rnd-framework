import csv
import numpy as np
from scipy.stats import linregress
import math
import os

scheme_name = "HLLC_Linear"

RUN_AMPLITUDE_TEST = False


Nw_list = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
CFL_list = [0.1, 0.3, 0.5]
Pe_list = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
tau_list = [10.0, 30.0, 60.0, 90.0]


u_fixed = 1.0            # Mean flow velocity (m/s).
B = 10.0                 # Channel width (m).
ini_H = 2.0              # Initial water depth (m).
x0_gauss_factor = 10.0   # Initial centroid factor; left_margin = 10*sigma0.
k_decay = 0.0
Manningn = 0.0
NZ = 20
HOL = 0.01


base_save_path = r'./outputs/gaussian_main_cases'

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


def unlimited_central(a, b, eps=1e-12):
    """
    Unlimited central MUSCL slope used as an unrestricted control.

    Here a and b are the left and right slopes:
       a = (U_i - U_{i-1}) / dx
       b = (U_{i+1} - U_i) / dx
    2) Return the central slope 0.5*(a+b) without TVD limiting;
    3) This is an unlimited central-slope MUSCL control,
       not a central-flux scheme; the advective flux is still computed by HLLC.
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        return 0.0
    return 0.5 * (a + b)



limiter_dict = {
    'unlimited_central': unlimited_central,
}

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


def calculate_peak_based_numerical_diffusion(c_ratio, sigma0_phys, D_phy, t_final):
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
    sigma_ana_sq = sigma0_phys ** 2 + 2.0 * D_phy * t_final
    sigma_eff_sq = sigma_ana_sq / (c_ratio ** 2)
    return (sigma_eff_sq - sigma_ana_sq) / (2.0 * t_final)


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

    key_fields = ['limiter', 'Nw', 'CFL', 'Pe', 'tau']
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
for limiter_name, limiter_func in limiter_dict.items():

    scheme_dir = os.path.join(base_save_path, f'{scheme_name}_{limiter_name}')
    if not os.path.exists(scheme_dir):
        os.makedirs(scheme_dir)

    amplitude_records = []


    summary_file = os.path.join(scheme_dir, f'Expansion_Study_{scheme_name}_{limiter_name}.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Nw', 'CFL', 'Pe', 'tau', 'M_total', 'C0_equiv', 't_final',
            'dt_actual', 'CFL_adv_actual', 'CFL_wave_actual', 'Fo_diff_actual',
            'D_phy', 'D_global', 'D_num_variance',
            'D_num_peak', 'D_num_peak_star', 'D_num_peak_over_D_phy', 'R_squared',
            'Kurtosis', 'Shape_Error',
            'Max_C_Ratio', 'Peak_Dissipation_Index',
            'Total_MSE', 'Diss_Error', 'Disp_Error', 'Diss_Ratio(%)', 'Disp_Ratio(%)',
            'L2_abs', 'L2_rel',

            'u_num_regression', 'delta_L_regression', 'Regression_Shift',

            'xc_num_final', 'xc_ana_final', 'delta_L_corrected', 'Corrected_Shift',

            'Normalized_Shift', 'Shift_sigma',
            'Final_Mass_Error(%)'
        ])

    for Nw in Nw_list:
        for CFL in CFL_list:
            for Pe in Pe_list:
                for tau in tau_list:
                    for M_total in M_total_list:
                        print("\n" + "=" * 65)
                        print(f"Starting case: {limiter_name} | M_total={M_total:.1e}, Nw={Nw}, CFL={CFL}, Pe={Pe}, tau={tau}")
                        print("=" * 65)


                        sigma0_phys = 1.0
                        dx = sigma0_phys / Nw
                        D_phy = u_fixed * dx / Pe
                        c_wave = math.sqrt(9.81 * ini_H)
                        max_celerity = u_fixed + c_wave
                        dt_max = dx ** 2 / (max_celerity * dx + 2.0 * D_phy)
                        dt_target = CFL * dt_max
                        dt_target = math.floor(dt_target * 10000) / 10000.0
                        t_total = tau * sigma0_phys / u_fixed






                        if dt_target <= 0.0:
                            raise ValueError("dt_target <= 0; check CFL, dx, and D_phy.")
                        NT = int(math.ceil(t_total / dt_target)) + 1
                        dt = t_total / (NT - 1)



                        CFL_adv_actual = u_fixed * dt / dx
                        CFL_wave_actual = max_celerity * dt / dx
                        Fo_diff_actual = D_phy * dt / (dx ** 2)

                        sigma_final = math.sqrt(sigma0_phys ** 2 + 2.0 * D_phy * t_total)
                        safe_factor = 10.0
                        left_margin = safe_factor * sigma0_phys
                        right_margin = u_fixed * t_total + safe_factor * sigma_final
                        L_total = left_margin + right_margin
                        x0_gauss_phys = left_margin
                        L_total = max(L_total, 200.0)
                        NE = int(L_total / dx) + 1
                        L_total = NE * dx

                        print(f"dx={dx:.3f} m, D_phy={D_phy:.4f} m²/s, dt={dt:.5f} s, NT={NT}, L_total={L_total:.1f} m")
                        print(f"Realized stability metrics: CFL_adv={CFL_adv_actual:.4f}, CFL_wave={CFL_wave_actual:.4f}, Fo_diff={Fo_diff_actual:.4f}")
                        print(f"sigma0_phys={sigma0_phys} m, x0_gauss={x0_gauss_phys:.1f} m, t_total={t_total:.1f} s")

                        DL_const = D_phy
                        A_fixed = B * ini_H
                        C0_equiv = M_total / (A_fixed * math.sqrt(2.0 * math.pi) * sigma0_phys)


                        amp_tag = format_amplitude_tag(M_total)
                        if INCLUDE_AMPLITUDE_TAG_IN_PATH:
                            save_path = os.path.join(scheme_dir, f'{amp_tag}_Nw_{Nw}_CFL_{CFL}_Pe_{Pe}_tau_{tau}')
                        else:
                            save_path = os.path.join(scheme_dir, f'Nw_{Nw}_CFL_{CFL}_Pe_{Pe}_tau_{tau}')
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
                            for j in range(int(R.NE)):
                                E = Element()
                                E.Q = u_fixed * B * ini_H
                                E.dx = tmpd
                                E.zs = E.zb + ini_H
                                x_left = j * tmpd
                                x_right = (j + 1) * tmpd
                                D_eff = D_phy
                                t_offset = sigma0_phys ** 2 / (2.0 * D_eff)
                                denominator = math.sqrt(4 * D_eff * t_offset)
                                erf_right = math.erf((x_right - x0_gauss_phys) / denominator)
                                erf_left = math.erf((x_left - x0_gauss_phys) / denominator)
                                mass_in_cell = (M_total / A_fixed) * 0.5 * (erf_right - erf_left)
                                E.C = mass_in_cell / tmpd
                                E.C = max(E.C, 0.0)
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

                            compute_gradients_MUSCL(Es, Fs, NE, limiter_func)

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

                                t_eff_current = current_time + (sigma0_phys ** 2) / (2.0 * D_phy)
                                center_current = x0_gauss_phys + u_fixed * current_time
                                denominator_cell = math.sqrt(4.0 * D_phy * t_eff_current)
                                current_analytical = []
                                for i in range(NE):
                                    x_left = i * dx
                                    x_right = (i + 1) * dx
                                    erf_right = math.erf((x_right - center_current) / denominator_cell)
                                    erf_left = math.erf((x_left - center_current) / denominator_cell)
                                    mass_in_cell = (M_total / A_fixed) * 0.5 * (erf_right - erf_left)
                                    c_exact = mass_in_cell / dx
                                    current_analytical.append(c_exact)
                                with open(Analytical_Profile_file, 'a', newline='') as f:
                                    csv.writer(f).writerow([itime, current_time] + current_analytical)


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


                        t_eff = t_final + (sigma0_phys ** 2) / (2.0 * D_phy)
                        center_pos = x0_gauss_phys + u_fixed * t_final
                        analytical_concs = []
                        denominator_final = math.sqrt(4 * D_phy * t_eff)
                        for i in range(NE):
                            x_left = i * dx
                            x_right = (i + 1) * dx
                            erf_right = math.erf((x_right - center_pos) / denominator_final)
                            erf_left = math.erf((x_left - center_pos) / denominator_final)
                            mass_in_cell = (M_total / A_fixed) * 0.5 * (erf_right - erf_left)
                            c_exact = mass_in_cell / dx
                            analytical_concs.append(c_exact)

                        total_mse, diss_err, disp_err, diss_ratio, disp_ratio = calculate_takacs_errors(analytical_concs, numerical_concs)
                        max_c_sim = max(numerical_concs) if numerical_concs else 0.0
                        max_c_ana = max(analytical_concs) if analytical_concs else 1e-12
                        c_ratio = max_c_sim / max_c_ana
                        peak_diss_index = 1.0 - c_ratio
                        final_kurt = record_kurtosis[-1] if record_kurtosis else 3.0
                        shape_error = abs(final_kurt - 3.0)

                        l2_abs = np.sqrt(total_mse)
                        l2_ana = np.sqrt(np.mean(np.array(analytical_concs) ** 2))
                        l2_rel = l2_abs / l2_ana if l2_ana > 1e-12 else 0.0


                        sigma_ana_sq = sigma0_phys ** 2 + 2.0 * D_phy * t_final
                        sigma_ana = math.sqrt(max(sigma_ana_sq, 1e-30))
                        D_num_peak = calculate_peak_based_numerical_diffusion(c_ratio, sigma0_phys, D_phy, t_final)
                        D_num_peak_star = D_num_peak / (u_fixed * dx) if np.isfinite(D_num_peak) and abs(u_fixed * dx) > 1e-30 else np.nan
                        D_num_peak_over_D_phy = D_num_peak / D_phy if np.isfinite(D_num_peak) and abs(D_phy) > 1e-30 else np.nan


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

                        print(f"=== Case complete ===")
                        print(f"D_num_variance = {D_numerical:.4f} | R² = {r_sq:.4f} | kurtosis = {final_kurt:.2f} | shape error = {shape_error:.4f}")
                        print(f"peak retention = {c_ratio * 100:.1f}% | peak attenuation index = {peak_diss_index:.4f} | D_num_peak = {D_num_peak:.4f} | D_num_peak* = {D_num_peak_star:.4f}")
                        print(f"Error components: dissipative share {diss_ratio:.1f}%, dispersive share {disp_ratio:.1f}%")
                        print(f"absolute L2 error = {l2_abs:.4e}, relative error = {l2_rel:.4f}")
                        print(f"regression displacement = {regression_shift:.4f} cells | centroid displacement = {corrected_shift:.4f} cells | width-normalized displacement = {shift_sigma:.4f} | mass error = {final_mass_error_pct:.4e}%")


                        with open(summary_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                Nw, CFL, Pe, tau, M_total, C0_equiv, t_final,
                                dt, CFL_adv_actual, CFL_wave_actual, Fo_diff_actual,
                                D_phy, D_global, D_numerical,
                                D_num_peak, D_num_peak_star, D_num_peak_over_D_phy, r_sq,
                                final_kurt, shape_error,
                                c_ratio, peak_diss_index,
                                total_mse, diss_err, disp_err, diss_ratio, disp_ratio,
                                l2_abs, l2_rel,
                                u_num_regression, delta_L_regression, regression_shift,
                                xc_num_final, xc_ana_final, delta_L_corrected, corrected_shift,
                                normalized_shift, shift_sigma,
                                final_mass_error_pct
                            ])

                        amplitude_records.append({
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

print(f"\n🎉 All selected schemes and cases completed.")
print(f"Summary tables: {base_save_path}/{{limiter}}/Expansion_Study_*.csv")
print("Normalized_Shift records centroid displacement; Regression_Shift records the velocity-regression estimate.")
