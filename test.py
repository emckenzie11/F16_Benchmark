import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft, rfftfreq

# ----------------- USER CONFIG -----------------
training_level = [3]      # level(s) used to train R(x,v)
validation_level = [2]    # level(s) used for validation
location_i = 2
location_j = 3

fs = 400.0
dt = 1.0 / fs

target_freq = 7.3
freq_low = 6.8
freq_high = 8.6

m_eff = 1.0
poly_order = 3  # Chebyshev order (1..poly_order used)
# ------------------------------------------------

# --------- I/O: adjust file paths if needed ----------
training_data = {
    1: pd.read_csv('BenchmarkData/F16Data_SineSw_Level1.csv'),
    3: pd.read_csv('BenchmarkData/F16Data_SineSw_Level3.csv'),
    5: pd.read_csv('BenchmarkData/F16Data_SineSw_Level5.csv'),
    7: pd.read_csv('BenchmarkData/F16Data_SineSw_Level7.csv')
}
validation_data = {
    2: pd.read_csv('BenchmarkData/F16Data_SineSw_Level2_Validation.csv'),
    4: pd.read_csv('BenchmarkData/F16Data_SineSw_Level4_Validation.csv'),
    6: pd.read_csv('BenchmarkData/F16Data_SineSw_Level6_Validation.csv'),
}

# Keep only requested levels
training_data = {lvl: training_data[lvl] for lvl in training_level}
validation_data = {lvl: validation_data[lvl] for lvl in validation_level}

# --------------- HELPERS ----------------
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, signal)

def band_limited_integrate_fft(a_t, fs, f_lo, f_hi, order=1):
    n = len(a_t)
    A = rfft(a_t - np.mean(a_t))
    f = rfftfreq(n, 1.0/fs)
    mask = (f >= f_lo) & (f <= f_hi)
    Y = np.zeros_like(A, dtype=np.complex128)
    if order == 0:
        Y[mask] = A[mask]
    else:
        omega = 2.0 * np.pi * f
        denom = (1j * omega)**order
        denom[0] = np.inf
        Y[mask] = A[mask] / denom[mask]
    x_t = irfft(Y, n=n)
    return np.real(x_t)

def compute_relative_motion(accel_i, accel_j, fs, f_lo, f_hi):
    a_rel = accel_i - accel_j
    v_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=1)
    x_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=2)
    return x_rel, v_rel

def compute_restoring_force_from_accel(a_i_bp, m_eff=1.0):
    # R(t) = - m * a(t)  (restoring force proxy)
    return -m_eff * a_i_bp

def scale_to_unit_interval(x):
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        # avoid div by zero if constant signal
        return np.zeros_like(x), x_min, x_max
    x_scaled = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    return x_scaled, x_min, x_max

# Chebyshev 1D basis (T1..Tn) for array x
def chebyshev_basis(x, order):
    N = len(x)
    basis = np.zeros((N, order))
    # T1 = x, T2 = 2x^2 - 1, Tn = 2x*T_{n-1} - T_{n-2}
    basis[:,0] = x
    if order > 1:
        basis[:,1] = 2*x**2 - 1
    for n in range(2, order):
        basis[:, n] = 2*x * basis[:, n-1] - basis[:, n-2]
    return basis

def build_2d_chebyshev_basis(x, v, order):
    bz = chebyshev_basis(x, order)
    bv = chebyshev_basis(v, order)
    N = len(x)
    basis_list = []
    for i in range(order):
        for j in range(order):
            basis_list.append(bz[:, i] * bv[:, j])
    H_poly = np.stack(basis_list, axis=1)  # shape (N, order**2)
    return H_poly

# predict R for scalar arrays x and v using fitted alpha
def predict_restoring_force(x, v, alpha, x_min, x_max, v_min, v_max, poly_order):
    # Convert to arrays (ensures length)
    x = np.atleast_1d(x)
    v = np.atleast_1d(v)

    # Scale to [-1,1]
    if x_max == x_min:
        x_scaled = np.zeros_like(x)
    else:
        x_scaled = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    if v_max == v_min:
        v_scaled = np.zeros_like(v)
    else:
        v_scaled = 2.0 * (v - v_min) / (v_max - v_min) - 1.0

    # Build basis
    H = build_2d_chebyshev_basis(x_scaled, v_scaled, poly_order)

    # Evaluate
    R = H.dot(alpha)

    # If input was scalar → return scalar
    if R.size == 1:
        return R[0]
    return R


# simple semi-implicit Euler integrator for simulation
def simulate_response(force_input, alpha, x_min, x_max, v_min, v_max, poly_order,
                      dt, m_eff, x0=0.0, v0=0.0, bandpass_force=True):
    N = len(force_input)
    x = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)
    x[0] = x0
    v[0] = v0

    # if bandpass_force True, restrict force to modal band (avoid out-of-band energy)
    if bandpass_force:
        F_bp = bandpass_filter(force_input, freq_low, freq_high, fs)
    else:
        F_bp = force_input.copy()

    for k in range(N-1):
        Rk = predict_restoring_force(x[k], v[k], alpha, x_min, x_max, v_min, v_max, poly_order)
        ak = (F_bp[k] - Rk) / m_eff
        v[k+1] = v[k] + ak * dt
        x[k+1] = x[k] + v[k+1] * dt    # semi-implicit: use updated velocity
        a[k] = ak

    # final a last sample
    a[-1] = (F_bp[-1] - predict_restoring_force(x[-1], v[-1], alpha, x_min, x_max, v_min, v_max, poly_order)) / m_eff
    return x, v, a

# ---------------- PROCESS TRAINING DATA ----------------
# We'll only use the first (and in your case only) training level to fit Chebyshev
train_level = training_level[0]

# get raw arrays
accel_i = training_data[train_level][f'Acceleration{location_i}'].to_numpy()
accel_j = training_data[train_level][f'Acceleration{location_j}'].to_numpy()
force_input_train = training_data[train_level]['Force'].to_numpy()

# bandpass accelerations (modal isolation)
a_i_bp = bandpass_filter(accel_i, freq_low, freq_high, fs)
a_j_bp = bandpass_filter(accel_j, freq_low, freq_high, fs)

# compute relative states (x_rel, v_rel) via band-limited FFT integration
x_rel_train, v_rel_train = compute_relative_motion(a_i_bp, a_j_bp, fs, freq_low, freq_high)

# compute restoring force target for training
f_rest_train = compute_restoring_force_from_accel(a_i_bp, m_eff)

# Save processed arrays back for reference
training_data[train_level]['x_rel'] = x_rel_train
training_data[train_level]['v_rel'] = v_rel_train
training_data[train_level]['f_rest'] = f_rest_train

# ---------------- SCALE STATES ----------------
x_scaled_train, x_min, x_max = scale_to_unit_interval(x_rel_train)
v_scaled_train, v_min, v_max = scale_to_unit_interval(v_rel_train)

# ---------------- BUILD CHEBYSHEV BASIS & FIT ----------------
H_poly = build_2d_chebyshev_basis(x_scaled_train, v_scaled_train, poly_order)
# Fit: f_rest = H_poly @ alpha
alpha, residuals, rank, s = np.linalg.lstsq(H_poly, f_rest_train, rcond=None)
print("-"*60)
print(f"Fitted {len(alpha)} Chebyshev coefficients (order={poly_order})")
print("Lstsq residual norm:", np.sum(residuals) if len(residuals)>0 else 0.0)

# Compute training prediction (for diagnostics)
F_train_pred = H_poly.dot(alpha)

# ------------------- PREPARE THRESHOLDS FOR SLICING -------------------
vel_thresh_frac = 0.02
disp_thresh_frac = 0.02
vel_thresh_min = 1e-6
disp_thresh_min = 1e-7

eps_v = max(vel_thresh_frac * np.max(np.abs(v_rel_train)), vel_thresh_min)
eps_x = max(disp_thresh_frac * np.max(np.abs(x_rel_train)), disp_thresh_min)
print(f"Using thresholds: eps_v={eps_v:.3e}, eps_x={eps_x:.3e}")

# ----------------- VALIDATION: process and simulate -----------------
for val_level in validation_level:
    # load raw validation signals
    accel_i_val = validation_data[val_level][f'Acceleration{location_i}'].to_numpy()
    accel_j_val = validation_data[val_level][f'Acceleration{location_j}'].to_numpy()
    force_input_val = validation_data[val_level]['Force'].to_numpy()

    # bandpass (same filter as training)
    a_i_bp_val = bandpass_filter(accel_i_val, freq_low, freq_high, fs)
    a_j_bp_val = bandpass_filter(accel_j_val, freq_low, freq_high, fs)

    # compute measured relative x, v for validation (to compare to sim)
    x_rel_val, v_rel_val = compute_relative_motion(a_i_bp_val, a_j_bp_val, fs, freq_low, freq_high)

    # compute measured restoring force (target) (shows what was observed)
    f_rest_val = compute_restoring_force_from_accel(a_i_bp_val, m_eff)

    # simulate response using learned Chebyshev model (use bandpassed force in sim)
    x_sim, v_sim, a_sim = simulate_response(force_input_val, alpha,
                                           x_min, x_max, v_min, v_max, poly_order,
                                           dt, m_eff, x0=0.0, v0=0.0,
                                           bandpass_force=True)

    # Save into validation_data for plotting convenience
    validation_data[val_level]['x_rel'] = x_rel_val
    validation_data[val_level]['v_rel'] = v_rel_val
    validation_data[val_level]['f_rest'] = f_rest_val
    validation_data[val_level]['x_sim'] = x_sim
    validation_data[val_level]['v_sim'] = v_sim
    validation_data[val_level]['a_sim'] = a_sim

    # ---------------- PLOT: time series comparison ----------------
    t = np.arange(len(x_rel_val)) / fs
    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax[0].plot(t, x_rel_val, label='Measured x_rel (val)', linewidth=1)
    ax[0].plot(t, x_sim, '--', label='Simulated x (model)', linewidth=1)
    ax[0].set_ylabel('Relative displacement [m]')
    ax[0].legend(); ax[0].grid(True)

    ax[1].plot(t, v_rel_val, label='Measured v_rel (val)', linewidth=1)
    ax[1].plot(t, v_sim, '--', label='Simulated v (model)', linewidth=1)
    ax[1].set_ylabel('Relative velocity [m/s]')
    ax[1].set_xlabel('Time [s]')
    ax[1].legend(); ax[1].grid(True)
    plt.suptitle(f'Validation Level {val_level}: Time-domain comparison')
    plt.tight_layout()
    plt.show()

    # ---------------- PLOT: RFS slices (validation: measured vs predicted) ----------------
    # build predicted restoring force at validation measured x,v for slice plots
    R_pred_val = predict_restoring_force(x_rel_val, v_rel_val, alpha, x_min, x_max, v_min, v_max, poly_order)

    # stiff slice (near-zero velocity)
    stiff_mask = np.abs(v_rel_val) <= eps_v
    disp_mask = np.abs(x_rel_val) <= eps_x

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

    ax1.scatter(x_rel_val[stiff_mask], f_rest_val[stiff_mask], s=8, alpha=0.5, label='Measured R (val)')
    ax1.scatter(x_rel_val[stiff_mask], R_pred_val[stiff_mask], s=6, alpha=0.9, marker='x', label='Predicted R (Cheb)')
    ax1.set_xlabel('x_rel [m]'); ax1.set_ylabel('R [N]'); ax1.set_title('Stiffness slice (near-zero velocity)'); ax1.grid(True); ax1.legend()

    ax2.scatter(v_rel_val[disp_mask], f_rest_val[disp_mask], s=8, alpha=0.5, label='Measured R (val)')
    ax2.scatter(v_rel_val[disp_mask], R_pred_val[disp_mask], s=6, alpha=0.9, marker='x', label='Predicted R (Cheb)')
    ax2.set_xlabel('v_rel [m/s]'); ax2.set_ylabel('R [N]'); ax2.set_title('Damping slice (near-zero displacement)'); ax2.grid(True); ax2.legend()

    plt.suptitle(f'Validation Level {val_level}: RFS slices')
    plt.tight_layout()
    plt.show()

    # ---------------- compute RMSE for the slices ----------------
    from math import sqrt
    rmse_stiff = None
    rmse_damp = None
    if np.sum(stiff_mask) > 5:
        e = R_pred_val[stiff_mask] - f_rest_val[stiff_mask]
        rmse_stiff = np.sqrt(np.mean(e**2))
        print(f"Level {val_level} stiffness slice RMSE: {rmse_stiff:.6e}")
    if np.sum(disp_mask) > 5:
        e = R_pred_val[disp_mask] - f_rest_val[disp_mask]
        rmse_damp = np.sqrt(np.mean(e**2))
        print(f"Level {val_level} damping slice RMSE:   {rmse_damp:.6e}")

    if (rmse_stiff is not None) and (rmse_damp is not None):
        overall_rmse = np.sqrt(0.5 * (rmse_stiff**2 + rmse_damp**2))
        print(f"Level {val_level} overall RMSE:         {overall_rmse:.6e}")

print("-"*60)
print("CHEBYSHEV TRAIN/VALIDATION COMPLETE")
print("-"*60)
