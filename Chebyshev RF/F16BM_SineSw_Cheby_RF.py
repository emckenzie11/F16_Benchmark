import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft, rfftfreq

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
levels_to_compute = [5]  # Levels to compute RFS
location_i = 2  # DOF i
location_j = 3  # DOF j

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step

# Mode isolation parameters
target_freq = 7.3     # Target mode frequency [Hz]
freq_low = 6.8        # Lower frequency bound [Hz]
freq_high = 8.6       # Upper frequency bound [Hz]

# Sampling parameters
fs = 400
dt = 1 / fs

# Effective mass
m_eff = 1.0

# Chebyshev polynomial order
poly_order = 3

# ----------- LOAD DATA -----------
data_dict = {
    1: pd.read_csv('BenchmarkData/F16Data_SineSw_Level1.csv'),
    3: pd.read_csv('BenchmarkData/F16Data_SineSw_Level3.csv'),
    5: pd.read_csv('BenchmarkData/F16Data_SineSw_Level5.csv'),
    7: pd.read_csv('BenchmarkData/F16Data_SineSw_Level7.csv')
}

# Keep only the levels requested
data_dict = {lvl: data_dict[lvl] for lvl in levels_to_compute}

# ----------- HELPER FUNCTIONS -----------
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass Butterworth filter to isolate a specific frequency band.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, signal)

def band_limited_integrate_fft(a_t, fs, f_lo, f_hi, order=1):
    """
    Band-limited integration via FFT division by (jω)^order inside [f_lo, f_hi].
    - a_t: array (time) acceleration signal
    - order: 1 -> velocity, 2 -> displacement; 0 -> band-pass only (no integration)
    Returns time-domain signal integrated within the band; zeros outside.
    """
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
        denom[0] = np.inf  # avoid DC
        Y[mask] = A[mask] / denom[mask]
    return irfft(Y, n=n)

def compute_relative_motion(accel_i, accel_j, fs, f_lo, f_hi):
    """
    Compute relative displacement and velocity from acceleration measurements
    using band-limited FFT integration.  <-- Change 2
    """
    # Relative acceleration
    a_rel = (accel_i - accel_j)
    # Band-limited integrations
    v_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=1)
    x_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=2)
    return x_rel, v_rel

def compute_restoring_force(accel_i_bp, m_eff=1.0):
    """
    Restoring force proxy using band-passed acceleration at DOF i:
      R(t) ≈ -m_eff * a_i_bp
    If you have measured input force F_i, prefer R = F_i_bp - m_eff * a_i_bp.
    """
    return -m_eff * accel_i_bp

# ----------- PROCESS DATA -----------
for level in levels_to_compute:
    print(f"Processing Level {level}...")
    data = data_dict[level]

    # Extract acceleration signals
    accel_i = data[f'Acceleration{location_i}'].to_numpy()
    accel_j = data[f'Acceleration{location_j}'].to_numpy()

    # Apply bandpass filter to isolate the target mode
    print(f"Level {level} - Applying bandpass filter ({freq_low}-{freq_high} Hz)...")
    a_i_bp = bandpass_filter(accel_i, freq_low, freq_high, fs)
    a_j_bp = bandpass_filter(accel_j, freq_low, freq_high, fs)

    # Compute relative motion using band-limited FFT integration
    x_rel, v_rel = compute_relative_motion(a_i_bp, a_j_bp, fs, freq_low, freq_high)

    # Compute restoring force
    f_rest = compute_restoring_force(a_i_bp, m_eff)

# ----------- SCALE RELATIVE STATES TO [-1,1] -----------
def scale_to_unit_interval(x):
    """Scale array to [-1,1] for Chebyshev polynomials."""
    x_min = np.min(x)
    x_max = np.max(x)
    x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1
    return x_scaled, x_min, x_max

x_scaled, x_min, x_max = scale_to_unit_interval(x_rel)
v_scaled, v_min, v_max = scale_to_unit_interval(v_rel)

# ----------- BUILD 2D CHEBYSHEV BASIS -----------
def chebyshev_basis(x, order):
    """Compute Chebyshev polynomials T1..Tn for 1D array x."""
    N = len(x)
    basis = np.zeros((N, order))
    basis[:, 0] = x  # T1
    if order > 1:
        basis[:, 1] = 2*x**2 - 1  # T2
    for n in range(2, order):
        basis[:, n] = 2*x*basis[:, n-1] - basis[:, n-2]
    return basis

def build_2d_chebyshev_basis(x, v, order):
    """Build 2D Chebyshev basis for all combinations of x^i * v^j"""
    bz = chebyshev_basis(x, order)
    bv = chebyshev_basis(v, order)
    N = len(x)
    basis_list = []
    for i in range(order):
        for j in range(order):
            basis_list.append(bz[:, i] * bv[:, j])
    H_poly = np.stack(basis_list, axis=1)
    return H_poly

H_poly = build_2d_chebyshev_basis(x_scaled, v_scaled, poly_order)

# ----------- FIT RESTORING FORCE COEFFICIENTS -----------

# Linear least squares: f_rest = H_poly @ alpha
alpha, residuals, rank, s = np.linalg.lstsq(H_poly, f_rest, rcond=None)

print("-"*60)
print(f"Fitted {len(alpha)} Chebyshev coefficients")

# ----------- PREDICT RESTORING FORCE USING CHEBYSHEV POLYNOMIALS -----------
F_pred = H_poly @ alpha

# ----------- SLICE ANALYSIS WITH 2% THRESHOLD -----------
# Thresholds for slicing (2% of maximum values)
vel_thresh_frac = 0.02    # 2% of max |v_rel| 
disp_thresh_frac = 0.02   # 2% of max |x_rel|

# Minimum floors in case of tiny signals
vel_thresh_min = 1e-6
disp_thresh_min = 1e-7

# Calculate adaptive thresholds
eps_v = max(vel_thresh_frac * np.max(np.abs(v_rel)), vel_thresh_min)
eps_x = max(disp_thresh_frac * np.max(np.abs(x_rel)), disp_thresh_min)

print("-"*60)
print(f"Velocity threshold: {eps_v:.6e} m/s ({vel_thresh_frac*100}% of max)")
print(f"Displacement threshold: {eps_x:.6e} m ({disp_thresh_frac*100}% of max)")

# Create masks for slicing
# Stiffness slice (R vs displacement): near-zero velocity
near_zero_vel_mask = np.abs(v_rel) <= eps_v
# Damping slice (R vs velocity): near-zero displacement  
near_zero_disp_mask = np.abs(x_rel) <= eps_x

# ----------- VISUALIZE SLICE PLOTS -----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Stiffness plot (Restoring Force vs Displacement)
ax1.plot(x_rel[near_zero_vel_mask], f_rest[near_zero_vel_mask], 
         'bo', markersize=3, alpha=0.6, markerfacecolor='none', 
         markeredgewidth=0.8, label='Measured')
ax1.plot(x_rel[near_zero_vel_mask], F_pred[near_zero_vel_mask], 
         'ro', markersize=2, alpha=0.8, label='Chebyshev Fit')
ax1.set_xlabel('Relative Displacement [m]')
ax1.set_ylabel('Restoring Force [m/s²]')
ax1.set_title(f'(a) Stiffness Characteristic - Level {levels_to_compute[0]}')
ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.grid(True, alpha=0.3)
ax1.legend()

# Damping plot (Restoring Force vs Velocity)
ax2.plot(v_rel[near_zero_disp_mask], f_rest[near_zero_disp_mask], 
         'bo', markersize=3, alpha=0.6, markerfacecolor='none', 
         markeredgewidth=0.8, label='Measured')
ax2.plot(v_rel[near_zero_disp_mask], F_pred[near_zero_disp_mask], 
         'ro', markersize=2, alpha=0.8, label='Chebyshev Fit')
ax2.set_xlabel('Relative Velocity [m/s]')
ax2.set_ylabel('Restoring Force [m/s²]')
ax2.set_title(f'(b) Damping Characteristic - Level {levels_to_compute[0]}')
ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# ----------- COMPUTE FIT QUALITY -----------
print("-"*60)
# Calculate RMSE
def rmse(y_true, y_pred):
    """
    Compute the Root Mean Square Error (RMSE) between predictions and true values.
    
    Parameters
    ----------
    y_true : array-like, shape (N,)
        True values
    y_pred : array-like, shape (N,)
        Predicted values
    
    Returns
    -------
    float
        RMSE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    error = y_pred - y_true
    mse = np.mean(error**2)   # Mean squared error
    return np.sqrt(mse)       # Root of mean squared error

if np.sum(near_zero_vel_mask) > 1:
    rmse_stiffness = rmse(f_rest[near_zero_vel_mask], F_pred[near_zero_vel_mask])
    print(f"Stiffness slice RMSE = {rmse_stiffness:.4f}")

if np.sum(near_zero_disp_mask) > 1:
    rmse_damping = rmse(f_rest[near_zero_disp_mask], F_pred[near_zero_disp_mask])
    print(f"Damping slice RMSE = {rmse_damping:.4f}")

# Calculate combined overall RMSE
rmse_combined = np.sqrt(0.5 * (rmse_stiffness**2 + rmse_damping**2))

print(f"Overall RMSE = {rmse_combined:.4f}")
print("-"*60)
print("ANALYSIS COMPLETE")
print("-"*60)







