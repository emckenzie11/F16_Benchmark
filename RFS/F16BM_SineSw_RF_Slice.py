import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft, rfftfreq

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Plot configuration
levels_to_compute = [1, 3, 5, 7]  # Levels to compute RFS (Options: 1, 3, 5, 7)
location_i = 2  # DOF i (location where acceleration is measured)
location_j = 3  # DOF j (location across the nonlinear connection)

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step

# Mode isolation parameters
target_freq = 7.3     # Target mode frequency [Hz]
freq_low = 6.8        # Lower frequency bound [Hz]
freq_high = 8.6       # Upper frequency bound [Hz]

# Thresholds - determines thickness of RFS slices
vel_thresh_frac = 1     # % of max |v_rel|
disp_thresh_frac = 1    # % of max |x_rel|

# Minimum floors in case of tiny signals
vel_thresh_min = 1e-6
disp_thresh_min = 1e-7

# Effective mass (normalized).
m_eff = 1.0

# Input CSVs
data_dict = {
    1: pd.read_csv('BenchmarkData/F16Data_SineSw_Level1.csv'),
    3: pd.read_csv('BenchmarkData/F16Data_SineSw_Level3.csv'),
    5: pd.read_csv('BenchmarkData/F16Data_SineSw_Level5.csv'),
    7: pd.read_csv('BenchmarkData/F16Data_SineSw_Level7.csv')
}
# ------------------------------------------

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

# Store results for all levels
rfs_data = {}

for level in levels_to_compute:
    data = data_dict[level]

    # Extract acceleration signals (assumed in m/s^2)
    accel_i_raw = data[f'Acceleration{location_i}'].to_numpy()
    accel_j_raw = data[f'Acceleration{location_j}'].to_numpy()

    # Apply bandpass filter to isolate the target mode (for consistency across all signals)
    print(f"Level {level} - Applying bandpass filter ({freq_low}-{freq_high} Hz)...")
    accel_i_bp = bandpass_filter(accel_i_raw, freq_low, freq_high, fs)
    accel_j_bp = bandpass_filter(accel_j_raw, freq_low, freq_high, fs)

    # Relative motion (subtract first, then integrate band-limited)  <-- Change 2
    rel_disp, rel_vel = compute_relative_motion(accel_i_bp, accel_j_bp, fs, freq_low, freq_high)

    # Restoring force proxy (use band-passed a_i; scale by mass if known)
    restoring_force = compute_restoring_force(accel_i_bp, m_eff=m_eff)

    # Adaptive, level-dependent thresholds 
    eps_v = max(vel_thresh_frac * np.max(np.abs(rel_vel)), vel_thresh_min)
    eps_x = max(disp_thresh_frac * np.max(np.abs(rel_disp)), disp_thresh_min)

    # Masks:
    # - Stiffness slice (R vs x): use near-zero velocity
    near_zero_vel_mask = np.abs(rel_vel) <= eps_v
    # - Damping slice (R vs v): use near-zero displacement
    near_zero_disp_mask = np.abs(rel_disp) <= eps_x

    # Store results
    rfs_data[level] = {
        'rel_disp': rel_disp,
        'rel_vel': rel_vel,
        'restoring_force': restoring_force,
        'vel_mask': near_zero_vel_mask,   # used for stiffness slice
        'disp_mask': near_zero_disp_mask   # used for damping slice
    }

# ---------------------- Plotting ----------------------

def plot_single_level(ax_stiff, ax_damp, level, title_prefix=('a', 'b')):
    """Plot a single level's stiffness and damping characteristics."""
    rel_disp = rfs_data[level]['rel_disp']
    rel_vel = rfs_data[level]['rel_vel']
    R = rfs_data[level]['restoring_force']
    mask_stiff = rfs_data[level]['vel_mask']
    mask_damp = rfs_data[level]['disp_mask']

    # Stiffness plot (R vs displacement)
    ax_stiff.plot(rel_disp[mask_stiff], R[mask_stiff],
                  'o', color='black', markersize=3, alpha=0.6,
                  markerfacecolor='none', markeredgewidth=0.5)
    ax_stiff.set_xlabel('Relative Displacement [m]')
    ax_stiff.set_ylabel('-Acceleration [m/s²]')
    ax_stiff.set_title(f'({title_prefix[0]}) Level {level} - Stiffness')
    ax_stiff.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax_stiff.grid(False)

    # Damping plot (R vs velocity)
    ax_damp.plot(rel_vel[mask_damp], R[mask_damp],
                 'o', color='black', markersize=3, alpha=0.6,
                 markerfacecolor='none', markeredgewidth=0.5)
    ax_damp.set_xlabel('Relative Velocity [m/s]')
    ax_damp.set_ylabel('-Acceleration [m/s²]')
    ax_damp.set_title(f'({title_prefix[1]}) Level {level} - Damping')
    ax_damp.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax_damp.grid(False)

# Dynamic plotting based on number of levels
n_levels = len(levels_to_compute)
figures = []  # Store all figures to show them simultaneously

if n_levels == 1:
    # Single level: 1 figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_single_level(axes[0], axes[1], levels_to_compute[0], title_prefix=('a', 'b'))
    plt.tight_layout()
    figures.append(fig)

elif n_levels == 2:
    # Two levels: 1 figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plot_single_level(axes[0, 0], axes[0, 1], levels_to_compute[0], title_prefix=('a', 'b'))
    plot_single_level(axes[1, 0], axes[1, 1], levels_to_compute[1], title_prefix=('c', 'd'))
    plt.tight_layout()
    figures.append(fig)

else:
    # Multiple levels: split into figures with max 2 levels per figure
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    title_idx = 0
    
    for i in range(0, n_levels, 2):
        # Determine how many levels for this figure
        levels_in_fig = levels_to_compute[i:i+2]
        
        if len(levels_in_fig) == 1:
            # Single level in this figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            plot_single_level(axes[0], axes[1], levels_in_fig[0], 
                            title_prefix=(alphabet[title_idx], alphabet[title_idx+1]))
            title_idx += 2
        else:
            # Two levels in this figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            plot_single_level(axes[0, 0], axes[0, 1], levels_in_fig[0], 
                            title_prefix=(alphabet[title_idx], alphabet[title_idx+1]))
            plot_single_level(axes[1, 0], axes[1, 1], levels_in_fig[1], 
                            title_prefix=(alphabet[title_idx+2], alphabet[title_idx+3]))
            title_idx += 4
        
        plt.tight_layout()
        figures.append(fig)

# Show all figures simultaneously
plt.show()
