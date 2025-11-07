import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft, rfftfreq

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
training_level = [1, 3, 5, 7]  # Levels to compute RFS (Options: 1, 3, 5, 7)
location_i = 2  # DOF i
location_j = 3  # DOF j

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step

# Mode isolation parameters
target_freq = 7.3     # Target mode frequency [Hz]
freq_low = 6.8        # Lower frequency bound [Hz]
freq_high = 8.6       # Upper frequency bound [Hz]

# Effective mass
m_eff = 1.0

# Thresholds - determines thickness of RFS slices
vel_thresh_frac = 0.02    # % of max |v_rel|
disp_thresh_frac = 0.02   # % of max |x_rel|

# Minimum floors in case of tiny signals
vel_thresh_min = 1e-6
disp_thresh_min = 1e-7

# ----------- LOAD DATA -----------
training_data = {
    1: pd.read_csv('BenchmarkData/F16Data_SineSw_Level1.csv'),
    3: pd.read_csv('BenchmarkData/F16Data_SineSw_Level3.csv'),
    5: pd.read_csv('BenchmarkData/F16Data_SineSw_Level5.csv'),
    7: pd.read_csv('BenchmarkData/F16Data_SineSw_Level7.csv')
}
    
# Keep only the levels requested
training_data = {level: training_data[level] for level in training_level}

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

def compute_restoring_force( accel_i_bp, accel_j_bp, m_eff=1.0):
    """
    Restoring force proxy using band-passed acceleration at DOF i:
      R(t) ≈ F_i_bp - m_eff * a_i_bp.
    """
    return - m_eff * (accel_i_bp - accel_j_bp)

# ----------- PROCESS DATA -----------
# Store results for all levels
rfs_data = {}

for level in training_level:
    # Extract acceleration signals from raw data
    accel_i = training_data[level][f'Acceleration{location_i}'].to_numpy()
    accel_j = training_data[level][f'Acceleration{location_j}'].to_numpy()
    force_input = training_data[level][f'Force'].to_numpy()

    # Apply bandpass filter to isolate the target mode
    a_i_bp = bandpass_filter(accel_i, freq_low, freq_high, fs)
    a_j_bp = bandpass_filter(accel_j, freq_low, freq_high, fs)
    F_i_bp = bandpass_filter(force_input, freq_low, freq_high, fs)

    # Compute relative motion using band-limited FFT integration
    x_rel, v_rel = compute_relative_motion(a_i_bp, a_j_bp, fs, freq_low, freq_high)

    # Compute restoring force
    f_rest = compute_restoring_force(a_i_bp, a_j_bp, m_eff)

    # Add processed data directly to the training_data dictionary
    training_data[level]['x_rel'] = x_rel
    training_data[level]['v_rel'] = v_rel
    training_data[level]['f_rest'] = f_rest

# ----------- SLICE ANALYSIS WITH 2% THRESHOLD -----------
# Thresholds for slicing (2% of maximum values)
print("-"*60)
print(f"Velocity threshold: {vel_thresh_frac*100}% of max |v_rel|")
print(f"Displacement threshold: {disp_thresh_frac*100}% of max |x_rel|")

# Store results for all levels
rfs_data = {}

for level in training_level:
    # Get processed data
    x_rel = training_data[level]['x_rel']
    v_rel = training_data[level]['v_rel']
    f_rest = training_data[level]['f_rest']

    # Calculate adaptive thresholds
    eps_v = max(vel_thresh_frac * np.max(np.abs(v_rel)), vel_thresh_min)
    eps_x = max(disp_thresh_frac * np.max(np.abs(x_rel)), disp_thresh_min)

    print(f"Level {level} - Velocity threshold: {eps_v:.6e} m/s")
    print(f"Level {level} - Displacement threshold: {eps_x:.6e} m")

    # Create masks for slicing
    # Stiffness slice (R vs displacement): near-zero velocity
    near_zero_vel_mask = np.abs(v_rel) <= eps_v
    # Damping slice (R vs velocity): near-zero displacement  
    near_zero_disp_mask = np.abs(x_rel) <= eps_x

    # Store results
    rfs_data[level] = {
        'x_rel': x_rel,
        'v_rel': v_rel,
        'f_rest': f_rest,
        'vel_mask': near_zero_vel_mask,   # used for stiffness slice
        'disp_mask': near_zero_disp_mask   # used for damping slice
    }

# ----------- VISUALIZE SLICE PLOTS -----------
figures = []  # Store all figures to show them at once

for level in training_level:
    # Get data for this level
    x_rel = rfs_data[level]['x_rel']
    v_rel = rfs_data[level]['v_rel']
    f_rest = rfs_data[level]['f_rest']
    near_zero_vel_mask = rfs_data[level]['vel_mask']
    near_zero_disp_mask = rfs_data[level]['disp_mask']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stiffness plot (Restoring Force vs Displacement)
    ax1.plot(x_rel[near_zero_vel_mask], f_rest[near_zero_vel_mask], 
             'bo', markersize=3, alpha=0.6, markerfacecolor='none', 
             markeredgewidth=0.8, label='Measured')
    ax1.set_xlabel('Relative Displacement [m]')
    ax1.set_ylabel('Restoring Force [N]')
    ax1.set_title(f'(a) Stiffness Characteristic - Level {level}')
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Damping plot (Restoring Force vs Velocity)
    ax2.plot(v_rel[near_zero_disp_mask], f_rest[near_zero_disp_mask], 
             'bo', markersize=3, alpha=0.6, markerfacecolor='none', 
             markeredgewidth=0.8, label='Measured')
    ax2.set_xlabel('Relative Velocity [m/s]')
    ax2.set_ylabel('Restoring Force [N]')
    ax2.set_title(f'(b) Damping Characteristic - Level {level}')
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    figures.append(fig)

# Show all figures at once
plt.show()

print("-"*60)
print("ANALYSIS COMPLETE")
print("-"*60)
