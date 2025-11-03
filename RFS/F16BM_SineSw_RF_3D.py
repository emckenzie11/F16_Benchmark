import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft, rfftfreq

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Plot configuration
levels_to_compute = [1]  # Levels to compute RFS (Options: 1, 3, 5, 7)
location_i = 2  # DOF i (location where acceleration is measured)
location_j = 3  # DOF j (location across the nonlinear connection)

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step

# Mode isolation parameters
target_freq = 7.3     # Target mode frequency [Hz]
freq_low = 6.8        # Lower frequency bound [Hz]
freq_high = 8.6       # Upper frequency bound [Hz]

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

print("="*60)
print("MODE ISOLATION AND RFS COMPUTATION")
print("="*60)
print(f"Target mode frequency range: {freq_low:.1f} - {freq_high:.1f} Hz")
print(f"Center frequency: {target_freq:.2f} Hz")
print("="*60)
print()

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

    # Store results (simplified for 3D plotting)
    rfs_data[level] = {
        'rel_disp': rel_disp,
        'rel_vel': rel_vel,
        'restoring_force': restoring_force
    }

    print(f"Level {level} processed:")
    print(f"  Relative displacement range: [{rel_disp.min():.6e}, {rel_disp.max():.6e}] m")
    print(f"  Relative velocity range: [{rel_vel.min():.6e}, {rel_vel.max():.6e}] m/s")
    print(f"  Restoring force proxy range: [{restoring_force.min():.6e}, {restoring_force.max():.6e}] {'N' if m_eff!=1.0 else 'm/s^2'}")
    print(f"  Total data points: {len(rel_disp)}")
    print()

# ---------------------- 3D Plotting ----------------------

def plot_3d_restoring_force_surface(level, title_suffix=""):
    """Plot 3D restoring force surface for a single level."""
    rel_disp = rfs_data[level]['rel_disp']
    rel_vel = rfs_data[level]['rel_vel']
    R = rfs_data[level]['restoring_force']
    
    print(f"Level {level}: Plotting {len(rel_disp)} points")
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with color mapping based on restoring force
    scatter = ax.scatter(rel_disp, rel_vel, R, c=R, cmap='viridis', 
                        s=20, alpha=0.6)
    
    # Labels and title
    ax.set_xlabel('Relative Displacement [m]')
    ax.set_ylabel('Relative Velocity [m/s]')
    ax.set_zlabel('-Acceleration [m/s²]')
    ax.set_title(f'Restoring Force Surface - Level {level}{title_suffix}')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('-Acceleration [m/s²]')
    
    # Format axes with scientific notation
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    return fig

# Dynamic plotting based on number of levels
n_levels = len(levels_to_compute)
figures = []  # Store all figures to show them simultaneously

print(f"\nGenerating 3D Restoring Force Surface plots for {n_levels} level(s)...")

if n_levels == 1:
    # Single level: one 3D plot
    fig = plot_3d_restoring_force_surface(levels_to_compute[0])
    figures.append(fig)
    
elif n_levels == 2:
    # Two levels: create individual 3D plots for each level
    for i, level in enumerate(levels_to_compute):
        fig = plot_3d_restoring_force_surface(level, f" ({chr(97+i)})")
        figures.append(fig)
        
else:
    # Multiple levels: create individual 3D plots for each level  
    for i, level in enumerate(levels_to_compute):
        fig = plot_3d_restoring_force_surface(level, f" ({chr(97+i)})")
        figures.append(fig)

# Show all figures simultaneously
plt.show()

print("\n" + "="*60)
print("3D Restoring Force Surface Computation Complete")
print("="*60)
print(f"Analyzed {len(levels_to_compute)} excitation levels")
print(f"Location i (acceleration measured): {location_i}")
print(f"Location j (across connection): {location_j}")
print(f"Mode isolated: {freq_low}-{freq_high} Hz")
print("All data points plotted in 3D surface")
print("="*60)