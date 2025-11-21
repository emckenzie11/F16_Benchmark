import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import griddata
import plotly.graph_objects as go
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

def plot_3d_restoring_force_surface(level, title_suffix="", grid_size=50):
    """Plot 3D restoring force surface for a single level using Plotly surface plot."""
    rel_disp = rfs_data[level]['rel_disp']
    rel_vel = rfs_data[level]['rel_vel']
    R = rfs_data[level]['restoring_force']
    
    print(f"Level {level}: Converting {len(rel_disp)} points to surface plot")
    
    # Create regular grid for surface plot
    x_min, x_max = rel_disp.min(), rel_disp.max()
    y_min, y_max = rel_vel.min(), rel_vel.max()
    
    # Create grid
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(xi, yi)
    
    # Stack the coordinates
    points = np.column_stack((rel_disp, rel_vel))
    
    # Interpolate using cubic method (fallback to linear if cubic fails)
    try:
        Z = griddata(points, R, (X, Y), method='cubic', fill_value=np.nan)
        # Fill any remaining NaN values with linear interpolation
        if np.any(np.isnan(Z)):
            Z_linear = griddata(points, R, (X, Y), method='linear', fill_value=np.nan)
            Z = np.where(np.isnan(Z), Z_linear, Z)
    except:
        # If cubic fails, use linear interpolation
        Z = griddata(points, R, (X, Y), method='linear', fill_value=np.nan)
    
    # Create Plotly 3D surface plot
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
    
    # Update layout with labels and title
    fig.update_layout(
        title=f'Restoring Force Surface - Level {level}{title_suffix}',
        scene=dict(
            xaxis_title='Relative Displacement [m]',
            yaxis_title='Relative Velocity [m/s]',
            zaxis_title='-Acceleration [m/s²]'
        ),
        width=1200,
        height=900
    )
    
    return fig

# Dynamic plotting based on number of levels
n_levels = len(levels_to_compute)
html_files = []  # Store HTML filenames
output_dir = "Figures/RestoringForce3D"
os.makedirs(output_dir, exist_ok=True)

for i, level in enumerate(levels_to_compute):
    title_suffix = f" ({chr(97+i)})" if n_levels > 1 else ""
    fig = plot_3d_restoring_force_surface(level, title_suffix)
    
    # Write HTML file
    html_filename = os.path.join(output_dir, f"RestoringForce_Level{level}_3D.html")
    fig.write_html(html_filename)
    html_files.append(html_filename)
    print(f"  -> Exported: {html_filename}")
