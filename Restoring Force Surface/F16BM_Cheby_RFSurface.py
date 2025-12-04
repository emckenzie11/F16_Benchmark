import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft, rfftfreq
import plotly.graph_objects as go

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Plot configuration
levels_to_compute = [7]  # Levels to compute RFS (Options: 1, 3, 5, 7)
location_i = 2  # DOF i (location where acceleration is measured)
location_j = 3  # DOF j (location across the nonlinear connection)

# Surface fitting parameters
chebyshev_order = 4   # Polynomial order for Chebyshev surface fit (lower = smoother, more stable)
grid_size = 50        # Grid resolution for surface plot
show_scatter = True   # Toggle scatter data points on/off in 3D plot
surface_on_grid = True  # If True, plot surface on grid; if False, plot surface only at data points

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step

# Mode isolation parameters
target_freq = 7.3     # Target mode frequency [Hz]
freq_low = 6.8        # Lower frequency bound [Hz]
freq_high = 8.6       # Upper frequency bound [Hz]

# Effective mass (normalized).
m_eff = 20.67

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

def scale_to_domain(x, x_min=None, x_max=None):
    """
    Scale data to [-1, 1] for Chebyshev polynomials.
    
    Returns:
        x_scaled: scaled data in [-1, 1]
        scale_params: (x_min, x_max) for inverse scaling
    """
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    
    x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1
    return x_scaled, (x_min, x_max)

def build_chebyshev_basis_1d(z, order):
    """
    Build 1D Chebyshev polynomial basis up to given order.
    Uses recurrence relation: T_{n+1}(z) = 2z*T_n(z) - T_{n-1}(z)
    
    Args:
        z: scaled coordinate in [-1, 1], shape (N,)
        order: maximum polynomial order
    
    Returns:
        T: matrix of shape (N, order+1) where T[:, n] = T_n(z)
    """
    N = len(z)
    T = np.zeros((N, order + 1))
    T[:, 0] = 1.0  # T_0(z) = 1
    if order >= 1:
        T[:, 1] = z  # T_1(z) = z
    for n in range(1, order):
        T[:, n+1] = 2 * z * T[:, n] - T[:, n-1]
    return T

def build_2d_chebyshev_basis(x_scaled, v_scaled, order):
    """
    Build 2D tensor-product Chebyshev basis.
    
    Φ_{pq}(x,v) = T_p(x) * T_q(v) for p,q = 0..order
    
    Args:
        x_scaled: displacement in [-1, 1], shape (N,)
        v_scaled: velocity in [-1, 1], shape (N,)
        order: polynomial order
    
    Returns:
        H: design matrix of shape (N, (order+1)^2)
    """
    T_x = build_chebyshev_basis_1d(x_scaled, order)  # (N, order+1)
    T_v = build_chebyshev_basis_1d(v_scaled, order)  # (N, order+1)
    
    N = len(x_scaled)
    n_terms = (order + 1) ** 2
    H = np.zeros((N, n_terms))
    
    idx = 0
    for p in range(order + 1):
        for q in range(order + 1):
            H[:, idx] = T_x[:, p] * T_v[:, q]
            idx += 1
    
    return H

def fit_chebyshev_surface(x, v, f, order, ridge_param=1e-3):
    """
    Fit a global Chebyshev polynomial surface to scattered data.
    
    Args:
        x: displacement data
        v: velocity data
        f: restoring force data
        order: Chebyshev polynomial order
        ridge_param: regularization parameter for stability (increased for high-order fits)
    
    Returns:
        coeffs: fitted coefficients
        scale_x: (x_min, x_max) scaling parameters
        scale_v: (v_min, v_max) scaling parameters
    """
    # Scale to [-1, 1]
    x_scaled, scale_x = scale_to_domain(x)
    v_scaled, scale_v = scale_to_domain(v)
    
    # Build design matrix
    H = build_2d_chebyshev_basis(x_scaled, v_scaled, order)
    
    # Solve with ridge regularization for stability
    # min ||H*a - f||^2 + ridge_param * ||a||^2
    HTH = H.T @ H
    HTf = H.T @ f
    coeffs = np.linalg.solve(HTH + ridge_param * np.eye(HTH.shape[0]), HTf)
    
    return coeffs, scale_x, scale_v

def evaluate_chebyshev_surface(X_grid, V_grid, coeffs, scale_x, scale_v, order):
    """
    Evaluate fitted Chebyshev surface on a grid.
    
    Args:
        X_grid: 2D grid of x values
        V_grid: 2D grid of v values
        coeffs: fitted coefficients
        scale_x: (x_min, x_max) from training
        scale_v: (v_min, v_max) from training
        order: polynomial order
    
    Returns:
        F_grid: evaluated surface values
    """
    # Flatten grids
    x_flat = X_grid.ravel()
    v_flat = V_grid.ravel()
    
    # Scale to [-1, 1]
    x_scaled = 2 * (x_flat - scale_x[0]) / (scale_x[1] - scale_x[0]) - 1
    v_scaled = 2 * (v_flat - scale_v[0]) / (scale_v[1] - scale_v[0]) - 1
    
    # Build basis
    H_grid = build_2d_chebyshev_basis(x_scaled, v_scaled, order)
    
    # Evaluate
    f_flat = H_grid @ coeffs
    F_grid = f_flat.reshape(X_grid.shape)
    
    return F_grid

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
    restoring_force = compute_restoring_force(accel_j_bp, m_eff=m_eff)

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

def plot_3d_restoring_force_surface(level, title_suffix="", grid_size=50, chebyshev_order=5):
    """Plot 3D restoring force surface for a single level using Chebyshev polynomial model."""
    rel_disp = rfs_data[level]['rel_disp']
    rel_vel = rfs_data[level]['rel_vel']
    R = rfs_data[level]['restoring_force']
    
    print(f"Level {level}: Chebyshev order {chebyshev_order}, {len(rel_disp)} points")
    
    # STEP 1-3: Fit Chebyshev polynomial surface for this specific level
    coeffs, scale_x, scale_v = fit_chebyshev_surface(rel_disp, rel_vel, R, chebyshev_order)
    
    # STEP 4: Evaluate on grid (optionally extrapolate beyond data bounds)
    x_min, x_max = rel_disp.min(), rel_disp.max()
    y_min, y_max = rel_vel.min(), rel_vel.max()
    x_min_ext = x_min
    x_max_ext = x_max
    y_min_ext = y_min
    y_max_ext = y_max
    
    # Create grid
    xi = np.linspace(x_min_ext, x_max_ext, grid_size)
    yi = np.linspace(y_min_ext, y_max_ext, grid_size)
    X, Y = np.meshgrid(xi, yi)
    
    # Evaluate Chebyshev surface on grid
    Z = evaluate_chebyshev_surface(X, Y, coeffs, scale_x, scale_v, chebyshev_order)
    
    print(f"  Grid: {grid_size}x{grid_size}, Z: [{Z.min():.2g}, {Z.max():.2g}]")
    
    fig = go.Figure()
    # State 1: Wireframe grid and vertical lines for positive F, with fixed camera orientation
    if surface_on_grid:
        # Wireframe grid
        for i in range(grid_size):
            fig.add_trace(go.Scatter3d(
                x=X[i, :],
                y=Y[i, :],
                z=Z[i, :],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))
        for j in range(grid_size):
            fig.add_trace(go.Scatter3d(
                x=X[:, j],
                y=Y[:, j],
                z=Z[:, j],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))
    elif not surface_on_grid:
        # Add Chebyshev surface only at data points, with contours
        Z_data = evaluate_chebyshev_surface(rel_disp, rel_vel, coeffs, scale_x, scale_v, chebyshev_order)
        print(f"  Data pts: {len(Z_data)}, Z: [{Z_data.min():.2g}, {Z_data.max():.2g}]")
        fig.add_trace(go.Mesh3d(
            x=rel_disp,
            y=rel_vel,
            z=Z_data,
            color='grey',
            opacity=1.0,
            name='Chebyshev Surface (data pts)',
            showscale=False,
            flatshading=False,
            lighting=dict(
                ambient=0.8,
                diffuse=0.9,
                specular=0.5,
                roughness=0.3,
                fresnel=0.2
            ),
            lightposition=dict(x=0, y=0, z=100)
        ))
    # Optionally add scatter data points (smaller, more transparent)
    if show_scatter:
        fig.add_trace(go.Scatter3d(
            x=rel_disp,
            y=rel_vel,
            z=R,
            mode='markers',
            marker=dict(size=1, color='red', opacity=0.5),
            name='Data Points'
        ))
    # Set custom camera angle for better perspective (State 1)
    fig.update_layout(
        scene_camera=dict(eye=dict(x=-2.5, y=-2.5, z=1.5)),  # Reference orientation
    )
    
    # Update layout with labels and title
    fig.update_layout(
        title=f'Chebyshev Restoring Force Surface_Grid (Order {chebyshev_order}) - Level {level}{title_suffix}',
        scene=dict(
            xaxis_title='Relative Displacement [m]',
            yaxis_title='Relative Velocity [m/s]',
            zaxis_title='-Acceleration [m/s²]',
            xaxis=dict(tickformat='.1e'),
            yaxis=dict(tickformat='.1e')
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
    fig = plot_3d_restoring_force_surface(level, title_suffix, grid_size=grid_size, chebyshev_order=chebyshev_order)
    
    # Write HTML file
    html_filename = os.path.join(output_dir, f"RestoringForce_Level{level}_3D_Grid.html")
    fig.write_html(html_filename)
    html_files.append(html_filename)
    print(f"  -> Exported: {html_filename}")
