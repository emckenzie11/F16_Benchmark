# F16 BENCHMARK - RUNGE-KUTTA RESTORING FORCE MODEL ANALYSIS
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import helper functions
from F16BM_RK45_helpers import *

# Start timing
start_time = time.time()
print("="*60)
print("F16 BENCHMARK - RUNGE-KUTTA RESTORING FORCE MODEL ANALYSIS")
print(f"Analysis started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("-"*60)

# ----------- User configuration -----------
training_level = [1]      # level(s) used to train nonlinear residuals R(x,v)
validation_level = [2]  # level(s) used for validation
location_i = 2
location_j = 3

fs = 400.0                # sampling frequency [Hz]
dt = 1.0 / fs             # time step

natural_freq = 7.3         # target mode frequency [Hz]
freq_low = 6.8
freq_high = 8.6

poly_order = 3            # Chebyshev order 
integration_method = 'RK45'  # integration method

# ----------- Load data -----------
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

# ----------- Process training data -----------
# First pass: process all levels to get relative motion data
for train_level in training_level:
    print(f"Processing training level {train_level}...")
    
    # Get raw arrays
    accel_raw_i = training_data[train_level][f'Acceleration{location_i}'].to_numpy()
    accel_raw_j = training_data[train_level][f'Acceleration{location_j}'].to_numpy()
    f_raw_train = training_data[train_level]['Force'].to_numpy()

    # Bandpassed signals
    accel_bp_i = bandpass_filter(accel_raw_i, freq_low, freq_high, fs)
    accel_bp_j = bandpass_filter(accel_raw_j, freq_low, freq_high, fs)
    f_bp_train = bandpass_filter(f_raw_train, freq_low, freq_high, fs)

    # Compute relative states (x_rel, v_rel) via band-limited FFT integration
    x_rel_train, v_rel_train = compute_relative_motion(accel_bp_i, accel_bp_j, fs, freq_low, freq_high)

    # Store processed arrays in training_data for this level
    training_data[train_level]['x_rel'] = x_rel_train
    training_data[train_level]['v_rel'] = v_rel_train
    training_data[train_level]['accel_bp_i'] = accel_bp_i
    training_data[train_level]['accel_bp_j'] = accel_bp_j
    training_data[train_level]['f_bp'] = f_bp_train

# ----------- Train linear baseline -----------
# Use level 1 only to establish baseline parameters
baseline_level = 1

x_rel_baseline = training_data[baseline_level]['x_rel']
v_rel_baseline = training_data[baseline_level]['v_rel'] 
accel_bp_i_baseline = training_data[baseline_level]['accel_bp_i']

# Estimate linear baseline from small-amplitude slices
vel_thresh_frac = 0.02
disp_thresh_frac = 0.02
vel_thresh_min = 1e-6
disp_thresh_min = 1e-7
eps_v = max(vel_thresh_frac * np.max(np.abs(v_rel_baseline)), vel_thresh_min)
eps_x = max(disp_thresh_frac * np.max(np.abs(x_rel_baseline)), disp_thresh_min)

# Build small-amplitude mask around origin in (x,v) plane
mask_baseline = (np.abs(v_rel_baseline) <= eps_v) | (np.abs(x_rel_baseline) <= eps_x)
X_baseline = np.column_stack([x_rel_baseline[mask_baseline], v_rel_baseline[mask_baseline]])
y_baseline = -accel_bp_i_baseline[mask_baseline]

# Least-squares for k0_norm, c0_norm (mass-normalised coefficients)
k0_norm, c0_norm = np.linalg.lstsq(X_baseline, y_baseline, rcond=None)[0]

# Convert to physical coefficients
omega_n = 2*np.pi*natural_freq
m_eff = k0_norm / (omega_n**2)
k0 = m_eff * k0_norm
c0 = m_eff * c0_norm
print(f"m_eff={m_eff:.3e} kg, k0={k0:.3e} N/m, c0={c0:.3e} N·s/m")

# Second pass: compute restoring forces using established m_eff
for train_level in training_level:
    accel_bp_i = training_data[train_level]['accel_bp_i']
    f_rest_train = compute_restoring_force_from_accel(accel_bp_i, m_eff)
    training_data[train_level]['f_rest'] = f_rest_train

# ----------- Pooled ridge: concatenate all training levels -----------
# Concatenate all chosen levels, compute global scaling, and fit single Chebyshev model
X, V, Rres = [], [], []

for lvl in training_level:
    # Get processed data for this level
    x_rel_train = training_data[lvl]['x_rel']
    v_rel_train = training_data[lvl]['v_rel']
    f_rest_train = training_data[lvl]['f_rest']
    
    # Compute nonlinear residual for this level
    R_residual = f_rest_train - (k0 * x_rel_train + c0 * v_rel_train)
    
    # Append to pooled lists
    X.append(x_rel_train)
    V.append(v_rel_train) 
    Rres.append(R_residual)

# Concatenate all levels into single arrays
x_pooled = np.concatenate(X)
v_pooled = np.concatenate(V) 
Rres_pooled = np.concatenate(Rres)

# ----------- Compute GLOBAL scaling factors -----------
s_x, b_x, x_scaled_pooled = compute_scaling_factors(x_pooled)
s_v, b_v, v_scaled_pooled = compute_scaling_factors(v_pooled)

# ----------- Build Chebyshev basis and fit with pooled data -----------
H_poly = build_2d_chebyshev_basis(x_scaled_pooled, v_scaled_pooled, poly_order, drop_const_and_linear=True)

# Fit: Rres = H_poly @ alpha (ridge regression)
lam = 1e-8
A = np.vstack([H_poly, np.sqrt(lam)*np.eye(H_poly.shape[1])])
b = np.hstack([Rres_pooled, np.zeros(H_poly.shape[1])])
alpha = np.linalg.lstsq(A, b, rcond=None)[0]

# Print scaling factors and Chebyshev polynomial with coefficients
print(f"Fitted {len(alpha)} Chebyshev coefficients (order = {poly_order})")
print("-"*60)
print("Scaling factors:")
print(f"  s_x = {s_x:12.6e}  (displacement scaling)")
print(f"  b_x = {b_x:12.6e}  (displacement offset)")
print(f"  s_v = {s_v:12.6e}  (velocity scaling)")
print(f"  b_v = {b_v:12.6e}  (velocity offset)")
print("\nUsage: x_scaled = s_x * x + b_x, v_scaled = s_v * v + b_v")
print("-"*60)
print("Chebyshev coefficients (alpha):")
print("Note: Constant (0,0), linear x (1,0), and linear v (0,1) terms are dropped")
idx = 0
for i in range(poly_order):
    for j in range(poly_order):
        # Skip the dropped terms: (0,0), (1,0), (0,1)
        if (i, j) in ((0, 0), (1, 0), (0, 1)):
            continue
        print(f"alpha[{i},{j}] = {alpha[idx]:12.6e}  (T{i}(x) * T{j}(v))")
        idx += 1

# Plot: Pooled training data restoring force slices
# Recompute thresholds using baseline level data for consistency
eps_v = max(vel_thresh_frac * np.max(np.abs(v_rel_baseline)), vel_thresh_min)
eps_x = max(disp_thresh_frac * np.max(np.abs(x_rel_baseline)), disp_thresh_min)

# masks for near-zero velocity and displacement using pooled data
vel_mask_pooled = np.abs(v_pooled) <= eps_v
disp_mask_pooled = np.abs(x_pooled) <= eps_x

# Calculate max values from pooled data for axis limits
x_max_pooled = np.max(np.abs(x_pooled[vel_mask_pooled]))
v_max_pooled = np.max(np.abs(v_pooled[disp_mask_pooled]))

# Compute pooled restoring force for plotting
f_rest_pooled = np.concatenate([training_data[lvl]['f_rest'] for lvl in training_level])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

ax1.scatter(x_pooled[vel_mask_pooled], f_rest_pooled[vel_mask_pooled], s=8, alpha=0.7, 
           c='blue', label=f'Pooled Training Restoring Force (Levels {training_level})')
ax1.set_xlabel('Relative Displacement [m]')
ax1.set_ylabel('Restoring Force [N]')
ax1.set_title('Pooled Training Data: Stiffness slice (near-zero velocity)')
ax1.set_xlim(-x_max_pooled * 1.1, x_max_pooled * 1.1)
ax1.grid(True)
ax1.legend()

ax2.scatter(v_pooled[disp_mask_pooled], f_rest_pooled[disp_mask_pooled], s=8, alpha=0.7, 
           c='blue', label=f'Pooled Training Restoring Force (Levels {training_level})')
ax2.set_xlabel('Relative Velocity [m/s]')
ax2.set_ylabel('Restoring Force [N]')
ax2.set_title('Pooled Training Data: Damping slice (near-zero displacement)')
ax2.set_xlim(-v_max_pooled * 1.1, v_max_pooled * 1.1)
ax2.grid(True)
ax2.legend()

plt.suptitle(f'Pooled Training Levels {training_level}: Restoring Force Slices')
plt.tight_layout()
plt.show()

# ----------- Validation -----------
for val_level in validation_level:
    # Load raw validation signals
    accel_raw_i_val = validation_data[val_level][f'Acceleration{location_i}'].to_numpy()
    accel_raw_j_val = validation_data[val_level][f'Acceleration{location_j}'].to_numpy()
    f_raw_val = validation_data[val_level]['Force'].to_numpy()

    # Bandpassed signals (same filter as training)
    accel_bp_i_val = bandpass_filter(accel_raw_i_val, freq_low, freq_high, fs)
    accel_bp_j_val = bandpass_filter(accel_raw_j_val, freq_low, freq_high, fs)
    f_bp_val = bandpass_filter(f_raw_val, freq_low, freq_high, fs)

    # Compute measured relative x, v for validation (to compare to sim)
    x_rel_val, v_rel_val = compute_relative_motion(accel_bp_i_val, accel_bp_j_val, fs, freq_low, freq_high)

    # Compute measured restoring force (to compare to sim)
    f_rest_val = compute_restoring_force_from_accel(accel_bp_i_val, m_eff)

    # Simulate response using learned Chebyshev model
    x_sim, v_sim, a_sim, f_rest_sim = simulate_response(
        f_bp_val, alpha, s_x, b_x, s_v, b_v, poly_order,
        dt, m_eff, integration_method, x0=0.0, v0=0.0,
        k0=k0, c0=c0)

    # Save into validation_data for plotting convenience
    validation_data[val_level]['x_rel'] = x_rel_val
    validation_data[val_level]['v_rel'] = v_rel_val
    validation_data[val_level]['f_rest'] = f_rest_val
    validation_data[val_level]['x_sim'] = x_sim
    validation_data[val_level]['v_sim'] = v_sim
    validation_data[val_level]['a_sim'] = a_sim
    validation_data[val_level]['f_rest_sim'] = f_rest_sim

# ----------- Prepare thresholds for slicing -----------
# Use baseline level for consistent thresholding (already computed above)
# eps_v and eps_x are already computed from baseline data
print("-"*60)
print(f"Slicing thresholds: eps_v={eps_v:.3e}, eps_x={eps_x:.3e}")

# ----------- End of processing -----------
# Calculate and display analysis time 
end_time = time.time()
analysis_duration = end_time - start_time
print("-"*60)
print(f"Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total analysis time: {analysis_duration:.2f} seconds ({analysis_duration/60:.2f} minutes)")
print("-"*60)
print("Generating plots...")

# ----------- Plots -----------
for val_level in validation_level:
    # Get validation data for plotting
    x_rel_val = validation_data[val_level]['x_rel']
    v_rel_val = validation_data[val_level]['v_rel'] 
    f_rest_val = validation_data[val_level]['f_rest']
    x_sim = validation_data[val_level]['x_sim']
    v_sim = validation_data[val_level]['v_sim']
    f_rest_sim = validation_data[val_level]['f_rest_sim']

    t = np.arange(len(x_rel_val)) / fs
    
    # Define time window for plotting (120-180 seconds)
    t_start, t_end = 120.0, 180.0
    idx_start = int(t_start * fs)
    idx_end = int(t_end * fs)
    
    # Restrict data to time window
    t_plot = t[idx_start:idx_end]
    x_rel_plot = x_rel_val[idx_start:idx_end]
    v_rel_plot = v_rel_val[idx_start:idx_end]
    x_sim_plot = x_sim[idx_start:idx_end]
    v_sim_plot = v_sim[idx_start:idx_end]
    f_bp_plot = f_bp_val[idx_start:idx_end]
    f_rest_plot = f_rest_val[idx_start:idx_end]
    f_rest_sim_plot = f_rest_sim[idx_start:idx_end]
    
    # Calculate scaling based on full dataset measured values 
    x_max_measured = np.max(np.abs(x_rel_val))
    v_max_measured = np.max(np.abs(v_rel_val))
    f_max_measured = np.max(np.abs(f_bp_val))
    f_rest_max_measured = np.max(np.abs(f_rest_val))
    
    fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Convert to mm and mm/s
    x_rel_plot_mm = x_rel_plot * 1000  # m to mm
    x_sim_plot_mm = x_sim_plot * 1000  # m to mm
    v_rel_plot_mms = v_rel_plot * 1000  # m/s to mm/s
    v_sim_plot_mms = v_sim_plot * 1000  # m/s to mm/s
    
    ax[0].plot(t_plot, x_rel_plot_mm, label='Measured', linewidth=1)
    ax[0].plot(t_plot, x_sim_plot_mm, '--', label='Simulated', linewidth=1)
    ax[0].set_ylabel('Relative displacement [mm]')
    ax[0].set_ylim(-x_max_measured * 1000 * 1.1, x_max_measured * 1000 * 1.1)  # Scale to measured max ± 10%
    ax[0].legend(); ax[0].grid(True)

    ax[1].plot(t_plot, v_rel_plot_mms, label='Measured', linewidth=1)
    ax[1].plot(t_plot, v_sim_plot_mms, '--', label='Simulated', linewidth=1)
    ax[1].set_ylabel('Relative velocity [mm/s]')
    ax[1].set_ylim(-v_max_measured * 1000 * 1.1, v_max_measured * 1000 * 1.1)  # Scale to measured max ± 10%
    ax[1].legend(); ax[1].grid(True)
    
    ax[2].plot(t_plot, f_bp_plot, 'g-', label='Measured', linewidth=1)
    ax[2].set_ylabel('Bandpassed Input Force [N]')
    ax[2].set_ylim(-f_max_measured * 1.1, f_max_measured * 1.1)  # Scale to measured max ± 10%
    ax[2].legend(); ax[2].grid(True)
    
    ax[3].plot(t_plot, f_rest_plot, label='Measured', linewidth=1)
    ax[3].plot(t_plot, f_rest_sim_plot, '--', label='Simulated', linewidth=1)
    ax[3].set_ylabel('Restoring Force [N]')
    ax[3].set_xlabel('Time [s]')
    ax[3].set_ylim(-f_rest_max_measured * 1.1, f_rest_max_measured * 1.1)  # Scale to measured max ± 10%
    ax[3].legend(); ax[3].grid(True)
    
    print(f"Plot scaling ({t_start}-{t_end}s) - Displacement: ±{x_max_measured:.2e} m, Velocity: ±{v_max_measured:.2e} m/s, Force: ±{f_max_measured:.2e} N")
    
    plt.suptitle(f'Validation Level {val_level}: Time-domain comparison ({t_start}-{t_end}s window)')
    plt.tight_layout()
    plt.show()

    # Plot: RFS slices (validation: measured vs predicted)
    # masks for near-zero velocity and displacement
    vel_mask = np.abs(v_rel_val) <= eps_v
    disp_mask = np.abs(x_rel_val) <= eps_x

    # Calculate max values from measured data for axis limits
    x_max_measured = np.max(np.abs(x_rel_val[vel_mask]))
    v_max_measured = np.max(np.abs(v_rel_val[disp_mask]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

    ax1.scatter(x_rel_val[vel_mask], f_rest_val[vel_mask], s=8, alpha=0.5, label='Measured Restoring Force')
    ax1.scatter(x_sim[vel_mask], f_rest_sim[vel_mask], s=6, alpha=0.9, marker='x', label='Predicted Restoring Force (Cheb)')
    ax1.set_xlabel('Relative Displacement [m]'); ax1.set_ylabel('Restoring Force [N]'); ax1.set_title('Stiffness slice (near-zero velocity)')
    ax1.set_xlim(-x_max_measured * 1.1, x_max_measured * 1.1)  # Set x-axis limits based on measured data
    ax1.grid(True); ax1.legend()

    ax2.scatter(v_rel_val[disp_mask], f_rest_val[disp_mask], s=8, alpha=0.5, label='Measured Restoring Force')
    ax2.scatter(v_sim[disp_mask], f_rest_sim[disp_mask], s=6, alpha=0.9, marker='x', label='Predicted Restoring Force (Cheb)')
    ax2.set_xlabel('Relative Velocity [m/s]'); ax2.set_ylabel('Restoring Force [N]'); ax2.set_title('Damping slice (near-zero displacement)')
    ax2.set_xlim(-v_max_measured * 1.1, v_max_measured * 1.1)  # Set x-axis limits based on measured data
    ax2.grid(True); ax2.legend()

    plt.suptitle(f'Validation Level {val_level}: RFS slices')
    plt.tight_layout()
    plt.show()

print("-"*60)
print("Plots Generated")
print("="*60)