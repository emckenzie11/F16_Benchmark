# F16 BENCHMARK - RUNGE-KUTTA RESTORING FORCE MODEL ANALYSIS
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from numpy.fft import rfft, irfft, rfftfreq

# Start timing
start_time = time.time()
print("="*60)
print("F16 BENCHMARK - RUNGE-KUTTA RESTORING FORCE MODEL ANALYSIS")
print(f"Analysis started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("-"*60)

# ----------- User configuration -----------
training_level = [1, 3]  # train on multiple levels
validation_level = [4]          # level(s) used for validation
location_i = 2
location_j = 3

fs = 400.0                # sampling frequency [Hz]
dt = 1.0 / fs             # time step

target_freq = 7.3         # target mode frequency [Hz]
freq_low = 6.8
freq_high = 8.6

m_eff = 1.0               # effective mass (normalized)
poly_order = 3        # Chebyshev order 
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

# ----------- Helper functions -----------
# Bandpass Butterworth filter for modal isolation
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, signal)

# Band-limited integration via FFT
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

# Compute relative displacement and velocity from acceleration measurements
def compute_relative_motion(accel_i, accel_j, fs, f_lo, f_hi):
    a_rel = accel_i - accel_j
    v_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=1)
    x_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=2)
    return x_rel, v_rel

# Compute restoring force
def compute_restoring_force_from_accel(a_i_bp, m_eff=1.0):
    # R(t) = - m * a(t)  (restoring force proxy)
    return -m_eff * a_i_bp

# ----------- Restoring force prediction -----------
# Compute scaling factors for Chebyshev basis
def compute_scaling_factors(x):
    """
    Compute scaling factors to map x to [-1,1] using x_scaled = s_x * x + b_x
    Returns s_x, b_x such that:
    - min(x) -> -1
    - max(x) -> +1
    """
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        # avoid div by zero if constant signal
        return 0.0, 0.0, np.zeros_like(x)
    
    s_x = 2.0 / (x_max - x_min)
    b_x = -(x_max + x_min) / (x_max - x_min)
    
    x_scaled = s_x * x + b_x
    return s_x, b_x, x_scaled

# Build 1D Chebyshev basis up to given order
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

# Combine 1D Chebyshev bases into 2D basis
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

# Predict restoring force using Chebyshev polynomials
def predict_restoring_force(x, v, alpha, s_x, b_x, s_v, b_v, poly_order):
    """
    Predict restoring force using scaling factors
    x_scaled = s_x * x + b_x
    v_scaled = s_v * v + b_v
    """
    # Convert to arrays (ensures length)
    x = np.atleast_1d(x)
    v = np.atleast_1d(v)

    # Scale to [-1,1] using linear transformation
    x_scaled = s_x * x + b_x
    v_scaled = s_v * v + b_v
    
    # Clamp to [-1,1] to prevent extreme extrapolation (optional)
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    v_scaled = np.clip(v_scaled, -1.0, 1.0)

    # Build basis
    H = build_2d_chebyshev_basis(x_scaled, v_scaled, poly_order)

    # Evaluate
    R = H.dot(alpha)

    # If input was scalar → return scalar
    if R.size == 1:
        return R[0]
    return R

# Simulate system response using configurable integration method
def simulate_response(force_input, alpha, s_x, b_x, s_v, b_v, poly_order,
                      dt, m_eff, x0=0.0, v0=0.0):
    """
    Simulate system response using RK (adaptive Runge-Kutta) integration.
    """
    N = len(force_input)
    
    # Apply bandpass filter to force
    F_bp = bandpass_filter(force_input, freq_low, freq_high, fs)
    
    # Create time vector
    t_eval = np.arange(N) * dt
    t_span = (0, t_eval[-1])
    
    # Create interpolation function for force
    force_interp = interp1d(t_eval, F_bp, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    
    def ode_system(t, y):
        """
        ODE system: dy/dt = [dx/dt, dv/dt]
        where y = [x, v]
        """
        x_val, v_val = y
        F_val = force_interp(t)
        R_val = predict_restoring_force(x_val, v_val, alpha, s_x, b_x, s_v, b_v, poly_order)
        
        dxdt = v_val
        dvdt = (F_val - R_val) / m_eff
        
        return [dxdt, dvdt]
    
    # Initial conditions [x0, v0]
    y0 = [x0, v0]
    
    # Solve ODE using selected integration method (adaptive Runge-Kutta)
    sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, 
                    method=integration_method, rtol=1e-8, atol=1e-10)
    
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    
    # Extract solution
    x = sol.y[0]  # displacement
    v = sol.y[1]  # velocity
    
    # Compute acceleration at each time point
    a = np.zeros(N)
    for i in range(N):
        F_val = F_bp[i]
        R_val = predict_restoring_force(x[i], v[i], alpha, s_x, b_x, s_v, b_v, poly_order)
        a[i] = (F_val - R_val) / m_eff
    
    return x, v, a

# ----------- Process training data (first pass: compute & store processed signals) -----------
# Compute x_rel, v_rel, f_rest for each training run and store those arrays.
for lvl in training_level:
    accel_i = training_data[lvl][f'Acceleration{location_i}'].to_numpy()
    accel_j = training_data[lvl][f'Acceleration{location_j}'].to_numpy()

    # Bandpass accelerations (modal isolation)
    a_i_bp = bandpass_filter(accel_i, freq_low, freq_high, fs)
    a_j_bp = bandpass_filter(accel_j, freq_low, freq_high, fs)

    # Compute relative states (x_rel, v_rel) via band-limited FFT integration
    x_rel, v_rel = compute_relative_motion(a_i_bp, a_j_bp, fs, freq_low, freq_high)

    # Compute restoring force target for training
    f_rest = compute_restoring_force_from_accel(a_i_bp, m_eff)

    # Save processed arrays back for reference (and for RLS training)
    training_data[lvl]['x_rel'] = x_rel
    training_data[lvl]['v_rel'] = v_rel
    training_data[lvl]['f_rest'] = f_rest

# ----------- Compute global scaling factors using all concatenated training runs -----------
all_x = [training_data[lvl]['x_rel'].to_numpy() for lvl in training_level]
all_v = [training_data[lvl]['v_rel'].to_numpy() for lvl in training_level]
x_rel_train_concat = np.concatenate(all_x)
v_rel_train_concat = np.concatenate(all_v)

s_x, b_x, _ = compute_scaling_factors(x_rel_train_concat)
s_v, b_v, _ = compute_scaling_factors(v_rel_train_concat)

print("-"*60)
print("Global scaling computed from all training runs:")
print(f"  s_x = {s_x:.6e}, b_x = {b_x:.6e}")
print(f"  s_v = {s_v:.6e}, b_v = {b_v:.6e}")

# ----------- RLS training across runs (sequential updates) -----------
# Dimensions
p = poly_order**2

# RLS hyperparameters 
lambda_f = 1.0          # forgetting factor (1.0 => no forgetting)
P_init = 1e6            # initial covariance scale
damping = 1e-8          # small regularization added to P at each step

# Initialize
alpha = np.zeros((p,))          # initial coefficients
P = np.eye(p) * P_init          # covariance matrix

# Process each training run sequentially
for lvl in training_level:
    x_rel = training_data[lvl]['x_rel'].to_numpy()
    v_rel = training_data[lvl]['v_rel'].to_numpy()
    f_rest = training_data[lvl]['f_rest'].to_numpy()

    # scale using global training scaling
    x_scaled = s_x * x_rel + b_x
    v_scaled = s_v * v_rel + b_v

    # build H for this run
    H_k = build_2d_chebyshev_basis(x_scaled, v_scaled, poly_order)  # shape (N_k, p)

    # Update alpha with RLS, sample-by-sample (you can batch if you prefer)
    for n in range(H_k.shape[0]):
        h = H_k[n, :].reshape((p,1))    # column vector
        y = np.array([f_rest[n]])       # scalar in 1-element array for consistent shapes

        # compute gain K = P h / (1 + h^T P h)
        Ph = P @ h                      # (p,1)
        denom = float(1.0 + (h.T @ Ph)) # scalar
        K = Ph / denom                  # (p,1)

        # innovation
        y_pred = float((h.T @ alpha.reshape((p,1))))  # scalar prediction
        resid = float(y - y_pred)                     # scalar residual

        # update alpha
        alpha = alpha + (K.flatten() * resid)         # (p,)

        # update covariance P
        P = (np.eye(p) - K @ h.T) @ P
        # apply forgetting factor and small damping for numerical stability
        P = P / lambda_f
        P += damping * np.eye(p)

    print(f"RLS: finished updating with training run {lvl}; current alpha norm = {np.linalg.norm(alpha):.6e}")

# Final alpha after RLS
print("-"*60)
print(f"RLS training complete. Final alpha length = {len(alpha)}")
idx = 0
print("Chebyshev coefficients (alpha):")
for i in range(poly_order):
    for j in range(poly_order):
        print(f"alpha[{i},{j}] = {alpha[idx]:12.6e}  (T{i}(x) * T{j}(v))")
        idx += 1

# ----------- Validation -----------
for val_level in validation_level:
    # Load raw validation signals
    accel_i_val = validation_data[val_level][f'Acceleration{location_i}'].to_numpy()
    accel_j_val = validation_data[val_level][f'Acceleration{location_j}'].to_numpy()
    force_input_val = validation_data[val_level]['Force'].to_numpy()

    # Bandpass (same filter as training)
    a_i_bp_val = bandpass_filter(accel_i_val, freq_low, freq_high, fs)
    a_j_bp_val = bandpass_filter(accel_j_val, freq_low, freq_high, fs)

    # Compute measured relative x, v for validation (to compare to sim)
    x_rel_val, v_rel_val = compute_relative_motion(a_i_bp_val, a_j_bp_val, fs, freq_low, freq_high)

    # Plot relative states vs time for validation dataset
    t_val = np.arange(len(x_rel_val)) / fs
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax[0].plot(t_val, x_rel_val, 'b-', linewidth=1, label='Relative displacement')
    ax[0].set_ylabel('Relative displacement [m]')
    ax[0].set_title(f'Validation Level {val_level}: Measured Relative States vs Time')
    ax[0].grid(True)
    ax[0].legend()
    
    ax[1].plot(t_val, v_rel_val, 'r-', linewidth=1, label='Relative velocity')
    ax[1].set_ylabel('Relative velocity [m/s]')
    ax[1].set_xlabel('Time [s]')
    ax[1].grid(True)
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

    # Compute measured restoring force (to compare to sim)
    f_rest_val = compute_restoring_force_from_accel(a_i_bp_val, m_eff)

    # Simulate response using learned Chebyshev model
    x_sim, v_sim, a_sim = simulate_response(force_input_val, alpha,
                                           s_x, b_x, s_v, b_v, poly_order,
                                           dt, m_eff, x0=0.0, v0=0.0)

    # Save into validation_data for plotting convenience
    validation_data[val_level]['x_rel'] = x_rel_val
    validation_data[val_level]['v_rel'] = v_rel_val
    validation_data[val_level]['f_rest'] = f_rest_val
    validation_data[val_level]['x_sim'] = x_sim
    validation_data[val_level]['v_sim'] = v_sim
    validation_data[val_level]['a_sim'] = a_sim

# ----------- Prepare thresholds for slicing -----------
vel_thresh_frac = 0.02
disp_thresh_frac = 0.02
vel_thresh_min = 1e-6
disp_thresh_min = 1e-7

# use training concatenated arrays for thresholds
eps_v = max(vel_thresh_frac * np.max(np.abs(v_rel_train_concat)), vel_thresh_min)
eps_x = max(disp_thresh_frac * np.max(np.abs(x_rel_train_concat)), disp_thresh_min)
print("-"*60)
print(f"Slicing thresholds: eps_v={eps_v:.3e}, eps_x={eps_x:.3e}")

# ----------- End of processing -----------
# Calculate and display analysis time (before plotting)
end_time = time.time()
analysis_duration = end_time - start_time
print("-"*60)
print("Runge-Kutta Complete")
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

    t = np.arange(len(x_rel_val)) / fs
    
    # Calculate scaling based on measured values
    x_max_measured = np.max(np.abs(x_rel_val))
    v_max_measured = np.max(np.abs(v_rel_val))
    
    # Plot full data (no time limit) with scaling based on measured values
    fig, ax = plt.subplots(2,1, figsize=(12,8), sharex=True)
    ax[0].plot(t, x_rel_val, label='Measured x_rel (val)', linewidth=1)
    ax[0].plot(t, x_sim, '--', label=f'Simulated x ({integration_method} model)', linewidth=1)
    ax[0].set_ylabel('Relative displacement [m]')
    ax[0].set_ylim(-x_max_measured * 1.1, x_max_measured * 1.1)  # Scale to measured max ± 10%
    ax[0].legend(); ax[0].grid(True)

    ax[1].plot(t, v_rel_val, label='Measured v_rel (val)', linewidth=1)
    ax[1].plot(t, v_sim, '--', label=f'Simulated v ({integration_method} model)', linewidth=1)
    ax[1].set_ylabel('Relative velocity [m/s]')
    ax[1].set_ylim(-v_max_measured * 1.1, v_max_measured * 1.1)  # Scale to measured max ± 10%
    ax[1].set_xlabel('Time [s]')
    ax[1].legend(); ax[1].grid(True)
    
    print(f"Plot scaling - Displacement: ±{x_max_measured:.2e} m, Velocity: ±{v_max_measured:.2e} m/s")
    
    plt.suptitle(f'Validation Level {val_level}: Time-domain comparison (full data)')
    plt.tight_layout()
    plt.show()

    # Plot: RFS slices (validation: measured vs predicted)
    # build predicted restoring force at validation measured x,v for slice plots
    R_pred_val = predict_restoring_force(x_rel_val, v_rel_val, alpha, s_x, b_x, s_v, b_v, poly_order)

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

    # Compute RMSE for the slices
    from math import sqrt
    rmse_stiff = None
    rmse_damp = None
    if np.sum(stiff_mask) > 5:
        e = R_pred_val[stiff_mask] - f_rest_val[stiff_mask]
        rmse_stiff = np.sqrt(np.mean(e**2))
        print(f"Level {val_level} stiffness slice RMSE: {rmse_stiff:.3g}")
    if np.sum(disp_mask) > 5:
        e = R_pred_val[disp_mask] - f_rest_val[disp_mask]
        rmse_damp = np.sqrt(np.mean(e**2))
        print(f"Level {val_level} damping slice RMSE:   {rmse_damp:.3g}")

    if (rmse_stiff is not None) and (rmse_damp is not None):
        overall_rmse = np.sqrt(0.5 * (rmse_stiff**2 + rmse_damp**2))
        print(f"Level {val_level} overall RMSE:         {overall_rmse:.3g}")

print("-"*60)
print("Plots Generated")
print("="*60)