import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from numpy.fft import rfft, irfft, rfftfreq
from scipy.interpolate import CubicSpline

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
def compute_relative_motion(accel_bp_i, accel_bp_j, fs, freq_low, freq_high):
    accel_rel = accel_bp_i - accel_bp_j
    v_rel = band_limited_integrate_fft(accel_rel, fs, freq_low, freq_high, order=1)
    x_rel = band_limited_integrate_fft(accel_rel, fs, freq_low, freq_high, order=2)
    return x_rel, v_rel

# Compute restoring force
def compute_restoring_force_from_accel(accel_bp, m_eff=1.0):
    # R(t) = - m_eff * a(t) 
    return -m_eff * accel_bp

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
    # T0 = 1, T1 = x, T2 = 2x^2 - 1, Tn = 2x*T_{n-1} - T_{n-2}
    basis[:,0] = 1.0  # T₀(x) = 1
    if order > 1:
        basis[:,1] = x  # T₁(x) = x
    if order > 2:
        basis[:,2] = 2*x**2 - 1  # T₂(x) = 2x² - 1
    for n in range(3, order):
        basis[:, n] = 2*x * basis[:, n-1] - basis[:, n-2]
    return basis

def build_2d_chebyshev_basis(x, v, order, drop_const_and_linear=True):
    bz = chebyshev_basis(x, order)   # shape (N, order), with bz[:,0]=T0=1, bz[:,1]=T1=x, ...
    bv = chebyshev_basis(v, order)   # shape (N, order)

    N = len(x)
    cols = []
    ij  = []

    # iterate i,j in [0, order-1]
    for i in range(order):
        for j in range(order):
            if drop_const_and_linear and (i, j) in ((0, 0), (1, 0), (0, 1)):
                continue
            cols.append(bz[:, i] * bv[:, j])
            ij.append((i, j))

    if len(cols) == 0:
        H_poly = np.zeros((N, 0))
    else:
        H_poly = np.column_stack(cols)  # shape (N, n_kept_terms)

    return H_poly

# Unified restoring force prediction function
def predict_RF(x, v, alpha, s_x, b_x, s_v, b_v, poly_order, k0=0.0, c0=0.0, include_linear=True):
    """
    Predict restoring force using Chebyshev polynomials with optional linear baseline.
    
    Parameters:
    -----------
    x, v : array-like
        Displacement and velocity values
    alpha : array
        Chebyshev coefficients
    s_x, b_x, s_v, b_v : float
        Scaling factors for x and v: x_scaled = s_x * x + b_x
    poly_order : int
        Order of Chebyshev polynomial
    k0, c0 : float, optional
        Linear baseline coefficients (default: 0.0)
    include_linear : bool, optional
        If True: returns k0*x + c0*v + R_nl(x,v) [total force]
        If False: returns only R_nl(x,v) [nonlinear residual]
        
    Returns:
    --------
    R : array or scalar
        Predicted restoring force or residual
    """
    # Convert to arrays (ensures length)
    x = np.atleast_1d(x)
    v = np.atleast_1d(v)

    # Scale to [-1,1] using linear transformation
    x_scaled = s_x * x + b_x
    v_scaled = s_v * v + b_v
    
    # Clamp to [-1,1] to prevent extrapolation issues
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    v_scaled = np.clip(v_scaled, -1.0, 1.0)

    # Build Chebyshev basis (drop const and linear terms to match training)
    H = build_2d_chebyshev_basis(x_scaled, v_scaled, poly_order, drop_const_and_linear=True)

    # Evaluate nonlinear residual
    R_nl = H.dot(alpha)
    
    # Add linear baseline if requested
    if include_linear:
        R_total = k0 * x + c0 * v + R_nl
        result = R_total
    else:
        result = R_nl

    # If input was scalar → return scalar
    if result.size == 1:
        return result[0]
    return result

def make_gate_from_envelope(F_bp, fs, frac_on=0.05, ramp_sec=20, ramp_out_sec=20):
    """
    Build a smooth gate with both ramp-in and ramp-out for the band-passed force.
    - frac_on: envelope fraction at which we ramp in/out (0..1)
    - ramp_sec: raised-cosine ramp-in length (seconds)
    - ramp_out_sec: raised-cosine ramp-out length (seconds)
    """
    t = np.arange(len(F_bp))/fs
    env = np.abs(hilbert(F_bp))
    peak = np.max(env)
    thr_on = frac_on * peak
    
    # Find ramp-in and ramp-out points
    idx_on = np.argmax(env >= thr_on)
    idx_off = len(env) - np.argmax(env[::-1] >= thr_on)
    
    t0, t1 = t[idx_on], t[idx_on] + ramp_sec
    t2, t3 = t[idx_off] - ramp_out_sec, t[idx_off]

    gate = np.zeros_like(t)
    
    # Ramp in
    in_ramp = (t >= t0) & (t <= t1)
    gate[in_ramp] = 0.5 - 0.5*np.cos(np.pi*(t[in_ramp]-t0)/(t1-t0))
    
    # Full on
    gate[t > t1] = 1.0
    
    # Ramp out
    out_ramp = (t >= t2) & (t <= t3)
    gate[out_ramp] = 0.5 + 0.5*np.cos(np.pi*(t[out_ramp]-t2)/(t3-t2))
    gate[t > t3] = 0.0
    
    return gate, (t0, t1, t2, t3)

# Simulate system response using configurable integration method
def simulate_response(force_input, alpha, s_x, b_x, s_v, b_v, poly_order,
                      dt, m_eff, integration_method='RK45', x0=0.0, v0=0.0,
                      k0=0.0, c0=0.0):
    """
    Simulate system response using configurable ODE integration method.
    Includes external forcing and restoring force.
    
    Parameters:
    -----------
    integration_method : str
        ODE solver method ('RK45', 'Radau', 'DOP853', 'BDF', etc.)
    """
    # Use full time series with gated force
    t_eval = np.arange(len(force_input)) * dt
    t_span = (0.0, t_eval[-1])
    
    print(f"Starting simulation: {len(force_input)} time points, {t_span[1]:.1f}s duration")

    # Smooth forcing for substeps
    force_spline = CubicSpline(t_eval, force_input, bc_type='natural')
    
    def ode_system(t, y):
        """
        ODE system: dy/dt = [dx/dt, dv/dt]
        where y = [x, v]
        Includes external forcing and restoring force.
        """
        x_val, v_val = y
        F_val = force_spline(t)  # External forcing
        R_val = predict_RF(x_val, v_val, alpha, s_x, b_x, s_v, b_v, poly_order, k0, c0, include_linear=True)
  
        dxdt = v_val
        dvdt = (F_val - R_val) / m_eff  # External force minus restoring force

        if abs(x_val) > 1e-2 or not np.isfinite(x_val):
            print(f"t={t:.3f}, x={x_val:.3e}, v={v_val:.3e}, F={F_val:.3e}, R={R_val:.3e}")
            raise SystemExit
    
        return [dxdt, dvdt]
    
    # Initial conditions [x0, v0]
    y0 = [x0, v0]
    
    # Set solver-specific parameters
    solver_params = {
        'rtol': 1e-6,
        'atol': 1e-9,
        'max_step': dt 
    }
    
    # Adjust parameters for specific solvers if needed
    if integration_method in ['Radau', 'BDF']:
        solver_params['rtol'] = 1e-7  # Tighter tolerance for implicit methods
        solver_params['atol'] = [1e-10, 1e-9]  # Different tolerances for x, v
    
    print(f"Starting ODE integration using {integration_method} method...")
    sol = solve_ivp(
        ode_system, t_span, y0, t_eval=t_eval,
        method=integration_method,
        **solver_params
    )
    print("ODE integration completed!")


    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    
    # Extract solution
    x = sol.y[0]  # displacement
    v = sol.y[1]  # velocity
    
    # Compute acceleration and restoring force at each time point
    a = np.zeros(len(x))
    R = np.zeros(len(x))
    for i in range(len(x)):
        F_val = force_input[i]
        R_val = predict_RF(x[i], v[i], alpha, s_x, b_x, s_v, b_v, poly_order, k0, c0, include_linear=True)
        a[i] = (F_val - R_val) / m_eff
        R[i] = R_val
    
    print("Simulation completed successfully!")
    return x, v, a, R
