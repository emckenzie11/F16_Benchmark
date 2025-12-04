import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft, rfftfreq
import plotly.graph_objects as go
from scipy.signal import savgol_filter

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------
levels_to_compute = [1]        # Levels to compute RFS (choose 1,3,5,7)
location_i = 2                 # DOF i (accel measurement)
location_j = 3                 # DOF j (across connection)
show_scatter = True

fs = 400
dt = 1/fs

target_freq = 7.3
freq_low = 6.8
freq_high = 8.6

m_eff = 20.67

grid = 20               # CO surface grid size

data_dict = {
    1: pd.read_csv("BenchmarkData/F16Data_SineSw_Level1.csv"),
    3: pd.read_csv("BenchmarkData/F16Data_SineSw_Level3.csv"),
    5: pd.read_csv("BenchmarkData/F16Data_SineSw_Level5.csv"),
    7: pd.read_csv("BenchmarkData/F16Data_SineSw_Level7.csv"),
}

# ------------------------------------------------------------
#  BANDPASS + INTEGRATION
# ------------------------------------------------------------
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    ny = 0.5*fs
    b,a = butter(order, [lowcut/ny, highcut/ny], btype='bandpass')
    return filtfilt(b,a,signal)

def band_limited_integrate_fft(a_t, fs, f_lo, f_hi, order=1):
    n = len(a_t)
    A = rfft(a_t - np.mean(a_t))
    f = rfftfreq(n, 1.0/fs)
    mask = (f>=f_lo)&(f<=f_hi)
    Y = np.zeros_like(A)
    if order == 0:
        Y[mask] = A[mask]
    else:
        omega = 2*np.pi*f
        denom = (1j*omega)**order
        denom[0] = np.inf
        Y[mask] = A[mask] / denom[mask]
    return irfft(Y, n=n)

def compute_relative_motion(accel_i_bp, accel_j_bp, fs, f_lo, f_hi):
    a_rel = accel_i_bp - accel_j_bp
    v_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=1)
    x_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=2)
    return x_rel, v_rel

def compute_restoring_force(accel_i_bp, m_eff=1.0):
    return -m_eff * accel_i_bp

# ------------------------------------------------------------
#   CRAWLEY/O'DONNELL FORCE–STATE SURFACE (FORTRAN-faithful)
# ------------------------------------------------------------
def crawley_odonnell_surface(states, force, ngrid=70, use_two_neighbour=True):
    """
    Reimplementation of GENERATE.F (binning + refinement)
    and SHOW_SURFACE.F (grid surface).
    """
    eps = 1e-4
    states = np.asarray(states)
    force = np.asarray(force)
    npts = len(force)

    d = states[:,0]
    v = states[:,1]

    dmin, dmax = d.min(), d.max()
    vmin, vmax = v.min(), v.max()

    dscale = 1.0 / (dmax - dmin)
    vscale = 1.0 / (vmax - vmin)

    surface = np.zeros((ngrid, ngrid))
    pop     = np.zeros((ngrid, ngrid), dtype=int)

    # --- Initial binning + incremental averaging ---
    for k in range(npts):
        dpt = dscale*(d[k] - dmin)
        vpt = vscale*(v[k] - vmin)

        ipt = int(ngrid*dpt - eps)
        jpt = int(ngrid*vpt - eps)

        ipt = min(max(ipt,0), ngrid-1)
        jpt = min(max(jpt,0), ngrid-1)

        popij = pop[ipt,jpt] + 1
        pop[ipt,jpt] = popij

        f1 = 1.0/popij
        f2 = f1*(popij-1)
        surface[ipt,jpt] = f1*force[k] + f2*surface[ipt,jpt]

    # Occupied → 1
    pop[pop>0] = 1

    # ---------------- Refinement ----------------
    def refine(neigh_required):
        nonlocal surface,pop
        while True:
            nchange = 0
            for i in range(1,ngrid-1):
                for j in range(1,ngrid-1):
                    if pop[i,j] == 0:
                        icount = pop[i+1,j] + pop[i-1,j] + pop[i,j+1] + pop[i,j-1]
                        if icount == neigh_required:
                            val = 0.0
                            if pop[i+1,j]: val += surface[i+1,j]
                            if pop[i-1,j]: val += surface[i-1,j]
                            if pop[i,j+1]: val += surface[i,j+1]
                            if pop[i,j-1]: val += surface[i,j-1]
                            surface[i,j] = val/neigh_required
                            pop[i,j] = 1
                            nchange += 1
            if nchange == 0:
                break

    refine(4)
    refine(3)
    if use_two_neighbour:
        refine(2)

    # --- Delete isolated bins (last part of FORTRAN) ---
    for i in range(ngrid):
        for j in range(ngrid):
            if pop[i,j] == 1:
                icount = 0
                if i+1<ngrid: icount += pop[i+1,j]
                if i-1>=0:    icount += pop[i-1,j]
                if j+1<ngrid: icount += pop[i,j+1]
                if j-1>=0:    icount += pop[i,j-1]
                if icount == 0:
                    pop[i,j] = 0
                    surface[i,j] = 0.0

    # Grid centres (like SHOW_SURFACE)
    x_edges = np.linspace(dmin, dmax, ngrid+1)
    v_edges = np.linspace(vmin, vmax, ngrid+1)
    xc = 0.5*(x_edges[:-1] + x_edges[1:])
    vc = 0.5*(v_edges[:-1] + v_edges[1:])
    Xc,Vc = np.meshgrid(xc,vc,indexing='ij')

    return Xc,Vc,surface,pop


# ------------------------------------------------------------
#     PROCESS LEVELS
# ------------------------------------------------------------
rfs_data = {}

for level in levels_to_compute:
    d = data_dict[level]

    accel_i = d[f"Acceleration{location_i}"].to_numpy()
    accel_j = d[f"Acceleration{location_j}"].to_numpy()

    accel_i_bp = bandpass_filter(accel_i, freq_low, freq_high, fs)
    accel_j_bp = bandpass_filter(accel_j, freq_low, freq_high, fs)

    x_rel, v_rel = compute_relative_motion(accel_i_bp, accel_j_bp, fs, freq_low, freq_high)
    R = compute_restoring_force(accel_j_bp, m_eff)
    
    # Smooth displacement, velocity, and force
    # Window length must be odd; tune between 51–201 depending on fs
    win = 101  
    poly = 3

    rel_disp_s = savgol_filter(x_rel, win, poly)
    rel_vel_s  = savgol_filter(v_rel,  win, poly)
    restoring_force_s = savgol_filter(R, win, poly)

    rfs_data[level] = {
        'rel_disp': rel_disp_s,
        'rel_vel': rel_vel_s,
        'restoring_force': restoring_force_s
    }

# ------------------------------------------------------------
#  3D PLOTTING
# ------------------------------------------------------------
def plot_CO_surface(level, grid_size=grid):
    rel_disp = rfs_data[level]['rel_disp']
    rel_vel  = rfs_data[level]['rel_vel']
    R        = rfs_data[level]['restoring_force']

    states = np.column_stack([rel_disp, rel_vel])

    Xc, Vc, Fgrid, pop = crawley_odonnell_surface(
        states, R, ngrid=grid_size, use_two_neighbour=False
    )

    fig = go.Figure()

    # Plot wireframe (CO surfaces were always a wire grid)
    for i in range(grid):
        zrow = Fgrid[i,:]
        mask = pop[i,:] == 1
        fig.add_trace(go.Scatter3d(
            x=Xc[i,mask], y=Vc[i,mask], z=zrow[mask],
            mode="lines", line=dict(color="black",width=1),
            showlegend=False
        ))

    for j in range(grid):
        zcol = Fgrid[:,j]
        mask = pop[:,j] == 1
        fig.add_trace(go.Scatter3d(
            x=Xc[mask,j], y=Vc[mask,j], z=zcol[mask],
            mode="lines", line=dict(color="black",width=1),
            showlegend=False
        ))

    # Optional scatter of raw points
    if show_scatter:
        fig.add_trace(go.Scatter3d(
            x=rel_disp, y=rel_vel, z=R,
            mode="markers",
            marker=dict(size=2, color="red", opacity=0.2)
        ))

    fig.update_layout(
        title=f"Crawley/O'Donnell Force–State Surface — Level {level}",
        scene_camera=dict(eye=dict(x=-2.5, y=-2.5, z=1.5)),  # match SineSw reference orientation
        scene=dict(
            xaxis_title="Relative Displacement [m]",
            yaxis_title="Relative Velocity [m/s]",
            zaxis_title="-Acceleration [m/s²]",
            xaxis=dict(tickformat='.1e'),
            yaxis=dict(tickformat='.1e')
        ),
        width=1200, height=900
    )
    return fig

# ------------------------------------------------------------
#   RUN + EXPORT
# ------------------------------------------------------------
os.makedirs("Figures/RestoringForce3D", exist_ok=True)

for level in levels_to_compute:
    fig = plot_CO_surface(level, grid_size=grid)
    fname = f"Figures/RestoringForce3D/RestoringForce_Level{level}_Crawley_Grid.html"
    fig.write_html(fname)
    print(f"Exported -> {fname}")
