import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Plot configuration
location = 3                       # Options: 1, 2, or 3 for different acceleration locations
levels_to_compare = [1, 2, 3]      # Options: list of levels to compare (e.g., [1, 2, 3])

# Frequency range for plotting
freq_min = 2               # Minimum frequency to plot [Hz]
freq_max = 15             # Maximum frequency to plot [Hz]

# Data parameters
fs = 400                                # Sampling frequency in Hz
dt = 1 / fs                             # Time step
N_per_period = 16384                    # Number of dpoints per period
N_periods = 3                           # Number of periods
N_total = N_per_period * N_periods      # Total number of data points
start_idx = N_total - N_per_period      # Start index for plotting
end_idx = N_total                       # End index for plotting
n_realisations = 9                      # Number of realisations to average

# Column configuration
force_start_col = 0        # Force columns start at column A (index 0)
accel_start_col = 18       # Column 'S' is the 19th column (0-based index = 18)
columns_per_location = 9   # 9 realizations per location

# Create time vector
t = np.linspace(0, (N_total-1)*dt, N_total)

# Function to get odd and even harmonic indices
def get_odd_even_harmonic_indices(freqs):
    """
    Returns indices corresponding to odd and even harmonics.
    
    Parameters:
    freqs : array
        Frequency vector
    
    Returns:
    odd_indices : array
        Indices of odd harmonics (1, 3, 5, 7, ...)
    even_indices : array
        Indices of even harmonics (2, 4, 6, 8, ...)
    """
    odd_indices = []
    even_indices = []
    for k in range(1, len(freqs)):
        if k % 2 == 1:  # Odd indices: 1, 3, 5, 7, ...
            odd_indices.append(k)
        else:  # Even indices: 2, 4, 6, 8, ...
            even_indices.append(k)
    return np.array(odd_indices), np.array(even_indices)

# Function to estimate local noise floor using even bins
def estimate_noise_floor(signal_fft, odd_indices, even_indices, window_size=5):
    """
    Estimate local noise floor at each odd frequency using nearby even bins.
    
    Parameters:
    signal_fft : array
        Complex FFT of the signal
    odd_indices : array
        Indices of odd harmonics
    even_indices : array
        Indices of even harmonics (noise estimates)
    window_size : int
        Number of nearby even bins to use for noise estimation
    
    Returns:
    noise_floor : array
        Estimated noise power at each odd frequency
    """
    signal_power = np.abs(signal_fft)**2
    noise_floor = np.zeros(len(odd_indices))
    
    for i, odd_idx in enumerate(odd_indices):
        # Find nearby even bins within window
        distance = np.abs(even_indices - odd_idx)
        nearby_even = even_indices[distance <= window_size]
        
        if len(nearby_even) > 0:
            # Use median of nearby even bin powers as noise estimate
            noise_floor[i] = np.median(signal_power[nearby_even])
        else:
            # Fallback: use minimum detectable level
            noise_floor[i] = 1e-20
    
    return noise_floor

# Function to process FRF with adaptive SNR gate for all realisations
def process_frf_averaged_realisations(data, location, snr_margin_db, n_realisations):
    # Initialize arrays to accumulate FRF results
    frf_sum = None
    excited_counts = None
    
    # Process each realisation
    for realisation in range(1, n_realisations + 1):
        # Get force and acceleration column indices
        force_idx = force_start_col + (location - 1) * columns_per_location + (realisation - 1)
        accel_idx = accel_start_col + (location - 1) * columns_per_location + (realisation - 1)
        
        force_column = data.columns[force_idx]
        accel_column = data.columns[accel_idx]

        # Extract segment
        force_seg = data[force_column][start_idx:end_idx].to_numpy()
        accel_seg = data[accel_column][start_idx:end_idx].to_numpy()
        
        # Compute FFTs
        force_fft = np.fft.fft(force_seg)
        accel_fft = np.fft.fft(accel_seg)
        
        # Frequency vector (positive frequencies only)
        freqs = np.fft.fftfreq(N_per_period, dt)
        freqs_pos = freqs[:N_per_period // 2]
        
        # Get odd and even harmonic indices
        odd_indices, even_indices = get_odd_even_harmonic_indices(freqs_pos)
        
        # Initialize count arrays on first iteration
        if excited_counts is None:
            excited_counts = np.zeros(len(freqs_pos), dtype=int)
            frf_sum = np.zeros(len(freqs_pos), dtype=complex)
        
        # Estimate noise floor for force signal using even bins
        force_noise_floor = estimate_noise_floor(force_fft[:N_per_period // 2], odd_indices, even_indices)
        
        # Calculate SNR at odd frequencies
        force_power_odd = np.abs(force_fft[odd_indices])**2
        snr_db = 10 * np.log10(force_power_odd / (force_noise_floor + 1e-20))
        
        # Keep only frequencies where SNR exceeds margin
        excited_mask = snr_db > snr_margin_db
        excited_odd_indices = odd_indices[excited_mask]
        
        # Accumulate counts
        excited_counts[excited_odd_indices] += 1
        
        # Calculate FRF = Acceleration / Force
        frf = accel_fft[:N_per_period // 2] / (force_fft[:N_per_period // 2] + 1e-20)
        frf_sum += frf
    
    # Average the FRF
    frf_avg = frf_sum / n_realisations
    
    # Calculate FRF magnitude in dB
    frf_mag_db = 20 * np.log10(np.abs(frf_avg) + 1e-10)
    
    # Calculate FRF phase in degrees
    frf_phase_deg = np.angle(frf_avg, deg=True)
    
    # Determine consensus excited indices
    # A frequency is considered excited if it was excited in majority of realisations
    threshold = n_realisations / 2
    excited_indices = np.where(excited_counts > threshold)[0]
    
    return freqs_pos, frf_mag_db, frf_phase_deg, excited_indices, odd_indices

# Create figure with 2 subplots (magnitude and phase)
fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(15, 10))

# Define colors for each level
colors = ['red', 'blue', 'green']

# Process and plot each level
for idx, level in enumerate(levels_to_compare):
    # Load data for this level
    if level == 1:
        RdataSpecialOddMSine = pd.read_csv('BenchmarkData/F16Data_SpecialOddMSine_Level1.csv')
        snr_margin_db = 20
    elif level == 2:
        RdataSpecialOddMSine = pd.read_csv('BenchmarkData/F16Data_SpecialOddMSine_Level2.csv')
        snr_margin_db = 25
    else:  # level == 3
        RdataSpecialOddMSine = pd.read_csv('BenchmarkData/F16Data_SpecialOddMSine_Level3.csv')
        snr_margin_db = 30
    
    # Process FRF averaged over all realisations
    freqs_pos, frf_mag_db, frf_phase_deg, excited_indices, odd_indices = \
        process_frf_averaged_realisations(RdataSpecialOddMSine, location, snr_margin_db, n_realisations)
    
    # FRF magnitude at excited odd frequencies only
    excited_plot = excited_indices[(freqs_pos[excited_indices] >= freq_min) & (freqs_pos[excited_indices] <= freq_max)]
    
    # Plot FRF magnitude
    ax_mag.plot(freqs_pos[excited_plot], frf_mag_db[excited_plot], '-', linewidth=2,
                color=colors[idx], label=f'Level {level}')
    
    # Plot FRF phase
    ax_phase.plot(freqs_pos[excited_plot], frf_phase_deg[excited_plot], '-', linewidth=2,          
                   color=colors[idx],  label=f'Level {level}')

# Configure magnitude plot
ax_mag.set_ylabel('FRF Magnitude [dB]')
ax_mag.set_title(f'Location {location} - FRF Magnitude (Robust Method)', pad=15)
ax_mag.set_xlim([freq_min, freq_max])
ax_mag.legend()
ax_mag.grid(True, which='both', alpha=0.3)

# Configure phase plot
ax_phase.set_xlabel('Frequency [Hz]')
ax_phase.set_ylabel('FRF Phase [deg]')
ax_phase.set_title(f'Location {location} - FRF Phase (Robust Method)', pad=15)
ax_phase.set_xlim([freq_min, freq_max])
ax_phase.set_ylim([-180, 180])
ax_phase.legend()
ax_phase.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()