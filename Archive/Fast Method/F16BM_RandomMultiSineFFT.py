import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Plot configuration
locations_to_compare = [1,2,3]  # Options: list of locations to compare (e.g., [1, 3])
realisation = 1                # Options: 1 through 9 for different realisations
level = 3                      # Options: 1 through 3 for different forcing amplitudes

# Frequency range for plotting
freq_min = 2               # Minimum frequency to plot [Hz]
freq_max = 15              # Maximum frequency to plot [Hz]

# SNR margin based on level
if level == 1:
    snr_margin_db = 35
elif level == 2:
    snr_margin_db = 32
else:  # level == 3
    snr_margin_db = 30

# Data parameters
fs = 400                                # Sampling frequency in Hz
dt = 1 / fs                             # Time step
N_per_period = 16384                    # Number of dpoints per period
N_periods = 3                           # Number of periods
N_total = N_per_period * N_periods      # Total number of data points
start_idx = N_total - N_per_period      # Start index for plotting
end_idx = N_total                       # End index for plotting

# Column configuration
force_start_col = 0        # Force columns start at column A (index 0)
accel_start_col = 18       # Column 'S' is the 19th column (0-based index = 18)
columns_per_location = 9   # 9 realizations per location

# Load Special Odd Multisine data
if level == 1:
    RdataSpecialOddMSine = pd.read_csv('BenchmarkData/F16Data_SpecialOddMSine_Level1.csv')
if level == 2:
    RdataSpecialOddMSine = pd.read_csv('BenchmarkData/F16Data_SpecialOddMSine_Level2.csv')
if level == 3:
    RdataSpecialOddMSine = pd.read_csv('BenchmarkData/F16Data_SpecialOddMSine_Level3.csv')

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

# Function to process FFT with adaptive SNR gate
def process_fft_with_adaptive_snr(data, location, realisation, snr_margin_db):
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
    
    # Estimate noise floor for force signal using even bins
    force_noise_floor = estimate_noise_floor(force_fft[:N_per_period // 2], odd_indices, even_indices)
    
    # Estimate noise floor for acceleration signal using even bins
    accel_noise_floor = estimate_noise_floor(accel_fft[:N_per_period // 2], odd_indices, even_indices)
    
    # Calculate SNR at odd frequencies
    force_power_odd = np.abs(force_fft[odd_indices])**2
    snr_db = 10 * np.log10(force_power_odd / (force_noise_floor + 1e-20))
    
    # Keep only frequencies where SNR exceeds margin
    excited_mask = snr_db > snr_margin_db
    excited_odd_indices = odd_indices[excited_mask]
    
    # Find rejected odd indices (those that didn't pass the SNR threshold)
    rejected_odd_indices = odd_indices[~excited_mask]
    
    # Get force magnitude at odd frequencies (convert to dB)
    X_mag_odd = np.abs(force_fft[odd_indices])
    X_mag_odd_db = 20 * np.log10(X_mag_odd + 1e-10)
    
    # Calculate adaptive threshold (noise floor + margin) in dB
    adaptive_threshold = np.sqrt(force_noise_floor) * 10**(snr_margin_db / 20)
    adaptive_threshold_db = 20 * np.log10(adaptive_threshold + 1e-10)
    
    # Get acceleration magnitude
    accel_mag = np.abs(accel_fft[:N_per_period // 2])
    accel_mag_db = 20 * np.log10(accel_mag + 1e-10)
    
    # Convert acceleration noise floor to dB (magnitude, not power)
    accel_noise_floor_db = 20 * np.log10(np.sqrt(accel_noise_floor) + 1e-10)
    
    return freqs_pos, accel_mag_db, excited_odd_indices, rejected_odd_indices, odd_indices, even_indices, X_mag_odd_db, adaptive_threshold_db, snr_db, accel_noise_floor_db

# Create subplots for comparison
n_locations = len(locations_to_compare)
fig, axes = plt.subplots(n_locations, 1, figsize=(15, 5 * n_locations))

# Make axes always iterable (even if only one location)
if n_locations == 1:
    axes = [axes]

# Process and plot each location
for idx, location in enumerate(locations_to_compare):
    # Process FFTs with adaptive SNR
    freqs_pos, accel_mag_db, excited_indices, rejected_odd_indices, odd_indices, even_indices, force_mag_odd_db, adaptive_threshold_db, snr_db, accel_noise_floor_db = \
        process_fft_with_adaptive_snr(RdataSpecialOddMSine, location, realisation, snr_margin_db)
    
    ax = axes[idx]
    
    # Acceleration FFT at excited odd frequencies only
    excited_plot = excited_indices[(freqs_pos[excited_indices] >= freq_min) & (freqs_pos[excited_indices] <= freq_max)]
    
    # Plot excited odd harmonics (black line only, no markers)
    ax.plot(freqs_pos[excited_plot], accel_mag_db[excited_plot], '-', linewidth=2, color='black', label='Excited Odd Harmonics')
    
    # Plot noise floor as green line
    freq_range_odd = (freqs_pos[odd_indices] >= freq_min) & (freqs_pos[odd_indices] <= freq_max)
    ax.plot(freqs_pos[odd_indices][freq_range_odd], accel_noise_floor_db[freq_range_odd], '-', linewidth=2, color='green', label='Noise Floor', alpha=0.7)
    
    # Plot rejected odd harmonics (noise detection lines) in frequency range (red)
    rejected_in_range = rejected_odd_indices[(freqs_pos[rejected_odd_indices] >= freq_min) & (freqs_pos[rejected_odd_indices] <= freq_max)]
    ax.plot(freqs_pos[rejected_in_range], accel_mag_db[rejected_in_range], 'o', color='red', markersize=5, label='Rejected Odd Harmonics', linestyle='None', markerfacecolor='none', markeredgewidth=1.5)
    
    # Plot even harmonics (also noise) in frequency range (blue)
    even_in_range = even_indices[(freqs_pos[even_indices] >= freq_min) & (freqs_pos[even_indices] <= freq_max)]
    ax.plot(freqs_pos[even_in_range], accel_mag_db[even_in_range], 'o', color='blue', markersize=4, label='Even Harmonics', linestyle='None', markerfacecolor='none', markeredgewidth=1)
    
    # Only show x-axis label on bottom plot
    if idx == n_locations - 1:
        ax.set_xlabel('Frequency [Hz]')
    
    ax.set_ylabel('Acceleration Magnitude [dB]')
    ax.set_title(f'Location {location} - Acceleration FFT (Realisation {realisation}, Level {level})', pad=15)
    ax.set_xlim([freq_min, freq_max])
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()