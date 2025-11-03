import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Data parameters
fs = 400                                # Sampling frequency in Hz
dt = 1 / fs                             # Time step
N_per_period = 8192                     # Number of dpoints per period
N_periods = 9                           # Number of periods
N_total = N_per_period * N_periods      # Total number of data points             
# ------------------------------------------

# Load Full Multisine data
RdataFullMSine_L1 = pd.read_csv('BenchmarkData/F16Data_FullMSine_Level1.csv')
RdataFullMSine_L5 = pd.read_csv('BenchmarkData/F16Data_FullMSine_Level5.csv')
RdataFullMSine_L7 = pd.read_csv('BenchmarkData/F16Data_FullMSine_Level7.csv')

# Create time vector
t = np.linspace(0, (N_total-1)*dt, N_total)

# Function to compute FRF for a dataset, discarding first period
def compute_frf(data, input_key='Force', output_key='Response'):
    """
    Computes the FRF for a dataset using periods 2 to 9.
    
    Parameters:
    data : DataFrame
        DataFrame containing 'Force' and 'Response' arrays
    input_key : str
        Key for input data
    output_key : str
        Key for output data
    """
    # Initialize sums for averaging FFTs
    X_sum = np.zeros(N_per_period, dtype=complex)
    Y_sum = np.zeros(N_per_period, dtype=complex)
    
    # Loop over periods 2 to 9 (indices 1 to 8)
    for n in range(1, 9):
        start_idx = N_per_period * n
        end_idx   = N_per_period * (n + 1)
        
        X_fft = np.fft.fft(data[input_key][start_idx:end_idx])
        Y_fft = np.fft.fft(data[output_key][start_idx:end_idx])
        
        X_sum += X_fft
        Y_sum += Y_fft
    
    # Average FFTs over periods
    X_avg = X_sum / 8
    Y_avg = Y_sum / 8
    
    # Compute FRF
    H = Y_avg / X_avg
    
    # Frequency vector (positive frequencies only)
    freqs = np.fft.fftfreq(N_per_period, dt)
    freqs_pos = freqs[:N_per_period // 2]
    
    # FRF magnitude and phase (positive frequencies)
    H_mag = np.abs(H[:N_per_period // 2])
    H_phase = np.angle(H[:N_per_period // 2])
    
    # Convert magnitude to dB
    H_mag_dB = 20 * np.log10(H_mag + 1e-10)
    
    return freqs_pos, H_mag_dB, H_phase

# Compute FRFs for Level 1
freqs, H1L1_mag, _ = compute_frf(RdataFullMSine_L1, input_key='Force', output_key='Acceleration1')
freqs, H2L1_mag, _ = compute_frf(RdataFullMSine_L1, input_key='Force', output_key='Acceleration2')
freqs, H3L1_mag, _ = compute_frf(RdataFullMSine_L1, input_key='Force', output_key='Acceleration3')

# Compute FRFs for Level 5
freqs, H1L5_mag, _ = compute_frf(RdataFullMSine_L5, input_key='Force', output_key='Acceleration1')
freqs, H2L5_mag, _ = compute_frf(RdataFullMSine_L5, input_key='Force', output_key='Acceleration2')
freqs, H3L5_mag, _ = compute_frf(RdataFullMSine_L5, input_key='Force', output_key='Acceleration3')

# Compute FRFs for Level 7
freqs, H1L7_mag, _ = compute_frf(RdataFullMSine_L7, input_key='Force', output_key='Acceleration1')
freqs, H2L7_mag, _ = compute_frf(RdataFullMSine_L7, input_key='Force', output_key='Acceleration2')
freqs, H3L7_mag, _ = compute_frf(RdataFullMSine_L7, input_key='Force', output_key='Acceleration3')

# Create two figures: one for levels comparison, one for locations comparison
freq_range = (freqs >= 2) & (freqs <= 15)

# Figure 1: Comparing locations for each level
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))

# Level 1 FRF Magnitude Plot
ax1.plot(freqs[freq_range], H1L1_mag[freq_range], label='Location 1')
ax1.plot(freqs[freq_range], H2L1_mag[freq_range], label='Location 2')
ax1.plot(freqs[freq_range], H3L1_mag[freq_range], label='Location 3')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('FRF Magnitude [dB]')
ax1.set_title('Level 1')
ax1.legend()
ax1.grid(True, which='both')

# Level 5 FRF Magnitude Plot
ax2.plot(freqs[freq_range], H1L5_mag[freq_range], label='Location 1')
ax2.plot(freqs[freq_range], H2L5_mag[freq_range], label='Location 2')
ax2.plot(freqs[freq_range], H3L5_mag[freq_range], label='Location 3')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('FRF Magnitude [dB]')
ax2.set_title('Level 5')
ax2.legend()
ax2.grid(True, which='both')

# Level 7 FRF Magnitude Plot
ax3.plot(freqs[freq_range], H1L7_mag[freq_range], label='Location 1')
ax3.plot(freqs[freq_range], H2L7_mag[freq_range], label='Location 2')
ax3.plot(freqs[freq_range], H3L7_mag[freq_range], label='Location 3')
ax3.set_xlabel('Frequency [Hz]')
ax3.set_ylabel('FRF Magnitude [dB]')
ax3.set_title('Level 7')
ax3.legend()
ax3.grid(True, which='both')

plt.tight_layout()
plt.show()
