import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Plot configuration
y_min = 0.3                            # Minimum y-axis value for coherence plots
y_max = 1.0                            # Maximum y-axis value for coherence plots

# Data parameters
fs = 400                                # Sampling frequency in Hz
dt = 1 / fs                             # Time step
N_per_period = 8192                    # Number of dpoints per period
N_periods = 9                           # Number of periods
N_total = N_per_period * N_periods      # Total number of data points             
# ------------------------------------------

# Load Full Multisine data
RdataFullMSine_L1 = pd.read_csv('BenchmarkData/F16Data_FullMSine_Level1.csv')
RdataFullMSine_L5 = pd.read_csv('BenchmarkData/F16Data_FullMSine_Level5.csv')
RdataFullMSine_L7 = pd.read_csv('BenchmarkData/F16Data_FullMSine_Level7.csv')

# Create time vector
t = np.linspace(0, (N_total-1)*dt, N_total)

def compute_coherence(data, input_key='Force', output_key='Response'):
    """
    Computes the coherence function using periods 2 to 9.
    
    Parameters:
    data : DataFrame
        DataFrame containing 'Force' and 'Response' arrays
    input_key : str
        Key for input data
    output_key : str
        Key for output data
    """
    # Initialize arrays for accumulating spectra
    Pxx_sum = np.zeros(N_per_period)
    Pyy_sum = np.zeros(N_per_period)
    Pxy_sum = np.zeros(N_per_period, dtype=complex)
    
    # Loop over periods 2 to 9 (indices 1 to 8)
    for n in range(1, 9):
        start_idx = N_per_period * n
        end_idx = N_per_period * (n + 1)
        
        # Get input and output data for this period
        x = data[input_key][start_idx:end_idx]
        y = data[output_key][start_idx:end_idx]
        
        # Compute FFTs
        X = np.fft.fft(x)
        Y = np.fft.fft(y)
        
        # Accumulate auto and cross spectra
        Pxx_sum += np.abs(X) ** 2
        Pyy_sum += np.abs(Y) ** 2
        Pxy_sum += X.conj() * Y
    
    # Average the spectra
    Pxx = Pxx_sum / 8
    Pyy = Pyy_sum / 8
    Pxy = Pxy_sum / 8
    
    # Compute coherence
    coherence = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    
    # Frequency vector (positive frequencies only)
    freqs = np.fft.fftfreq(N_per_period, dt)
    freqs_pos = freqs[:N_per_period // 2]
    
    return freqs_pos, coherence[:N_per_period // 2]

# Compute coherence for Level 1
freqs, C1L1 = compute_coherence(RdataFullMSine_L1, input_key='Force', output_key='Acceleration1')
freqs, C2L1 = compute_coherence(RdataFullMSine_L1, input_key='Force', output_key='Acceleration2')
freqs, C3L1 = compute_coherence(RdataFullMSine_L1, input_key='Force', output_key='Acceleration3')

# Compute coherence for Level 5
freqs, C1L5 = compute_coherence(RdataFullMSine_L5, input_key='Force', output_key='Acceleration1')
freqs, C2L5 = compute_coherence(RdataFullMSine_L5, input_key='Force', output_key='Acceleration2')
freqs, C3L5 = compute_coherence(RdataFullMSine_L5, input_key='Force', output_key='Acceleration3')

# Compute coherence for Level 7
freqs, C1L7 = compute_coherence(RdataFullMSine_L7, input_key='Force', output_key='Acceleration1')
freqs, C2L7 = compute_coherence(RdataFullMSine_L7, input_key='Force', output_key='Acceleration2')
freqs, C3L7 = compute_coherence(RdataFullMSine_L7, input_key='Force', output_key='Acceleration3')

# Create two figures: one for levels comparison, one for locations comparison
freq_range = (freqs >= 2) & (freqs <= 15)

# Figure 1: Comparing locations for each level
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))
fig1.suptitle('F16 Benchmark Multi-Sine Test: Input-Output Coherence', fontsize=14)

# Level 1 Coherence Plot
ax1.plot(freqs[freq_range], C1L1[freq_range], label='Location 1')
ax1.plot(freqs[freq_range], C2L1[freq_range], label='Location 2')
ax1.plot(freqs[freq_range], C3L1[freq_range], label='Location 3')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Coherence')
ax1.set_ylim([y_min, y_max])  # User-defined coherence range
ax1.set_title('Level 1')
ax1.legend()
ax1.grid(True)

# Level 5 Coherence Plot
ax2.plot(freqs[freq_range], C1L5[freq_range], label='Location 1')
ax2.plot(freqs[freq_range], C2L5[freq_range], label='Location 2')
ax2.plot(freqs[freq_range], C3L5[freq_range], label='Location 3')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Coherence')
ax2.set_ylim([y_min, y_max])
ax2.set_title('Level 5')
ax2.legend()
ax2.grid(True)

# Level 7 Coherence Plot
ax3.plot(freqs[freq_range], C1L7[freq_range], label='Location 1')
ax3.plot(freqs[freq_range], C2L7[freq_range], label='Location 2')
ax3.plot(freqs[freq_range], C3L7[freq_range], label='Location 3')
ax3.set_xlabel('Frequency [Hz]')
ax3.set_ylabel('Coherence')
ax3.set_ylim([y_min, y_max])
ax3.set_title('Level 7')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
