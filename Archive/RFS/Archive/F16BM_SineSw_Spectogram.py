import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Plot configuration
level = 1            # Options: 1, 3, 5, 7
channel = 'Acceleration3'  # Options: 'Voltage', 'Force', 'Acceleration1', 'Acceleration2', 'Acceleration3'

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step
N = 108477           # Number of data points
# ------------------------------------------

# Load CSV data
RdataSineSw_L1 = pd.read_csv('BenchmarkData/F16Data_SineSw_Level1.csv')
RdataSineSw_L3 = pd.read_csv('BenchmarkData/F16Data_SineSw_Level3.csv')
RdataSineSw_L5 = pd.read_csv('BenchmarkData/F16Data_SineSw_Level5.csv')
RdataSineSw_L7 = pd.read_csv('BenchmarkData/F16Data_SineSw_Level7.csv')

# Select data based on level
if level == 1:
    data = RdataSineSw_L1
    title = 'Sine Sweep - Level 1'
elif level == 3:
    data = RdataSineSw_L3
    title = 'Sine Sweep - Level 3'
elif level == 5:
    data = RdataSineSw_L5
    title = 'Sine Sweep - Level 5'
else:  # level 7
    data = RdataSineSw_L7
    title = 'Sine Sweep - Level 7'

# Extract signal and create time vector
signal_data = data[channel].to_numpy()
t = np.linspace(0, (len(signal_data)-1)*dt, len(signal_data))

# --- Spectrogram, ridge detection, and time-gating ---

# Spectrogram parameters (tune if needed)
nperseg = 2048            # window length in samples (trade time vs freq resolution)
noverlap = nperseg // 2   # 50% overlap
mode = 'magnitude'        # get magnitude spectrogram
scaling = 'spectrum'      # power-like scaling (useful for ridge detection)
fmax_plot = 50.0          # plotting upper freq limit (Hz)

# compute spectrogram
f_s, t_s, Sxx = spectrogram(signal_data, fs=fs, window='hann',
                           nperseg=nperseg, noverlap=noverlap,
                           scaling=scaling, mode=mode)

# convert to dB for plotting (avoid log of zero)
Sdb = 20.0 * np.log10(Sxx + 1e-20)

# Plot spectrogram (time x frequency)
plt.figure(figsize=(10, 4))
plt.pcolormesh(t_s, f_s, Sdb, shading='gouraud', cmap='viridis')
plt.colorbar(label='Magnitude (dB)')
plt.ylim(0, min(fmax_plot, f_s.max()))
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title(f'Spectrogram — {channel} (Level {level})')
plt.tight_layout()
plt.show()

