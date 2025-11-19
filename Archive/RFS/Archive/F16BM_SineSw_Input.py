import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Plot configuration
level = 7            # Options: 1, 3, 5, 7
channel = 'Acceleration3'  # Options: 'Voltage', 'Force', 'Acceleration1', 'Acceleration2', 'Acceleration3'

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step
N = 108477        # Number of data points

# Sweep parameters
f_start = 15          # Start frequency [Hz]
f_end = 2             # End frequency [Hz]
sweep_rate = -0.05    # Frequency decay rate [Hz/s]

# Highlight region (optional - set to None to disable)
highlight_f_min = 6.5 # Minimum frequency to highlight [Hz]
highlight_f_max = 8   # Maximum frequency to highlight [Hz]
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

# Extract signal
signal_data = data[channel].to_numpy()

# Create time vector
t = np.linspace(0, (N-1)*dt, N)

# Calculate instantaneous frequency: f(t) = f_start + sweep_rate * t
instantaneous_freq = f_start + sweep_rate * t

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Fill the entire area in black
ax.fill_between(instantaneous_freq, signal_data, color='black', alpha=1.0)

# Add highlighted region if specified
if highlight_f_min is not None and highlight_f_max is not None:
    # Find indices in the highlight frequency range
    highlight_mask = (instantaneous_freq >= highlight_f_min) & (instantaneous_freq <= highlight_f_max)
    
    # Overlay the highlighted region in blue
    ax.fill_between(instantaneous_freq[highlight_mask], signal_data[highlight_mask], 
                    color='cornflowerblue', alpha=1.0)

# Set labels and title
ax.set_xlabel('Sweep Frequency [Hz]')
ax.set_ylabel('Acceleration [m/s²]')
ax.set_title(f'Time series of vertical acceleration measured at location 3\nSine sweep excitation ({f_start} Hz to {f_end} Hz at {sweep_rate} Hz/s)')

# Invert x-axis to show frequency sweep from high to low
ax.invert_xaxis()

# Set limits
ax.set_xlim([f_start, f_end])
ax.set_ylim([signal_data.min() * 1.1, signal_data.max() * 1.1])

# Remove grid
ax.grid(False)

plt.tight_layout()
plt.show()

# Print sweep information
print(f"\n{'='*60}")
print(f"Sine Sweep Information")
print(f"{'='*60}")
print(f"Sweep rate: {sweep_rate} Hz/s")
print(f"Start frequency: {f_start} Hz")
print(f"End frequency: {f_end} Hz")
print(f"Total sweep duration: {(f_end - f_start) / sweep_rate:.2f} s")
print(f"Data duration: {t[-1]:.2f} s")
print(f"Final instantaneous frequency: {instantaneous_freq[-1]:.2f} Hz")
print(f"{'='*60}\n")