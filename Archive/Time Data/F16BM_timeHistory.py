
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Plot configuration
plot_type = 'grid'  # Options: 'grid' for 3x3 grid, 'single' for individual wide plots
input_type = 'full'  # Options: 'sweep', 'full', 'special'
level = 1            # Options: 1, 3, 5
channel = 'Acceleration3'  # Options: 'Voltage', 'Force', 'Acceleration1', 'Acceleration2', 'Acceleration3'

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step
N = 10000             # Number of data points
start_idx = 2000      # Start index for plotting
end_idx = 3200       # End index for plotting
# ------------------------------------------

# Load CSV data for all levels
# Sine Sweep data
RdataSineSw_L1 = pd.read_csv('BenchmarkData/F16Data_SineSw_Level1.csv')
RdataSineSw_L3 = pd.read_csv('BenchmarkData/F16Data_SineSw_Level3.csv')
RdataSineSw_L5 = pd.read_csv('BenchmarkData/F16Data_SineSw_Level5.csv')

# Full Multisine data
RdataFullMSine_L1 = pd.read_csv('BenchmarkData/F16Data_FullMSine_Level1.csv')
RdataFullMSine_L3 = pd.read_csv('BenchmarkData/F16Data_FullMSine_Level3.csv')
RdataFullMSine_L5 = pd.read_csv('BenchmarkData/F16Data_FullMSine_Level5.csv')

# Special Odd Multisine data
RdataSpecialOddMSine_L1 = pd.read_csv('BenchmarkData/F16Data_SpecialOddMSine_Level1.csv')
RdataSpecialOddMSine_L3 = pd.read_csv('BenchmarkData/F16Data_SpecialOddMSine_Level2.csv')
RdataSpecialOddMSine_L5 = pd.read_csv('BenchmarkData/F16Data_SpecialOddMSine_Level3.csv')

# Create time vector
t = np.linspace(0, (N-1)*dt, N)

# Print plotting range information
print("\nTime range being plotted:")
print(f"From {t[start_idx]:.3f}s to {t[end_idx]:.3f}s")

# Set ylabel based on channel
if 'Acceleration' in channel:
    ylabel = f'{channel} (m/s²)'
elif channel == 'Force':
    ylabel = 'Force (N)'
elif channel == 'Voltage':
    ylabel = 'Voltage (V)'

# Select data based on input type and level
if plot_type == 'single':
    # Configure single wide plot
    fig, ax = plt.subplots(figsize=(18, 6))
    
    if input_type == 'sweep':
        if level == 1:
            data = RdataSineSw_L1
            color = 'red'
            title = 'Sine Sweep - Level 1'
        elif level == 3:
            data = RdataSineSw_L3
            color = 'red'
            title = 'Sine Sweep - Level 3'
        else:  # level 5
            data = RdataSineSw_L5
            color = 'red'
            title = 'Sine Sweep - Level 5'
    elif input_type == 'full':
        if level == 1:
            data = RdataFullMSine_L1
            color = 'blue'
            title = 'Full Multisine - Level 1'
        elif level == 3:
            data = RdataFullMSine_L3
            color = 'blue'
            title = 'Full Multisine - Level 3'
        else:  # level 5
            data = RdataFullMSine_L5
            color = 'blue'
            title = 'Full Multisine - Level 5'
    else:  # special
        if level == 1:
            data = RdataSpecialOddMSine_L1
            color = 'green'
            title = 'Special Odd Multisine - Level 1'
        elif level == 3:
            data = RdataSpecialOddMSine_L3
            color = 'green'
            title = 'Special Odd Multisine - Level 3'
        else:  # level 5
            data = RdataSpecialOddMSine_L5
            color = 'green'
            title = 'Special Odd Multisine - Level 5'
    
    # Create single plot
    ax.plot(t[start_idx:end_idx], data[channel][start_idx:end_idx], color=color)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} - {channel}")
    ax.grid(True)
    
else:  # grid mode
    # Create a tiled plot with 3x3 grid (3 levels x 3 input types)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex='col')

    # Column titles for input types
    plt.figtext(0.20, 0.95, 'Sine Sweep', ha='center', va='center', fontsize=12, weight='bold')
    plt.figtext(0.53, 0.95, 'Full Multisine', ha='center', va='center', fontsize=12, weight='bold')
    plt.figtext(0.85, 0.95, 'Random Multisine', ha='center', va='center', fontsize=12, weight='bold')

    # Sine Sweep plots (first column)
    axes[0,0].plot(t[start_idx:end_idx], RdataSineSw_L1[channel][start_idx:end_idx], color='red')
    axes[0,0].set_ylabel(f'{ylabel}\nLevel 1')
    axes[0,0].grid(True)

    axes[1,0].plot(t[start_idx:end_idx], RdataSineSw_L3[channel][start_idx:end_idx], color='red')
    axes[1,0].set_ylabel(f'{ylabel}\nLevel 3')
    axes[1,0].grid(True)

    axes[2,0].plot(t[start_idx:end_idx], RdataSineSw_L5[channel][start_idx:end_idx], color='red')
    axes[2,0].set_xlabel('Time (s)')
    axes[2,0].set_ylabel(f'{ylabel}\nLevel 5')
    axes[2,0].grid(True)

    # Full Multisine plots (middle column)
    axes[0,1].plot(t[start_idx:end_idx], RdataFullMSine_L1[channel][start_idx:end_idx], color='blue')
    axes[0,1].grid(True)

    axes[1,1].plot(t[start_idx:end_idx], RdataFullMSine_L3[channel][start_idx:end_idx], color='blue')
    axes[1,1].grid(True)

    axes[2,1].plot(t[start_idx:end_idx], RdataFullMSine_L5[channel][start_idx:end_idx], color='blue')
    axes[2,1].set_xlabel('Time (s)')
    axes[2,1].grid(True)

    # Special Odd Multisine plots (last column)
    axes[0,2].plot(t[start_idx:end_idx], RdataSpecialOddMSine_L1[channel][start_idx:end_idx], color='green')
    axes[0,2].grid(True)

    axes[1,2].plot(t[start_idx:end_idx], RdataSpecialOddMSine_L3[channel][start_idx:end_idx], color='green')
    axes[1,2].grid(True)

    axes[2,2].plot(t[start_idx:end_idx], RdataSpecialOddMSine_L5[channel][start_idx:end_idx], color='green')
    axes[2,2].set_xlabel('Time (s)')
    axes[2,2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for column titles

plt.show()  # Display the figure