import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from architecture import cVAE, cvae_loss
from helpers import process_data, normalise_data, downsample_signal, slice_period_data

# ----------- USER CONFIGURATION -----------
# Validation level
validation_level = [2, 4, 6]  # Level to use for validation (Options: 2, 4, 6)

# User-configurable data-shape parameters
training_period_indices = list(range(3, 10))  # periods 3 through 9 inclusive 
downsample_factor_accel = 2   # Downsampling for acceleration signals
downsample_factor_force = 2   # Downsampling for force signal 

# Training parameters
num_epochs = 150
beta = 0.5  
minibatch_size = 32

# ----------- PARAMETERS -----------
# Data levels and locations
training_level = [1, 3, 5, 7]
location_i = 2  
location_j = 3  

# Data parameters
fs = 400             
dt = 1 / fs          

# Data shape parameters
points_per_period = 8192

# Model parameters
latent_dim = 32               

# ----------- LOAD DATA -----------
training_data = {
    1: pd.read_csv('BenchmarkData/F16Data_FullMSine_Level1.csv'), # 12.4N
    3: pd.read_csv('BenchmarkData/F16Data_FullMSine_Level3.csv'), # 36.8N
    5: pd.read_csv('BenchmarkData/F16Data_FullMSine_Level5.csv'), # 73.6N
    7: pd.read_csv('BenchmarkData/F16Data_FullMSine_Level7.csv')  # 97.8N
}

validation_data = {
    2: pd.read_csv('BenchmarkData/F16Data_FullMSine_Level2_Validation.csv'), # 24.6N
    4: pd.read_csv('BenchmarkData/F16Data_FullMSine_Level4_Validation.csv'), # 61.4N
    6: pd.read_csv('BenchmarkData/F16Data_FullMSine_Level6_Validation.csv'), # 85.7N
}

# Keep only requested levels
training_data = {lvl: training_data[lvl] for lvl in training_level}
validation_data = {lvl: validation_data[lvl] for lvl in validation_level}

# ----------- PROCESS TRAINING DATA (UNSLICED) -----------
# Extract per-period samples for training
X_raw, F_raw = process_data(training_data, training_level,
                             period_indices=training_period_indices)

# Downsample signals
X_raw = downsample_signal(X_raw, downsample_factor_accel, axis=-1)
F_raw = downsample_signal(F_raw, downsample_factor_force, axis=-1)

# ----------- SLICE DATA -----------
slice_length = 256
step = 128

X_sliced_list = []
F_sliced_list = []

for i in range(X_raw.shape[0]):   # loop over all periods
    X_period = X_raw[i]           # (2, T)
    F_period = F_raw[i]           # (T,)

    X_slices, F_slices = slice_period_data(X_period, F_period,
                                           slice_length=slice_length,
                                           step=step)

    X_sliced_list.append(X_slices)
    F_sliced_list.append(F_slices)

# Stack across all periods
X_sliced = np.vstack(X_sliced_list)   # (N_slices_total, 2, slice_length)
F_sliced = np.vstack(F_sliced_list)   # (N_slices_total, slice_length)

# Normalise entire sliced dataset
X_norm, X_mean, X_std = normalise_data(X_sliced)
F_norm, F_mean, F_std = normalise_data(F_sliced)

# Convert to tensors
X_train = torch.tensor(X_norm, dtype=torch.float32)
F_train = torch.tensor(F_norm, dtype=torch.float32)

# Create dataset
train_dataset = TensorDataset(X_train, F_train)

# Create dataloader
train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)

print("=" * 60)
print("TRAINING")
print("=" * 60)
print("Training Data Shape (Acceleration):", X_train.shape)
print("Conditioning Data Shape (Force):", F_train.shape)

# ----------- TRAINING SETUP -----------
model = cVAE(latent_dim=latent_dim, n_locations=2, 
             sequence_length_accel=slice_length,
             sequence_length_force=slice_length)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("-" * 60)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ----------- TRAINING -----------
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0

    for X_batch, F_batch in train_loader:
        # Forward pass
        X_recon, mu, logvar, z = model(X_batch, F_batch)

        # Loss for this minibatch
        loss, recon_loss, kl_loss = cvae_loss(X_recon, X_batch, mu, logvar, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate epoch stats
        epoch_loss += loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: "
              f"Loss={epoch_loss:.3f}  "
              f"Recon={epoch_recon:.3f}  "
              f"KL={epoch_kl:.3f}")

# ----------- LOOP THROUGH VALIDATION LEVELS -----------
print("=" * 60)
print("SIMULATION")
print("=" * 60)
# Initialize storage for all validation levels
all_val_data = {}

for val_level in validation_level:
    # ----------- PROCESS VALIDATION DATA (UNSLICED) -----------
    # Extract per-period samples for this validation level
    X_val_raw, F_val_raw = process_data({val_level: validation_data[val_level]}, [val_level],
                                         period_indices=training_period_indices)

    X_val_raw = downsample_signal(X_val_raw, downsample_factor_accel, axis=-1)
    F_val_raw = downsample_signal(F_val_raw, downsample_factor_force, axis=-1)

    #----------- SLICE VALIDATION DATA -----------
    period_index = 5  # 0-based index for 6th period

    X_slices_val, F_slices_val = slice_period_data(
           X_val_raw[period_index],
           F_val_raw[period_index],
           slice_length=256,
           step=128
    )

    # Normalise validation force data
    F_slices_val_norm = (F_slices_val - F_mean) / F_std

    # Convert to tensor
    F_slices_val_norm = torch.tensor(F_slices_val_norm, dtype=torch.float32)

    # --------------- SIMULATE THE MODEL USING DECODER ONLY -----------
    print("Simulating Level:", val_level)
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        # Generate multiple stochastic reconstructions using decoder only
        n_samples = 50
        n_slices = F_slices_val_norm.shape[0]
        
        # Store samples: (n_slices, n_samples, 2, slice_length)
        X_sim_all_sliced = []
        
        # Outer loop: iterate over each slice
        for i in range(n_slices):
            c_slice = F_slices_val_norm[i:i+1]   # selects singular slice
            
            # Inner loop: iterate over each sample
            X_sim_samples_for_slice = []
            for _ in range(n_samples):
                z = torch.randn(1, latent_dim) # reparameterisation trick
                X_sim_slice = model.decoder(z, c_slice)[0]   # pass through decoder only
                X_sim_samples_for_slice.append(X_sim_slice.cpu().numpy())  # append sample
            
            # Stack samples for this slice
            X_sim_samples_for_slice = np.array(X_sim_samples_for_slice)
            X_sim_all_sliced.append(X_sim_samples_for_slice)
        
        # Stack all slices
        X_sim_all_sliced = np.array(X_sim_all_sliced)

    # --------------- STITCH SLICES BACK TOGETHER -----------
    T = X_val_raw.shape[2]                # e.g., 4096
    n_slices = X_sim_all_sliced.shape[0]  # number of slices in this period
    n_samples = X_sim_all_sliced.shape[1] # Monte Carlo samples per slice

    # This will hold final generated full signals:
    X_sim_samples_for_period = []

    # Reconstruct FULL PERIOD for each Monte Carlo sample
    for sample_idx in range(n_samples):

        # Empty reconstruction buffers
        X_recon = np.zeros((2, T))
        counts  = np.zeros(T)

        # Loop through each slice
        for slice_idx in range(n_slices):

            # Predicted slice for this sample: shape (2, slice_length)
            sl = X_sim_all_sliced[slice_idx, sample_idx]

            # Compute placement in the final signal
            start = slice_idx * step
            end   = start + slice_length

            # Accumulate values
            X_recon[:, start:end] += sl
            counts[start:end]     += 1

        # Avoid division by zero (normally unnecessary)
        counts[counts == 0] = 1

        # Average overlapping regions
        X_recon /= counts

        # Save this sample
        X_sim_samples_for_period.append(X_recon)

    # Convert to array → shape (n_samples, 2, T)
    X_sim_all = np.array(X_sim_samples_for_period)

    # Compute mean across Monte-Carlo samples
    X_sim_mean = X_sim_all.mean(axis=0)  # shape (2, T)

    # --------------- DENORMALISE FOR ANALYSIS -----------
    # Denormalise the mean simulation back to original scale
    X_sim_mean_denorm = X_sim_mean * X_std + X_mean

    # Denormalise all simulation samples for uncertainty analysis
    X_sim_samples_denorm = X_sim_all * X_std + X_mean
    
    # Store results for this validation level
    all_val_data[val_level] = {
        'X_val_raw': X_val_raw,
        'X_sim_mean_denorm': X_sim_mean_denorm,
        'X_sim_samples_denorm': X_sim_samples_denorm,
        'period_index': period_index
    }

# ----------- PLOTTING ALL VALIDATION LEVELS ON ONE FIGURE -----------
print("Plotting All Validation Levels")
# Downsampled sampling frequency used for plotting 
fs_downsampled = fs / downsample_factor_accel

# Zoom settings
zoom_window_seconds = 1.0  # seconds of data to display

# Location names for plotting
location_names = [f'Location {location_i} (Wing Side)', f'Location {location_j} (Payload Side)']

# Create single figure with 3 rows (one per validation level) x 2 columns (one per location)
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
fig.suptitle('cVAE Reconstruction: Simulated vs Validation Data', fontsize=18, fontweight='bold')

# Plot each validation level
for row_idx, val_level in enumerate(validation_level):
    data = all_val_data[val_level]
    X_val_raw = data['X_val_raw']
    X_sim_mean_denorm = data['X_sim_mean_denorm']
    X_sim_samples_denorm = data['X_sim_samples_denorm']
    period_index = data['period_index']
    
    T_full = X_val_raw.shape[2]
    t = np.arange(T_full) / fs_downsampled
    
    # Compute zoom window (centered at midpoint)
    half_window_samples = int((zoom_window_seconds / 2) * fs_downsampled)
    mid_idx = len(t) // 2
    start_idx = max(mid_idx - half_window_samples, 0)
    end_idx = min(mid_idx + half_window_samples, len(t))
    
    t_zoom = t[start_idx:end_idx]

    # Plot both locations for this level
    for loc_idx in range(2):
        ax = axes[row_idx, loc_idx]
        
        # Plot validation (true) data for this period (zoomed)
        ax.plot(t_zoom, X_val_raw[period_index, loc_idx, start_idx:end_idx], 
                color='black', linewidth=1.5, label='Validation (True)', alpha=0.8)
        
        # Plot mean simulation for this period (zoomed)
        ax.plot(t_zoom, X_sim_mean_denorm[loc_idx, start_idx:end_idx], 
                color='red', linewidth=1.5, label='cVAE Mean', linestyle='--', alpha=0.9)
        
        # Plot variance as confidence bands (std across samples for this period)
        sim_sample_std = X_sim_samples_denorm[:, loc_idx, start_idx:end_idx].std(axis=0)
        
        ax.fill_between(t_zoom, X_sim_mean_denorm[loc_idx, start_idx:end_idx] - sim_sample_std, 
                        X_sim_mean_denorm[loc_idx, start_idx:end_idx] + sim_sample_std,
                        color='blue', alpha=0.2, label='±1σ Uncertainty')
        ax.fill_between(t_zoom, X_sim_mean_denorm[loc_idx, start_idx:end_idx] - 2*sim_sample_std, 
                        X_sim_mean_denorm[loc_idx, start_idx:end_idx] + 2*sim_sample_std,
                        color='blue', alpha=0.1, label='±2σ Uncertainty')
        
        # Labels and formatting
        if row_idx == 2:  # Only bottom row gets x-label
            ax.set_xlabel('Time [s]')
        ax.set_ylabel('Acceleration [m/s²]')
        
        # Title: location name for top row, level number on left column
        if row_idx == 0:
            ax.set_title(f'{location_names[loc_idx]}', fontsize=14)
        if loc_idx == 0:
            ax.text(-0.15, 0.5, f'Level {val_level}', transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='center', rotation=90)
        
        ax.grid(True, alpha=0.3)
        if row_idx == 0 and loc_idx == 1:  # Legend only on top-right
            ax.legend(loc='upper right')
        
        ax.ticklabel_format(style='plain', useOffset=False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

plt.tight_layout()
plt.show()

# ----------- RMSE CALCULATION FOR ALL LEVELS -----------
print("=" * 60)
print("RMSE")
print("=" * 60)

for val_level in validation_level:
    data = all_val_data[val_level]
    X_val_raw = data['X_val_raw']
    X_sim_mean_denorm = data['X_sim_mean_denorm']
    period_index = data['period_index']
    
    print(f"Level {val_level}:")
    rmse_results = {}
    for loc_idx in range(2):
        y_pred = X_sim_mean_denorm[loc_idx, :]
        y_true = X_val_raw[period_index, loc_idx, :]
        y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().numpy()
        y_true_np = y_true if isinstance(y_true, np.ndarray) else y_true.detach().numpy()
        rmse = np.sqrt(np.mean((y_pred_np - y_true_np)**2))
        rmse_results[f'rmse_location_{loc_idx+1}'] = rmse
        print(f"{location_names[loc_idx]} RMSE: {rmse:.6f}")
    print("-" * 60)
        


