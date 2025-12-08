import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from architecture import cVAE, cvae_loss
from helpers import process_data, normalise_data, downsample_signal

# ----------- USER CONFIGURATION -----------
# Validation level
validation_level = [6]  # Level to use for validation (Options: 2, 4, 6)

# User-configurable data-shape parameters
training_period_indices = list(range(3, 10))  # periods 3 through 9 inclusive 
downsample_factor_accel = 2   # Downsampling for acceleration signals
downsample_factor_force = 8   # Downsampling for force signal 

# Training parameters
num_epochs = 1200
beta = 0.5  # Weight for KL divergence in cVAE loss

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
sequence_length_accel = points_per_period // downsample_factor_accel
sequence_length_force = points_per_period // downsample_factor_force         

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

# ----------- PROCESS DATA -----------
# Extract per-period samples for training
X_raw, F_raw = process_data(training_data, training_level,
                             period_indices=training_period_indices)

# Validation data
X_val_raw, F_val_raw = process_data(validation_data, validation_level,
                                     period_indices=training_period_indices)

# Downsample signals
X_raw = downsample_signal(X_raw, downsample_factor_accel, axis=-1)
F_raw = downsample_signal(F_raw, downsample_factor_force, axis=-1)
X_val_raw = downsample_signal(X_val_raw, downsample_factor_accel, axis=-1)
F_val_raw = downsample_signal(F_val_raw, downsample_factor_force, axis=-1)

# Normalise acceleration data and force signals
X_normalised, X_mean, X_std = normalise_data(X_raw)
F_normalised, F_mean, F_std = normalise_data(F_raw)

# Store normalisation parameters
normalisation_params = {
    'X_mean': X_mean, 'X_std': X_std,
    'F_mean': F_mean, 'F_std': F_std,
}

# Normalise validation force using training statistics
F_val_normalised = (F_val_raw - F_mean) / F_std
F_val = torch.tensor(F_val_normalised, dtype=torch.float32)  # (n_val_samples, sequence_length)

# ----------- PREPARE TRAINING DATA -----------
print("-" * 60)
print(f"Training Data Shape: {X_normalised.shape}")
print(f"Force Condition shape: {F_normalised.shape}")

# Convert to PyTorch tensors
X_train = torch.tensor(X_normalised, dtype=torch.float32)  # (n_samples, 2, sequence_length)
F_train = torch.tensor(F_normalised, dtype=torch.float32)  # (n_samples, sequence_length)

# ----------- TRAINING SETUP -----------
model = cVAE(latent_dim=latent_dim, n_locations=2, 
             sequence_length_accel=sequence_length_accel, 
             sequence_length_force=sequence_length_force)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("-" * 60)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ----------- TRAINING -----------
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass: model learns in physical space with force signal conditioning
    X_train_recon, mu_train, logvar_train, z_train = model(X_train, F_train)
    
    # Compute cVAE loss in physical space
    total_loss, recon_loss, kl_loss = cvae_loss(X_train_recon, X_train, mu_train, logvar_train, beta=beta)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}: Loss={total_loss.item():.3f} Recon={recon_loss.item():.3f} KL={kl_loss.item():.3f}")
        
# --------------- SIMULATE THE MODEL USING DECODER ONLY -----------
model.eval()
with torch.no_grad():
    # Generate multiple stochastic reconstructions using decoder only
    n_samples = 50
    X_sim_list = []  # Store in physical space

    # Use first validation sample's force signal for conditioning
    F_val_single = F_val[0:1]  # (1, sequence_length)

    for _ in range(n_samples):
        z_sim = torch.randn(1, latent_dim, device=F_val_single.device)
        X_sim = model.decoder(z_sim, F_val_single)  # Decode to physical coordinates (B, 2, sequence_length)
        X_sim_list.append(X_sim.cpu().numpy())

    # Stack samples
    X_sim = np.array(X_sim_list)  # (n_samples, batch, 2, sequence_length) in physical space

    # Calculate mean reconstruction from samples
    X_sim_mean = X_sim.mean(axis=0)                 

# --------------- DENORMALISE FOR ANALYSIS -----------
# Denormalise the mean simulation back to original scale
X_sim_mean_denorm = X_sim_mean * normalisation_params['X_std'] + normalisation_params['X_mean']

# Denormalise all simulation samples for uncertainty analysis
X_sim_samples_denorm = X_sim * normalisation_params['X_std'] + normalisation_params['X_mean']
    
# ----------- PLOTTING -----------
# Downsampled sampling frequency used for plotting (account for downsample_factor_accel)
fs_downsampled = fs / downsample_factor_accel
t = np.arange(sequence_length_accel) / fs_downsampled

# Setup figure with subplots (2 locations x 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'cVAE Reconstruction: Simulated vs Validation Data\nLevel {validation_level[0]}', fontsize=16, fontweight='bold')

# Location names for plotting (only i and j remain)
location_names = [f'Location {location_i} (Wing Side)', f'Location {location_j} (Payload Side)']

# Plot both locations
for loc_idx in range(2):
    # Time domain plot with variance
    ax_time = axes[loc_idx, 0]
    
    # Plot validation (true) data
    ax_time.plot(t, X_val_raw[0, loc_idx, :], 
                 color='black', linewidth=1.5, label='Validation (True)', alpha=0.8)
    
    # Plot mean simulation
    ax_time.plot(t, X_sim_mean_denorm[0, loc_idx, :], 
                 color='red', linewidth=1.5, label='cVAE Mean', linestyle='--', alpha=0.9)
    
    # Plot variance as confidence bands (use simulated samples)
    sim_sample_std = X_sim_samples_denorm[:, 0, loc_idx, :].std(axis=0)
    
    ax_time.fill_between(t, X_sim_mean_denorm[0, loc_idx, :] - sim_sample_std, X_sim_mean_denorm[0, loc_idx, :] + sim_sample_std,
                         color='red', alpha=0.2, label='±1σ Uncertainty')
    ax_time.fill_between(t, X_sim_mean_denorm[0, loc_idx, :] - 2*sim_sample_std, X_sim_mean_denorm[0, loc_idx, :] + 2*sim_sample_std,
                         color='red', alpha=0.1, label='±2σ Uncertainty')
    
    if loc_idx == 1:  # Only bottom plots get x-label
        ax_time.set_xlabel('Time [s]')
    ax_time.set_ylabel('Acceleration [m/s²]')
    ax_time.set_title(f'{location_names[loc_idx]}')
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc='upper right')
    
    # Zoomed time domain plot (centered at midpoint ±1 second)
    ax_zoom = axes[loc_idx, 1]

    # Compute window of ±0.5 second around the midpoint (using downsampled fs)
    half_window = int(0.5 * fs_downsampled)  # samples per 0.5 second at downsampled rate
    mid_idx = len(t) // 2
    start_idx = max(mid_idx - half_window, 0)
    end_idx = min(mid_idx + half_window, len(t))

    t_zoom = t[start_idx:end_idx]
    
    # Plot zoomed validation (true) data
    ax_zoom.plot(t_zoom, X_val_raw[0, loc_idx, start_idx:end_idx], 
                 color='black', linewidth=1.5, label='Validation (True)', alpha=0.8)
    
    # Plot zoomed mean simulation
    ax_zoom.plot(t_zoom, X_sim_mean_denorm[0, loc_idx, start_idx:end_idx], 
                 color='red', linewidth=1.5, label='cVAE Mean', linestyle='--', alpha=0.9)
    
    # Plot zoomed uncertainty bands
    zoom_sample_mean = X_sim_mean_denorm[0, loc_idx, start_idx:end_idx]
    zoom_sample_std = sim_sample_std[start_idx:end_idx]
    
    ax_zoom.fill_between(t_zoom, zoom_sample_mean - zoom_sample_std, zoom_sample_mean + zoom_sample_std,
                         color='red', alpha=0.2, label='±1σ Uncertainty')
    ax_zoom.fill_between(t_zoom, zoom_sample_mean - 2*zoom_sample_std, zoom_sample_mean + 2*zoom_sample_std,
                         color='red', alpha=0.1, label='±2σ Uncertainty')
    
    if loc_idx == 1:  # Only bottom plots get x-label
        ax_zoom.set_xlabel('Time [s]')
    ax_zoom.set_ylabel('Acceleration [m/s²]')
    ax_zoom.set_title(f'{location_names[loc_idx]}')
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc='upper right')

# Set consistent number formatting for all plots
for ax in axes.flat:
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

plt.tight_layout()
plt.show()

# ----------- RMSE CALCULATION -----------
print("-" * 60)
rmse_results = {}
for loc_idx in range(2):
    y_pred = X_sim_mean_denorm[0, loc_idx, :]
    y_true = X_val_raw[0, loc_idx, :]
    y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().numpy()
    y_true_np = y_true if isinstance(y_true, np.ndarray) else y_true.detach().numpy()
    rmse = np.sqrt(np.mean((y_pred_np - y_true_np)**2))
    rmse_results[f'rmse_location_{loc_idx+1}'] = rmse
    print(f"{location_names[loc_idx]} RMSE: {rmse:.6f}")
    
print("-" * 60)

    