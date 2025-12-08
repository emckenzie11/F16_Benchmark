import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from architecture import cVAE, cvae_loss
from helpers import process_acceleration_data, normalise_data

# ----------- USER CONFIGURATION -----------
# Validation level
validation_level = [2]  # Level to use for validation (Options: 2, 4, 6)

# User-configurable data-shape parameters
training_period_indices = list(range(3, 10))  # periods 3 through 9 inclusive 
downsample_factor = 2   

# Training parameters
num_epochs = 1200

# Loss weights
weight_recon = 1.0      # Reconstruction loss weight
weight_rf = 0.5         # Restoring force loss weight
weight_kl = 0.5         # KL divergence weight  

# ----------- PARAMETERS -----------
# Data levels and locations
training_level = [1, 3, 5, 7]
location_i = 2  
location_j = 3  

# Force mapping for training levels (N)
force_map = {1: 12.4, 3: 36.8, 5: 73.6, 7: 97.8}

# Force mapping for validation levels (N)
force_levels = {2: 24.6, 4: 61.4, 6: 85.7} 

# Effective mass
m_eff_j = 1.0 

# Data parameters
fs = 400             
dt = 1 / fs

# Restoring force parameters
freq_low = 6.8          # Band-pass filter low frequency
freq_high = 8.6         # Band-pass filter high frequency           

# Data shape parameters
points_per_period = 8192

# Model parameters
latent_dim = 32         
sequence_length = points_per_period // downsample_factor
n_locations = 2         
condition_dim = 1        

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
# Extract per-period samples for training. Each period becomes an independent sample.
X_raw = process_acceleration_data(training_data, training_level,
                                  location_i=location_i, location_j=location_j,
                                  points_per_period=points_per_period,
                                  downsample_factor=downsample_factor,
                                  period_indices=training_period_indices)

# Validation: keep previous behaviour (extract contiguous block or single period)
X_val_raw = process_acceleration_data(validation_data, validation_level,
                                      location_i=location_i, location_j=location_j,
                                      points_per_period=points_per_period,
                                      downsample_factor=downsample_factor,
                                      period_indices=training_period_indices)

# Expand condition vector so each period is an independent sample at the same force level
n_periods_used = len(training_period_indices)
C_raw = np.repeat([force_map[l] for l in training_level], repeats=n_periods_used)

# Normalise acceleration data and conditions
X_normalised, X_mean, X_std = normalise_data(X_raw)
C_normalised, C_mean, C_std = normalise_data(C_raw)

# Store normalisation parameters
normalisation_params = {
    'X_mean': X_mean, 'X_std': X_std,
    'C_mean': C_mean, 'C_std': C_std,
}

# Normalise validation using training statistics to keep conditioning consistent
C_val_raw = np.array([force_levels[validation_level[0]]])  # Get condition  for selected validation level
C_val_normalised = (C_val_raw - C_mean) / C_std
C_val = torch.tensor(C_val_normalised, dtype=torch.float32).unsqueeze(-1)  # Force conditions (1, 1)

# ----------- PREPARE TRAINING DATA -----------
print("-" * 60)
print(f"Training on {X_normalised.shape[0]} samples directly in physical space")
print(f"Sample shape: {X_normalised.shape[1:]} (n_locations, sequence_length)")

# Convert to PyTorch tensors - work directly in physical space
X_train = torch.tensor(X_normalised, dtype=torch.float32)  # (n_samples, 2, sequence_length)

# Expand and normalise conditioning to match training samples
C_normalised_expanded = (C_raw - C_mean) / C_std
C_train = torch.tensor(C_normalised_expanded, dtype=torch.float32).unsqueeze(-1)  # Force conditions (n_samples, 1)

# ----------- TRAINING SETUP -----------
# Ensure model input dimension matches (n_locations * sequence_length)
input_dim = n_locations * sequence_length
model = cVAE(latent_dim=latent_dim, input_dim=input_dim, n_locations=n_locations, sequence_length=sequence_length)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("-" * 60)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ----------- TRAINING -----------
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass: model learns in physical space
    X_train_recon, mu_train, logvar_train, z_train = model(X_train, C_train)
    
    # Extract restoring forces from location_j (index 1 in 2-location setup)
    RF_true = -m_eff_j*X_train[:, 1, :]  # (n_samples, sequence_length)
    RF_recon = -m_eff_j*X_train_recon[:, 1, :]  # (n_samples, sequence_length)
    
    # Compute loss using cvae_loss function with restoring force term
    total_loss, recon_loss, rf_loss, kl_loss = cvae_loss(
        X_train_recon, X_train, RF_recon, RF_true, mu_train, logvar_train,
        weight_recon=weight_recon,
        weight_rf=weight_rf,
        weight_kl=weight_kl
    )
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}: Loss={total_loss.item():.3f} Recon={recon_loss.item():.3f} RF={rf_loss.item():.3f} KL={kl_loss.item():.3f}")
        
# --------------- SIMULATE THE MODEL USING DECODER ONLY -----------
model.eval()
with torch.no_grad():
    # Generate multiple stochastic reconstructions using decoder only
    n_samples = 50
    X_sim_list = []  # Store in physical space

    for _ in range(n_samples):
        z_sim = torch.randn(1, latent_dim, device=C_val.device)
        X_sim = model.decoder(z_sim, C_val)  # Decode to physical coordinates (B, 2, 4096)
        X_sim_list.append(X_sim.cpu().numpy())

    # Stack samples
    X_sim = np.array(X_sim_list)  # (n_samples, batch, 2, 4096) in physical space

    # Calculate mean reconstruction from samples
    X_sim_mean = X_sim.mean(axis=0)                 

# --------------- DENORMALISE FOR ANALYSIS -----------
# Denormalise the mean simulation back to original scale
X_sim_mean_denorm = X_sim_mean * normalisation_params['X_std'] + normalisation_params['X_mean']

# Denormalise all simulation samples for uncertainty analysis
X_sim_samples_denorm = X_sim * normalisation_params['X_std'] + normalisation_params['X_mean']
    
# ----------- PLOTTING -----------
# Downsampled sampling frequency used for plotting (account for downsample_factor)
fs_downsampled = fs / downsample_factor
t = np.arange(sequence_length) / fs_downsampled

# Setup figure with subplots (n_locations x 2 columns)
fig, axes = plt.subplots(n_locations, 2, figsize=(15, 5 * n_locations))
fig.suptitle(f'cVAE Reconstruction: Simulated vs Validation Data\nLevel {validation_level[0]} (Force: {C_val_raw[0]:.1f}N)', fontsize=16, fontweight='bold')

# Location names for plotting (only i and j remain)
location_names = [f'Location {location_i} (Wing Side)', f'Location {location_j} (Payload Side)']

# Plot both locations
for loc_idx in range(n_locations):
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
    
    if loc_idx == n_locations - 1:  # Only bottom plots get x-label
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
    
    if loc_idx == n_locations - 1:  # Only bottom plots get x-label
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
for loc_idx in range(n_locations):
    y_pred = X_sim_mean_denorm[0, loc_idx, :]
    y_true = X_val_raw[0, loc_idx, :]
    y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().numpy()
    y_true_np = y_true if isinstance(y_true, np.ndarray) else y_true.detach().numpy()
    rmse = np.sqrt(np.mean((y_pred_np - y_true_np)**2))
    rmse_results[f'rmse_location_{loc_idx+1}'] = rmse
    print(f"{location_names[loc_idx]} RMSE: {rmse:.6f}")
    
print("-" * 60)

# ----------- SAVE RESULTS TO TRACKER -----------
from helpers import save_results_to_tracker

# Collect all experiment parameters and results
results_dict = {
    # Model parameters
    'latent_dim': latent_dim,
    'sequence_length': sequence_length,
    'n_locations': n_locations,
    'condition_dim': condition_dim,
    
    # Training parameters
    'num_epochs': num_epochs,
    'learning_rate': 1e-3,
    'weight_recon': weight_recon,
    'weight_rf': weight_rf,
    'weight_kl': weight_kl,
    
    # Data configuration
    'training_levels': str(training_level),
    'validation_level': validation_level[0],
    'validation_force': C_val_raw[0],
    'location_i': location_i,
    'location_j': location_j,
    
    # Results
    **rmse_results,
    'total_rmse': np.mean(list(rmse_results.values())),
    
    # Model info
    'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    'loss_computation': 'physical_space_with_RF',
    
    # Final training loss (if available)
    'final_total_loss': total_loss.item() if 'total_loss' in locals() else None,
    'final_recon_loss': recon_loss.item() if 'recon_loss' in locals() else None,
    'final_rf_loss': rf_loss.item() if 'rf_loss' in locals() else None,
    'final_kl_loss': kl_loss.item() if 'kl_loss' in locals() else None,
}

save_results_to_tracker(results_dict)
    