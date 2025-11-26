import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from architecture import cVAE, cvae_loss
from helpers import process_acceleration_data, normalize_data

# ----------- USER CONFIGURATION -----------
# Data levels and locations
training_level = [1, 3, 5, 7]
validation_level = [2]
location_i = 2  # DOF i (wing side of NL connection)
location_j = 3  # DOF j (payload side of NL connection)
force_levels = {2: 24.6, 4: 61.4, 6: 85.7}  # Level to force mapping 

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step

# Model parameters
latent_dim = 64          # Size of latent space
sequence_length = 4096   # Downsampled length
n_locations = 2          # Number of acceleration locations
condition_dim = 1        # Single force amplitude condition

# Training parameters
num_epochs = 1200
beta_max = 0.04
warmup_epochs = 100  # Number of epochs to reach full beta

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
X_raw = process_acceleration_data(training_data, training_level, location_i=location_i, location_j=location_j)
X_val_raw = process_acceleration_data(validation_data, validation_level, location_i=location_i, location_j=location_j)

C_raw = np.array([12.4, 36.8, 73.6, 97.8])  # Force levels for conditioning  
C_val_raw = np.array([force_levels[validation_level[0]]])  # Get condition  for selected validation level
 
# Normalise acceleration data and conditions
X_normalized, X_mean, X_std = normalize_data(X_raw)
C_normalized, C_mean, C_std = normalize_data(C_raw)
X_val_normalized, X_val_mean, X_val_std = normalize_data(X_val_raw)
C_val_normalized, C_val_mean, C_val_std = normalize_data(C_val_raw)

# Store normalisation parameters
normalisation_params = {
    'X_mean': X_mean, 'X_std': X_std,
    'C_mean': C_mean, 'C_std': C_std,
    'X_val_mean': X_val_mean, 'X_val_std': X_val_std,
    'C_val_mean': C_val_mean, 'C_val_std': C_val_std,
}

C_val = torch.tensor(C_val_normalized, dtype=torch.float32).unsqueeze(-1)  # Force conditions (1, 1)
X_val = torch.tensor(X_val_normalized, dtype=torch.float32)  # Normalized accelerations (1, 2, 4096)

# ----------- POD COMPUTATION -----------
# Concatenate acceleration data from all training levels
# X_normalized shape: (4, 2, 4096) -> extract each level and concatenate
X1 = X_normalized[0]  # Level 1: (2, 4096)
X3 = X_normalized[1]  # Level 3: (2, 4096)
X5 = X_normalized[2]  # Level 5: (2, 4096)
X7 = X_normalized[3]  # Level 7: (2, 4096)

S = np.concatenate([X1, X3, X5, X7], axis=1)

# Compute SVD: S = U * Σ * V^T
U, Sigma, VT = np.linalg.svd(S, full_matrices=False)

# POD Modes
print(f"U: {U}")
energies = Sigma**2
energy_frac = energies / energies.sum()
print(f"Mode energy fraction: {energy_frac}")

# Modal coefficients & stack
a1 = U.T @ X1      # shape (2, 4096)
a3 = U.T @ X3
a5 = U.T @ X5
a7 = U.T @ X7
A = np.stack([a1, a3, a5, a7], axis=0)  # Shape: (4, 2, 4096)

# Convert to PyTorch tensors
U_torch = torch.tensor(U, dtype=torch.float32)
A_train = torch.tensor(A, dtype=torch.float32)  # Modal coefficients (4, 2, 4096)
C_train = torch.tensor(C_normalized, dtype=torch.float32).unsqueeze(-1)  # Force conditions (4, 1)

# ----------- TRAINING SETUP -----------
model = cVAE(latent_dim=latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"Training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters for {num_epochs} epochs")
print(f"Beta warmup: 0 → {beta_max} over {warmup_epochs} epochs")

# ----------- TRAINING -----------
for epoch in range(num_epochs):
    model.train()
    
    # Beta warmup schedule
    if epoch < warmup_epochs:
        beta = beta_max * (epoch / warmup_epochs)  # Linear warmup
    else:
        beta = beta_max
    
    # Forward pass: model learns in POD modal space
    A_train_recon, mu_train, logvar_train, z_train = model(A_train, C_train)
    
    # Transform reconstructed modal coordinates back to physical space
    X_train_recon = torch.einsum('cr,brt->bct', U_torch, A_train_recon)  # (B, 2, 4096)
    
    # Convert normalized training data to tensor for loss computation
    X_train_phys = torch.tensor(X_normalized, dtype=torch.float32)  # (4, 2, 4096)
    
    # Loss in physical space (acceleration coordinates)
    total_loss, recon_loss, kl_loss = cvae_loss(X_train_recon, X_train_phys, mu_train, logvar_train, beta)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss.item():.1f} (Recon = {recon_loss.item():.1f}, KL = {kl_loss.item():.1f}, β = {beta:.3f})")
        
# --------------- SIMULATE THE MODEL USING DECODER ONLY -----------
model.eval()
with torch.no_grad():
    # Generate multiple stochastic reconstructions using decoder only
    n_samples = 50
    X_sim_modal = []  # Store in modal space
    X_sim_phys = []   # Store in physical space

    for _ in range(n_samples):
        z_sim = torch.randn(X_val.shape[0], latent_dim, device=C_val.device)  # Sample from prior ~ N(0, I)
        a_sim = model.decoder(z_sim, C_val)  # Decode to modal coordinates (B, 2, 4096)
        
        # Reconstruct back to physical DOFs: X = U @ a
        X_sim = torch.einsum('cr,brt->bct', U_torch, a_sim)  # (B, 2, 4096)
        
        X_sim_modal.append(a_sim.cpu().numpy())
        X_sim_phys.append(X_sim.cpu().numpy())

    # Stack samples
    X_sim_modal = np.array(X_sim_modal)  # (n_samples, batch, 2, 4096) in modal space
    X_sim = np.array(X_sim_phys)          # (n_samples, batch, 2, 4096) in physical space

    # Calculate mean reconstruction from samples (in physical space)
    X_sim_mean = X_sim.mean(axis=0)                 

# --------------- DENORMALIZE FOR ANALYSIS -----------
# Denormalize the mean simulation back to original scale
X_sim_mean_denorm = X_sim_mean * normalisation_params['X_std'] + normalisation_params['X_mean']

# Denormalize all simulation samples for uncertainty analysis  
X_sim_samples_denorm = X_sim * normalisation_params['X_std'] + normalisation_params['X_mean']
    
# ----------- PLOTTING -----------
fs_downsampled = fs / 2
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
rmse_results = {}
for loc_idx in range(n_locations):
    y_pred = X_sim_mean_denorm[0, loc_idx, :]
    y_true = X_val_raw[0, loc_idx, :]
    y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().numpy()
    y_true_np = y_true if isinstance(y_true, np.ndarray) else y_true.detach().numpy()
    rmse = np.sqrt(np.mean((y_pred_np - y_true_np)**2))
    rmse_results[f'rmse_location_{loc_idx+1}'] = rmse
    print(f"{location_names[loc_idx]} RMSE: {rmse:.6f}")

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
    'beta_max': beta_max,
    'warmup_epochs': warmup_epochs,
    'learning_rate': 1e-3,
    
    # Data configuration
    'training_levels': str(training_level),
    'validation_level': validation_level[0],
    'validation_force': C_val_raw[0],
    'location_h': None,
    'location_i': location_i,
    'location_j': location_j,
    
    # Results
    **rmse_results,
    'total_rmse': np.mean(list(rmse_results.values())),
    
    # Model info
    'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    'loss_computation': 'physical_space',
    
    # Final training loss (if available)
    'final_total_loss': total_loss.item() if 'total_loss' in locals() else None,
    'final_recon_loss': recon_loss.item() if 'recon_loss' in locals() else None,
    'final_kl_loss': kl_loss.item() if 'kl_loss' in locals() else None,
}

save_results_to_tracker(results_dict)
    
