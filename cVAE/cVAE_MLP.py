import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ----------- USER CONFIGURATION -----------
training_level = [1, 3, 5, 7] 
validation_level = [6]
location_i = 2  # DOF i (wing side of NL connection)
location_j = 3  # DOF j (payload side of NL connection)

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step

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

# ----------- PROCESS TRAINING DATA -----------
X = []  # Will store acceleration matrices
C = np.array([12.4, 36.8, 73.6, 97.8])  # Force levels for conditioning 

for level in training_level:
    accel_matrix = np.vstack([
        training_data[level][f'Acceleration{location_i}'].to_numpy(),  # Row 0: Location i
        training_data[level][f'Acceleration{location_j}'].to_numpy(),  # Row 1: Location j  
        ])
    
    # Extract second to last period and downsample
    points_per_period = 8192
    start_idx = 7 * points_per_period
    end_idx = 8 * points_per_period # Second to last period

    accel_matrix = accel_matrix[:, start_idx:end_idx][:, ::2]  # Extract 8th period and downsample by factor of 2 
    
    X.append(accel_matrix)

# Convert to 3D array
X = np.array(X)   
 
# Normalise acceleration data
X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std

# Normalise conditions 
C_mean = np.mean(C)
C_std = np.std(C)
C_normalized = (C - C_mean) / C_std

# Store normalisation parameters
normalisation_params = {
    'X_mean': X_mean, 'X_std': X_std,
    'C_mean': C_mean, 'C_std': C_std
}

# Convert to PyTorch format
X_train_torch = torch.FloatTensor(X_normalized)
C_train_torch = torch.FloatTensor(C_normalized.reshape(-1, 1))  # (4,) → (4,1) for proper input

# ----------- PARAMETERS -----------
latent_dim = 32          # Size of latent space
sequence_length = 4096   # Downsampled length
n_locations = 2          # Number of acceleration locations
condition_dim = 1        # Single force amplitude condition

# ----------- ENCODER ARCHITECTURE -----------
class Encoder(nn.Module):
    def __init__(self, input_dim=8192, condition_dim=1, latent_dim=32):
        super().__init__()
        
        # Acceleration branch
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)

        # Condition branch
        self.fc_cond = nn.Linear(condition_dim, 16)

        # Combined layers
        self.fc_combined = nn.Linear(256 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, c):
        # Flatten acceleration
        x = x.view(x.size(0), -1)           # (batch, 8192)

        # Acceleration MLP
        x = F.relu(self.fc1(x))             # (batch, 1024)
        x = F.relu(self.fc2(x))             # (batch, 256)

        # Condition MLP
        c = F.relu(self.fc_cond(c))         # (batch, 16)

        # Combine
        h = torch.cat([x, c], dim=1)        # (batch, 272)
        h = F.relu(self.fc_combined(h))     # (batch, 128)

        # Latent mean and variance
        mu = self.fc_mu(h)                  # (batch, latent_dim)
        logvar = self.fc_logvar(h)          # (batch, latent_dim)

        return mu, logvar

# ----------- DECODER ARCHITECTURE -----------
class Decoder(nn.Module):
    def __init__(self, output_dim=8192, condition_dim=1, latent_dim=32):
        super().__init__()

        # Process latent + condition together
        self.fc_cond = nn.Linear(condition_dim, 16)
        self.fc_z = nn.Linear(latent_dim, 64)
        self.fc_combined = nn.Linear(64 + 16, 256)

        # MLP to expand to output size
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, output_dim)

    def forward(self, z, c):
        # Condition embedding
        c = F.relu(self.fc_cond(c))         # (batch, 16)
        
        # Latent embedding
        z = F.relu(self.fc_z(z))            # (batch, 64)

        # Combine
        h = torch.cat([z, c], dim=1)        # (batch, 80)
        h = F.relu(self.fc_combined(h))     # (batch, 256)

        # MLP expansion
        h = F.relu(self.fc1(h))             # (batch, 1024)
        x_hat = self.fc2(h)                 # (batch, 8192)

        # Reshape back to (batch, 2, 4096)
        return x_hat.view(z.size(0), 2, 4096)

# ----------- cVAE ARCHITECTURE -----------
class cVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
    
    def reparameterize(self, mu, logvar):
        # Reparameterisation trick: z = μ + σ·ε
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, c):
        # Encode inputs to latent distribution
        mu, logvar = self.encoder(x, c)

        # Sample latent vector
        z = self.reparameterize(mu, logvar)

        # Decode latent vector to reconstruction
        x_recon = self.decoder(z, c)

        return x_recon, mu, logvar, z

# ----------- LOSS FUNCTION -----------
def cvae_loss(x_recon, x, mu, logvar, beta=1.0):
    """cVAE loss: reconstruction + KL divergence"""
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence loss: KL(q(z|x,c) || p(z))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

# ----------- TRAINING SETUP -----------
model = cVAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 2000
beta = 0.1
print(f"Training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters for {num_epochs} epochs")

# ----------- TRAINING -----------
for epoch in range(num_epochs):
    model.train()
    x_train_recon, mu_train, logvar_train, z_train = model(X_train_torch, C_train_torch)
    total_loss, recon_loss, kl_loss = cvae_loss(x_train_recon, X_train_torch, mu_train, logvar_train, beta)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss.item():.1f} (Recon = {recon_loss.item():.1f}, KL = {kl_loss.item():.1f})")

# ----------- VALIDATION DATA PROCESSING (FOR ERROR CALCULATION) -----------
# Initialise
X_val = []  # Will store acceleration matrices

for level in validation_level:
    accel_matrix = np.vstack([
        validation_data[level][f'Acceleration{location_i}'].to_numpy(),  # Row 0: Location i
        validation_data[level][f'Acceleration{location_j}'].to_numpy(),  # Row 1: Location j  
        ])
    
    # Extract second to last period and downsample
    points_per_period = 8192
    start_idx = 7 * points_per_period
    end_idx = 8 * points_per_period # Second to last period

    accel_matrix = accel_matrix[:, start_idx:end_idx][:, ::2]  # Extract 8th period and downsample by factor of 2 
    
    X_val.append(accel_matrix)

# Convert to arrays and normalize using TRAINING parameters 
X_val_raw = np.array(X_val)
X_val_normalized = (X_val_raw - normalisation_params['X_mean']) / normalisation_params['X_std']

# Validation force levels and normalize using TRAINING parameters  
force_levels = {2: 24.6, 4: 61.4, 6: 85.7}  # Level to force mapping
C_val_raw = np.array([force_levels[validation_level[0]]])  # Get force for selected validation level
C_val_normalized = (C_val_raw - normalisation_params['C_mean']) / normalisation_params['C_std']

# Convert to PyTorch tensors
X_val_torch = torch.FloatTensor(X_val_normalized)
C_val_torch = torch.FloatTensor(C_val_normalized.reshape(-1, 1))

# --------------- SIMULATE THE MODEL -----------
model.eval()
with torch.no_grad():

    # Encode to get latent distribution parameters
    mu_val, logvar_val = model.encoder(X_val_torch, C_val_torch)
    std_val = torch.exp(0.5 * logvar_val)

    # Generate multiple stochastic reconstruction
    n_samples = 50
    val_samples = []

    for _ in range(n_samples):
        eps = torch.randn_like(std_val)          # Sample epsilon
        z_val_sample = mu_val + eps * std_val    # Sample z
        x_val_sample = model.decoder(z_val_sample, C_val_torch)  # Decode sample

        val_samples.append(x_val_sample.cpu().numpy())

    # Stack samples: (n_samples, batch, 2, 4096)
    val_samples = np.array(val_samples)

    # Mean reconstruction using sample mean
    x_val_recon_mean = val_samples.mean(axis=0)         # (batch, 2, 4096)

    # Calculte deterministic mean 
    x_val_mean = model.decoder(mu_val, C_val_torch)     # (batch, 2, 4096)

    # Compare deterministic vs sample-based means
    reconstruction_rmse = np.sqrt(np.mean((x_val_recon_mean - x_val_mean.cpu().numpy())**2))
    print(f"RMSE between deterministic and sample-based reconstruction: {reconstruction_rmse:.6f}")
    print(f"Deterministic mean range: [{x_val_mean.min().item():.4f}, {x_val_mean.max().item():.4f}]")
    print(f"Sample-based mean range: [{x_val_recon_mean.min():.4f}, {x_val_recon_mean.max():.4f}]")

    # Standard deviation for uncertainty bands
    x_val_recon_std = val_samples.std(axis=0)           # (batch, 2, 4096)

    # For loss calculation, use the mean reconstruction
    # Convert mean back to torch
    x_val_recon_mean_torch = torch.from_numpy(x_val_recon_mean).to(X_val_torch.device)

    val_total_loss, val_recon_loss, val_kl_loss = cvae_loss(x_val_recon_mean_torch, X_val_torch, mu_val, logvar_val, beta)

# --------------- DENORMALIZE FOR ANALYSIS -----------
# Denormalize validation data back to original scale
X_val_true = X_val_torch * normalisation_params['X_std'] + normalisation_params['X_mean']
X_val_recon = x_val_recon_mean * normalisation_params['X_std'] + normalisation_params['X_mean']

# Denormalize samples for uncertainty analysis
val_samples_denorm = val_samples * normalisation_params['X_std'] + normalisation_params['X_mean']

# ----------- PLOTTING -----------
fs_downsampled = fs / 2
t = np.arange(sequence_length) / fs_downsampled

# Setup figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'cVAE Reconstruction: Simulated vs Validation Data\nLevel 2 (Force: 24.6N)', fontsize=16, fontweight='bold')

# Location names for plotting
location_names = [f'Location {location_i} (Wing Side)', f'Location {location_j} (Payload Side)']

# Plot both locations
for loc_idx in range(n_locations):
    # Time domain plot with variance
    ax_time = axes[loc_idx, 0]
    
    # Plot validation (true) data
    ax_time.plot(t, X_val_true[0, loc_idx, :], 
                 color='black', linewidth=1.5, label='Validation (True)', alpha=0.8)
    
    # Plot mean reconstruction
    ax_time.plot(t, X_val_recon[0, loc_idx, :], 
                 color='red', linewidth=1.5, label='cVAE Mean', linestyle='--', alpha=0.9)
    
    # Plot variance as confidence bands (use pre-computed samples)
    val_sample_mean = val_samples_denorm[:, 0, loc_idx, :].mean(axis=0)
    val_sample_std = val_samples_denorm[:, 0, loc_idx, :].std(axis=0)
    
    ax_time.fill_between(t, val_sample_mean - val_sample_std, val_sample_mean + val_sample_std,
                         color='red', alpha=0.2, label='±1σ Uncertainty')
    ax_time.fill_between(t, val_sample_mean - 2*val_sample_std, val_sample_mean + 2*val_sample_std,
                         color='red', alpha=0.1, label='±2σ Uncertainty')
    
    if loc_idx == 1:  # Only bottom plots get x-label
        ax_time.set_xlabel('Time [s]')
    ax_time.set_ylabel('Acceleration [m/s²]')
    ax_time.set_title(f'{location_names[loc_idx]}')
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()
    
    # Zoomed time domain plot (1 second of data)
    ax_zoom = axes[loc_idx, 1]
    
    # Select 1 second of data (200 samples at 200 Hz)
    zoom_samples = int(1.0 * fs_downsampled)  # 200 samples for 1 second
    start_idx = len(t) // 4  # Start at 1/4 through the signal
    end_idx = start_idx + zoom_samples
    
    t_zoom = t[start_idx:end_idx]
    
    # Plot zoomed validation (true) data
    ax_zoom.plot(t_zoom, X_val_true[0, loc_idx, start_idx:end_idx], 
                 color='black', linewidth=1.5, label='Validation (True)', alpha=0.8)
    
    # Plot zoomed mean reconstruction
    ax_zoom.plot(t_zoom, X_val_recon[0, loc_idx, start_idx:end_idx], 
                 color='red', linewidth=1.5, label='cVAE Mean', linestyle='--', alpha=0.9)
    
    # Plot zoomed uncertainty bands
    zoom_sample_mean = val_sample_mean[start_idx:end_idx]
    zoom_sample_std = val_sample_std[start_idx:end_idx]
    
    ax_zoom.fill_between(t_zoom, zoom_sample_mean - zoom_sample_std, zoom_sample_mean + zoom_sample_std,
                         color='red', alpha=0.2, label='±1σ Uncertainty')
    ax_zoom.fill_between(t_zoom, zoom_sample_mean - 2*zoom_sample_std, zoom_sample_mean + 2*zoom_sample_std,
                         color='red', alpha=0.1, label='±2σ Uncertainty')
    
    if loc_idx == 1:  # Only bottom plots get x-label
        ax_zoom.set_xlabel('Time [s]')
    ax_zoom.set_ylabel('Acceleration [m/s²]')
    ax_zoom.set_title(f'{location_names[loc_idx]}')
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend()

# Set consistent number formatting for all plots
for ax in axes.flat:
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

plt.tight_layout()
plt.show()

# ----------- RMSE CALCULATION -----------
for loc_idx in range(n_locations):
    y_pred = X_val_recon[0, loc_idx, :]
    y_true = X_val_true[0, loc_idx, :]
    y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().numpy()
    y_true_np = y_true if isinstance(y_true, np.ndarray) else y_true.detach().numpy()
    rmse = np.sqrt(np.mean((y_pred_np - y_true_np)**2))
    print(f"{location_names[loc_idx]} RMSE: {rmse:.6f}")
    
