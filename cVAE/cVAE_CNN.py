import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ----------- USER CONFIGURATION -----------
training_level = [1, 3, 5, 7] 
validation_level = [2]
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
X_train = torch.FloatTensor(X_normalized)
C_train = torch.FloatTensor(C_normalized) 
C_train = C_train.unsqueeze(1)  # (4,) → (4,1) for proper input

# ----------- PARAMETERS -----------
latent_dim = 32          # Size of latent space
sequence_length = 4096   # Downsampled length
n_locations = 2          # Number of acceleration locations
condition_dim = 1        # Single force amplitude condition

# ----------- ENCODER ARCHITECTURE -----------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Single convolution for acceleration processing
        self.conv1 = nn.Conv1d(n_locations, 32, kernel_size=16, padding=8)  # (n_locations,sequence_length) → (32,sequence_length)
        self.pool1 = nn.MaxPool1d(8)                              # (32,4096) → (32,512)
        self.global_pool = nn.AdaptiveAvgPool1d(1)                # (32,512) → (32,1) → (32,)
        
        # Condition processing
        self.cond_fc = nn.Linear(condition_dim, 16)               # (condition_dim,) → (16,)
        
        # Combine and output
        self.fc_combined = nn.Linear(32 + 16, 64)                 # (48,) → (64,)
        self.fc_mu = nn.Linear(64, latent_dim)                    # (64,) → (32,)
        self.fc_logvar = nn.Linear(64, latent_dim)                # (64,) → (32,)
    
    def forward(self, x, c):
        # Branch 1: Process acceleration sequences
        x = F.relu(self.conv1(x))          # (batch, 32, 4096)
        x = self.pool1(x)                  # (batch, 32, 512)
        x = self.global_pool(x)            # (batch, 32, 1)
        x = x.view(x.size(0), -1)          # (batch, 32) - Flatten
        
        # Branch 2: Process condition
        c = F.relu(self.cond_fc(c))        # (batch, 16)
        
        # Combine branches
        combined = torch.cat([x, c], dim=1)  # (batch, 48) 
        combined = F.relu(self.fc_combined(combined))  # (batch, 64)
        
        # Output latent parameters
        mu = self.fc_mu(combined)          # (batch, 32)
        logvar = self.fc_logvar(combined)  # (batch, 32)
        
        return mu, logvar

# ----------- DECODER ARCHITECTURE -----------
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Combine latent + condition
        self.fc_input = nn.Linear(latent_dim + condition_dim, 64)  # (33,) → (64,)
        
        # Expand to feature maps
        self.fc_expand = nn.Linear(64, 32 * 512)  # (64,) → (16384,)
        
        # Upsample back to original sequence length
        self.upsample = nn.Upsample(scale_factor=8, mode='linear', align_corners=False)  # (32,512) → (32,4096)
        
        # Final convolution to get back to n_locations acceleration channels  
        self.final_conv = nn.Conv1d(32, n_locations, kernel_size=16, padding=8)  # (32,sequence_length) → (n_locations,sequence_length+1)
    
    def forward(self, z, c):
        # Combine latent sample with condition
        combined = torch.cat([z, c], dim=1)  # (batch, 33)
        
        # Process through dense layers
        x = F.relu(self.fc_input(combined))   # (batch, 64)
        x = F.relu(self.fc_expand(x))         # (batch, 16384)
        
        # Reshape to feature maps
        x = x.view(x.size(0), 32, 512)       # (batch, 32, 512)
        
        # Upsample to full sequence length
        x = self.upsample(x)                 # (batch, 32, 4096)
        
        # Final convolution to reconstruction
        x = self.final_conv(x)               # (batch, n_locations, sequence_length+1)
        x = x[:, :, :sequence_length]        # Crop to (batch, n_locations, sequence_length)
        
        return x

# Quick architecture validation
print(f"Data shapes: X={X_train.shape}, C={C_train.shape}")

# ----------- cVAE ARCHITECTURE -----------
class cVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE sampling"""
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        eps = torch.randn_like(std)    # Sample noise from N(0,1)
        z = mu + eps * std             # z = μ + ε·σ
        return z
    
    def forward(self, x, c):
        """Complete forward pass: encode → sample → decode"""
        # Encode
        mu, logvar = self.encoder(x, c)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z, c)
        
        return x_recon, mu, logvar
    
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
num_epochs = 1000
beta = 0.5
print(f"Training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters for {num_epochs} epochs")

# ----------- TRAINING -----------
for epoch in range(num_epochs):
    model.train()
    x_recon, mu, logvar = model(X_train, C_train)
    total_loss, recon_loss, kl_loss = cvae_loss(x_recon, X_train, mu, logvar, beta)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}: Loss={total_loss.item():.1f} (Recon={recon_loss.item():.1f}, KL={kl_loss.item():.1f})")

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
X_val = np.array(X_val)
X_val_normalized = (X_val - normalisation_params['X_mean']) / normalisation_params['X_std']

# Validation force levels and normalize using TRAINING parameters  
C_val = np.array([24.6])  # Level 2 force amplitude
C_val_normalized = (C_val - normalisation_params['C_mean']) / normalisation_params['C_std']

# Convert to PyTorch tensors
X_val_torch = torch.FloatTensor(X_val_normalized)
C_val_torch = torch.FloatTensor(C_val_normalized.reshape(-1, 1))

# --------------- SIMULATE THE MODEL -----------
print(f"\n🧪 Testing cVAE on Validation Data (Level {validation_level[0]})...")

model.eval()
with torch.no_grad():
    # Forward pass through trained model
    x_val_recon, mu_val, logvar_val = model(X_val_torch, C_val_torch)
    
    # Compute validation losses
    val_total_loss, val_recon_loss, val_kl_loss = cvae_loss(x_val_recon, X_val_torch, mu_val, logvar_val, beta)
    
    # Compare with training performance
    model.train()
    with torch.no_grad():
        x_train_recon, mu_train, logvar_train = model(X_train, C_train)
        train_total_loss, train_recon_loss, train_kl_loss = cvae_loss(x_train_recon, X_train, mu_train, logvar_train, beta)


# --------------- DENORMALIZE FOR ANALYSIS -----------
print(f"\n🔄 Denormalizing results for physical interpretation...")

# Denormalize validation data back to original scale
X_val_original = X_val_torch * normalisation_params['X_std'] + normalisation_params['X_mean']
X_val_recon_original = x_val_recon * normalisation_params['X_std'] + normalisation_params['X_mean']

# ----------- PLOTTING -----------
fs_downsampled = fs / 2
t = np.arange(sequence_length) / fs_downsampled

# Setup figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'cVAE Reconstruction: Simulated vs Validation Data\nLevel 2 (Force: 24.6N)', fontsize=16, fontweight='bold')

# Location names for plotting
location_names = [f'Location {location_i} (Wing Side)', f'Location {location_j} (Payload Side)']
colors = ['#1f77b4', '#ff7f0e']  # Blue and orange

# Plot both locations
for loc_idx in range(n_locations):
    # Time domain plot
    ax_time = axes[loc_idx, 0]
    ax_time.plot(t, X_val_original[0, loc_idx, :], 
                 color=colors[loc_idx], linewidth=1.5, label='Validation (True)', alpha=0.8)
    ax_time.plot(t, X_val_recon_original[0, loc_idx, :], 
                 color='red', linewidth=1, label='Simulated (cVAE)', linestyle='--', alpha=0.9)
    
    ax_time.set_xlabel('Time [s]')
    ax_time.set_ylabel('Acceleration [m/s²]')
    ax_time.set_title(f'{location_names[loc_idx]} - Time Domain')
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()
    
    # Frequency domain plot
    ax_freq = axes[loc_idx, 1]
    
    # Compute FFTs
    fft_val = np.fft.fft(X_val_original[0, loc_idx, :])
    fft_sim = np.fft.fft(X_val_recon_original[0, loc_idx, :])
    freqs = np.fft.fftfreq(sequence_length, 1/fs_downsampled)
    
    # Plot only positive frequencies up to Nyquist
    nyquist_idx = len(freqs) // 2
    ax_freq.loglog(freqs[1:nyquist_idx], np.abs(fft_val[1:nyquist_idx]), 
                   color=colors[loc_idx], linewidth=1.5, label='Validation (True)', alpha=0.8)
    ax_freq.loglog(freqs[1:nyquist_idx], np.abs(fft_sim[1:nyquist_idx]), 
                   color='red', linewidth=1, label='Simulated (cVAE)', linestyle='--', alpha=0.9)
    
    ax_freq.set_xlabel('Frequency [Hz]')
    ax_freq.set_ylabel('Magnitude')
    ax_freq.set_title(f'{location_names[loc_idx]} - Frequency Domain')
    ax_freq.grid(True, alpha=0.3)
    ax_freq.legend()

plt.tight_layout()
plt.show()

# ----------- RMSE CALCULATION -----------
for loc_idx in range(n_locations):
    y_mod = X_val_recon_original[0, loc_idx, :]
    y_t = X_val_original[0, loc_idx, :]
    y_mod_np = y_mod.detach().numpy() if hasattr(y_mod, 'detach') else y_mod
    y_t_np = y_t.detach().numpy() if hasattr(y_t, 'detach') else y_t
    rmse = np.sqrt(np.mean((y_mod_np - y_t_np)**2))
    print(f"{location_names[loc_idx]} RMSE: {rmse:.6f}")
    
