import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- ENCODER ARCHITECTURE -----------
class Encoder(nn.Module):
    """Encoder: maps (x, c) -> q(z|x,c) with parameters (mu, logvar)"""
    def __init__(self, n_locations=2, sequence_length_accel=256, sequence_length_force=256, latent_dim = 32):
        super().__init__()
        
        # Conv1D feature extraction
        self.conv_net = nn.Sequential(
            nn.Conv1d(n_locations, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        
        conv_output_size = 32 * sequence_length_accel
        
        # Acceleration MLP
        self.accel_mlp = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Force conditioning MLP
        self.force_net = nn.Sequential(
            nn.Linear(sequence_length_force, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Combined embedding
        self.combined_net = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
        )
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x, c):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = self.accel_mlp(x)
        c = self.force_net(c)
        
        h = torch.cat([x, c], dim=1)
        h = self.combined_net(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


# ----------- DECODER ARCHITECTURE -----------
class Decoder(nn.Module):
    """Decoder: maps (z, c) -> p(x|z,c)"""
    def __init__(self, n_locations=2, sequence_length_accel=256, sequence_length_force=256, latent_dim = 32):
        super().__init__()
        self.n_locations = int(n_locations)
        self.sequence_length = int(sequence_length_accel)
        
        # Force conditioning MLP
        self.force_net = nn.Sequential(
            nn.Linear(sequence_length_force, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Latent embedding
        self.latent_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
        )
        
        # Expansion MLP
        conv_input_size = 32 * sequence_length_accel  
        self.combined_net = nn.Sequential(
            nn.Linear(64 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, conv_input_size),
            nn.ReLU(),
        )
        
        # Conv1D reconstruction
        self.deconv_net = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, n_locations, kernel_size=5, padding=2),
        )

    def forward(self, z, c):
        c = self.force_net(c)
        z = self.latent_net(z)

        h = torch.cat([z, c], dim=1)
        h = self.combined_net(h)
        h = h.view(z.size(0), 32, self.sequence_length)
        
        x_hat = self.deconv_net(h)
        return x_hat


# ----------- cVAE ARCHITECTURE -----------
class cVAE(nn.Module):
    """Conditional VAE: q(z|x,c) encoder, p(x|z,c) decoder, N(0,I) prior"""
    def __init__(self, n_locations=2, sequence_length_accel=256, sequence_length_force=256, latent_dim = 32):
        super().__init__()
        self.encoder = Encoder(n_locations=n_locations, sequence_length_accel=sequence_length_accel,
                               sequence_length_force=sequence_length_force, latent_dim=latent_dim)
        self.decoder = Decoder(n_locations=n_locations, sequence_length_accel=sequence_length_accel, 
                               sequence_length_force=sequence_length_force, latent_dim=latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar, z


# ----------- LOSS FUNCTION -----------
def cvae_loss(x_recon, x, mu, logvar, beta=0.5):
    """cVAE loss: reconstruction (MSE) + beta * KL divergence"""
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss