import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- ENCODER ARCHITECTURE -----------
class Encoder(nn.Module):
    def __init__(self, input_dim=8192, condition_dim=1, latent_dim=16):
        super().__init__()
        
        # Acceleration branch
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)

        # Condition branch
        self.fc_cond = nn.Linear(condition_dim, 8)

        # Combined layers
        self.fc_combined = nn.Linear(128 + 8, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x, c):
        # Flatten acceleration
        x = x.view(x.size(0), -1)           # (batch, input_dim)

        # Acceleration MLP
        x = F.relu(self.fc1(x))             # (batch, 512)
        x = F.relu(self.fc2(x))             # (batch, 128)

        # Condition MLP
        c = F.relu(self.fc_cond(c))         # (batch, 8)

        # Combine
        h = torch.cat([x, c], dim=1)        # (batch, 136)
        h = F.relu(self.fc_combined(h))     # (batch, 256)

        # Latent mean and variance
        mu = self.fc_mu(h)                  # (batch, latent_dim)
        logvar = self.fc_logvar(h)          # (batch, latent_dim)

        return mu, logvar


# ----------- DECODER ARCHITECTURE -----------
class Decoder(nn.Module):
    def __init__(self, output_dim=8192, condition_dim=1, latent_dim=16):
        super().__init__()

        # Process latent + condition together
        self.fc_cond = nn.Linear(condition_dim, 8)
        self.fc_z = nn.Linear(latent_dim, 128)
        self.fc_combined = nn.Linear(128 + 8, 256)

        # MLP to expand to output size
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, z, c):
        # Condition embedding
        c = F.relu(self.fc_cond(c))         # (batch, 8)
        
        # Latent embedding
        z = F.relu(self.fc_z(z))            # (batch, 128)

        # Combine
        h = torch.cat([z, c], dim=1)        # (batch, 136)
        h = F.relu(self.fc_combined(h))     # (batch, 256)

        # MLP expansion
        h = F.relu(self.fc1(h))             # (batch, 512)
        x_hat = self.fc2(h)                 # (batch, 12288)

        # Reshape back to (batch, 2, 4096)
        return x_hat.view(z.size(0), 2, 4096)


# ----------- cVAE ARCHITECTURE -----------
class cVAE(nn.Module):
    def __init__(self, latent_dim=16, input_dim=8192):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(output_dim=input_dim, latent_dim=latent_dim)
    
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
