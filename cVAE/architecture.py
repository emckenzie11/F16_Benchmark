import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- ENCODER ARCHITECTURE -----------
class Encoder(nn.Module):
    def __init__(self, n_locations=2, sequence_length_accel=4096, sequence_length_force=4096, latent_dim=16):
        super().__init__()
        
        input_dim = n_locations * sequence_length_accel
        
        # Acceleration branch (smaller MLP)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)

        # Condition branch (force signal)
        self.fc_cond1 = nn.Linear(sequence_length_force, 32)
        self.fc_cond2 = nn.Linear(32, 8)

        # Combined layers
        self.fc_combined = nn.Linear(64 + 8, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, c):
        # Flatten acceleration
        x = x.view(x.size(0), -1)           # (batch, input_dim)

        # Acceleration MLP
        x = F.relu(self.fc1(x))             # (batch, 256)
        x = F.relu(self.fc2(x))             # (batch, 64)

        # Condition MLP (process force signal)
        c = F.relu(self.fc_cond1(c))        # (batch, 32)
        c = F.relu(self.fc_cond2(c))        # (batch, 8)

        # Combine
        h = torch.cat([x, c], dim=1)        # (batch, 72)
        h = F.relu(self.fc_combined(h))     # (batch, 128)

        # Latent mean and variance
        mu = self.fc_mu(h)                  # (batch, latent_dim)
        logvar = self.fc_logvar(h)          # (batch, latent_dim)

        return mu, logvar


# ----------- DECODER ARCHITECTURE -----------
class Decoder(nn.Module):
    def __init__(self, n_locations=2, sequence_length_accel=4096, sequence_length_force=4096, latent_dim=8):
        super().__init__()
        # Save shape params for reshape
        self.n_locations = int(n_locations)
        self.sequence_length = int(sequence_length_accel)
        
        output_dim = n_locations * sequence_length_accel

        # Process latent + condition together
        self.fc_cond1 = nn.Linear(sequence_length_force, 32)
        self.fc_cond2 = nn.Linear(32, 8)
        self.fc_z = nn.Linear(latent_dim, 64)
        self.fc_combined = nn.Linear(64 + 8, 128)

        # MLP to expand to output size (mirror of encoder)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, z, c):
        # Condition embedding (process force signal)
        c = F.relu(self.fc_cond1(c))        # (batch, 32)
        c = F.relu(self.fc_cond2(c))        # (batch, 8)
        
        # Latent embedding
        z = F.relu(self.fc_z(z))            # (batch, 64)

        # Combine
        h = torch.cat([z, c], dim=1)        # (batch, 72)
        h = F.relu(self.fc_combined(h))     # (batch, 128)

        # MLP expansion
        h = F.relu(self.fc1(h))             # (batch, 256)
        x_hat = self.fc2(h)                 # (batch, output_dim)

        # Reshape back to (batch, n_locations, sequence_length)
        batch_size = z.size(0)
        expected = self.n_locations * self.sequence_length
        if x_hat.numel() != batch_size * expected:
            raise RuntimeError(f"Decoder produced {x_hat.numel()} elements but expected {batch_size * expected} "
                               f"(batch={batch_size}, n_locations={self.n_locations}, sequence_length={self.sequence_length})")

        return x_hat.view(batch_size, self.n_locations, self.sequence_length)


# ----------- cVAE ARCHITECTURE -----------
class cVAE(nn.Module):
    def __init__(self, latent_dim=8, n_locations=2, sequence_length_accel=4096, sequence_length_force=4096):
        super().__init__()
        self.encoder = Encoder(n_locations=n_locations, sequence_length_accel=sequence_length_accel,
                               sequence_length_force=sequence_length_force, latent_dim=latent_dim)
        self.decoder = Decoder(n_locations=n_locations, sequence_length_accel=sequence_length_accel, 
                               sequence_length_force=sequence_length_force, latent_dim=latent_dim)
    
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
def cvae_loss(x_recon, x, mu, logvar, beta=0.5):
    """
    cVAE loss with reconstruction, restoring force, and KL divergence terms.
    
    Args:
        x_recon: reconstructed accelerations (batch, n_locations, sequence_length)
        x: true accelerations (batch, n_locations, sequence_length)
        mu: encoder mean (batch, latent_dim)
        logvar: encoder log-variance (batch, latent_dim)
        beta: weight for KL divergence loss
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE on all accelerations)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence loss: KL(q(z|x,c) || p(z))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total weighted loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss