import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- ENCODER ARCHITECTURE -----------
class Encoder(nn.Module):
    def __init__(self, input_dim=8192, condition_dim=1, latent_dim=8):
        super().__init__()
        
        # Acceleration branch (smaller MLP)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)

        # Condition branch
        self.fc_cond = nn.Linear(condition_dim, 8)

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

        # Condition MLP
        c = F.relu(self.fc_cond(c))         # (batch, 8)

        # Combine
        # x: (batch, 64), c: (batch, 8) -> concat (batch, 72)
        h = torch.cat([x, c], dim=1)        # (batch, 72)
        h = F.relu(self.fc_combined(h))     # (batch, 128)

        # Latent mean and variance
        mu = self.fc_mu(h)                  # (batch, latent_dim)
        logvar = self.fc_logvar(h)          # (batch, latent_dim)

        return mu, logvar


# ----------- DECODER ARCHITECTURE -----------
class Decoder(nn.Module):
    def __init__(self, output_dim=8192, n_locations=2, sequence_length=4096, condition_dim=1, latent_dim=8):
        super().__init__()
        # Save shape params for reshape
        self.n_locations = int(n_locations)
        self.sequence_length = int(sequence_length)

        # Process latent + condition together
        self.fc_cond = nn.Linear(condition_dim, 8)
        self.fc_z = nn.Linear(latent_dim, 64)
        self.fc_combined = nn.Linear(64 + 8, 128)

        # MLP to expand to output size (mirror of encoder)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, z, c):
        # Condition embedding
        c = F.relu(self.fc_cond(c))         # (batch, 8)
        
        # Latent embedding
        z = F.relu(self.fc_z(z))            # (batch, 64)

        # Combine
        # z: (batch,64), c: (batch,8) -> concat (batch,72)
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
    def __init__(self, latent_dim=8, input_dim=8192, n_locations=2, sequence_length=4096):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(output_dim=input_dim, n_locations=n_locations, sequence_length=sequence_length, latent_dim=latent_dim)
    
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
def cvae_loss(x_recon, x, RF_recon, RF_true, mu, logvar, weight_recon=1.0, weight_rf=0.5, weight_kl=0.5):
    """
    cVAE loss with reconstruction, restoring force, and KL divergence terms.
    
    Args:
        x_recon: reconstructed accelerations (batch, n_locations, sequence_length)
        x: true accelerations (batch, n_locations, sequence_length)
        RF_recon: reconstructed restoring force (batch, sequence_length)
        RF_true: true restoring force (batch, sequence_length)
        mu: encoder mean (batch, latent_dim)
        logvar: encoder log-variance (batch, latent_dim)
        weight_recon: weight for reconstruction loss
        weight_rf: weight for restoring force loss
        weight_kl: weight for KL divergence loss
    
    Returns:
        total_loss, recon_loss, rf_loss, kl_loss
    """
    # Reconstruction loss (MSE on all accelerations)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # Restoring force loss: MSE between reconstructed and true restoring forces
    rf_loss = F.mse_loss(RF_recon, RF_true, reduction='sum')
    
    # KL divergence loss: KL(q(z|x,c) || p(z))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total weighted loss
    total_loss = weight_recon * recon_loss + weight_rf * rf_loss + weight_kl * kl_loss
    
    return total_loss, recon_loss, rf_loss, kl_loss
