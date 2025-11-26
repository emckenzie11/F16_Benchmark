"""Network architecture for the conditional variational autoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- ENCODER ARCHITECTURE -----------
class Encoder(nn.Module):
    """Maps acceleration data and conditions to latent-space statistics."""

    def __init__(self, input_dim: int = 8192, condition_dim: int = 1, latent_dim: int = 8):
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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode acceleration ``x`` with condition ``c`` into mean and log-variance."""

        # Flatten acceleration
        x = x.view(x.size(0), -1)  # (batch, input_dim)

        # Acceleration MLP
        x = F.relu(self.fc1(x))  # (batch, 256)
        x = F.relu(self.fc2(x))  # (batch, 64)

        # Condition MLP
        c = F.relu(self.fc_cond(c))  # (batch, 8)

        # Combine acceleration and conditioning
        h = torch.cat([x, c], dim=1)  # (batch, 72)
        h = F.relu(self.fc_combined(h))  # (batch, 128)

        # Latent mean and variance
        mu = self.fc_mu(h)  # (batch, latent_dim)
        logvar = self.fc_logvar(h)  # (batch, latent_dim)

        return mu, logvar


# ----------- DECODER ARCHITECTURE -----------
class Decoder(nn.Module):
    """Reconstructs acceleration sequences from latent vectors and conditions."""

    def __init__(
        self,
        output_dim: int = 8192,
        n_locations: int = 2,
        sequence_length: int = 4096,
        condition_dim: int = 1,
        latent_dim: int = 8,
    ):
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

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode latent vector ``z`` with condition ``c`` to acceleration output."""

        # Condition embedding
        c = F.relu(self.fc_cond(c))  # (batch, 8)

        # Latent embedding
        z = F.relu(self.fc_z(z))  # (batch, 64)

        # Combine
        h = torch.cat([z, c], dim=1)  # (batch, 72)
        h = F.relu(self.fc_combined(h))  # (batch, 128)

        # MLP expansion
        h = F.relu(self.fc1(h))  # (batch, 256)
        x_hat = self.fc2(h)  # (batch, output_dim)

        # Reshape back to (batch, n_locations, sequence_length)
        batch_size = z.size(0)
        expected = self.n_locations * self.sequence_length
        if x_hat.numel() != batch_size * expected:
            raise RuntimeError(
                "Decoder produced {produced} elements but expected {expected} (batch={batch}, n_locations={n_locations}, "
                "sequence_length={sequence_length})".format(
                    produced=x_hat.numel(),
                    expected=batch_size * expected,
                    batch=batch_size,
                    n_locations=self.n_locations,
                    sequence_length=self.sequence_length,
                )
            )

        return x_hat.view(batch_size, self.n_locations, self.sequence_length)


# ----------- cVAE ARCHITECTURE -----------
class cVAE(nn.Module):
    """Conditional VAE working on modal acceleration data."""

    def __init__(self, latent_dim: int = 8, input_dim: int = 8192, n_locations: int = 2, sequence_length: int = 4096, condition_dim: int = 1):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, condition_dim=condition_dim, latent_dim=latent_dim)
        self.decoder = Decoder(
            output_dim=input_dim,
            n_locations=n_locations,
            sequence_length=sequence_length,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Apply the reparameterisation trick: :math:`z = \mu + \sigma\epsilon`."""

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode ``x`` and ``c`` then decode a sample from the latent distribution."""

        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar, z


# ----------- LOSS FUNCTION -----------
def cvae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reconstruction plus KL divergence losses for the cVAE."""

    recon_loss = F.mse_loss(x_recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
