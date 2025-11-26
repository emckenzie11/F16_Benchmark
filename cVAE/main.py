"""End-to-end script for training and evaluating the cVAE model.

The original notebook-style script has been organised into smaller functions to
make the workflow clearer: configuration, data preparation, POD computation,
training, simulation, plotting, and result tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from architecture import cVAE, cvae_loss
from helpers import normalize_data, process_acceleration_data, save_results_to_tracker


# ----------- CONFIGURATION -----------
@dataclass
class DataConfig:
    training_levels: list[int]
    validation_level: int
    location_i: int
    location_j: int
    points_per_period: int = 8192
    period_number: int = 7
    n_periods: int = 1
    downsample_factor: int = 2
    fs: int = 400
    sequence_length: int | None = None
    n_locations: int = 2


@dataclass
class ModelConfig:
    latent_dim: int = 8
    sequence_length: int = 4096
    n_locations: int = 2
    condition_dim: int = 1


@dataclass
class TrainingConfig:
    num_epochs: int = 1200
    beta_max: float = 0.04
    warmup_epochs: int = 100
    learning_rate: float = 1e-3


FORCE_LEVELS: Dict[int, float] = {2: 24.6, 4: 61.4, 6: 85.7}


# ----------- DATA LOADING -----------
def load_data(training_levels: Iterable[int], validation_level: int) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Load the required training and validation CSV files."""

    training_data = {
        1: pd.read_csv("BenchmarkData/F16Data_FullMSine_Level1.csv"),  # 12.4N
        3: pd.read_csv("BenchmarkData/F16Data_FullMSine_Level3.csv"),  # 36.8N
        5: pd.read_csv("BenchmarkData/F16Data_FullMSine_Level5.csv"),  # 73.6N
        7: pd.read_csv("BenchmarkData/F16Data_FullMSine_Level7.csv"),  # 97.8N
    }

    validation_data = {
        2: pd.read_csv("BenchmarkData/F16Data_FullMSine_Level2_Validation.csv"),  # 24.6N
        4: pd.read_csv("BenchmarkData/F16Data_FullMSine_Level4_Validation.csv"),  # 61.4N
        6: pd.read_csv("BenchmarkData/F16Data_FullMSine_Level6_Validation.csv"),  # 85.7N
    }

    training_data = {lvl: training_data[lvl] for lvl in training_levels}
    validation_data = {lvl: validation_data[lvl] for lvl in [validation_level]}
    return training_data, validation_data


# ----------- DATA PREPARATION -----------
def prepare_acceleration_datasets(data_cfg: DataConfig, force_levels: Iterable[float]) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray, torch.Tensor, torch.Tensor]:
    """Process raw CSV data, normalise, and prepare tensors for training."""

    training_data, validation_data = load_data(data_cfg.training_levels, data_cfg.validation_level)

    X_raw = process_acceleration_data(
        training_data,
        data_cfg.training_levels,
        location_i=data_cfg.location_i,
        location_j=data_cfg.location_j,
        points_per_period=data_cfg.points_per_period,
        period_number=data_cfg.period_number,
        n_periods=data_cfg.n_periods,
        downsample_factor=data_cfg.downsample_factor,
    )
    X_val_raw = process_acceleration_data(
        validation_data,
        [data_cfg.validation_level],
        location_i=data_cfg.location_i,
        location_j=data_cfg.location_j,
        points_per_period=data_cfg.points_per_period,
        period_number=data_cfg.period_number,
        n_periods=data_cfg.n_periods,
        downsample_factor=data_cfg.downsample_factor,
    )

    C_raw = np.array(list(force_levels))
    C_val_raw = np.array([FORCE_LEVELS[data_cfg.validation_level]])

    X_normalized, X_mean, X_std = normalize_data(X_raw)
    C_normalized, C_mean, C_std = normalize_data(C_raw)

    normalisation_params = {
        "X_mean": X_mean,
        "X_std": X_std,
        "C_mean": C_mean,
        "C_std": C_std,
    }

    A_train = torch.tensor(X_normalized, dtype=torch.float32)
    C_train = torch.tensor(C_normalized, dtype=torch.float32).unsqueeze(-1)

    return X_normalized, X_val_raw, normalisation_params, C_val_raw, A_train, C_train


# ----------- POD COMPUTATION -----------
def compute_pod_modes(X_normalized: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Compute POD modes and modal coefficients from normalised acceleration data."""

    X1, X3, X5, X7 = X_normalized
    S = np.concatenate([X1, X3, X5, X7], axis=1)

    U, Sigma, _ = np.linalg.svd(S, full_matrices=False)
    energies = Sigma ** 2
    energy_frac = energies / energies.sum()
    print(f"POD: S shape={S.shape}, U shape={U.shape}, n_modes={len(Sigma)}")
    print(f"Mode energy fraction (first 5): {energy_frac[:5]}")

    a1 = U.T @ X1
    a3 = U.T @ X3
    a5 = U.T @ X5
    a7 = U.T @ X7
    A = np.stack([a1, a3, a5, a7], axis=0)

    U_torch = torch.tensor(U, dtype=torch.float32)
    A_train_modal = torch.tensor(A, dtype=torch.float32)
    return U, U_torch, A_train_modal


# ----------- TRAINING -----------
def train_cvae(
    model: cVAE,
    A_train_modal: torch.Tensor,
    C_train: torch.Tensor,
    X_normalized: np.ndarray,
    U_torch: torch.Tensor,
    train_cfg: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train the model with a linear beta warmup schedule."""

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    X_train_phys = torch.tensor(X_normalized, dtype=torch.float32)

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        f"Training for {train_cfg.num_epochs} epochs, beta warmup -> {train_cfg.beta_max} over {train_cfg.warmup_epochs} epochs\n"
        f"Shapes: input_dim={A_train_modal.shape[1] * A_train_modal.shape[2]}, sequence_length={A_train_modal.shape[2]}, n_locations={A_train_modal.shape[1]}\n"
        f"A_train.shape={A_train_modal.shape}, C_train.shape={C_train.shape}"
    )

    total_loss = recon_loss = kl_loss = torch.tensor(0.0)

    for epoch in range(train_cfg.num_epochs):
        model.train()

        beta = train_cfg.beta_max * (epoch / train_cfg.warmup_epochs) if epoch < train_cfg.warmup_epochs else train_cfg.beta_max

        A_train_recon, mu_train, logvar_train, _ = model(A_train_modal, C_train)
        X_train_recon = torch.einsum("cr,brt->bct", U_torch, A_train_recon)

        total_loss, recon_loss, kl_loss = cvae_loss(X_train_recon, X_train_phys, mu_train, logvar_train, beta)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(
                f"Epoch {epoch+1}: Loss={total_loss.item():.3f} "
                f"Recon={recon_loss.item():.3f} KL={kl_loss.item():.3f} β={beta:.3f}"
            )

    return total_loss, recon_loss, kl_loss


# ----------- SIMULATION -----------
def simulate_cvae(
    model: cVAE,
    U_torch: torch.Tensor,
    latent_dim: int,
    C_val: torch.Tensor,
    normalisation_params: dict[str, float],
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate stochastic reconstructions and denormalise them."""

    model.eval()
    X_sim_modal = []
    X_sim_phys = []

    with torch.no_grad():
        for _ in range(n_samples):
            z_sim = torch.randn(1, latent_dim, device=C_val.device)
            a_sim = model.decoder(z_sim, C_val)
            X_sim = torch.einsum("cr,brt->bct", U_torch, a_sim)
            X_sim_modal.append(a_sim.cpu().numpy())
            X_sim_phys.append(X_sim.cpu().numpy())

    X_sim_modal = np.array(X_sim_modal)
    X_sim = np.array(X_sim_phys)
    X_sim_mean = X_sim.mean(axis=0)

    X_sim_mean_denorm = X_sim_mean * normalisation_params["X_std"] + normalisation_params["X_mean"]
    X_sim_samples_denorm = X_sim * normalisation_params["X_std"] + normalisation_params["X_mean"]

    return X_sim_mean_denorm, X_sim_samples_denorm, X_sim


# ----------- PLOTTING -----------
def plot_results(
    X_sim_mean_denorm: np.ndarray,
    X_sim_samples_denorm: np.ndarray,
    X_val_raw: np.ndarray,
    data_cfg: DataConfig,
    C_val_raw: np.ndarray,
) -> None:
    """Plot validation data against simulated reconstructions and uncertainty bands."""

    fs_downsampled = data_cfg.fs / data_cfg.downsample_factor
    t = np.arange(data_cfg.sequence_length) / fs_downsampled

    fig, axes = plt.subplots(data_cfg.n_locations, 2, figsize=(15, 5 * data_cfg.n_locations))
    fig.suptitle(
        "cVAE Reconstruction: Simulated vs Validation Data\n"
        f"Level {data_cfg.validation_level} (Force: {C_val_raw[0]:.1f}N)",
        fontsize=16,
        fontweight="bold",
    )

    location_names = [
        f"Location {data_cfg.location_i} (Wing Side)",
        f"Location {data_cfg.location_j} (Payload Side)",
    ]

    for loc_idx in range(data_cfg.n_locations):
        ax_time = axes[loc_idx, 0]
        ax_time.plot(t, X_val_raw[0, loc_idx, :], color="black", linewidth=1.5, label="Validation (True)", alpha=0.8)
        ax_time.plot(
            t,
            X_sim_mean_denorm[0, loc_idx, :],
            color="red",
            linewidth=1.5,
            label="cVAE Mean",
            linestyle="--",
            alpha=0.9,
        )

        sim_sample_std = X_sim_samples_denorm[:, 0, loc_idx, :].std(axis=0)
        ax_time.fill_between(
            t,
            X_sim_mean_denorm[0, loc_idx, :] - sim_sample_std,
            X_sim_mean_denorm[0, loc_idx, :] + sim_sample_std,
            color="red",
            alpha=0.2,
            label="±1σ Uncertainty",
        )
        ax_time.fill_between(
            t,
            X_sim_mean_denorm[0, loc_idx, :] - 2 * sim_sample_std,
            X_sim_mean_denorm[0, loc_idx, :] + 2 * sim_sample_std,
            color="red",
            alpha=0.1,
            label="±2σ Uncertainty",
        )

        if loc_idx == data_cfg.n_locations - 1:
            ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel("Acceleration [m/s²]")
        ax_time.set_title(f"{location_names[loc_idx]}")
        ax_time.grid(True, alpha=0.3)
        ax_time.legend(loc="upper right")

        ax_zoom = axes[loc_idx, 1]
        half_window = int(0.5 * fs_downsampled)
        mid_idx = len(t) // 2
        start_idx = max(mid_idx - half_window, 0)
        end_idx = min(mid_idx + half_window, len(t))
        t_zoom = t[start_idx:end_idx]

        ax_zoom.plot(
            t_zoom,
            X_val_raw[0, loc_idx, start_idx:end_idx],
            color="black",
            linewidth=1.5,
            label="Validation (True)",
            alpha=0.8,
        )
        ax_zoom.plot(
            t_zoom,
            X_sim_mean_denorm[0, loc_idx, start_idx:end_idx],
            color="red",
            linewidth=1.5,
            label="cVAE Mean",
            linestyle="--",
            alpha=0.9,
        )

        zoom_sample_mean = X_sim_mean_denorm[0, loc_idx, start_idx:end_idx]
        zoom_sample_std = sim_sample_std[start_idx:end_idx]
        ax_zoom.fill_between(
            t_zoom,
            zoom_sample_mean - zoom_sample_std,
            zoom_sample_mean + zoom_sample_std,
            color="red",
            alpha=0.2,
            label="±1σ Uncertainty",
        )
        ax_zoom.fill_between(
            t_zoom,
            zoom_sample_mean - 2 * zoom_sample_std,
            zoom_sample_mean + 2 * zoom_sample_std,
            color="red",
            alpha=0.1,
            label="±2σ Uncertainty",
        )

        if loc_idx == data_cfg.n_locations - 1:
            ax_zoom.set_xlabel("Time [s]")
        ax_zoom.set_ylabel("Acceleration [m/s²]")
        ax_zoom.set_title(f"{location_names[loc_idx]}")
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.legend(loc="upper right")

    for ax in axes.flat:
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

    plt.tight_layout()
    plt.show()


# ----------- METRICS -----------
def compute_rmse(X_sim_mean_denorm: np.ndarray, X_val_raw: np.ndarray, data_cfg: DataConfig) -> dict[str, float]:
    """Compute RMSE between predicted and validation accelerations for each location."""

    rmse_results: dict[str, float] = {}
    for loc_idx in range(data_cfg.n_locations):
        y_pred = X_sim_mean_denorm[0, loc_idx, :]
        y_true = X_val_raw[0, loc_idx, :]
        y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().numpy()
        y_true_np = y_true if isinstance(y_true, np.ndarray) else y_true.detach().numpy()
        rmse = float(np.sqrt(np.mean((y_pred_np - y_true_np) ** 2)))
        rmse_results[f"rmse_location_{loc_idx + 1}"] = rmse
        print(f"Location {loc_idx + 1} RMSE: {rmse:.6f}")
    return rmse_results


# ----------- MAIN WORKFLOW -----------
def run_experiment() -> None:
    data_cfg = DataConfig(
        training_levels=[1, 3, 5, 7],
        validation_level=6,
        location_i=2,
        location_j=3,
        period_number=7,
        n_periods=1,
        downsample_factor=2,
        sequence_length=(8192 * 1) // 2,
        n_locations=2,
    )

    model_cfg = ModelConfig(
        latent_dim=8,
        sequence_length=(data_cfg.points_per_period * data_cfg.n_periods) // data_cfg.downsample_factor,
        n_locations=data_cfg.n_locations,
        condition_dim=1,
    )

    train_cfg = TrainingConfig(num_epochs=1200, beta_max=0.04, warmup_epochs=100, learning_rate=1e-3)

    X_normalized, X_val_raw, normalisation_params, C_val_raw, _X_train_tensor, C_train = prepare_acceleration_datasets(
        data_cfg,
        force_levels=[12.4, 36.8, 73.6, 97.8],
    )

    _U, U_torch, A_train_modal = compute_pod_modes(X_normalized)

    model = cVAE(
        latent_dim=model_cfg.latent_dim,
        input_dim=model_cfg.n_locations * model_cfg.sequence_length,
        n_locations=model_cfg.n_locations,
        sequence_length=model_cfg.sequence_length,
        condition_dim=model_cfg.condition_dim,
    )

    total_loss, recon_loss, kl_loss = train_cvae(
        model,
        A_train_modal,
        C_train,
        X_normalized,
        U_torch,
        train_cfg,
    )

    C_val = torch.tensor((np.array([C_val_raw[0]]) - normalisation_params["C_mean"]) / normalisation_params["C_std"], dtype=torch.float32).unsqueeze(-1)
    X_sim_mean_denorm, X_sim_samples_denorm, _ = simulate_cvae(
        model,
        U_torch,
        model_cfg.latent_dim,
        C_val,
        normalisation_params,
    )

    plot_results(X_sim_mean_denorm, X_sim_samples_denorm, X_val_raw, data_cfg, C_val_raw)
    rmse_results = compute_rmse(X_sim_mean_denorm, X_val_raw, data_cfg)

    results_dict = {
        "latent_dim": model_cfg.latent_dim,
        "sequence_length": model_cfg.sequence_length,
        "n_locations": model_cfg.n_locations,
        "condition_dim": model_cfg.condition_dim,
        "num_epochs": train_cfg.num_epochs,
        "beta_max": train_cfg.beta_max,
        "warmup_epochs": train_cfg.warmup_epochs,
        "learning_rate": train_cfg.learning_rate,
        "training_levels": str(data_cfg.training_levels),
        "validation_level": data_cfg.validation_level,
        "validation_force": C_val_raw[0],
        "location_i": data_cfg.location_i,
        "location_j": data_cfg.location_j,
        **rmse_results,
        "total_rmse": np.mean(list(rmse_results.values())),
        "model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "loss_computation": "physical_space",
        "final_total_loss": total_loss.item(),
        "final_recon_loss": recon_loss.item(),
        "final_kl_loss": kl_loss.item(),
    }

    save_results_to_tracker(results_dict)


if __name__ == "__main__":
    run_experiment()
