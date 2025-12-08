import numpy as np
import pandas as pd
from datetime import datetime

def downsample_signal(signal, factor, axis=-1):
    """
    Downsample a signal by taking every nth sample along specified axis.
    
    Args:
        signal: Input signal (numpy array)
        factor: Downsampling factor (integer)
        axis: Axis along which to downsample (default: -1, last axis)
    
    Returns:
        Downsampled signal
    """
    if factor == 1:
        return signal
    
    # Create slicing tuple to downsample along specified axis
    slices = [slice(None)] * signal.ndim
    slices[axis] = slice(None, None, factor)
    
    return signal[tuple(slices)]


def process_data(data_dict, levels, location_i=2, location_j=3,
                 points_per_period=8192, period_indices=None):
    """
    Extract acceleration and force data from multiple levels and multiple periods.

    Args:
        data_dict: Dictionary mapping level numbers to pandas DataFrames
        levels: List of levels to process (order determines sample order in output)
        location_i: First DOF location to extract (wing side, default 2)
        location_j: Second DOF location to extract (payload side, default 3)
        points_per_period: Number of samples per period in the raw data (default 8192)
        period_indices: REQUIRED list of 1-based period indices to extract individually

    Returns:
        tuple: (X_samples, F_samples)
            - X_samples: Array of shape (n_samples, 2, points_per_period) containing acceleration data
            - F_samples: Array of shape (n_samples, points_per_period) containing force data
        where n_samples = n_levels * len(period_indices)
    """

    X_samples = []
    F_samples = []

    for level in levels:
        # Build list of acceleration arrays in order: [location_i, location_j]
        accel_arrays = [
            data_dict[level][f'Acceleration{location_i}'].to_numpy(),  # Row 0: Location i (wing side)
            data_dict[level][f'Acceleration{location_j}'].to_numpy(),  # Row 1: Location j (payload side)
        ]
        
        # Extract force signal
        force_array = data_dict[level]['Force'].to_numpy()

        accel_matrix = np.vstack(accel_arrays)  # shape (2, total_samples)

        # Extract each specified period as an independent sample
        for p in period_indices:
            # p is 1-based
            start_idx = (p - 1) * points_per_period
            end_idx = start_idx + points_per_period
            
            # Extract acceleration block
            accel_block = accel_matrix[:, start_idx:end_idx]
            X_samples.append(accel_block)
            
            # Extract force block
            force_block = force_array[start_idx:end_idx]
            F_samples.append(force_block)

    return np.array(X_samples), np.array(F_samples)


def normalise_data(data, mean=None, std=None):
    """
    normalise data using z-score normalization.
    
    Args:
        data: Data to normalise (numpy array or scalar)
        mean: Pre-computed mean (if None, computed from data)
        std: Pre-computed standard deviation (if None, computed from data)
    
    Returns:
        tuple: (normalised_data, mean, std)
            - normalised_data: normalised data
            - mean: Mean used for normalization
            - std: Standard deviation used for normalization
    """
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    
    # Avoid division by zero
    if std == 0:
        std = 1.0
    
    normalised = (data - mean) / std
    
    return normalised, mean, std