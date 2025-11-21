import numpy as np


def process_acceleration_data(data_dict, levels, location_i, location_j):
    """
    Extract and process acceleration data from multiple levels.
    
    Args:
        data_dict: Dictionary mapping level numbers to pandas DataFrames
        levels: List of levels to process
        location_i: First DOF location (wing side)
        location_j: Second DOF location (payload side)
    
    Returns:
        np.ndarray: Array of shape (n_levels, 2, 4096) containing processed acceleration data
    """
    X = []
    
    for level in levels:
        accel_matrix = np.vstack([
            data_dict[level][f'Acceleration{location_i}'].to_numpy(),  # Row 0: Location i
            data_dict[level][f'Acceleration{location_j}'].to_numpy(),  # Row 1: Location j  
        ])
        
        # Extract second to last period and downsample
        points_per_period = 8192
        start_idx = 7 * points_per_period
        end_idx = 8 * points_per_period  # Second to last period
        
        accel_matrix = accel_matrix[:, start_idx:end_idx][:, ::2]  # Extract 8th period and downsample by factor of 2 
        
        X.append(accel_matrix)
    
    return np.array(X)


def normalize_data(data, mean=None, std=None):
    """
    Normalize data using z-score normalization.
    
    Args:
        data: Data to normalize (numpy array or scalar)
        mean: Pre-computed mean (if None, computed from data)
        std: Pre-computed standard deviation (if None, computed from data)
    
    Returns:
        tuple: (normalized_data, mean, std)
            - normalized_data: Normalized data
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
    
    normalized = (data - mean) / std
    
    return normalized, mean, std
