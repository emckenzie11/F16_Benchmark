import numpy as np
import pandas as pd
from datetime import datetime
import os
import openpyxl

def save_results_to_tracker(results_dict, tracker_file='cVAE/results.xlsx'):
    """
    Save experiment results to Excel tracker file.
    
    Args:
        results_dict: Dictionary containing experiment results and parameters
        tracker_file: Path to Excel tracker file (relative to parent directory)
    """
    # Determine next test number
    test_number = 1
    if os.path.exists(tracker_file):
        try:
            existing_df = pd.read_excel(tracker_file, sheet_name=0)
            if len(existing_df) > 0 and 'test_number' in existing_df.columns:
                test_number = existing_df['test_number'].max() + 1
            else:
                test_number = len(existing_df) + 1
        except Exception:
            test_number = 1
    
    # Add test number and timestamp as first columns
    results_dict_with_metadata = {
        'test_number': test_number,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    results_dict_with_metadata.update(results_dict)
    
    # Check if tracker file exists
    if os.path.exists(tracker_file):
        try:
            # Read existing tracker
            existing_df = pd.read_excel(tracker_file, sheet_name=0)

            # Create new row DataFrame
            new_row_df = pd.DataFrame([results_dict_with_metadata])

            # Align dtypes and columns to avoid concat dtype-inference issues
            existing_cols = list(existing_df.columns)
            new_cols = [col for col in new_row_df.columns if col not in existing_cols]

            # Ensure new_row_df has all existing columns with matching dtypes
            for col in existing_cols:
                if col not in new_row_df.columns:
                    # create a single-row column with same dtype as existing_df[col]
                    try:
                        new_row_df[col] = pd.Series([pd.NA], dtype=existing_df[col].dtype)
                    except Exception:
                        new_row_df[col] = pd.NA
                else:
                    # Try casting to the existing dtype to keep types consistent
                    try:
                        new_row_df[col] = new_row_df[col].astype(existing_df[col].dtype)
                    except Exception:
                        pass

            # For any new columns present only in new_row_df, add them to existing_df with matching dtype
            for col in new_cols:
                try:
                    existing_df[col] = pd.Series([pd.NA] * len(existing_df), dtype=new_row_df[col].dtype)
                except Exception:
                    existing_df[col] = pd.NA

            # Maintain column order
            column_order = existing_cols + new_cols
            existing_df = existing_df.reindex(columns=column_order)
            new_row_df = new_row_df.reindex(columns=column_order)

            # Combine dataframes
            combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)

        except Exception:
            combined_df = pd.DataFrame([results_dict_with_metadata])
    else:
        combined_df = pd.DataFrame([results_dict_with_metadata])
    
    # Save to Excel
    try:
        combined_df.to_excel(tracker_file, index=False, engine='openpyxl')
    except Exception:
        # Fallback to CSV if Excel fails
        csv_file = tracker_file.replace('.xlsx', '.csv')
        combined_df.to_csv(csv_file, index=False)


def process_acceleration_data(data_dict, levels, location_i=None, location_j=None,
                              points_per_period=8192, period_number=8, n_periods=1, downsample_factor=2):
    """
    Extract and process acceleration data from multiple levels.

    Args:
        data_dict: Dictionary mapping level numbers to pandas DataFrames
        levels: List of levels to process
        location_i: First DOF location to extract (wing side)
        location_j: Second DOF location to extract (payload side)
        points_per_period: Number of samples per period in the raw data (default 8192)
        period_number: 1-based index of the first period to extract (default 8)
        n_periods: Number of consecutive periods to extract (default 1)
        downsample_factor: Integer downsampling factor to apply after extraction (default 2)

    Returns:
        np.ndarray: Array of shape (n_levels, 2, sequence_length) containing processed acceleration data
                    where sequence_length = (points_per_period * n_periods) // downsample_factor
    """
    X = []
    
    for level in levels:
        # Always build list of acceleration arrays in order: [location_i, location_j]
        accel_arrays = [
            data_dict[level][f'Acceleration{location_i}'].to_numpy(),  # Row 0: Location i (wing side)
            data_dict[level][f'Acceleration{location_j}'].to_numpy(),  # Row 1: Location j (payload side)
        ]
        
        accel_matrix = np.vstack(accel_arrays)

        # Determine indices for requested period(s)
        # period_number is 1-based, so convert to 0-based index
        start_idx = (period_number - 1) * points_per_period
        end_idx = start_idx + n_periods * points_per_period

        # Extract requested period block and downsample by provided factor
        accel_matrix = accel_matrix[:, start_idx:end_idx][:, ::downsample_factor]

        X.append(accel_matrix)
    
    return np.array(X)


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