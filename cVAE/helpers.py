"""Helper utilities for cVAE training and evaluation."""

from __future__ import annotations

from datetime import datetime
import os
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd


def save_results_to_tracker(results_dict: dict[str, Any], tracker_file: str = "cVAE/results.xlsx") -> None:
    """Append experiment metadata to an Excel tracker (or CSV fallback).

    The function is intentionally defensive: it tolerates missing columns and
    dtype mismatches so that the tracker remains usable even if new fields are
    added over time.
    """

    # Determine next test number
    test_number = 1
    if os.path.exists(tracker_file):
        try:
            existing_df = pd.read_excel(tracker_file, sheet_name=0)
            if len(existing_df) > 0 and "test_number" in existing_df.columns:
                test_number = existing_df["test_number"].max() + 1
            else:
                test_number = len(existing_df) + 1
        except Exception:
            test_number = 1

    # Add test number and timestamp as first columns
    results_dict_with_metadata = {
        "test_number": test_number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_dict_with_metadata.update(results_dict)

    # Check if tracker file exists
    if os.path.exists(tracker_file):
        try:
            existing_df = pd.read_excel(tracker_file, sheet_name=0)
            new_row_df = pd.DataFrame([results_dict_with_metadata])

            # Align dtypes and columns to avoid concat dtype-inference issues
            existing_cols = list(existing_df.columns)
            new_cols = [col for col in new_row_df.columns if col not in existing_cols]

            for col in existing_cols:
                if col not in new_row_df.columns:
                    try:
                        new_row_df[col] = pd.Series([pd.NA], dtype=existing_df[col].dtype)
                    except Exception:
                        new_row_df[col] = pd.NA
                else:
                    try:
                        new_row_df[col] = new_row_df[col].astype(existing_df[col].dtype)
                    except Exception:
                        pass

            for col in new_cols:
                try:
                    existing_df[col] = pd.Series([pd.NA] * len(existing_df), dtype=new_row_df[col].dtype)
                except Exception:
                    existing_df[col] = pd.NA

            column_order = existing_cols + new_cols
            existing_df = existing_df.reindex(columns=column_order)
            new_row_df = new_row_df.reindex(columns=column_order)

            combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)

        except Exception:
            combined_df = pd.DataFrame([results_dict_with_metadata])
    else:
        combined_df = pd.DataFrame([results_dict_with_metadata])

    # Save to Excel
    try:
        combined_df.to_excel(tracker_file, index=False, engine="openpyxl")
    except Exception:
        # Fallback to CSV if Excel fails
        csv_file = tracker_file.replace(".xlsx", ".csv")
        combined_df.to_csv(csv_file, index=False)


def process_acceleration_data(
    data_dict: dict[int, pd.DataFrame],
    levels: Iterable[int],
    location_i: int | None = None,
    location_j: int | None = None,
    points_per_period: int = 8192,
    period_number: int = 8,
    n_periods: int = 1,
    downsample_factor: int = 2,
) -> np.ndarray:
    """Extract and downsample acceleration data from multiple levels."""

    X = []

    for level in levels:
        # Always build list of acceleration arrays in order: [location_i, location_j]
        accel_arrays = [
            data_dict[level][f"Acceleration{location_i}"].to_numpy(),  # Row 0: Location i (wing side)
            data_dict[level][f"Acceleration{location_j}"].to_numpy(),  # Row 1: Location j (payload side)
        ]

        accel_matrix = np.vstack(accel_arrays)

        # Determine indices for requested period(s)
        start_idx = (period_number - 1) * points_per_period
        end_idx = start_idx + n_periods * points_per_period

        # Extract requested period block and downsample by provided factor
        accel_matrix = accel_matrix[:, start_idx:end_idx][:, ::downsample_factor]

        X.append(accel_matrix)

    return np.array(X)


def normalize_data(data: np.ndarray, mean: float | None = None, std: float | None = None) -> Tuple[np.ndarray, float, float]:
    """Normalize data using z-score normalization."""

    if mean is None:
        mean = float(np.mean(data))
    if std is None:
        std = float(np.std(data))

    if std == 0:
        std = 1.0

    normalized = (data - mean) / std

    return normalized, mean, std
