# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import os
import pathlib
import pickle
import warnings
from copy import deepcopy
from datetime import datetime

import numpy as np
import roboverse.constants as c
from tqdm import tqdm

from rv_train.utils.train_utils import ForkedPdb as debug  # noqa


def get_dataset_stats(dataset, keys, alpha=0.99, sample_size=None):
    """
    Calculates statistics for specified keys over an entire dataset by iteration.

    This function assumes input data for keys has ndim >= 1 (not scalar).

    It iterates through the dataset once. For each specified key,
    it computes the overall minimum and maximum values encountered. It also
    calculates the exponential moving average (EMA) of the mean and the
    standard deviation.

    The statistics (min, max, mean, std) are computed such that they represent
    the variation *across the dataset* for each element in the *last dimension*
    of the data associated with the key.

    It achieves this by reshaping the data to 2D (-1, D), where D is the size
    of the original last dimension (even for 1D input), and then aggregating
    along the first axis (axis=0).

    The EMA calculation uses the formula:
    EMA_new = alpha * current_value + (1 - alpha) * EMA_old

    The standard deviation EMA is derived from the EMA of the mean and the
    EMA of the mean of squares:
    var_ema = ema(mean_of_squares) - (ema(mean))^2
    std_ema = sqrt(max(0, var_ema))

    :param dataset: An iterable (e.g., list, PyTorch DataLoader, tf.data.Dataset)
                    where each item provides data (e.g., a dictionary)
                    containing the specified keys. Data must have ndim >= 1.
    :param keys: A list or tuple of keys (strings) for which to calculate stats.
                 The data associated with these keys should be numerical numpy arrays
                 with ndim >= 1.
    :param alpha: The smoothing factor for the Exponential Moving Average
                  (float, between 0 and 1). A higher value gives more weight
                  to recent data points. Defaults to 0.99.
    :param sample_size: The number of samples to use for the stats calculation. If None, all samples are used.
    :return: A dictionary where keys are the input `keys`, and values are
             dictionaries containing the calculated statistics: 'min', 'max',
             'mean_ema', 'std_ema'. Each statistic is a NumPy array whose
             shape corresponds to the last dimension of the data.
    :raises TypeError: If the dataset is not iterable or keys is not a list/tuple.
    :raises ValueError: If alpha is not between 0 and 1.
    :raises KeyError: If a key is not found in a dataset item.
    :raises TypeError: If the data associated with a key is not numerical or cannot be reshaped.
    :raises AssertionError: If data associated with a key has ndim == 0 (is scalar).
    """
    # Current date: Friday, April 25, 2025 (as per context)

    if not isinstance(keys, (list, tuple)):
        raise TypeError("Keys must be a list or tuple.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (exclusive).")

    stats = {}

    # for item_idx, item in tqdm(enumerate(dataset_iterator), total=len(dataset), desc="Calculating dataset stats"):
    if sample_size is not None:
        item_indices = np.random.choice(len(dataset), sample_size, replace=False)
    else:
        item_indices = range(len(dataset))
    for item_idx in tqdm(item_indices, desc="Calculating dataset stats"):
        item = dataset[int(item_idx)]
        for key in keys:
            if key not in item:
                raise KeyError(
                    f"Key '{key}' not found in dataset item {item_idx}: {item}"
                )

            try:
                data = np.asarray(item[key])
            except Exception as e:
                raise TypeError(
                    f"Could not convert data for key '{key}' in item {item_idx} to NumPy array. Error: {e}"
                )

            if not np.issubdtype(data.dtype, np.number):
                raise TypeError(
                    f"Data for key '{key}' in item {item_idx} is not numerical (dtype: {data.dtype})."
                )

            # --- Assertion and Stat Calculation (assuming ndim >= 1) ---
            # Enforce assumption that data is at least 1D
            assert (
                data.ndim >= 1
            ), f"Data for key '{key}' in item {item_idx} must have ndim >= 1, but got shape {data.shape}"

            # Handle empty arrays within an item gracefully
            if data.size == 0:
                print(
                    f"Warning: Empty array encountered for key '{key}' in item {item_idx}. Skipping stats calculation for this item."
                )
                continue

            # This block now handles ndim >= 1 (including 1D case)
            try:
                # Reshape to 2D: (product of other dims, last_dim_size)
                # For 1D (C,), last_dim_size=C, reshape is (-1, C) -> (1, C)
                last_dim_size = data.shape[-1]
                reshaped_data = data.reshape(-1, last_dim_size)
            except (
                ValueError
            ) as e:  # Should only happen if data.shape is somehow invalid
                raise TypeError(
                    f"Could not reshape data for key '{key}' (shape: {data.shape}) in item {item_idx}. Error: {e}"
                )
            except IndexError:  # Should now be impossible due to ndim >= 1 assert
                # This error path is less likely now but kept for safety
                raise TypeError(
                    f"Could not get last dimension size for key '{key}' (shape: {data.shape}) in item {item_idx}."
                )

            # Calculate stats along axis 0 (over flattened dimensions)
            # For input (B, ..., D), result shape: (D,)
            # For input (C,), reshaped to (1, C), result shape: (C,)
            current_min = np.min(reshaped_data, axis=0)
            current_max = np.max(reshaped_data, axis=0)
            current_mean = np.mean(reshaped_data, axis=0)
            current_mean_sq = np.mean(np.square(reshaped_data), axis=0)
            # --- End of stats calculation ---

            # Initialize or update stats dictionary
            if key not in stats:
                stats[key] = {
                    "min": current_min,
                    "max": current_max,
                    "mean_ema": current_mean,
                    "_mean_sq_ema": current_mean_sq,
                }
            else:
                # Check for shape consistency before updating
                current_shape = np.shape(current_min)
                expected_shape = np.shape(stats[key]["min"])
                if current_shape != expected_shape:
                    print(
                        f"Warning: Shape mismatch for key '{key}' between items. "
                        f"Expected shape: {expected_shape}, "
                        f"Current item shape: {current_shape}. Skipping update for item {item_idx}."
                    )
                    continue

                stats[key]["min"] = np.minimum(stats[key]["min"], current_min)
                stats[key]["max"] = np.maximum(stats[key]["max"], current_max)

                stats[key]["mean_ema"] = (
                    alpha * current_mean + (1 - alpha) * stats[key]["mean_ema"]
                )
                stats[key]["_mean_sq_ema"] = (
                    alpha * current_mean_sq + (1 - alpha) * stats[key]["_mean_sq_ema"]
                )

    # Final standard deviation calculation
    for key in list(stats.keys()):
        if "_mean_sq_ema" in stats[key]:
            mean_ema = stats[key]["mean_ema"]
            mean_sq_ema = stats[key]["_mean_sq_ema"]

            var_ema = mean_sq_ema - np.square(mean_ema)
            # Ensure variance is non-negative for numerical stability
            var_ema = np.where(var_ema < 0, 0, var_ema)
            stats[key]["std_ema"] = np.sqrt(var_ema)

            del stats[key]["_mean_sq_ema"]
        else:
            print(f"Warning: Could not calculate std_ema for key '{key}'.")
            if "mean_ema" in stats[key]:
                stats[key]["std_ema"] = np.zeros_like(stats[key]["mean_ema"])
            else:
                del stats[key]  # Remove key if no stats could be computed

    return stats


def remove_key_if_exists_from_cfg(cfg, keys_to_remove):
    """
    Remove keys from a cfg.
    """
    for key in keys_to_remove:
        if "/" in key:
            key1, key2 = key.split("/")
            if key1 in cfg:
                if key2 in cfg[key1]:
                    del cfg[key1][key2]
        else:
            if key in cfg:
                del cfg[key]
    return cfg


def check_cfg_similar(cfg1, cfg2):
    """
    Check if two configs are similar. It ignore the keys that do not affect the stats.
    """
    keys_to_ignore = [
        "horizon",
        "IMAGE/brightness_aug",
        "IMAGE/contrast_aug",
        "IMAGE/saturation_aug",
        "IMAGE/hue_aug",
        "IMAGE/img_size",
        "IMAGE/crop_img",
        "IMAGE/cam_list",
        "LEROBOT/le_cam_list",
        "LEROBOT/fps",
    ]

    if cfg1.datasets != cfg2.datasets:
        return False

    if "lerobot_libero" not in cfg1.datasets:
        keys_to_ignore.extend(["LEROBOT_LIBERO"])
    if "lerobot" not in cfg1.datasets:
        keys_to_ignore.extend(["LEROBOT"])
    if "robotest" not in cfg1.datasets:
        keys_to_ignore.extend(["ROBOTEST"])
    if "metaworld" not in cfg1.datasets:
        keys_to_ignore.extend(["METAWORLD"])
    if "simpler" not in cfg1.datasets:
        keys_to_ignore.extend(["SIMPLER"])
    if "libero" not in cfg1.datasets:
        keys_to_ignore.extend(["LIBERO"])
    if "robomimic" not in cfg1.datasets:
        keys_to_ignore.extend(["ROBOMIMIC"])

    if cfg1.unifier == c.IMAGE:
        keys_to_ignore.extend(["PC", "PATH2D"])

    _cfg1 = deepcopy(cfg1)
    _cfg2 = deepcopy(cfg2)
    _cfg1 = remove_key_if_exists_from_cfg(_cfg1, keys_to_ignore)
    _cfg2 = remove_key_if_exists_from_cfg(_cfg2, keys_to_ignore)

    return _cfg1 == _cfg2


def get_unifier_stats(dataset, cfg, stat_keys):
    """
    Get the stats of the unifier.
    :param dataset: The dataset to get the stats of.
    :param cfg: The config to get the stats of.
    :param stat_keys: The keys to get the stats of.
    :return: A dictionary of the stats.
    """

    # checking if the stats are already present in the dataset
    # only works for single dataset unifiers
    if len(dataset.datasets) == 1:
        _dataset = list(dataset.datasets.values())[0]
        if hasattr(_dataset, "stats"):
            if all(key in _dataset.stats for key in stat_keys):
                return {key: _dataset.stats[key] for key in stat_keys}

    dataset_len = len(dataset)
    # cache the stats based on string of cfg and dataset len
    cache_dir = os.path.join(
        pathlib.Path(__file__).parent.parent.parent, "data", "stats"
    )
    os.makedirs(cache_dir, exist_ok=True)

    def get_cached_stats(cfg, dataset_len):
        """
        Check if stats exist in cache_dir and if they do, return them.
        If they don't exist, return None.
        """
        stats = None
        for file in os.listdir(cache_dir):
            if file.endswith(".pkl"):
                stored_cfg, stored_len, stored_stats = pickle.load(
                    open(os.path.join(cache_dir, file), "rb")
                )
                # check if the cfg is the same, if not, then we need to recalculate the stats
                if check_cfg_similar(stored_cfg, cfg):
                    if dataset_len != stored_len:
                        warnings.warn(
                            f"Dataset length mismatch for {file}. Expected {stored_len}, got {dataset_len}. Make sure you understand what you are doing."
                        )
                    cache_file = os.path.join(cache_dir, file)
                    warnings.warn(f"Loading stats from cache file {cache_file}")
                    stats = stored_stats
                    break
        return stats

    stats = get_cached_stats(cfg, dataset_len)

    if stats is None:
        warnings.warn(f"Calculating stats for {dataset_len} samples")
        new_stats = get_dataset_stats(dataset, stat_keys)

        # rechecking if stats exist now; if something already exists, then we should not overwrite it
        old_stats = get_cached_stats(cfg, dataset_len)
        if old_stats is not None:
            stats = old_stats
        else:
            stats = new_stats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_file = os.path.join(cache_dir, f"{timestamp}.pkl")
            pickle.dump((cfg, dataset_len, stats), open(cache_file, "wb"))
            warnings.warn(f"Dumped stats to cache file {cache_file}")

    return stats


if __name__ == "__main__":
    dummy_dataset = [
        {
            "data_a": np.random.rand(10, 5) * 10,  # 2D -> stats shape (5,)
            "data_b": np.random.rand(10, 4, 4, 3) + 5,  # 4D -> stats shape (3,)
            "data_c": np.random.randn(20),  # 1D -> stats shape (20,)
        },
        {
            "data_a": np.random.rand(10, 5) * 10,  # 2D -> stats shape (5,)
            "data_b": np.random.rand(10, 4, 4, 3) + 5,  # 4D -> stats shape (3,)
            "data_c": np.random.randn(20),  # 1D -> stats shape (20,)
        },
        {
            "data_a": np.random.rand(10, 5) * 10,  # 2D -> stats shape (5,)
            "data_b": np.random.rand(10, 4, 4, 3) + 5,  # 4D -> stats shape (3,)
            "data_c": np.random.randn(20),  # 1D -> stats shape (20,)
        },
    ]

    # Keys we want stats for (must correspond to data with ndim >= 1)
    keys_to_analyze = ["data_a", "data_b", "data_c"]
    dataset_stats = get_dataset_stats(dummy_dataset, keys_to_analyze, alpha=0.9)

    print("Calculated Dataset Statistics (ndim >= 1 asserted):")
    print(f"(Current Date: {np.datetime64('today')})")  # Example of using current date
    for key, stats_dict in dataset_stats.items():
        print(f"\n--- Stats for key: '{key}' ---")
        for stat_name, stat_value in stats_dict.items():
            # np.shape works for both scalars and arrays
            shape_str = str(np.shape(stat_value))
            if shape_str == "()":
                shape_str = "scalar"  # Prettier printing for scalar shapes

            # Handle potential 0-size arrays if key had only empty arrays
            if hasattr(stat_value, "size") and stat_value.size == 0:
                value_str = "N/A (empty)"
            else:
                value_str = str(np.round(stat_value, 3))
            print(f"  {stat_name}: shape={shape_str}, value={value_str}")
