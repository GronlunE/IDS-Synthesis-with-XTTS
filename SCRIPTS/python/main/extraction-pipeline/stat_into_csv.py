"""
Created on 15.11.2024

@author: GronlunE

Description:
    This script processes reference and synthesized audio files to calculate various statistics at both the phrase and clip levels.
    The statistics are extracted from `.mat` files containing data on fundamental frequency (f0), spectral tilt, syllable durations,
    and f0 deltas for each phrase in the audio files. The script calculates mean, standard deviation, percentile ranges, and log-transformed
    values for these features and aggregates them at both the phrase and clip levels. The resulting statistics are then saved to CSV files.

    The `create_dataframe` function converts the dictionary of statistics into a Pandas DataFrame. The `save_to_csv` function handles
    the saving of the statistics DataFrame into a CSV file. The `calculate_relevant_statistics` function computes various statistical
    measures for the provided data, including handling log transformations and delta data for specific features. The `main` function loads
    the `.mat` file, processes the reference and synthesized files, calculates the relevant statistics, and exports them to CSV files.

    The script processes reference and synthesized audio files in subdirectories of a root directory, loads corresponding data from a
    `.mat` file, and handles the extraction of statistics for each phrase. The statistics for each phrase are then aggregated at the
    clip level, and both sets of statistics (reference and synthesized) are saved to separate CSV files.

Usage:
    - Set the following directory paths:
      - `REF_DIR`: Path to the root directory containing reference audio files.
      - `SYNTH_DIR`: Path to the root directory containing synthesized audio files.
      - `REF_PHRASE_DIR`: Path to the directory containing reference phrase audio files.
      - `SYNTH_PHRASE_DIR`: Path to the directory containing synthesized phrase audio files.
      - `MAT_INPUT`: Path to the `.mat` file containing the feature data.
      - `CSV_OUTPUT_DIR`: Directory where the resulting CSV files will be saved.
    - Ensure that `.wav` files are organized in subdirectories (`original`, `denoised`, `enhanced`) within `REF_DIR` and `SYNTH_DIR`.
    - The script will load the `.mat` data and calculate statistics for each phrase in the reference and synthesized audio files.
    - The resulting statistics will be saved to two separate CSV files: "references.csv" and "syntheses.csv".

Dependencies:
    - `numpy` for numerical calculations and array operations.
    - `pandas` for creating and saving dataframes.
    - `scipy.io.loadmat` for loading `.mat` files.
    - `os` for file and directory operations.
    - `tqdm` for showing progress bars during file processing.

Notes:
    - This script assumes that the `.mat` file follows a specific structure (with keys like 'references' and 'syntheses') and contains data
      for the features (e.g., 'f0', 'f0_delta', 'syllable_durations', etc.) at both the phrase and clip levels.
    - The script handles empty or zero data in the `.mat` file by skipping those entries and printing a count of how many empty data
      points were encountered.
    - The calculated statistics are saved as CSV files with columns for each feature and calculated statistic (mean, standard deviation,
      percentiles, etc.).
    - It processes audio files in the "original", "denoised", and "enhanced" subdirectories of `REF_DIR` and `SYNTH_DIR`.

"""


import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from tqdm import tqdm

# Define directories and subdirectories
REF_DIR = r"G:\Research\XTTS_Test\DATA\IDS-with-XTTS\references"
SYNTH_DIR = r"G:\Research\XTTS_Test\DATA\IDS-with-XTTS\syntheses"
REF_PHRASE_DIR = r"G:\Research\XTTS_Test\DATA\IDS-with-XTTS\phrases\references"
SYNTH_PHRASE_DIR = r"G:\Research\XTTS_Test\DATA\IDS-with-XTTS\phrases\syntheses"
MAT_INPUT = r"G:\Research\XTTS_Test\CODE\python\output\data\mat\test\IDSXTTS.mat"
CSV_OUTPUT_DIR = r"G:\Research\XTTS_Test\CODE\python\output\fast_data"
SUB_DIRS = ["original", "denoised", "enhanced"]


def create_dataframe(stat_dict):
    records = []
    for file_key, stats in stat_dict.items():
        record = {'file_name': file_key}
        record.update(stats)
        records.append(record)

    return pd.DataFrame(records)


def save_to_csv(df, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, file_name)
    df.to_csv(csv_file_path, index=False)


def calculate_relevant_statistics(data, stat_name, is_f0=False, is_delta=False, is_syllable=False, is_phrase_level=False):
    stats = {}

    # Append "_phrase" if we're calculating at the phrase level
    suffix = "_phrase" if is_phrase_level else ""

    mean = np.mean(data)
    std = np.std(data)
    min_5 = np.percentile(data, 5)
    max_95 = np.percentile(data, 95)
    range_val = max_95 - min_5

    stats[f"{stat_name}_mean{suffix}"] = mean
    stats[f"{stat_name}_std{suffix}"] = std
    stats[f"{stat_name}_range{suffix}"] = range_val
    stats[f"{stat_name}_min5{suffix}"] = min_5
    stats[f"{stat_name}_max95{suffix}"] = max_95

    if is_f0 or is_syllable:
        log_data = np.log(data)
        stats[f"{stat_name}_log_mean{suffix}"] = np.mean(log_data)
        stats[f"{stat_name}_log_std{suffix}"] = np.std(log_data)
        stats[f"{stat_name}_log_min5{suffix}"] = np.percentile(log_data, 5)
        stats[f"{stat_name}_log_max95{suffix}"] = np.percentile(log_data, 95)
        stats[f"{stat_name}_log_range{suffix}"] = stats[f"{stat_name}_log_max95{suffix}"] - stats[f"{stat_name}_log_min5{suffix}"]

    if is_delta:
        abs_data = np.abs(data)
        log_abs_data = np.log(abs_data + 1e-10)
        stats[f"{stat_name}_abs_log_mean{suffix}"] = np.mean(log_abs_data)
        stats[f"{stat_name}_abs_log_std{suffix}"] = np.std(log_abs_data)
        stats[f"{stat_name}_abs_log_min_5{suffix}"] = np.percentile(log_abs_data, 5)
        stats[f"{stat_name}_abs_log_max_95{suffix}"] = np.percentile(log_abs_data, 95)
        stats[f"{stat_name}_abs_log_range{suffix}"] = stats[f"{stat_name}_abs_log_max_95{suffix}"] - stats[f"{stat_name}_abs_log_min_5{suffix}"]

    return stats


def main():
    # Load data from .mat files
    mat_data_dict = loadmat(MAT_INPUT, simplify_cells=True)
    mat_data_dict = mat_data_dict["DATA"]

    all_ref_stats = {}
    all_synth_stats = {}

    ref_files = []
    synth_files = []

    for sub_dir in SUB_DIRS:
        ref_files.extend([f for f in os.listdir(os.path.join(REF_DIR, sub_dir)) if f.endswith('.wav')])
        synth_files.extend([f for f in os.listdir(os.path.join(SYNTH_DIR, sub_dir)) if f.endswith('.wav')])

    ref_files = np.array(ref_files).flatten()
    synth_files = np.array(synth_files).flatten()

    # Initialize empty data count
    empty_data_count = 0

    # Process statistics for references
    for file_name in tqdm(ref_files, desc="Processing Reference Files"):
        base_file_name = os.path.splitext(file_name)[0]  # Remove .wav extension
        category = base_file_name.split("_")[0]
        phrases = mat_data_dict['references'][category][base_file_name].keys()

        # Initialize phrase-level data collections
        clip_level_data = {
            'f0': np.array([]),
            'f0_delta': np.array([]),
            'spectral_tilt': np.array([]),
            'syllable_durations': np.array([])
        }

        # Initialize for gathering phrase-level stats
        phrase_stats = {}

        # Process each phrase for the current file
        for phrase_key in phrases:
            for stat_name in ['f0', 'f0_delta', 'spectral_tilt', 'syllable_durations']:
                data = mat_data_dict['references'][category][base_file_name][phrase_key][stat_name]

                is_f0 = stat_name == 'f0'
                is_delta = stat_name in ['f0_delta', 'f0_delta_thrline', 'f0_delta_thrchange']
                is_syllable = stat_name == 'syllable_durations'

                # Check if data is empty and skip if so
                if isinstance(data, (list, np.ndarray)) and (len(data) == 0):
                    empty_data_count += 1
                    # print(f"Empty or zero data for file: {file_name}, phrase: {phrase_key}, count: {empty_data_count}")
                    continue  # Skip to the next iteration
                elif is_delta and (data.size == 3 and data[0] == 0 and data[1] == 0):
                    # Handle the case where data has three elements and the first two are zeros
                    empty_data_count += 1
                    # print(f"Data with zeros for file: {file_name}, phrase: {phrase_key}, count: {empty_data_count}")
                    continue  # Skip to the next iteration

                data = np.array([data]).flatten()
                clip_level_data[stat_name] = np.concatenate((clip_level_data[stat_name], data))

                # Collect phrase-level statistics
                measurements = calculate_relevant_statistics(data, stat_name, is_f0, is_delta, is_syllable, is_phrase_level=True)

                for data_key in measurements.keys():
                    if data_key not in phrase_stats.keys():
                        # Initialize as a numpy array
                        phrase_stats[data_key] = np.array([measurements[data_key]])
                    else:
                        # Concatenate the new data to the existing array
                        phrase_stats[data_key] = np.concatenate((phrase_stats[data_key], np.array([measurements[data_key]])))
        clip_stats = {}

        # Once all phrases are processed for the file, aggregate data across phrases to compute clip-level statistics
        for stat_name in clip_level_data.keys():
            is_f0 = stat_name == 'f0'
            is_delta = stat_name in ['f0_delta', 'f0_delta_thrline', 'f0_delta_thrchange']
            is_syllable = stat_name == 'syllable_durations'
            data = clip_level_data[stat_name]
            measurements = calculate_relevant_statistics(data, stat_name, is_f0, is_delta, is_syllable, is_phrase_level=False)
            for data_key in measurements.keys():
                clip_stats[data_key] = measurements[data_key]

        # Aggregate phrase-level statistics (mean and std across phrases)
        for stat_key in phrase_stats.keys():
            phrase_stats[stat_key] = np.mean(np.array(phrase_stats[stat_key]))

        # Add both clip-level and phrase-level aggregated stats to the global dict
        all_ref_stats[file_name] = {**clip_stats, **phrase_stats}

    # Process statistics for syntheses (similar to above)
    for file_name in tqdm(synth_files, desc="Processing Synthesized Files"):
        base_file_name = os.path.splitext(file_name)[0].replace("xtts_", "")  # Remove .wav extension and "xtts"
        category = base_file_name.split("_")[0]
        phrases = mat_data_dict['syntheses'][category][base_file_name].keys()

        # Initialize phrase-level data collections
        clip_level_data = {
            'f0': np.array([]),
            'f0_delta': np.array([]),
            'spectral_tilt': np.array([]),
            'syllable_durations': np.array([])
        }
        # Initialize for gathering phrase-level stats
        phrase_stats = {}

        # Process each phrase for the current file
        # Process each phrase for the current file
        for phrase_key in phrases:
            for stat_name in ['f0', 'f0_delta', 'spectral_tilt', 'syllable_durations']:
                data = mat_data_dict['syntheses'][category][base_file_name][phrase_key][stat_name]

                is_f0 = stat_name == 'f0'
                is_delta = stat_name in ['f0_delta', 'f0_delta_thrline', 'f0_delta_thrchange']
                is_syllable = stat_name == 'syllable_durations'

                # Check if data is empty and skip if so
                if isinstance(data, (list, np.ndarray)) and (len(data) == 0):
                    empty_data_count += 1
                    # print(f"Empty or zero data for file: {file_name}, phrase: {phrase_key}, count: {empty_data_count}")
                    continue  # Skip to the next iteration
                elif is_delta and (data.size == 3 and data[0] == 0 and data[1] == 0):
                    # Handle the case where data has three elements and the first two are zeros
                    empty_data_count += 1
                    # print(f"Data with zeros for file: {file_name}, phrase: {phrase_key}, count: {empty_data_count}")
                    continue  # Skip to the next iteration

                data = np.array([data]).flatten()
                clip_level_data[stat_name] = np.concatenate((clip_level_data[stat_name], data))

                # Collect phrase-level statistics
                measurements = calculate_relevant_statistics(data, stat_name, is_f0, is_delta, is_syllable,
                                                             is_phrase_level=True)

                for data_key in measurements.keys():
                    if data_key not in phrase_stats.keys():
                        # Initialize as a numpy array
                        phrase_stats[data_key] = np.array([measurements[data_key]])
                    else:
                        # Concatenate the new data to the existing array
                        phrase_stats[data_key] = np.concatenate(
                            (phrase_stats[data_key], np.array([measurements[data_key]])))
        clip_stats = {}

        # Once all phrases are processed for the file, aggregate data across phrases to compute clip-level statistics
        for stat_name in clip_level_data.keys():
            is_f0 = stat_name == 'f0'
            is_delta = stat_name in ['f0_delta', 'f0_delta_thrline', 'f0_delta_thrchange']
            is_syllable = stat_name == 'syllable_durations'
            data = clip_level_data[stat_name]
            measurements = calculate_relevant_statistics(data, stat_name, is_f0, is_delta, is_syllable,
                                                         is_phrase_level=False)
            for data_key in measurements.keys():
                clip_stats[data_key] = measurements[data_key]

        # Aggregate phrase-level statistics (mean and std across phrases)
        for stat_key in phrase_stats.keys():
            phrase_stats[stat_key] = np.mean(np.array(phrase_stats[stat_key]))

        # Add both clip-level and phrase-level aggregated stats to the global dict
        all_synth_stats[file_name] = {**clip_stats, **phrase_stats}

    print(f"Empty data count: {empty_data_count}")
    # Create dataframes for both reference and synthesized statistics
    ref_df = create_dataframe(all_ref_stats)
    synth_df = create_dataframe(all_synth_stats)

    # Save both to CSV files
    save_to_csv(ref_df, CSV_OUTPUT_DIR, "references.csv")
    save_to_csv(synth_df, CSV_OUTPUT_DIR, "syntheses.csv")


if __name__ == "__main__":
    main()
