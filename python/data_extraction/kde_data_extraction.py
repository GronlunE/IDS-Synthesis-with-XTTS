"""
Created on 31.8. 2024

@author: GronlunE

Description:

This script processes audio files to compute various acoustic statistics and saves the results in both pickle and MAT file formats. It handles the extraction of pitch, spectral tilt, pitch delta, and syllable duration statistics from audio files in specified directories.

The script provides the following functionality:
- `process_files`: Processes audio files in given subdirectories, computes various statistics, and stores the results in dictionaries.
- `save_results`: Saves the computed statistics to both `.pkl` and `.mat` files in a specified output directory.
- `calculate_kde_data`: Orchestrates the processing of reference and synthesized audio files, computes statistics, and saves the results.

Usage:
- Ensure the required modules (`librosa`, `parselmouth`, `scipy`, and custom functions from `csv_data_extraction`) are installed and accessible.
- Define the appropriate directory paths for input and output.
- Run the script to process the audio files and save the computed statistics.

Dependencies:
- `os`, `pickle`, `librosa`, `parselmouth`, `tqdm` for file handling, data manipulation, and progress indication.
- `csv_data_extraction` module for obtaining statistical functions.
- `scipy.io.savemat` for saving data in MAT format.

"""

import os
import pickle
import librosa
import parselmouth
from tqdm import tqdm
from csv_data_extraction import get_f0_statistics, get_f0_delta_statistics, get_spectral_tilt_statistics, get_syllable_duration_statistics
from parselmouth.praat import call
from scipy.io import savemat


def process_files(base_dir, category, f0_dict, spectral_tilt_dict, f0_delta_dict, syllable_duration_dict, sub_dirs):
    """
    Process audio files in specified subdirectories to compute and store various statistics.

    :param base_dir: Base directory containing the subdirectories with audio files.
    :param category: The category of files being processed (e.g., "Reference" or "Synthesis").
    :param f0_dict: Dictionary to store computed F0 statistics.
    :param spectral_tilt_dict: Dictionary to store computed spectral tilt statistics.
    :param f0_delta_dict: Dictionary to store computed F0 delta statistics.
    :param syllable_duration_dict: Dictionary to store computed syllable duration statistics.
    :param sub_dirs: List of subdirectories to process.
    :return: None
    """
    for sub_dir in sub_dirs:
        directory = os.path.join(base_dir, sub_dir)
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]
        files.sort()

        for file in tqdm(files, desc=f"Processing {category} - {sub_dir}", leave=False):

            # Compute statistics
            target_sr = 16000  # Target sampling rate
            y, sr = librosa.load(file, sr=target_sr)
            snd = parselmouth.Sound(y, sr)
            pitch = call(snd, "To Pitch", 0.0, 75, 600)

            f0_values = get_f0_statistics(pitch)[-1]
            spectral_tilts = get_spectral_tilt_statistics(y, pitch, sr)[-1]
            f0_delta_values = get_f0_delta_statistics(f0_values)[-1]
            syllable_durations = get_syllable_duration_statistics(file)[-1]

            # Store the results
            f0_dict[category][sub_dir].append(f0_values)
            spectral_tilt_dict[category][sub_dir].append(spectral_tilts)
            f0_delta_dict[category][sub_dir].append(f0_delta_values)
            syllable_duration_dict[category][sub_dir].append(syllable_durations)


def save_results(stat_name, stat_dict, output_dir):
    """
    Save the computed statistics to both .pkl and .mat files.

    :param stat_name: The base name for the output files.
    :param stat_dict: Dictionary containing the statistics to be saved.
    :param output_dir: Directory where the output files will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define file paths
    pkl_file = os.path.join(output_dir, f"{stat_name}_data.pkl")
    mat_file = os.path.join(output_dir, f"{stat_name}_data.mat")

    # Save to .pkl file
    with open(pkl_file, "wb") as f:
        pickle.dump(stat_dict, f)

    # Save to .mat file
    savemat(mat_file, stat_dict)


def calculate_kde_data():
    """
    Calculate statistics from reference and synthesized audio files and save the results.

    :return: None
    """
    # Define the directories
    ref_dir = r"synthesis_stage/references"
    synth_dir = r"synthesis_stage/synthesized"
    sub_dirs = ["denoised", "enhanced", "original"]
    output_dir = r"plot_data/kde"

    # Initialize the results dictionary
    f0_dict = {
        "Reference": {sub_dir: [] for sub_dir in sub_dirs},
        "Synthesis": {sub_dir: [] for sub_dir in sub_dirs}
    }

    spectral_tilt_dict = {
        "Reference": {sub_dir: [] for sub_dir in sub_dirs},
        "Synthesis": {sub_dir: [] for sub_dir in sub_dirs}
    }
    f0_delta_dict = {
        "Reference": {sub_dir: [] for sub_dir in sub_dirs},
        "Synthesis": {sub_dir: [] for sub_dir in sub_dirs}
    }

    syllable_duration_dict = {
        "Reference": {sub_dir: [] for sub_dir in sub_dirs},
        "Synthesis": {sub_dir: [] for sub_dir in sub_dirs}
    }

    # Process both reference and synthesized files
    process_files(ref_dir, "Reference", f0_dict, spectral_tilt_dict, f0_delta_dict, syllable_duration_dict, sub_dirs)
    process_files(synth_dir, "Synthesis", f0_dict, spectral_tilt_dict, f0_delta_dict, syllable_duration_dict, sub_dirs)

    # Save each type of statistic
    save_results('f0', f0_dict, output_dir)
    save_results('spectral_tilt', spectral_tilt_dict, output_dir)
    save_results('f0_delta', f0_delta_dict, output_dir)
    save_results('syllable_duration', syllable_duration_dict, output_dir)
