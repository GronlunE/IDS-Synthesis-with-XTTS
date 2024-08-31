"""

"""
import os
import pickle
import numpy as np
import librosa
import parselmouth
from tqdm import tqdm
from csv_data_extraction import get_f0_statistics, get_f0_delta_statistics, get_spectral_tilt_statistics, get_syllable_duration_statistics
import parselmouth
from parselmouth.praat import call
from scipy.io import savemat


def process_files(base_dir, category, f0_dict, spectral_tilt_dict, f0_delta_dict, syllable_duration_dict, sub_dirs):
    """

    :param base_dir:
    :param category:
    :param f0_dict:
    :param spectral_tilt_dict:
    :param f0_delta_dict:
    :param syllable_duration_dict:
    :param sub_dirs:
    :return:
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
    Save the results to both .pkl and .mat files in the specified output directory.

    :param stat_name: The base name for the output files.
    :param stat_dict: The dictionary of data to save.
    :param output_dir: The directory where the files should be saved.
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

    :return:
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
    process_files(synth_dir, "Synthesis",f0_dict, spectral_tilt_dict, f0_delta_dict, syllable_duration_dict, sub_dirs)

    # Save each type of statistic
    save_results('f0', f0_dict, output_dir)
    save_results('spectral_tilt', spectral_tilt_dict, output_dir)
    save_results('f0_delta', f0_delta_dict, output_dir)
    save_results('syllable_duration', syllable_duration_dict, output_dir)
