"""
Created on 15.11.2024

@author: GronlunE

Description:
    This script processes speech data by extracting various statistical features from reference and synthesized audio files.
    It computes pitch, F0 (fundamental frequency), spectral tilt, and syllable durations for each file, storing the results in a dictionary structure.
    The script processes the audio files, organizes the extracted features by base file and phrase, and then saves the results in the .mat file format.

    The `process_files` function processes the audio files in specified directories, computes the desired statistics,
    and stores them in a dictionary under the corresponding base file and phrase. A progress bar is displayed during file processing.

    The `collect` function calls the `process_files` function for both reference and synthesized files, then saves the extracted features to a .mat file.

Usage:
    - Ensure the necessary audio files are located in the `references` and `syntheses` directories, along with the corresponding phrases.
    - Set the directories and paths according to your project setup.
    - The script will process the audio files, compute statistical features, and save them in a `.mat` file under a top-level key "DATA".

Dependencies:
    - `scipy` for saving and loading MATLAB files.
    - `librosa` for audio processing and feature extraction.
    - `parselmouth` for pitch extraction and analysis.
    - `tqdm` for progress bar visualization.
    - Custom functions from the `data_extraction_functions` module for feature extraction (e.g., pitch, spectral tilt, syllable durations).
    - Standard Python libraries (`os`, `glob`).

Notes:
    - The target sampling rate for all audio files is set to 16 kHz.
    - The script assumes that audio files are in `.wav` format and follow a specific naming convention (e.g., `xtts_<base_file>_phrase_<number>.wav`).
    - The extracted features are saved under the "DATA" key in the output .mat file.

"""

from scipy.io import savemat, loadmat
from data_extraction_functions import *
import os
from tqdm import tqdm
import librosa
import parselmouth


# Define directories and subdirectories
REF_DIR = r"G:\Research\XTTS_Test\DATA\IDS-with-XTTS\references"
SYNTH_DIR = r"G:\Research\XTTS_Test\DATA\IDS-with-XTTS\syntheses"
REF_PHRASE_DIR = r"G:\Research\XTTS_Test\DATA\IDS-with-XTTS\phrases\references"
SYNTH_PHRASE_DIR = r"G:\Research\XTTS_Test\DATA\IDS-with-XTTS\phrases\syntheses"
SUB_DIRS = ["original", "denoised", "enhanced"]
MAT_OUTPUT_DIR = r"G:\Research\XTTS_Test\CODE\python\output\data\mat\test"
SYLDURS = loadmat(r"G:\Research\XTTS_Test\CODE\python\output\data\mat\test\syllable_durations_all.mat", simplify_cells=True)


def process_files(base_dir, category, sub_dirs, results_dicts):
    for sub_dir in sub_dirs:
        directory = os.path.join(base_dir, sub_dir)
        files = [f for f in os.listdir(directory) if f.endswith(".wav")]

        for file in tqdm(files, desc=f"Processing {category} - {sub_dir}", leave=False):
            file_path = os.path.join(directory, file)

            # Extract base file and phrase from the filename
            base_file, phrase = extract_base_and_phrase(file)

            # Compute statistics
            target_sr = 16000  # Target sampling rate
            y, sr = librosa.load(file_path, sr=target_sr)
            snd = parselmouth.Sound(y, sr)
            pitch = call(snd, "To Pitch", 0.0, 75, 500)

            # Compute statistics
            f0_values = get_f0_statistics(pitch)
            delta_values = get_f0_delta_statistics(pitch)
            spectral_tilt_values = get_spectral_tilt_statistics(y, pitch, sr)
            syllable_durations = get_syllable_duration_statistics(file_path, SYLDURS)

            # Store the results under base_file and phrase
            results_dicts[category][sub_dir][base_file][phrase] = {
                'f0': f0_values,
                'f0_delta': delta_values,
                'spectral_tilt': spectral_tilt_values,
                'syllable_durations': syllable_durations
            }


def save_results(ref_stat_dict, synth_stat_dict, output_dir):
    """
    Save the results to .mat file format under a top key "DATA".
    """
    os.makedirs(output_dir, exist_ok=True)

    # Combine both dictionaries under a top key "DATA"
    combined_dict = {
        'DATA': {
            'references': ref_stat_dict,
            'syntheses': synth_stat_dict
        }
    }

    mat_file = os.path.join(output_dir, "IDSXTTS.mat")
    savemat(mat_file, combined_dict)


def extract_base_and_phrase(filename):
    """
    Extracts the base file name and phrase number from the filename,
    removing the 'xtts_' prefix and the '.wav' suffix.
    """
    filename = filename.replace('.wav', '')
    if filename.startswith('xtts_'):
        filename = filename[5:]  # Remove the 'xtts_' prefix

    base_file = '_'.join(filename.split('_')[:-2])  # Extract everything before "_phrase_n"
    phrase = filename.split('_')[-2] + '_' + filename.split('_')[-1]  # Get "phrase_n"
    return base_file, phrase


def preallocate_results(base_dir, category, sub_dirs):
    """
    Preallocates the results_dicts with appropriate base file and phrase keys.
    """
    results_dicts = {category: {sub_dir: {} for sub_dir in sub_dirs}}

    for sub_dir in sub_dirs:
        directory = os.path.join(base_dir, sub_dir)
        files = [f for f in os.listdir(directory) if f.endswith(".wav")]

        for file in files:
            base_file, phrase = extract_base_and_phrase(file)

            # Check if base_file key exists, if not, create it
            if base_file not in results_dicts[category][sub_dir]:
                results_dicts[category][sub_dir][base_file] = {}

            # Check if phrase key exists for this base_file, if not, create it
            if phrase not in results_dicts[category][sub_dir][base_file]:
                results_dicts[category][sub_dir][base_file][phrase] = {
                    'f0': [],
                    'f0_delta': [],
                    'spectral_tilt': [],
                    'syllable_duration': []
                }

    return results_dicts


def collect():
    # Preallocate the results dictionary for both reference and synthesis files
    ref_results_dicts = preallocate_results(REF_PHRASE_DIR, "Reference", SUB_DIRS)
    synth_results_dicts = preallocate_results(SYNTH_PHRASE_DIR, "Synthesis", SUB_DIRS)

    # Process both reference and synthesized files
    process_files(REF_PHRASE_DIR, "Reference", SUB_DIRS, ref_results_dicts)
    process_files(SYNTH_PHRASE_DIR, "Synthesis", SUB_DIRS, synth_results_dicts)

    # Save each type of statistic to .mat files
    save_results(ref_results_dicts["Reference"], synth_results_dicts["Synthesis"], MAT_OUTPUT_DIR)


collect()
