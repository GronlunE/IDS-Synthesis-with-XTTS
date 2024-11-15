import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

# Define file paths and output directory
SYNTHESIZED_FILE = r"G:\Research\XTTS_Test\DATA\IDS-ADS\IDS-ADS_syntheses.csv"
REFERENCES_FILE = r"G:\Research\XTTS_Test\DATA\IDS-ADS\IDS-ADS_references.csv"
OUTPUT_DIR = r"G:\Research\XTTS_Test\DATA\IDS-ADS\figures\kde"

color_map = {
    'ADS_original_reference': 'green',
    'IDS_original_reference': 'lightgreen',
    'ADS_original_synthesis': 'green',
    'IDS_original_synthesis': 'lightgreen',
    'ADS_denoised_reference': 'cornflowerblue',
    'IDS_denoised_reference': 'lightblue',
    'ADS_denoised_synthesis': 'cornflowerblue',
    'IDS_denoised_synthesis': 'lightblue',
    'ADS_enhanced_reference': 'darkred',
    'IDS_enhanced_reference': 'lightcoral',
    'ADS_enhanced_synthesis': 'darkred',
    'IDS_enhanced_synthesis': 'lightcoral',
}


def load_data(synthesized_file, references_file):
    synthesized_df = pd.read_csv(synthesized_file)
    references_df = pd.read_csv(references_file)
    return synthesized_df, references_df


def prepare_data(synthesized_df, references_df):
    synthesized_df['category'] = synthesized_df['file_name'].str.extract(r'(denoised|enhanced|original)')[0]
    references_df['category'] = references_df['file_name'].str.extract(r'(denoised|enhanced|original)')[0]

    synthesized_df['speaker'] = synthesized_df['file_name'].str.extract(r'Baby (\d+)')[0]
    references_df['speaker'] = references_df['file_name'].str.extract(r'Baby (\d+)')[0]

    return synthesized_df, references_df


def plot_kde_subplot(ax, plot_data, labels, title):
    """Plot KDE for the provided data with the specified color map."""
    handles = []
    for data, label in zip(plot_data, labels):
        if data.size > 0:  # Check if there is data to plot
            kde = gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 1000)
            line_style = '--' if 'synthesis' in label else '-'
            line, = ax.plot(x, kde(x), label=label, color=color_map[label], linewidth=2.5, linestyle=line_style)
            handles.append(line)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(handles=handles, loc='best')


def plot_all_kdes(synthesized_df, references_df, variables_with_units, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    speakers = synthesized_df['speaker'].unique()

    for speaker in speakers:
        synthesized_speaker_df = synthesized_df[synthesized_df['speaker'] == speaker]
        references_speaker_df = references_df[references_df['speaker'] == speaker]

        for var_name, (title, unit) in variables_with_units.items():
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            plt.suptitle(f'KDE Plot for {title} - Speaker {speaker}', fontsize=16)

            # 1. Original Data
            original_data = [
                references_speaker_df[references_speaker_df['file_name'].str.contains("ADS") & (
                            references_speaker_df['category'] == "original")][var_name].dropna().values,
                references_speaker_df[references_speaker_df['file_name'].str.contains("IDS") & (
                            references_speaker_df['category'] == "original")][var_name].dropna().values,
                synthesized_speaker_df[synthesized_speaker_df['file_name'].str.contains("ADS") & (
                            synthesized_speaker_df['category'] == "original")][var_name].dropna().values,
                synthesized_speaker_df[synthesized_speaker_df['file_name'].str.contains("IDS") & (
                            synthesized_speaker_df['category'] == "original")][var_name].dropna().values
            ]

            original_labels = [
                'ADS_original_reference',
                'IDS_original_reference',
                'ADS_original_synthesis',
                'IDS_original_synthesis',
            ]
            plot_kde_subplot(axs[0, 0], original_data, original_labels, 'Original Data')

            # 2. Denoised Data
            denoised_data = [
                references_speaker_df[references_speaker_df['file_name'].str.contains("ADS") & (
                            references_speaker_df['category'] == "denoised")][var_name].dropna().values,
                references_speaker_df[references_speaker_df['file_name'].str.contains("IDS") & (
                            references_speaker_df['category'] == "denoised")][var_name].dropna().values,
                synthesized_speaker_df[synthesized_speaker_df['file_name'].str.contains("ADS") & (
                            synthesized_speaker_df['category'] == "denoised")][var_name].dropna().values,
                synthesized_speaker_df[synthesized_speaker_df['file_name'].str.contains("IDS") & (
                            synthesized_speaker_df['category'] == "denoised")][var_name].dropna().values
            ]

            denoised_labels = [
                'ADS_denoised_reference',
                'IDS_denoised_reference',
                'ADS_denoised_synthesis',
                'IDS_denoised_synthesis',
            ]
            plot_kde_subplot(axs[0, 1], denoised_data, denoised_labels, 'Denoised Data')

            # 3. Enhanced Data
            enhanced_data = [
                references_speaker_df[references_speaker_df['file_name'].str.contains("ADS") & (
                            references_speaker_df['category'] == "enhanced")][var_name].dropna().values,
                references_speaker_df[references_speaker_df['file_name'].str.contains("IDS") & (
                            references_speaker_df['category'] == "enhanced")][var_name].dropna().values,
                synthesized_speaker_df[synthesized_speaker_df['file_name'].str.contains("ADS") & (
                            synthesized_speaker_df['category'] == "enhanced")][var_name].dropna().values,
                synthesized_speaker_df[synthesized_speaker_df['file_name'].str.contains("IDS") & (
                            synthesized_speaker_df['category'] == "enhanced")][var_name].dropna().values
            ]

            enhanced_labels = [
                'ADS_enhanced_reference',
                'IDS_enhanced_reference',
                'ADS_enhanced_synthesis',
                'IDS_enhanced_synthesis',
            ]
            plot_kde_subplot(axs[1, 0], enhanced_data, enhanced_labels, 'Enhanced Data')

            # 4. Original vs Enhanced
            combined_data = [
                references_speaker_df[references_speaker_df['file_name'].str.contains("ADS") & (
                            references_speaker_df['category'] == "original")][var_name].dropna().values,
                references_speaker_df[references_speaker_df['file_name'].str.contains("IDS") & (
                            references_speaker_df['category'] == "original")][var_name].dropna().values,
                references_speaker_df[references_speaker_df['file_name'].str.contains("ADS") & (
                            references_speaker_df['category'] == "enhanced")][var_name].dropna().values,
                references_speaker_df[references_speaker_df['file_name'].str.contains("IDS") & (
                            references_speaker_df['category'] == "enhanced")][var_name].dropna().values,
                synthesized_speaker_df[synthesized_speaker_df['file_name'].str.contains("ADS") & (
                            synthesized_speaker_df['category'] == "original")][var_name].dropna().values,
                synthesized_speaker_df[synthesized_speaker_df['file_name'].str.contains("IDS") & (
                            synthesized_speaker_df['category'] == "original")][var_name].dropna().values,
                synthesized_speaker_df[synthesized_speaker_df['file_name'].str.contains("ADS") & (
                            synthesized_speaker_df['category'] == "enhanced")][var_name].dropna().values,
                synthesized_speaker_df[synthesized_speaker_df['file_name'].str.contains("IDS") & (
                            synthesized_speaker_df['category'] == "enhanced")][var_name].dropna().values,
            ]

            combined_labels = [
                'ADS_original_reference',
                'IDS_original_reference',
                'ADS_enhanced_reference',
                'IDS_enhanced_reference',
                'ADS_original_synthesis',
                'IDS_original_synthesis',
                'ADS_enhanced_synthesis',
                'IDS_enhanced_synthesis',
            ]
            plot_kde_subplot(axs[1, 1], combined_data, combined_labels, 'Original vs Enhanced')

            # Save the plot to the specified path
            filename = f"Baby{speaker}_{var_name}_kde.pdf"
            file_path = os.path.join(output_dir, filename)
            plt.savefig(file_path)
            plt.close()

    print("KDE plots for individual speakers have been generated and saved.")


def plot_kdes_for_all_speakers(synthesized_df, references_df, variables_with_units, output_dir):
    """Plot KDEs for all variables combined across all speakers."""
    os.makedirs(output_dir, exist_ok=True)

    for var_name, (title, unit) in variables_with_units.items():
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        plt.suptitle(f'KDE Plot for {title} - All Speakers', fontsize=16)

        # Original Data
        original_data = [
            references_df[references_df['file_name'].str.contains("ADS") & (references_df['category'] == "original")][
                var_name].dropna().values,
            references_df[references_df['file_name'].str.contains("IDS") & (references_df['category'] == "original")][
                var_name].dropna().values,
            synthesized_df[
                synthesized_df['file_name'].str.contains("ADS") & (synthesized_df['category'] == "original")][
                var_name].dropna().values,
            synthesized_df[
                synthesized_df['file_name'].str.contains("IDS") & (synthesized_df['category'] == "original")][
                var_name].dropna().values
        ]

        original_labels = [
            'ADS_original_reference',
            'IDS_original_reference',
            'ADS_original_synthesis',
            'IDS_original_synthesis',
        ]
        plot_kde_subplot(axs[0, 0], original_data, original_labels, 'Original Data')

        # Denoised Data
        denoised_data = [
            references_df[references_df['file_name'].str.contains("ADS") & (references_df['category'] == "denoised")][
                var_name].dropna().values,
            references_df[references_df['file_name'].str.contains("IDS") & (references_df['category'] == "denoised")][
                var_name].dropna().values,
            synthesized_df[
                synthesized_df['file_name'].str.contains("ADS") & (synthesized_df['category'] == "denoised")][
                var_name].dropna().values,
            synthesized_df[
                synthesized_df['file_name'].str.contains("IDS") & (synthesized_df['category'] == "denoised")][
                var_name].dropna().values
        ]

        denoised_labels = [
            'ADS_denoised_reference',
            'IDS_denoised_reference',
            'ADS_denoised_synthesis',
            'ADS_denoised_synthesis',
        ]
        plot_kde_subplot(axs[0, 1], denoised_data, denoised_labels, 'Denoised Data')

        # Enhanced Data
        enhanced_data = [
            references_df[references_df['file_name'].str.contains("ADS") & (references_df['category'] == "enhanced")][
                var_name].dropna().values,
            references_df[references_df['file_name'].str.contains("IDS") & (references_df['category'] == "enhanced")][
                var_name].dropna().values,
            synthesized_df[
                synthesized_df['file_name'].str.contains("ADS") & (synthesized_df['category'] == "enhanced")][
                var_name].dropna().values,
            synthesized_df[
                synthesized_df['file_name'].str.contains("IDS") & (synthesized_df['category'] == "enhanced")][
                var_name].dropna().values
        ]

        enhanced_labels = [
            'ADS_enhanced_reference',
            'IDS_enhanced_reference',
            'ADS_enhanced_synthesis',
            'IDS_enhanced_synthesis',
        ]
        plot_kde_subplot(axs[1, 0], enhanced_data, enhanced_labels, 'Enhanced Data')

        # Original vs Enhanced
        combined_data = [
            references_df[references_df['file_name'].str.contains("ADS") & (references_df['category'] == "original")][
                var_name].dropna().values,
            references_df[references_df['file_name'].str.contains("IDS") & (references_df['category'] == "original")][
                var_name].dropna().values,
            references_df[references_df['file_name'].str.contains("ADS") & (references_df['category'] == "enhanced")][
                var_name].dropna().values,
            references_df[references_df['file_name'].str.contains("IDS") & (references_df['category'] == "enhanced")][
                var_name].dropna().values,
            synthesized_df[
                synthesized_df['file_name'].str.contains("ADS") & (synthesized_df['category'] == "original")][
                var_name].dropna().values,
            synthesized_df[
                synthesized_df['file_name'].str.contains("IDS") & (synthesized_df['category'] == "original")][
                var_name].dropna().values,
            synthesized_df[
                synthesized_df['file_name'].str.contains("ADS") & (synthesized_df['category'] == "enhanced")][
                var_name].dropna().values,
            synthesized_df[
                synthesized_df['file_name'].str.contains("IDS") & (synthesized_df['category'] == "enhanced")][
                var_name].dropna().values,
        ]

        combined_labels = [
            'ADS_original_reference',
            'IDS_original_reference',
            'ADS_enhanced_reference',
            'IDS_enhanced_reference',
            'ADS_original_synthesis',
            'IDS_original_synthesis',
            'ADS_enhanced_synthesis',
            'IDS_enhanced_synthesis',
        ]
        plot_kde_subplot(axs[1, 1], combined_data, combined_labels, 'Original vs Enhanced')

        # Save the plot to the specified path
        filename = f"AllSpeakers_{var_name}_kde.pdf"
        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path)
        plt.close()

    print("KDE plots for all speakers have been generated and saved.")


def create_kde_of_csv_stats():
    synthesized_df, references_df = load_data(SYNTHESIZED_FILE, REFERENCES_FILE)
    synthesized_df, references_df = prepare_data(synthesized_df, references_df)

    variables_with_units = {
        "f0_log_std": ("F0 Standard Deviation (Log)", "Log(Hz)"),
        "f0_log_mean": ("F0 Mean (Log)", "Log(Hz)"),
        "f0_delta_abs_log_std": ("Absolute Delta F0 Standard Deviation (Log)", "Log(Hz)"),
        "f0_delta_abs_log_mean": ("Absolute Delta F0 Mean (Log)", "Log(Hz)"),
        "spectral_tilt_std": ("Spectral Tilt Standard Deviation", "dB/freqbin"),
        "spectral_tilt_mean": ("Spectral Tilt Mean", "dB/freqbin"),
        "syllable_durations_log_std": ("Syllable Durations Standard Deviation (Log)", "Log(s)"),
        "syllable_durations_log_mean": ("Syllable Durations Mean (Log)", "Log(s)"),
        "f0_log_std_phrase": ("F0 Standard Deviation (Log, Phrase)", "Log(Hz)"),
        "f0_log_mean_phrase": ("F0 Mean (Log, Phrase)", "Log(Hz)"),
        "f0_delta_std_phrase": ("Delta F0 Standard Deviation (Phrase)", "Log(Hz)"),
        "f0_delta_mean_phrase": ("Delta F0 Mean (Phrase)", "Log(Hz)"),
        "f0_delta_abs_log_std_phrase": ("Absolute Delta F0 Standard Deviation (Log, Phrase)", "Log(Hz)"),
        "f0_delta_abs_log_mean_phrase": ("Absolute Delta F0 Mean (Log, Phrase)", "Log(Hz)"),
        "spectral_tilt_std_phrase": ("Spectral Tilt Standard Deviation (Phrase)", "dB/freqbin"),
        "spectral_tilt_mean_phrase": ("Spectral Tilt Mean (Phrase)", "dB/freqbin"),
        "syllable_durations_log_std_phrase": ("Syllable Durations Standard Deviation (Log, Phrase)", "Log(s)"),
        "syllable_durations_log_mean_phrase": ("Syllable Durations Mean (Log, Phrase)", "Log(s)"),
        "f0_log_range": ("F0 (Log) Range", "Log(Hz)"),
        "f0_delta_abs_log_range": ("Absolute Delta F0 Range (Log)", "Log(Hz)"),
        "spectral_tilt_range": ("Spectral Tilt Range", "dB/freqbin"),
        "syllable_durations_log_range": ("Syllable Durations Range (Log)", "Log(s)"),
        "f0_log_range_phrase": ("F0 Range (Log, Phrase)", "Log(Hz)"),
        "f0_delta_abs_log_range_phrase": ("Absolute Delta F0 Range (Log, Phrase)", "Log(Hz)"),
        "spectral_tilt_range_phrase": ("Spectral Tilt Range (Phrase)", "dB/freqbin"),
        "syllable_durations_log_range_phrase": ("Syllable Durations Range (Log, Phrase)", "Log(s)"),
    }

    # Plot KDEs for individual speakers
    plot_all_kdes(synthesized_df, references_df, variables_with_units, OUTPUT_DIR)

    # Plot KDEs for all speakers combined
    plot_kdes_for_all_speakers(synthesized_df, references_df, variables_with_units, OUTPUT_DIR)


if __name__ == "__main__":
    create_kde_of_csv_stats()
