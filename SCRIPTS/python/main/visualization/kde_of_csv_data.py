import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os


def load_data(synthesized_file, references_file):
    """
    Load synthesized and reference data from CSV files.

    :param synthesized_file: Path to the synthesized data CSV
    :param references_file: Path to the reference data CSV
    :return: DataFrames for synthesized and reference data
    """
    synthesized_df = pd.read_csv(synthesized_file)
    references_df = pd.read_csv(references_file)
    return synthesized_df, references_df


def prepare_data(synthesized_df, references_df):
    """
    Process the data by extracting categories and adding color information.

    :param synthesized_df: DataFrame with synthesized data
    :param references_df: DataFrame with reference data
    :return: Processed DataFrames with added color information
    """
    synthesized_df['category'] = synthesized_df['file_name'].str.extract(r'(denoised|enhanced|original)')[0]
    references_df['category'] = references_df['file_name'].str.extract(r'(denoised|enhanced|original)')[0]

    color_map = {
        "original": 'tab:green',
        "denoised": 'tab:blue',
        "enhanced": 'tab:orange'
    }
    synthesized_df['color'] = synthesized_df['category'].map(color_map)
    references_df['color'] = references_df['category'].map(color_map)

    return synthesized_df, references_df


def plot_kde_subplot(data_name, unit, ax, data_dict, title, linestyle='-', include_all=False):
    """
    Plot KDE on the provided axis for the given data.

    :param data_name: Name of the data variable
    :param unit: Unit of the data
    :param ax: Axis to plot on
    :param data_dict: Dictionary with data to plot
    :param title: Title of the subplot
    :param linestyle: Line style for the plot
    :param include_all: Flag to include all sub_dirs
    """
    linestyle_labels = {
        '-': 'Reference',
        '--': 'Synthesis'
    }
    color_map = {
        "original": 'tab:green',
        "denoised": 'tab:blue',
        "enhanced": 'tab:orange'
    }

    for sub_dir, color in color_map.items():
        if include_all and sub_dir not in data_dict:
            continue

        if sub_dir in data_dict:
            plot_data = np.concatenate(data_dict[sub_dir], axis=None)

            # Kernel density estimation
            kde = gaussian_kde(plot_data)

            # Adjust x to be within the specified range
            x = np.linspace(plot_data.min(), plot_data.max(), 1000)
            kde_values = kde(x)

            # Use the provided linestyle and map it to the legend label
            label = linestyle_labels.get(linestyle, 'Unknown')

            # Plot KDEs
            ax.plot(x, kde_values, label=f"{label} ({sub_dir.capitalize()})", color=color,
                    linestyle=linestyle, linewidth=2.5)

    ax.set_xlabel(f"{unit}")
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def plot_all_kdes(synthesized_df, references_df, variables_with_units, output_dir):
    """
    Plot KDEs for all variables and save plots to the specified directory.

    :param synthesized_df: DataFrame with synthesized data
    :param references_df: DataFrame with reference data
    :param variables_with_units: Dictionary of variables with their titles and units
    :param output_dir: Directory to save the output plots
    """
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        "original": 'tab:green',
        "denoised": 'tab:blue',
        "enhanced": 'tab:orange'
    }

    for var_name, (title, unit) in variables_with_units.items():
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        plt.suptitle(f'KDE Plot for {title}', fontsize=16)

        # 1. All Reference data together
        ref_data = {
            sub_dir: references_df[references_df['category'] == sub_dir][var_name].dropna().values
            for sub_dir in ["denoised", "enhanced", "original"]
        }
        plot_kde_subplot(var_name, unit, axs[0, 0], ref_data, f'References', linestyle='-')

        # 2. Denoised data (Reference and Synthesis)
        for category, linestyle in [("Reference", '-'), ("Synthesis", '--')]:
            if category == "Reference":
                data_dict = {
                    "denoised": references_df[references_df['category'] == "denoised"][var_name].dropna().values
                }
            else:
                data_dict = {
                    "denoised": synthesized_df[synthesized_df['category'] == "denoised"][var_name].dropna().values
                }
            plot_kde_subplot(var_name, unit, axs[0, 1], data_dict, f"Denoised Data ({category})", linestyle=linestyle)

        # 3. Enhanced data (Reference and Synthesis)
        for category, linestyle in [("Reference", '-'), ("Synthesis", '--')]:
            if category == "Reference":
                data_dict = {
                    "enhanced": references_df[references_df['category'] == "enhanced"][var_name].dropna().values
                }
            else:
                data_dict = {
                    "enhanced": synthesized_df[synthesized_df['category'] == "enhanced"][var_name].dropna().values
                }
            plot_kde_subplot(var_name, unit, axs[1, 0], data_dict, f"Enhanced Data ({category})", linestyle=linestyle)

        # 4. All Synthesis and Original Reference Data
        combined_data_map = {
            "Reference (Original)": "original",
            "Synthesis (Denoised)": "denoised",
            "Synthesis (Enhanced)": "enhanced",
            "Synthesis (Original)": "original"
        }

        for label, sub_dir in combined_data_map.items():
            if "Reference" in label:
                combined_data = references_df[references_df['category'] == sub_dir][var_name].dropna().values
            elif "Synthesis" in label:
                combined_data = synthesized_df[synthesized_df['category'] == sub_dir][var_name].dropna().values

            plot_data = np.concatenate(combined_data, axis=None) if len(combined_data) > 0 else np.array([])

            if len(plot_data) > 0:
                kde = gaussian_kde(plot_data)
                x = np.linspace(plot_data.min(), plot_data.max(), 1000)
                kde_values = kde(x)
                linestyle = '-' if 'Reference' in label else '--'
                color = color_map.get(sub_dir, 'black')  # Use the color for the sub_dir
                axs[1, 1].plot(x, kde_values, label=label, linestyle=linestyle, color=color, linewidth=2.5)

        axs[1, 1].set_xlabel(f"{unit}")
        axs[1, 1].set_ylabel('Density')
        axs[1, 1].set_title("Original reference with all syntheses")
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        # Save the plot to the specified path
        filename = f"{var_name}_kde.pdf"
        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path)
        plt.close()

    print("KDE plots have been generated and saved.")


def create_kde_of_csv_stats():
    # File paths
    synthesized_file = r"G:\Research\XTTS_Test\CODE\python\output\fast_data\syntheses.csv"
    references_file = r"G:\Research\XTTS_Test\CODE\python\output\fast_data\references.csv"
    output_dir = r"G:\Research\XTTS_Test\CODE\python\output\plots\kde\csv"

    # Load and prepare data
    synthesized_df, references_df = load_data(synthesized_file, references_file)
    synthesized_df, references_df = prepare_data(synthesized_df, references_df)

    # Define variables and their units
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

        # Adding ranges
        "f0_log_range": ("F0 (Log) Range", "Log(Hz)"),
        "f0_delta_abs_log_range": ("Absolute Delta F0 Range (Log)", "Log(Hz)"),
        "spectral_tilt_range": ("Spectral Tilt Range", "dB/freqbin"),
        "syllable_durations_log_range": ("Syllable Durations Range (Log)", "Log(s)"),
        "f0_log_range_phrase": ("F0 Range (Log, Phrase)", "Log(Hz)"),
        "f0_delta_abs_log_range_phrase": ("Absolute Delta F0 Range (Log, Phrase)", "Log(Hz)"),
        "spectral_tilt_range_phrase": ("Spectral Tilt Range (Phrase)", "dB/freqbin"),
        "syllable_durations_log_range_phrase": ("Syllable Durations Range (Log, Phrase)", "Log(s)"),
    }

    # Plot KDEs
    plot_all_kdes(synthesized_df, references_df, variables_with_units, output_dir)


create_kde_of_csv_stats()
