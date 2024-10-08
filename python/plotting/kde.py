"""
Created on 31.8. 2024

@author: GronlunE

Description:

This script generates Kernel Density Estimation (KDE) plots for various acoustic measurements, including pitch (F0), pitch delta, spectral tilt, and syllable duration. The data for these measurements are loaded from pickle files, and the script produces both linear and logarithmic scale plots to visualize the distribution of these measurements.

The script provides the following functionality:
- `plot_kde_subplot`: Plots KDEs for a specific subplot in a grid, distinguishing between reference and synthesis data with different line styles and colors.
- `plot_kde`: Creates KDE plots for multiple categories and saves them in both linear and logarithmic scales.
- `plot_all_kde`: Loads data from pickle files and calls `plot_kde` to generate plots for each type of measurement.

Usage:
- Ensure the required libraries (`pickle`, `numpy`, `matplotlib`, `scipy`, and `tqdm`) are installed.
- Place the pickle files containing the measurement data in the specified paths.
- Run the script to generate and save KDE plots for each measurement.

Dependencies:
- `pickle` for loading data from files.
- `numpy` for numerical operations.
- `matplotlib` for plotting.
- `scipy` for KDE estimation.
- `tqdm` for progress indication during plotting.

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm


def plot_kde_subplot(data_name, unit, ax, data_dict, title, linestyle='-', include_all=False):
    """
    Plots KDEs for a specific subplot in a grid.

    :param data_name: The name of the data being plotted (e.g., 'f0', 'f0_delta').
    :param unit: The unit of measurement for the data (e.g., 'Hz', 'Hz/s').
    :param ax: The axis object on which to plot.
    :param data_dict: A dictionary containing data arrays to plot.
    :param title: The title of the subplot.
    :param linestyle: The line style for plotting (default is '-') (e.g., '-' for solid, '--' for dashed).
    :param include_all: Whether to include all sub-directories or just those in data_dict.
    :return: None
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
            f0_data = np.concatenate(data_dict[sub_dir], axis=None)
            kde = gaussian_kde(f0_data)

            # Adjust x to be within the specified range
            x = np.linspace(f0_data.min(), f0_data.max(), 1000)
            kde_values = kde(x)

            # Use the provided linestyle and map it to the legend label
            label = linestyle_labels.get(linestyle, 'Unknown')

            # Plot KDEs
            ax.plot(x, kde_values, label=f"{label} ({sub_dir.capitalize()})", color=color,
                    linestyle=linestyle, linewidth=2.5)

    ax.set_xlabel(f"{data_name} ({unit})")
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def plot_kde(data, data_name, unit):
    """
    Creates KDE plots for multiple categories and saves them in both linear and logarithmic scales.

    :param data: A dictionary containing the measurement data for different categories.
    :param data_name: The name of the data being plotted (e.g., 'f0', 'f0_delta').
    :param unit: The unit of measurement for the data (e.g., 'Hz', 'Hz/s').
    :return: None
    """
    # Define a custom color map for distinct colors
    color_map = {
        "original": 'tab:green',
        "denoised": 'tab:blue',
        "enhanced": 'tab:orange'
    }
    # Create a single figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    fig.tight_layout(pad=5.0)

    # 1. All Reference data together
    for sub_dir in ["denoised", "enhanced", "original"]:
        plot_kde_subplot(data_name, unit, axs[0, 0], {
            sub_dir: data["Reference"].get(sub_dir, [])
        }, f"References", linestyle='-')

    # 2. Denoised data (Reference and Synthesis)
    for category, linestyle in [("Reference", '-'), ("Synthesis", '--')]:
        plot_kde_subplot(data_name, unit, axs[0, 1], {
            "denoised": data[category].get("denoised", [])
        }, f"Denoised Data", linestyle=linestyle)

    # 3. Enhanced data (Reference and Synthesis)
    for category, linestyle in [("Reference", '-'), ("Synthesis", '--')]:
        plot_kde_subplot(data_name, unit, axs[1, 0], {
            "enhanced": data[category].get("enhanced", [])
        }, f"Enhanced Data", linestyle=linestyle)

    # 4. All Synthesis and Original Reference Data
    for label, sub_dirs in {
        "Reference (Original)": ["original"],
        "Synthesis (Denoised)": ["denoised"],
        "Synthesis (Enhanced)": ["enhanced"],
        "Synthesis (Original)": ["original"]
    }.items():
        combined_data = []
        if label == "Reference (Original)":
            combined_data = data["Reference"].get("original", [])
        elif label == "Synthesis (Denoised)":
            combined_data = data["Synthesis"].get("denoised", [])
        elif label == "Synthesis (Enhanced)":
            combined_data = data["Synthesis"].get("enhanced", [])
        elif label == "Synthesis (Original)":
            combined_data = (data["Synthesis"].get("original", []))

        f0_data = np.concatenate(combined_data, axis=None) if combined_data else np.array([])
        kde = gaussian_kde(f0_data)
        x = np.linspace(f0_data.min(), f0_data.max(), 1000)
        kde_values = kde(x)
        linestyle = '-' if 'Reference' in label else '--'
        color = color_map.get(sub_dirs[0], 'black')  # Use the color for the first sub_dir in the list
        axs[1, 1].plot(x, kde_values, label=label, linestyle=linestyle, color=color,
                       linewidth=2.5)

    axs[1, 1].set_xlabel(f"{data_name} ({unit})")
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].set_title("Original reference with all syntheses")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    filename = f"{data_name}_kde_plot.pdf"
    plt.savefig(filename)
    plt.close()

    # --------------------------------------------------LOGARITHMIC----------------------------------------------

    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    fig.tight_layout(pad=5.0)

    # Apply the same plotting logic but with logarithmic scaling
    for i in range(2):
        for j in range(2):
            axs[i, j].set_xscale('log')
            axs[i, j].set_yscale('log')

    # 1. All Reference data together in log scale
    for sub_dir in ["denoised", "enhanced", "original"]:
        plot_kde_subplot(data_name, unit, axs[0, 0], {
            sub_dir: data["Reference"].get(sub_dir, [])
        }, f"References (Logarithmic)", linestyle='-')

    # 2. Denoised data (Reference and Synthesis) in log scale
    for category, linestyle in [("Reference", '-'), ("Synthesis", '--')]:
        plot_kde_subplot(data_name, unit, axs[0, 1], {
            "denoised": data[category].get("denoised", [])
        }, f"Denoised Data (Logarithmic)", linestyle=linestyle)

    # 3. Enhanced data (Reference and Synthesis) in log scale
    for category, linestyle in [("Reference", '-'), ("Synthesis", '--')]:
        plot_kde_subplot(data_name, unit, axs[1, 0], {
            "enhanced": data[category].get("enhanced", [])
        }, f"Enhanced Data (Logarithmic)", linestyle=linestyle)

    # 4. All Synthesis and Original Reference Data in log scale
    for label, sub_dirs in {
        "Reference (Original)": ["original"],
        "Synthesis (Denoised)": ["denoised"],
        "Synthesis (Enhanced)": ["enhanced"],
        "Synthesis (Original)": ["original"]
    }.items():
        combined_data = []
        if label == "Reference (Original)":
            combined_data = data["Reference"].get("original", [])
        elif label == "Synthesis (Denoised)":
            combined_data = data["Synthesis"].get("denoised", [])
        elif label == "Synthesis (Enhanced)":
            combined_data = data["Synthesis"].get("enhanced", [])
        elif label == "Synthesis (Original)":
            combined_data = data["Synthesis"].get("original", [])

        f0_data = np.concatenate(combined_data, axis=None) if combined_data else np.array([])
        kde = gaussian_kde(f0_data)
        x = np.linspace(f0_data.min(), f0_data.max(), 1000)
        kde_values = kde(x)
        linestyle = '-' if 'Reference' in label else '--'
        color = color_map.get(sub_dirs[0], 'black')  # Use the color for the first sub_dir in the list
        axs[1, 1].plot(x, kde_values, label=label, linestyle=linestyle, color=color,
                       linewidth=2.5)

    axs[1, 1].set_xlabel(f"{data_name} ({unit})")
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].set_title("Original reference with all syntheses (Logarithmic)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # Save the logarithmic plot
    log_filename = f"{data_name}_logarithmic_kde_plot.pdf"
    plt.savefig(log_filename)
    plt.close()


def plot_all_kde():
    """
    Loads data from pickle files and generates KDE plots for each measurement type.

    :return: None
    """
    # List of stat names and corresponding data files
    stat_files = {
        'f0': {"file": "../../f0_data.pkl", "unit": "Hz"},
        'f0_delta': {"file": "../../f0_delta_data.pkl", "unit": "Hz/s"},
        'spectral_tilt': {"file": "../../spectral_tilt_data.pkl", "unit": "dB"},
        'syllable_duration': {"file": "../../syllable_duration_data.pkl", "unit": "s"}
    }

    # Loop through each stat, load the data, and plot KDE
    for stat_name, stat_info in tqdm(stat_files.items(), desc="Creating figures", unit="figure"):
        with open(stat_info["file"], 'rb') as file:
            data = pickle.load(file)
        plot_kde(data, stat_name, stat_info["unit"])
