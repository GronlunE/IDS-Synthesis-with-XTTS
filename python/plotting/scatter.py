"""
Created on 31.8. 2024

@author: GronlunE

Description:

This script generates scatter plots to visualize and compare acoustic measurements from synthesized and reference data. It includes the following functionalities:
- `add_ellipse`: Adds an ellipse to a scatter plot based on the covariance of the data points.
- `draw_scatter_plot`: Creates scatter plots to visualize synthesized and reference data, and overlays ellipses to represent data distributions.
- `plot_scatters`: Main function to load data, process it, and generate scatter plots for various variables.

Dependencies:
- `pandas` for data manipulation.
- `matplotlib` for plotting.
- `numpy` for numerical operations.
- `scipy` for statistical functions.

The "GILES" in the code refers to GILES generated text samples.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2


def add_ellipse(ax, x_data, y_data, color, linestyle, label=None):
    """
    Add an ellipse to the plot representing the covariance of the data points.

    :param ax: Matplotlib axis object where the ellipse will be added.
    :param x_data: Array-like, x coordinates of the data points.
    :param y_data: Array-like, y coordinates of the data points.
    :param color: Color of the ellipse.
    :param linestyle: Style of the ellipse line (e.g., '--', '-').
    :param label: Optional label to annotate the ellipse.
    """
    if len(x_data) < 2 or len(y_data) < 2:
        return  # Not enough data to draw an ellipse

    # Remove NaN and infinite values
    valid_idx = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_idx]
    y_data = y_data[valid_idx]

    if len(x_data) < 2 or len(y_data) < 2:
        return  # Still not enough data after cleaning

    mean = np.array([x_data.mean(), y_data.mean()])
    covariance = np.cov(x_data, y_data)

    try:
        eigvals, eigvecs = np.linalg.eig(covariance)
    except np.linalg.LinAlgError:
        return  # Covariance matrix is not valid

    # Compute the angle and width/height of the ellipse
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    width, height = 2 * np.sqrt(eigvals * chi2.ppf(0.95, 2))

    # Create and add ellipse
    ellipse = Ellipse(mean, width, height, angle=np.degrees(angle), color=color, fill=False, linestyle=linestyle,
                      linewidth=2)
    ax.add_patch(ellipse)
    if label:
        ax.text(mean[0], mean[1], label, color=color, fontsize=12, verticalalignment='bottom',
                horizontalalignment='right')


def draw_scatter_plot(x_var, y_var, var_name, color_map, synthesized_df, references_df, categories, n_categories, giles_color_map, giles_numbers, giles_colors, colors):
    """
    Generate scatter plots comparing synthesized and reference data, including ellipses to represent data distributions.

    :param x_var: Name of the column to plot on the x-axis.
    :param y_var: Name of the column to plot on the y-axis.
    :param var_name: Identifier for the variable being plotted, used in the output file name.
    :param color_map: Dictionary mapping categories to colors.
    :param synthesized_df: DataFrame containing synthesized data.
    :param references_df: DataFrame containing reference data.
    :param categories: List of categories for plotting.
    :param n_categories: Number of unique categories.
    :param giles_color_map: Dictionary mapping GILES numbers to colors.
    :param giles_numbers: List of unique GILES numbers.
    :param giles_colors: List of colors corresponding to GILES numbers.
    :param colors: List of colors for plotting categories.
    """
    plt.figure(figsize=(18, 12))

    # Plot 1: Synthesized Data - X vs. Y by Category
    plt.subplot(2, 2, 1)
    for category, color in color_map.items():
        category_data = synthesized_df[synthesized_df['category'] == category]
        x_data = category_data[x_var].values
        y_data = category_data[y_var].values

        plt.scatter(x_data, y_data, color=color, label=f"{category}", alpha=0.7)
        add_ellipse(plt.gca(), x_data, y_data, color, linestyle='--')

    plt.xlabel(x_var.replace('_', ' ').title())
    plt.ylabel(y_var.replace('_', ' ').title())
    plt.title(f'Synthesized Data: {x_var.replace("_", " ").title()} vs {y_var.replace("_", " ").title()} by Category')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, linestyle='') for color in
               colors[:n_categories]]
    plt.legend(handles, categories, title="Category")
    plt.grid(True)

    # Plot 2: Synthesized Data - X vs. Y by GILES number
    plt.subplot(2, 2, 2)
    for giles_number, color in giles_color_map.items():
        giles_data = synthesized_df[synthesized_df['giles_number'] == giles_number]
        x_data = giles_data[x_var].values
        y_data = giles_data[y_var].values

        plt.scatter(x_data, y_data, color=color, label=f"GILES_{giles_number}", alpha=0)
        add_ellipse(plt.gca(), x_data, y_data, color, linestyle='--')

    plt.xlabel(x_var.replace('_', ' ').title())
    plt.ylabel(y_var.replace('_', ' ').title())
    plt.title(f'Synthesized Data: {x_var.replace("_", " ").title()} vs {y_var.replace("_", " ").title()} by GILES Number')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, linestyle='') for color in
               giles_colors[:len(giles_numbers)]]
    plt.legend(handles, [f"GILES_{n}" for n in giles_numbers], title="GILES Sample")
    plt.grid(True)

    # Plot 3: Concatenated Data - X vs. Y by Category
    plt.subplot(2, 2, 3)
    for category, color in color_map.items():
        category_data = references_df[references_df['category'] == category]
        x_data = category_data[x_var].values
        y_data = category_data[y_var].values

        plt.scatter(x_data, y_data, color=color, marker='x', label=f"{category}", alpha=0.7)
        add_ellipse(plt.gca(), x_data, y_data, color, linestyle='-')

    plt.xlabel(x_var.replace('_', ' ').title())
    plt.ylabel(y_var.replace('_', ' ').title())
    plt.title(f'Reference Data: {x_var.replace("_", " ").title()} vs {y_var.replace("_", " ").title()} by Category')
    handles = [plt.Line2D([0], [0], marker='X', color='w', markerfacecolor=color, markersize=10, linestyle='') for color in
               colors[:n_categories]]
    plt.legend(handles, categories, title="Category")
    plt.grid(True)

    # Plot 4: Combined Synthesized and Concatenated Data
    plt.subplot(2, 2, 4)
    for category, color in color_map.items():
        # Synthesized data
        cat_synth_data = synthesized_df[synthesized_df['category'] == category]
        cat_concat_data = references_df[references_df['category'] == category]

        if not cat_synth_data.empty:
            plt.scatter(cat_synth_data[x_var], cat_synth_data[y_var], color=color, marker='o', alpha=0,
                        label=f"{category} (Synthesized)")
            add_ellipse(plt.gca(), cat_synth_data[x_var], cat_synth_data[y_var], color, linestyle='--')

        if not cat_concat_data.empty:
            plt.scatter(cat_concat_data[x_var], cat_concat_data[y_var], color=color, marker='x', alpha=0,
                        label=f"{category} (Reference)")
            add_ellipse(plt.gca(), cat_concat_data[x_var], cat_concat_data[y_var], color, linestyle='-')

    plt.xlabel(x_var.replace('_', ' ').title())
    plt.ylabel(y_var.replace('_', ' ').title())
    plt.title(f'Combined Data: {x_var.replace("_", " ").title()} vs {y_var.replace("_", " ").title()}')
    handles = [plt.Line2D([0], [0], color=color, linestyle='-', linewidth=2, label=f"{category} (Reference)")
               for category, color in color_map.items()]
    handles += [plt.Line2D([0], [0], color=color, linestyle='--', linewidth=2, label=f"{category} (Synthesized)")
                for category, color in color_map.items()]
    plt.legend(handles,
               [f"{category} (Reference)" for category in categories] + [f"{category} (Synthesized)" for category in
                                                                           categories], title="Category")
    plt.grid(True)

    plt.tight_layout()
    filename = f"{var_name}_sd_mean_scatter.pdf"
    plt.savefig(filename)
    plt.close()


def plot_scatters():
    """
    Load data, process it, and generate scatter plots for various acoustic measurements.

    This function performs the following steps:
    1. Load synthesized and reference data from CSV files.
    2. Extract categories and GILES numbers from file names.
    3. Define color maps for categories and GILES numbers.
    4. Plot scatter plots for different variables using the `draw_scatter_plot` function.
    """
    # Load synthesized data
    synthesized = r"plot_data/scatter/synthesized.csv"
    synthesized_df = pd.read_csv(synthesized)

    # Load concatenated data
    references = r"plot_data/scatter/references.csv"
    references_df = pd.read_csv(references)

    # Extract categories and GILES numbers for synthesized data
    synthesized_df['category'] = synthesized_df['file_name'].str.extract(r'(denoised|enhanced|original)')[0]
    synthesized_df['giles_number'] = synthesized_df['file_name'].str.extract(r'GILES_(\d+)')[0].astype(int)

    # Define color maps
    categories = synthesized_df['category'].unique()
    n_categories = len(categories)
    colors = plt.get_cmap('tab10').colors  # Use 'tab10' colormap for up to 10 distinct colors

    # Ensure we have enough colors
    if n_categories > len(colors):
        raise ValueError("Number of categories exceeds available colors in 'tab10' colormap.")

    color_map = dict(zip(categories, colors[:n_categories]))
    synthesized_df['color'] = synthesized_df['category'].map(color_map)

    # Define color map for GILES numbers
    giles_numbers = synthesized_df['giles_number'].unique()
    giles_colors = plt.get_cmap('tab20').colors  # Use 'tab20' colormap for up to 20 distinct colors

    # Ensure we have enough colors
    if len(giles_numbers) > len(giles_colors):
        raise ValueError("Number of GILES numbers exceeds available colors in 'tab20' colormap.")

    giles_color_map = dict(zip(giles_numbers, giles_colors[:len(giles_numbers)]))
    synthesized_df['giles_color'] = synthesized_df['giles_number'].map(giles_color_map)

    # Extract categories for concatenated data
    references_df['category'] = references_df['file_name'].str.extract(r'(denoised|enhanced|original)')[0]

    # Convert categories to color codes for concatenated data
    references_df['color'] = references_df['category'].map(color_map)

    # Variables to plot
    variables = {
        'f0': ('f0_sd', 'f0_mean'),
        'f0_delta': ('f0_delta_sd', 'f0_delta_mean'),
        'spectral_tilt': ('spectral_tilt_sd', 'spectral_tilt_mean'),
        'syllable_duration': ('syllable_duration_sd', 'syllable_duration_mean')
    }

    # Generate scatter plots for each variable
    for var_name, (x_var, y_var) in variables.items():
        draw_scatter_plot(x_var, y_var, var_name,  color_map, synthesized_df, references_df, categories, n_categories, giles_color_map, giles_numbers, giles_colors, colors)
