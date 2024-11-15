import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2
import os


# Function to add ellipses for 95% confidence interval
def add_ellipse(ax, x_data, y_data, color, linestyle, label=None):
    """

    :param ax:
    :param x_data:
    :param y_data:
    :param color:
    :param linestyle:
    :param label:
    :return:
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

    :return:
    """
    # Plot 1: Synthesized Data - X vs. Y by Category
    fig = plt.figure(figsize=(18, 12))

    fig.suptitle(f"Scatter and ellipse fit of {var_name.capitalize()} for References and Syntheses", fontsize=16)

    color_map = {
        "original": 'tab:green',
        "denoised": 'tab:blue',
        "enhanced": 'tab:orange'
    }

    # Assuming 'categories' is a list containing the keys of color_map
    categories = list(color_map.keys())
    # Ensuring the same colors are applied
    synthesized_df['color'] = synthesized_df['category'].map(color_map)

    # Plot the data
    plt.subplot(2, 2, 1)
    for category in synthesized_df['category'].unique():
        # Get the corresponding color from the color_map
        color = color_map.get(category, 'tab:gray')  # Use a default color if the category is not found
        category_data = synthesized_df[synthesized_df['category'] == category]
        x_data = category_data[x_var].values
        y_data = category_data[y_var].values

        plt.scatter(x_data, y_data, color=color, label=f"{category}", alpha=0.7)
        add_ellipse(plt.gca(), x_data, y_data, color, linestyle='--')

    # Set axis labels and title
    plt.xlabel(x_var.replace('_', ' ').title())
    plt.ylabel(y_var.replace('_', ' ').title())
    plt.title(f'Synthesis Data: {x_var.replace("_", " ").title()} vs {y_var.replace("_", " ").title()} by Category')

    # Construct legend handles based on the color_map
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[category], markersize=10, linestyle='')
        for category in color_map.keys()]

    # Add the legend with proper labels
    plt.legend(handles, color_map.keys(), title="Category")
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
    plt.title(f'Synthesis Data: {x_var.replace("_", " ").title()} vs {y_var.replace("_", " ").title()} by GILES Number')
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

    # Set axis labels and title
    plt.xlabel(x_var.replace('_', ' ').title())
    plt.ylabel(y_var.replace('_', ' ').title())
    plt.title(f'Reference Data: {x_var.replace("_", " ").title()} vs {y_var.replace("_", " ").title()} by Category')

    # Construct legend handles based on the color_map
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[category], markersize=10, linestyle='')
        for category in color_map.keys()
    ]

    # Add the legend with proper labels
    plt.legend(handles, color_map.keys(), title="Category")
    plt.grid(True)

    # Plot 4: Combined Synthesized and Concatenated Data
    plt.subplot(2, 2, 4)
    for category, color in color_map.items():
        # Synthesized data
        cat_synth_data = synthesized_df[synthesized_df['category'] == category]
        cat_concat_data = references_df[references_df['category'] == category]

        if not cat_synth_data.empty:
            plt.scatter(cat_synth_data[x_var], cat_synth_data[y_var], color=color, marker='o', alpha=0,
                        label=f"{category} (Syntheses)")
            add_ellipse(plt.gca(), cat_synth_data[x_var], cat_synth_data[y_var], color, linestyle='--',)

        if not cat_concat_data.empty:
            plt.scatter(cat_concat_data[x_var], cat_concat_data[y_var], color=color, marker='x', alpha=0,
                        label=f"{category} (Reference)")
            add_ellipse(plt.gca(), cat_concat_data[x_var], cat_concat_data[y_var], color, linestyle='-')

    plt.xlabel(x_var.replace('_', ' ').title())
    plt.ylabel(y_var.replace('_', ' ').title())
    plt.title(f'Combined Data: {x_var.replace("_", " ").title()} vs {y_var.replace("_", " ").title()}')
    handles = [plt.Line2D([0], [0], color=color, linestyle='-', linewidth=2, label=f"{category} (Reference)")
               for category, color in color_map.items()]
    handles += [plt.Line2D([0], [0], color=color, linestyle='--', linewidth=2, label=f"{category} (Syntheses)")
                for category, color in color_map.items()]
    plt.legend(handles,
               [f"{category} (Reference)" for category in categories] + [f"{category} (Synthesized)" for category in
                                                                           categories], title="Category")
    plt.grid(True)

    plt.tight_layout()
    # Define the output directory for scatter plots
    output_dir = r"G:\Research\XTTS_Test\CODE\python\output\fast_data\scatter"

    # Create the filename
    filename = f"{var_name}_scatter.pdf"

    # Combine the output directory and filename
    file_path = os.path.join(output_dir, filename)

    # Save the scatter plot to the specified path
    plt.savefig(file_path)
    plt.close()


def plot_all_scatter():
    # Load synthesized data
    synthesized_file = r"G:\Research\XTTS_Test/CODE/python/output/fast_data/syntheses.csv"
    synthesized_df = pd.read_csv(synthesized_file)

    # Load concatenated data
    references_file = r"G:\Research\XTTS_Test/CODE/python/output/fast_data/references.csv"
    references_df = pd.read_csv(references_file)

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

    color_map = {
        "original": 'tab:green',
        "denoised": 'tab:blue',
        "enhanced": 'tab:orange'
    }

    # Assuming 'categories' is a list containing the keys of color_map
    categories = list(color_map.keys())
    # Ensuring the same colors are applied
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

    # Your variables as a dictionary
    variables = {
        # Standard deviation / Mean pairs
        'f0_ln_sd_mean': ('f0_ln_std', 'f0_ln_mean'),
        'f0_delta_abs_ln_sd_mean': ('f0_delta_abs_ln_std', 'f0_delta_abs_ln_mean'),
        'spectral_tilt_sd_mean': ('spectral_tilt_std', 'spectral_tilt_mean'),
        'syllable_duration_ln_sd_mean': ('syllable_durations_ln_std', 'syllable_durations_ln_mean'),

        # Min / Max pairs
        'f0_ln_range': ('f0_ln_min5', 'f0_ln_max95'),
        'f0_delta_abs_ln_range': ('f0_delta_abs_ln_min_5', 'f0_delta_abs_ln_max_95'),
        'spectral_tilt_range': ('spectral_tilt_min5', 'spectral_tilt_max95'),
        'syllable_duration_ln_range': ('syllable_durations_ln_min5', 'syllable_durations_ln_max95'),

        # Phrase variants - Standard deviation / Mean pairs
        'f0_ln_sd_mean_phrase': ('f0_ln_std_phrase', 'f0_ln_mean_phrase'),
        'f0_delta_abs_ln_sd_mean_phrase': ('f0_delta_abs_ln_std_phrase', 'f0_delta_abs_ln_mean_phrase'),
        'spectral_tilt_sd_mean_phrase': ('spectral_tilt_std_phrase', 'spectral_tilt_mean_phrase'),
        'syllable_duration_ln_sd_mean_phrase': ('syllable_durations_ln_std_phrase', 'syllable_durations_ln_mean_phrase'),

        # Phrase variants - Min / Max pairs
        'f0_ln_range_phrase': ('f0_ln_min5_phrase', 'f0_ln_max95_phrase'),
        'f0_delta_abs_ln_range_phrase': ('f0_delta_abs_ln_min_5_phrase', 'f0_delta_abs_ln_max_95_phrase'),
        'spectral_tilt_range_phrase': ('spectral_tilt_min5_phrase', 'spectral_tilt_max95_phrase'),
        'syllable_duration_ln_range_phrase': ('syllable_durations_ln_min5_phrase', 'syllable_durations_ln_max95_phrase'),
    }

    # Loop through the dictionary and plot
    for var_name, (x_var, y_var) in variables.items():
        draw_scatter_plot(x_var, y_var, var_name,  color_map, synthesized_df, references_df, categories, n_categories, giles_color_map, giles_numbers, giles_colors, colors)


plot_all_scatter()
