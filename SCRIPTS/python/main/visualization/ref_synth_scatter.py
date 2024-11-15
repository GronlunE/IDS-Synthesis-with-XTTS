"""
Created on 15.11.2024

@author: GronlunE

Description:
    This script generates scatter plots comparing reference and synthesized data for various audio features.
    The scatter plots visualize the relationship between reference values (for enhanced and original references)
    and synthesized values (for enhanced syntheses) across several audio features. The analysis is performed for
    features such as pitch, spectral tilt, and syllable duration statistics. The script calculates Pearson correlation
    coefficients and fits a linear regression line to assess the correlation between reference and synthesized data.

    The main functionalities of the script include:
    - Loading synthesized and reference data from CSV files.
    - Extracting relevant categories (e.g., "enhanced", "original") and assigning appropriate colors for GILES numbers.
    - Creating scatter plots for each feature in the `variables_list`, comparing enhanced references vs. enhanced syntheses
      and original references vs. enhanced syntheses.
    - Annotating the plots with linear fit lines and Pearson correlation values.
    - Saving the scatter plots as PDF files in the specified output directory.

Usage:
    - Set the input file paths for the synthesized data (`syntheses.csv`) and reference data (`references.csv`).
    - Define the output directory (`OUTPUT_DIR`) where the PDF plots will be saved.
    - The script loops through the variables listed in `variables_list`, generating scatter plots for each feature.
    - The generated plots will be saved as PDF files in the output directory.

Dependencies:
    - `pandas` for data manipulation and handling CSV files.
    - `matplotlib` for plotting scatter plots and saving them as PDFs.
    - `numpy` for numerical operations and generating linear fit lines.
    - `scipy.stats` for Pearson correlation and linear regression.
    - `os` for file and directory operations.
    - `tqdm` for progress bar display during processing.

Notes:
    - The script expects CSV files containing columns for different audio features, with columns named similarly to those
      listed in the `variables_list`.
    - It processes GILES numbers (from 1 to 10) and assigns distinct colors for each GILES number to help visualize individual
      points in the scatter plots.
    - The output plots will show the comparison between reference and synthesized data, and will include Pearson correlation
      coefficients and linear fit lines.

"""


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2
import os
from tqdm import tqdm
from scipy.stats import linregress, pearsonr
from matplotlib.lines import Line2D

# Set output directory
OUTPUT_DIR = r"/CODE/python/output/fast_data/ref_synth_scatter"

# Load data
SYNTHDATA = pd.read_csv(r"/CODE/python/output/fast_data/syntheses.csv")
REFDATA = pd.read_csv(r"/CODE/python/output/fast_data/references.csv")

# Extract categories and GILES numbers for synthesized data
SYNTHDATA['category'] = SYNTHDATA['file_name'].str.extract(r'(denoised|enhanced|original)')[0]
SYNTHDATA['giles_number'] = SYNTHDATA['file_name'].str.extract(r'GILES_(\d+)')[0].astype(int)

# Define color maps
color_map = {
    "original": 'tab:green',
    "denoised": 'tab:blue',
    "enhanced": 'tab:orange'
}
SYNTHDATA['color'] = SYNTHDATA['category'].map(color_map)

# Define color map for GILES numbers
giles_numbers = SYNTHDATA['giles_number'].unique()
giles_colors = plt.get_cmap('tab20').colors  # Use 'tab20' colormap for up to 20 distinct colors
giles_color_map = dict(zip(giles_numbers, giles_colors[:len(giles_numbers)]))
SYNTHDATA['giles_color'] = SYNTHDATA['giles_number'].map(giles_color_map)

# Extract categories for concatenated data
REFDATA['category'] = REFDATA['file_name'].str.extract(r'(denoised|enhanced|original)')[0]
REFDATA['color'] = REFDATA['category'].map(color_map)


variables_list = [
    'f0_log_std', 'f0_log_mean',
    'f0_delta_abs_log_std', 'f0_delta_abs_log_mean',
    'spectral_tilt_std', 'spectral_tilt_mean',
    'syllable_durations_log_std', 'syllable_durations_log_mean',
    'f0_log_std_phrase', 'f0_log_mean_phrase',
    'f0_delta_abs_log_std_phrase', 'f0_delta_abs_log_mean_phrase',
    'spectral_tilt_std_phrase', 'spectral_tilt_mean_phrase',
    'syllable_durations_log_std_phrase', 'syllable_durations_log_mean_phrase',
]


def draw_ref_synth_scatter(reference_var, synthesis_var, feature_name, references_df, synthesized_df):
    """Draw scatter plot comparing reference and synthesized data and save as PDF."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    ax1, ax2 = axes.flatten()

    # Lists to hold values for setting the same scale
    ref_values_all = []
    synth_values_all = []

    scatter_handles = []

    # Plot reference data (scatter plot) on the first axis
    for _, ref_row in references_df.iterrows():
        ref_base_filename = ref_row['file_name'].split('\\')[-1].replace('.wav', '')
        if "enhanced" in ref_base_filename:
            ref_value = ref_row[reference_var]
            for i in range(1, 11):  # Loop through GILES 1 to 10
                synth_filename = f"{ref_base_filename}_GILES_{i}.wav"
                synth_row = synthesized_df[synthesized_df['file_name'].str.contains(synth_filename)]

                if not synth_row.empty:
                    synth_value = synth_row[synthesis_var].values[0]
                    giles_color = giles_color_map.get(i, 'tab:gray')

                    # Scatter plot on the first axis and store handle for legend
                    handle = ax1.scatter(ref_value, synth_value, color=giles_color, marker='o', alpha=0.5)

                    if len(scatter_handles) < 10:
                        scatter_handles.append(handle)

                    # Store values for setting scale
                    ref_values_all.append(ref_value)
                    synth_values_all.append(synth_value)

    # Calculate Pearson correlation coefficient and overall linear fit using linregress
    r, p_value = pearsonr(ref_values_all, synth_values_all)
    slope, intercept, _, _, _ = linregress(ref_values_all, synth_values_all)

    # Extend the line to cover the entire axis range
    min_val, max_val = min(ref_values_all + synth_values_all), max(ref_values_all + synth_values_all)
    range_val = max_val - min_val
    buffer = range_val * 0.1
    x_range = np.linspace(min_val - buffer, max_val + buffer, 100)
    line_fit = slope * x_range + intercept

    # Plot the extended linear fit line
    ax1.plot(x_range, line_fit, color='black', linestyle='--')

    # Set labels and title for the first scatter plot
    ax1.set_xlabel("Enhanced references")
    ax1.set_ylabel("Enhanced syntheses")
    ax1.set_title(f"{feature_name.replace('_', ' ').title()} - Scatter - Enhanced references vs. Enhanced syntheses")

    # Set limits with buffer for the first plot
    ax1.set_xlim(min_val - buffer, max_val + buffer)
    ax1.set_ylim(min_val - buffer, max_val + buffer)

    # Manually create a handle for the black fit line for the legend
    fit_handle = Line2D([0], [0], color='black', linestyle='--',
                        label=f'Linear fit (r={r:.2f}, p={p_value:.4f})')

    # Combine the scatter handles and the fit line into one unified legend
    giles_labels = [f'GILES {i}' for i in range(1, 11)]

    # Combine both the GILES scatter handles and the fit line handle
    combined_handles = scatter_handles + [fit_handle]
    combined_labels = giles_labels + [f'Linear fit (r={r:.2f}, p={p_value:.4f})']

    # Create a single legend
    ax1.legend(combined_handles, combined_labels, loc='best', fontsize='small')

    # Lists to hold values for setting the same scale
    ref_values_all = []
    synth_values_all = []

    scatter_handles = []

    # Plot reference data (scatter plot) on the first axis
    for _, ref_row in references_df.iterrows():

        ref_base_filename = ref_row['file_name'].split('\\')[-1].replace('.wav', '')

        if "enhanced" in ref_base_filename:

            # Generate the corresponding original filename
            original_base_filename = ref_base_filename.replace("enhanced", "original") + '.wav'

            # Find the original reference value from references_df
            original_row = references_df[references_df['file_name'].str.contains(original_base_filename)]

            ref_value = original_row[reference_var].iloc[0]

            for i in range(1, 11):  # Loop through GILES 1 to 10
                synth_filename = f"{ref_base_filename}_GILES_{i}.wav"
                synth_row = synthesized_df[synthesized_df['file_name'].str.contains(synth_filename)]

                if not synth_row.empty:
                    synth_value = synth_row[synthesis_var].values[0]
                    giles_color = giles_color_map.get(i, 'tab:gray')

                    # Scatter plot on the first axis and store handle for legend
                    handle = ax2.scatter(ref_value, synth_value, color=giles_color, marker='o', alpha=0.5)

                    if len(scatter_handles) < 10:
                        scatter_handles.append(handle)

                    # Store values for setting scale
                    ref_values_all.append(ref_value)
                    synth_values_all.append(synth_value)

    # Calculate Pearson correlation coefficient and overall linear fit using linregress
    r, p_value = pearsonr(ref_values_all, synth_values_all)
    slope, intercept, _, _, _ = linregress(ref_values_all, synth_values_all)

    # Extend the line to cover the entire axis range
    min_val, max_val = min(ref_values_all + synth_values_all), max(ref_values_all + synth_values_all)
    range_val = max_val - min_val
    buffer = range_val * 0.1
    x_range = np.linspace(min_val - buffer, max_val + buffer, 100)
    line_fit = slope * x_range + intercept

    # Plot the extended linear fit line
    ax2.plot(x_range, line_fit, color='black', linestyle='--')

    # Set labels and title for the first scatter plot
    ax2.set_xlabel("Original references")
    ax2.set_ylabel("Enhanced syntheses")
    ax2.set_title(f"{feature_name.replace('_', ' ').title()} - Scatter - Original references vs. Enhanced syntheses")

    # Set limits with buffer for the first plot
    ax2.set_xlim(min_val - buffer, max_val + buffer)
    ax2.set_ylim(min_val - buffer, max_val + buffer)

    # Manually create a handle for the black fit line for the legend
    fit_handle = Line2D([0], [0], color='black', linestyle='--',
                        label=f'Linear fit (r={r:.2f}, p={p_value:.4f})')

    # Combine the scatter handles and the fit line into one unified legend
    giles_labels = [f'GILES {i}' for i in range(1, 11)]

    # Combine both the GILES scatter handles and the fit line handle
    combined_handles = scatter_handles + [fit_handle]
    combined_labels = giles_labels + [f'Linear fit (r={r:.2f}, p={p_value:.4f})']

    # Create a single legend
    ax2.legend(combined_handles, combined_labels, loc='best', fontsize='small')

    # Adjust layout and save the plot as a PDF
    plt.tight_layout()
    pdf_filename = os.path.join(OUTPUT_DIR, f"{feature_name}_scatter.pdf")
    plt.savefig(pdf_filename, format='pdf')
    plt.close(fig)  # Close the figure to free memory


# Loop through the variables list to create scatter plots with a progress bar
for variable in tqdm(variables_list, desc="Processing Variables", unit="variable"):
    draw_ref_synth_scatter(reference_var=variable, synthesis_var=variable, feature_name=variable,
                           references_df=REFDATA, synthesized_df=SYNTHDATA)
