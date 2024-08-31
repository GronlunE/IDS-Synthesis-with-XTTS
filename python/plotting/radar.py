"""
Created on 31.8. 2024

@author: GronlunE

Description:

This script generates Radar plots for comparing acoustic measurements across various categories between original, manipulated, and synthesized data. The data for these measurements are loaded from CSV files, and the script produces radar plots for each type of manipulation (e.g., 'enhanced', 'denoised').

The script provides the following functionality:
- `create_radar_plot`: Creates a radar plot comparing original data, manipulated data, and synthesized data (both original and manipulated). The plot is saved as a PDF file.
- `process_files`: Processes data files for a given manipulation type to generate radar plots. It extracts the relevant data and calculates mean values for synthesized data before creating plots.
- `plot_radar_plots`: Orchestrates the processing of data files for multiple manipulation types by calling `process_files`.

Usage:
- Ensure the required libraries (`pandas`, `numpy`, `matplotlib`, `tqdm`) are installed.
- Place the CSV files containing the measurement data in the specified paths.
- Run the script to generate and save radar plots for each manipulation type.

Dependencies:
- `pandas` for data manipulation and analysis.
- `numpy` for numerical operations.
- `matplotlib` for plotting radar charts.
- `tqdm` for progress indication during file processing.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
from tqdm import tqdm

# Load synthesized data from CSV file into a pandas DataFrame
synthesized_file = r"G:\Research\XTTS_Test\VISUALIZATION\graph_data\radar\synthesized_normalized.csv"
synthesized_df = pd.read_csv(synthesized_file)

# Load reference (concatenated) data from CSV file into a pandas DataFrame
concatenated_file = r"G:\Research\XTTS_Test\VISUALIZATION\graph_data\radar\references_normalized.csv"
references_df = pd.read_csv(concatenated_file)

# Define the categories of features for the radar plot
categories = [
    'f0_mean',                 # Mean of the fundamental frequency
    'f0_sd',                   # Standard deviation of the fundamental frequency
    'f0_max',                  # Maximum value of the fundamental frequency
    'f0_min',                  # Minimum value of the fundamental frequency
    'f0_range',                # Range of the fundamental frequency (max - min)
    'f0_delta_mean',           # Mean of the delta (change) in fundamental frequency
    'f0_delta_sd',             # Standard deviation of the delta in fundamental frequency
    'spectral_tilt_mean',      # Mean of the spectral tilt
    'spectral_tilt_sd',        # Standard deviation of the spectral tilt
    'syllable_duration_mean',  # Mean duration of syllables
    'syllable_duration_sd'     # Standard deviation of syllable duration
]

# Filter DataFrames to include only the relevant columns
references_df = references_df[['file_name'] + categories]
synthesized_df = synthesized_df[['file_name'] + categories]


def create_radar_plot(data_original, data_manipulated, data_synthesized_original, data_synthesized_manipulated, categories, output_file, manipulation):
    """
    Create and save a radar plot comparing original data, manipulated data,
    and synthesized data (both original and manipulated).

    :param data_original: Array of feature values for the original data
    :param data_manipulated: Array of feature values for the manipulated data
    :param data_synthesized_original: Mean values of synthesized data (original)
    :param data_synthesized_manipulated: Mean values of synthesized data (manipulated)
    :param categories: List of feature category names
    :param output_file: Path to save the radar plot image
    :param manipulation: Type of manipulation for labeling (e.g., 'enhanced', 'denoised')
    """
    num_categories = len(categories)

    # Append the first value to the end of each data array to close the radar plot loop
    data_original = np.concatenate((data_original, [data_original[0]]))
    data_manipulated = np.concatenate((data_manipulated, [data_manipulated[0]]))
    data_synthesized_original = np.concatenate((data_synthesized_original, [data_synthesized_original[0]]))
    data_synthesized_manipulated = np.concatenate((data_synthesized_manipulated, [data_synthesized_manipulated[0]]))

    # Calculate the angles for each axis of the radar plot
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop for the radar plot

    # Create a radar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot the data on the radar chart
    ax.plot(angles, data_original, 'o-', label='Original', linewidth=2, markersize=8)
    ax.plot(angles, data_manipulated, 'o-', label=manipulation.capitalize(), linewidth=2, markersize=8)
    ax.plot(angles, data_synthesized_original, 'o-', label='Synthesized (Original)', linewidth=2, markersize=8)
    ax.plot(angles, data_synthesized_manipulated, 'o-', label=f'Synthesized ({manipulation.capitalize()})', linewidth=2, markersize=8)

    # Fill the area under each line
    ax.fill(angles, data_original, alpha=0.25)
    ax.fill(angles, data_manipulated, alpha=0.25)
    ax.fill(angles, data_synthesized_original, alpha=0.25)
    ax.fill(angles, data_synthesized_manipulated, alpha=0.25)

    # Customize the radar plot
    ax.set_yticklabels([])  # Remove y-axis labels
    ax.set_xticks(angles[:-1])  # Set x-ticks
    ax.set_xticklabels(categories, rotation=45, ha='right')  # Set x-tick labels

    # Set plot title
    plt.title('Radar Plot of Normalized Features')

    # Calculate figure size in inches
    fig_width_inch, fig_height_inch = fig.get_size_inches()

    # Convert cm to inches for positioning the legend
    cm_to_inch = 0.393701
    horizontal_offset_inch = 5 * cm_to_inch
    vertical_offset_inch = 2 * cm_to_inch

    # Calculate the position for the legend
    bbox_to_anchor_x = 1 - horizontal_offset_inch / fig_width_inch
    bbox_to_anchor_y = 1 + vertical_offset_inch / fig_height_inch

    # Position the legend outside the plot area
    plt.legend(loc='upper left', bbox_to_anchor=(bbox_to_anchor_x, bbox_to_anchor_y))

    # Save the radar plot to a file
    plt.savefig(output_file)
    plt.close()


# Output directory where radar plots will be saved
output_dir = r"G:\Research\XTTS_Test\VISUALIZATION\radar_plots"
os.makedirs(output_dir, exist_ok=True)


def process_files(manipulation_type):
    """
    Process files for a given manipulation type to generate radar plots.

    :param manipulation_type: Type of manipulation to process (e.g., 'enhanced', 'denoised')
    """
    # Filter reference data to include only files with the specified manipulation type
    manipulation_files = references_df[references_df['file_name'].str.contains(manipulation_type)]

    # Iterate over each file for the given manipulation type
    for _, row in tqdm(manipulation_files.iterrows(), total=len(manipulation_files), desc=f"Processing {manipulation_type} files"):
        manipulated_file_name = row['file_name']
        manipulated_data = row[categories].values  # Extract feature data for the manipulated file

        # Determine the corresponding original file name
        concat_n = manipulated_file_name.split('_')[-1]  # Extract the 'concat_n' part
        original_file_name = manipulated_file_name.replace(manipulation_type, 'original')

        # Retrieve original data based on the original file name
        original_data_row = references_df[references_df['file_name'] == original_file_name]
        if original_data_row.empty:
            continue  # Skip if no matching original data is found
        original_data = original_data_row[categories].values.flatten()

        # Filter synthesized data for the original and manipulated file types
        synthesized_original_filtered = synthesized_df[synthesized_df['file_name'].str.contains(f"{original_file_name[:-4]}_")]
        synthesized_manipulated_filtered = synthesized_df[synthesized_df['file_name'].str.contains(f"{manipulated_file_name[:-4]}_")]

        # Calculate the mean values for synthesized data
        synthesized_original_mean = synthesized_original_filtered[categories].mean().values
        synthesized_manipulated_mean = synthesized_manipulated_filtered[categories].mean().values

        # Generate the radar plot and save it to a file
        radar_plot_file = os.path.join(output_dir, f"{manipulation_type}_concat_{concat_n}_radar_plot.pdf")
        create_radar_plot(original_data, manipulated_data, synthesized_original_mean, synthesized_manipulated_mean, categories, radar_plot_file, manipulation_type)


def plot_radar_plots():
    """
    Generate radar plots for all manipulation types by calling `process_files` for each.
    """
    # Process 'enhanced' files
    process_files('enhanced')

    # Process 'denoised' files
    process_files('denoised')
