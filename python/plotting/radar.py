import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
from tqdm import tqdm

# Load synthesized data
synthesized_file = r"G:\Research\XTTS_Test\VISUALIZATION\graph_data\radar\synthesized_normalized.csv"
synthesized_df = pd.read_csv(synthesized_file)

# Load concatenated data
concatenated_file = r"G:\Research\XTTS_Test\VISUALIZATION\graph_data\radar\references_normalized.csv"
references_df = pd.read_csv(concatenated_file)

# Define categories, excluding 'range_f0'
categories = [
    'f0_mean',
    'f0_sd',
    'f0_max',
    'f0_min',
    'f0_range',
    'f0_delta_mean',
    'f0_delta_sd',
    'spectral_tilt_mean',
    'spectral_tilt_sd',
    'syllable_duration_mean',
    'syllable_duration_sd'
]

# Adjust DataFrames to only include relevant columns
references_df = references_df[['file_name'] + categories]
synthesized_df = synthesized_df[['file_name'] + categories]


def create_radar_plot(data_original, data_manipulated, data_synthesized_original, data_synthesized_manipulated, categories, output_file, manipulation):
    num_categories = len(categories)

    # Close the loop for radar plot
    data_original = np.concatenate((data_original, [data_original[0]]))
    data_manipulated = np.concatenate((data_manipulated, [data_manipulated[0]]))
    data_synthesized_original = np.concatenate((data_synthesized_original, [data_synthesized_original[0]]))
    data_synthesized_manipulated = np.concatenate((data_synthesized_manipulated, [data_synthesized_manipulated[0]]))

    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.plot(angles, data_original, 'o-', label='Original', linewidth=2, markersize=8)
    ax.plot(angles, data_manipulated, 'o-', label=manipulation.capitalize(), linewidth=2, markersize=8)
    ax.plot(angles, data_synthesized_original, 'o-', label='Synthesized (Original)', linewidth=2, markersize=8)
    ax.plot(angles, data_synthesized_manipulated, 'o-', label=f'Synthesized ({manipulation.capitalize()})', linewidth=2, markersize=8)

    ax.fill(angles, data_original, alpha=0.25)
    ax.fill(angles, data_manipulated, alpha=0.25)
    ax.fill(angles, data_synthesized_original, alpha=0.25)
    ax.fill(angles, data_synthesized_manipulated, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, rotation=45, ha='right')

    plt.title('Radar Plot of Normalized Features')

    # Get figure size in inches
    fig_width_inch, fig_height_inch = fig.get_size_inches()

    # Convert cm to inches
    cm_to_inch = 0.393701
    horizontal_offset_inch = 5 * cm_to_inch
    vertical_offset_inch = 2 * cm_to_inch

    # Calculate the position for the legend
    bbox_to_anchor_x = 1 - horizontal_offset_inch / fig_width_inch
    bbox_to_anchor_y = 1 + vertical_offset_inch / fig_height_inch

    # Position legend 5 cm to the left and 2 cm up of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(bbox_to_anchor_x, bbox_to_anchor_y))

    plt.savefig(output_file)
    plt.close()


# Output directory
output_dir = r"G:\Research\XTTS_Test\VISUALIZATION\radar_plots"
os.makedirs(output_dir, exist_ok=True)


# Function to process a batch of files
def process_files(manipulation_type):
    manipulation_files = references_df[references_df['file_name'].str.contains(manipulation_type)]

    for _, row in tqdm(manipulation_files.iterrows(), total=len(manipulation_files), desc=f"Processing {manipulation_type} files"):
        manipulated_file_name = row['file_name']
        manipulated_data = row[categories].values  # Get data excluding the 'file_name' column

        # Get the corresponding original file
        concat_n = manipulated_file_name.split('_')[-1]  # Get the 'concat_n' part
        original_file_name = manipulated_file_name.replace(manipulation_type, 'original')

        original_data_row = references_df[references_df['file_name'] == original_file_name]
        if original_data_row.empty:
            continue
        original_data = original_data_row[categories].values.flatten()

        # Filter synthesized data for original and manipulated files
        synthesized_original_filtered = synthesized_df[synthesized_df['file_name'].str.contains(f"{original_file_name[:-4]}_")]
        synthesized_manipulated_filtered = synthesized_df[synthesized_df['file_name'].str.contains(f"{manipulated_file_name[:-4]}_")]

        # Calculate mean of synthesized data for original and manipulated
        synthesized_original_mean = synthesized_original_filtered[categories].mean().values
        synthesized_manipulated_mean = synthesized_manipulated_filtered[categories].mean().values

        # Create radar plot with the specified filename
        radar_plot_file = os.path.join(output_dir, f"{manipulation_type}_concat_{concat_n}_radar_plot.pdf")
        create_radar_plot(original_data, manipulated_data, synthesized_original_mean, synthesized_manipulated_mean, categories, radar_plot_file, manipulation_type)

def plot_radar_plots():
    # Process 'enhanced' files
    process_files('enhanced')

    # Process 'denoised' files
    process_files('denoised')
