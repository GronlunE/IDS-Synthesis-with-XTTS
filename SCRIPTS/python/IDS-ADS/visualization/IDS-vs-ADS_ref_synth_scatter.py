import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr
import os
from tqdm import tqdm
from matplotlib.lines import Line2D

# Set output directory and file paths
OUTPUT_DIR = r"G:\Research\XTTS_Test\DATA\IDS-ADS\figures\scatter\ref_synth"
SYNTHDATA_FILE = r"G:\Research\XTTS_Test\DATA\IDS-ADS\IDS-ADS_syntheses.csv"
REFDATA_FILE = r"G:\Research\XTTS_Test\DATA\IDS-ADS\IDS-ADS_references.csv"

# Load data
synth_data = pd.read_csv(SYNTHDATA_FILE)
ref_data = pd.read_csv(REFDATA_FILE)

# Extract categories and speakers for synthesized data
synth_data['category'] = (synth_data['file_name'].str.extract(r'(ADS|IDS)_(original|enhanced|denoised)')[0] + '_' +
                                                              synth_data['file_name'].str.extract(r'(ADS|IDS)_(original|enhanced|denoised)')[1] + '_synthesis')
synth_data['speaker'] = synth_data['file_name'].str.extract(r'Baby (\d+)')[0]

# Define color map for GILES numbers
giles_numbers = synth_data['file_name'].str.extract(r'GILES_(\d+)')[0].unique()  # Extract GILES numbers from file names
giles_colors = plt.get_cmap('tab20').colors  # Use 'tab20' colormap for up to 20 distinct colors
giles_color_map = dict(zip(giles_numbers.astype(int), giles_colors[:len(giles_numbers)]))  # Ensure GILES numbers are integers
synth_data['giles_color'] = synth_data['file_name'].str.extract(r'GILES_(\d+)')[0].astype(int).map(giles_color_map)

# Extract categories and speakers for reference data
ref_data['category'] = (ref_data['file_name'].str.extract(r'(ADS|IDS)_(original|enhanced|denoised)')[0] + '_' +
                                                            ref_data['file_name'].str.extract(r'(ADS|IDS)_(original|enhanced|denoised)')[1] + '_reference')
ref_data['speaker'] = ref_data['file_name'].str.extract(r'Baby (\d+)')[0]

# List of variables to process
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

def draw_ref_synth_scatter_for_speaker(speaker, ref_data, synth_data, variable):
    """Draw scatter plot comparing reference and synthesized data for a specific speaker."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    ax1, ax2 = axes.flatten()

    # Filter data by speaker
    speaker_ref_data = ref_data[ref_data['speaker'] == speaker]
    speaker_synth_data = synth_data[synth_data['speaker'] == speaker]

    # Function to plot data for a specific type (Enhanced vs Enhanced or Original vs Enhanced)
    def plot_data(ax, references_df, synthesized_df, feature_name, is_enhanced=True):
        # Lists to hold values for IDS and ADS
        ids_ref_values = []
        ids_synth_values = []
        ads_ref_values = []
        ads_synth_values = []
        scatter_handles = []

        # Plot for both ADS and IDS
        for _, ref_row in references_df.iterrows():
            ref_base_filename = ref_row['file_name'].split('\\')[-1].replace('.wav', '')
            if "enhanced" in ref_base_filename and is_enhanced:
                ref_value = ref_row[variable]

                for i in range(1, 11):  # Loop through GILES 1 to 10
                    synth_filename = f"{ref_base_filename}_GILES_{i}.wav"
                    synth_row = synthesized_df[synthesized_df['file_name'].str.contains(synth_filename)]

                    if not synth_row.empty:
                        synth_value = synth_row[variable].values[0]
                        giles_color = giles_color_map.get(i, 'tab:gray')

                        # Determine if it's ADS or IDS based on filename
                        data_type = 'ADS' if 'ADS' in ref_base_filename else 'IDS'
                        marker = 'o' if data_type == 'IDS' else 'x'
                        handle = ax.scatter(ref_value, synth_value, color=giles_color, marker=marker, alpha=0.5)

                        if len(scatter_handles) < 10:
                            scatter_handles.append(handle)

                        # Store values for setting scale
                        if data_type == 'IDS':
                            ids_ref_values.append(ref_value)
                            ids_synth_values.append(synth_value)
                        else:
                            ads_ref_values.append(ref_value)
                            ads_synth_values.append(synth_value)

        # Calculate Pearson correlation and linear fit for IDS
        if ids_ref_values and ids_synth_values:  # Check if lists are not empty
            r_ids, p_value_ids = pearsonr(ids_ref_values, ids_synth_values)
            slope_ids, intercept_ids, _, _, _ = linregress(ids_ref_values, ids_synth_values)

            # Extend line for plotting
            min_val, max_val = min(ids_ref_values + ids_synth_values), max(ids_ref_values + ids_synth_values)
            range_val = max_val - min_val
            buffer = range_val * 0.1
            x_range = np.linspace(min_val - buffer, max_val + buffer, 100)
            line_fit_ids = slope_ids * x_range + intercept_ids

            # Plot the extended linear fit line for IDS
            ax.plot(x_range, line_fit_ids, color='black', linestyle='-', label=f'IDS Fit Line (r={r_ids:.2f}, p={p_value_ids:.4f})')

        # Calculate Pearson correlation and linear fit for ADS
        if ads_ref_values and ads_synth_values:  # Check if lists are not empty
            r_ads, p_value_ads = pearsonr(ads_ref_values, ads_synth_values)
            slope_ads, intercept_ads, _, _, _ = linregress(ads_ref_values, ads_synth_values)

            # Extend line for plotting
            min_val, max_val = min(ads_ref_values + ads_synth_values), max(ads_ref_values + ads_synth_values)
            range_val = max_val - min_val
            buffer = range_val * 0.1
            x_range = np.linspace(min_val - buffer, max_val + buffer, 100)
            line_fit_ads = slope_ads * x_range + intercept_ads

            # Plot the extended linear fit line for ADS
            ax.plot(x_range, line_fit_ads, color='black', linestyle='--', label=f'ADS Fit Line (r={r_ads:.2f}, p={p_value_ads:.4f})')

        # Set labels and title for the plot
        ax.set_xlabel("References")
        ax.set_ylabel("Syntheses")
        ax.set_title(f"{feature_name.replace('_', ' ').title()} - Scatter - Enhanced References vs. Enhanced Syntheses, x for ADS, o for IDS")

        # Set limits with buffer
        ax.set_xlim(min_val - buffer, max_val + buffer)
        ax.set_ylim(min_val - buffer, max_val + buffer)

        # Combine scatter handles into one legend
        combined_handles = scatter_handles + [
            Line2D([0], [0], color='black', linestyle='-', label=f'IDS Fit Line (r={r_ids:.2f}, p={p_value_ids:.4f})'),
            Line2D([0], [0], color='black', linestyle='--', label=f'ADS Fit Line (r={r_ads:.2f}, p={p_value_ads:.4f})')
        ]
        combined_labels = [f'GILES {i}' for i in range(1, 11)] + [f'IDS Fit Line (r={r_ids:.2f}, p={p_value_ids:.4f})', f'ADS Fit Line (r={r_ads:.2f}, p={p_value_ads:.4f})']

        # Create a single legend
        ax.legend(combined_handles, combined_labels, loc='best', fontsize='small')

    # Plot Enhanced vs Enhanced
    plot_data(ax1, speaker_ref_data, speaker_synth_data, variable, is_enhanced=True)

    # Plot Original vs Enhanced
    ids_ref_values = []
    ids_synth_values = []
    ads_ref_values = []
    ads_synth_values = []
    scatter_handles = []

    for _, ref_row in speaker_ref_data.iterrows():
        ref_base_filename = ref_row['file_name'].split('\\')[-1].replace('.wav', '')

        if "enhanced" not in ref_base_filename:  # Use original reference
            # Generate the corresponding enhanced filename
            enhanced_base_filename = ref_base_filename.replace("original", "enhanced") + '.wav'

            # Find the enhanced reference value from references_df
            enhanced_row = speaker_ref_data[speaker_ref_data['file_name'].str.contains(enhanced_base_filename)]

            if not enhanced_row.empty:
                ref_value = enhanced_row[variable].iloc[0]

                for i in range(1, 11):  # Loop through GILES 1 to 10
                    synth_filename = f"{ref_base_filename}_GILES_{i}.wav"
                    synth_row = speaker_synth_data[speaker_synth_data['file_name'].str.contains(synth_filename)]

                    if not synth_row.empty:
                        synth_value = synth_row[variable].values[0]
                        giles_color = giles_color_map.get(i, 'tab:gray')

                        # Determine if it's ADS or IDS based on filename
                        data_type = 'ADS' if 'ADS' in ref_base_filename else 'IDS'
                        marker = 'o' if data_type == 'IDS' else 'x'
                        handle = ax2.scatter(ref_value, synth_value, color=giles_color, marker=marker, alpha=0.5)

                        if len(scatter_handles) < 10:
                            scatter_handles.append(handle)

                        # Store values for setting scale
                        if data_type == 'IDS':
                            ids_ref_values.append(ref_value)
                            ids_synth_values.append(synth_value)
                        else:
                            ads_ref_values.append(ref_value)
                            ads_synth_values.append(synth_value)

    # Calculate Pearson correlation and linear fit for IDS in Original vs Enhanced plot
    if ids_ref_values and ids_synth_values:  # Check if lists are not empty
        r_ids, p_value_ids = pearsonr(ids_ref_values, ids_synth_values)
        slope_ids, intercept_ids, _, _, _ = linregress(ids_ref_values, ids_synth_values)

        # Extend line for plotting
        min_val, max_val = min(ids_ref_values + ids_synth_values), max(ids_ref_values + ids_synth_values)
        range_val = max_val - min_val
        buffer = range_val * 0.1
        x_range = np.linspace(min_val - buffer, max_val + buffer, 100)
        line_fit_ids = slope_ids * x_range + intercept_ids

        # Plot the extended linear fit line for IDS
        ax2.plot(x_range, line_fit_ids, color='black', linestyle='-', label=f'IDS Fit Line (r={r_ids:.2f}, p={p_value_ids:.4f})')

    # Calculate Pearson correlation and linear fit for ADS in Original vs Enhanced plot
    if ads_ref_values and ads_synth_values:  # Check if lists are not empty
        r_ads, p_value_ads = pearsonr(ads_ref_values, ads_synth_values)
        slope_ads, intercept_ads, _, _, _ = linregress(ads_ref_values, ads_synth_values)

        # Extend line for plotting
        min_val, max_val = min(ads_ref_values + ads_synth_values), max(ads_ref_values + ads_synth_values)
        range_val = max_val - min_val
        buffer = range_val * 0.1
        x_range = np.linspace(min_val - buffer, max_val + buffer, 100)
        line_fit_ads = slope_ads * x_range + intercept_ads

        # Plot the extended linear fit line for ADS
        ax2.plot(x_range, line_fit_ads, color='black', linestyle='--', label=f'ADS Fit Line (r={r_ads:.2f}, p={p_value_ads:.4f})')

    # Set labels and title for the second scatter plot
    ax2.set_xlabel("Original references")
    ax2.set_ylabel("Enhanced syntheses")
    ax2.set_title(f"{variable.replace('_', ' ').title()} - Scatter - Original References vs. Enhanced Syntheses, x for ADS, o for IDS")

    # Set limits with buffer for the second plot
    ax2.set_xlim(min_val - buffer, max_val + buffer)
    ax2.set_ylim(min_val - buffer, max_val + buffer)

    # Combine scatter handles into one legend
    combined_handles = scatter_handles + [
        Line2D([0], [0], color='black', linestyle='-', label=f'IDS Fit Line (r={r_ids:.2f}, p={p_value_ids:.4f})'),
        Line2D([0], [0], color='black', linestyle='--', label=f'ADS Fit Line (r={r_ads:.2f}, p={p_value_ads:.4f})')
    ]
    combined_labels = [f'GILES {i}' for i in range(1, 11)] + [f'IDS Fit Line (r={r_ids:.2f}, p={p_value_ids:.4f})', f'ADS Fit Line (r={r_ads:.2f}, p={p_value_ads:.4f})']

    # Create a single legend
    ax2.legend(combined_handles, combined_labels, loc='best', fontsize='small')

    # Adjust layout and save the plot as a PDF
    pdf_filename = os.path.join(OUTPUT_DIR, f"Baby{speaker}_{variable}_ref_synth_scatter.pdf")
    plt.tight_layout()
    plt.savefig(pdf_filename, format='pdf')
    plt.close(fig)  # Close the figure to free memory


# Process all speakers
speakers = ref_data['speaker'].unique()
"""
for speaker in tqdm(speakers):
    for variable in variables_list:
        draw_ref_synth_scatter_for_speaker(speaker, ref_data, synth_data, variable)
"""


def draw_combined_ref_synth_scatter(ref_data, synth_data, variable):
    """Draw scatter plot combining all speakers' data for a specific variable."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    ax1, ax2 = axes.flatten()

    # Function to plot data for a specific type (Enhanced vs Enhanced or Original vs Enhanced)
    def plot_combined_data(ax, references_df, synthesized_df, feature_name, is_enhanced=True):
        # Lists to hold values for IDS and ADS
        ids_ref_values = []
        ids_synth_values = []
        ads_ref_values = []
        ads_synth_values = []
        scatter_handles = []

        # Plot for both ADS and IDS
        for _, ref_row in references_df.iterrows():
            ref_base_filename = ref_row['file_name'].split('\\')[-1].replace('.wav', '')
            if ("enhanced" in ref_base_filename and is_enhanced) or ("enhanced" not in ref_base_filename and not is_enhanced):
                ref_value = ref_row[variable]

                for i in range(1, 11):  # Loop through GILES 1 to 10
                    synth_filename = f"{ref_base_filename}_GILES_{i}.wav"
                    synth_row = synthesized_df[synthesized_df['file_name'].str.contains(synth_filename)]

                    if not synth_row.empty:
                        synth_value = synth_row[variable].values[0]
                        giles_color = giles_color_map.get(i, 'tab:gray')

                        # Determine if it's ADS or IDS based on filename
                        data_type = 'ADS' if 'ADS' in ref_base_filename else 'IDS'
                        marker = 'o' if data_type == 'IDS' else 'x'
                        handle = ax.scatter(ref_value, synth_value, color=giles_color, marker=marker, alpha=0.5)

                        if len(scatter_handles) < 10:
                            scatter_handles.append(handle)

                        # Store values for setting scale
                        if data_type == 'IDS':
                            ids_ref_values.append(ref_value)
                            ids_synth_values.append(synth_value)
                        else:
                            ads_ref_values.append(ref_value)
                            ads_synth_values.append(synth_value)

        # Calculate Pearson correlation and linear fit for IDS
        if ids_ref_values and ids_synth_values:  # Check if lists are not empty
            r_ids, p_value_ids = pearsonr(ids_ref_values, ids_synth_values)
            slope_ids, intercept_ids, _, _, _ = linregress(ids_ref_values, ids_synth_values)

            # Extend line for plotting
            min_val, max_val = min(ids_ref_values + ids_synth_values), max(ids_ref_values + ids_synth_values)
            range_val = max_val - min_val
            buffer = range_val * 0.1
            x_range = np.linspace(min_val - buffer, max_val + buffer, 100)
            line_fit_ids = slope_ids * x_range + intercept_ids

            # Plot the extended linear fit line for IDS
            ax.plot(x_range, line_fit_ids, color='black', linestyle='-', label=f'IDS Fit Line (r={r_ids:.2f}, p={p_value_ids:.4f})')

        # Calculate Pearson correlation and linear fit for ADS
        if ads_ref_values and ads_synth_values:  # Check if lists are not empty
            r_ads, p_value_ads = pearsonr(ads_ref_values, ads_synth_values)
            slope_ads, intercept_ads, _, _, _ = linregress(ads_ref_values, ads_synth_values)

            # Extend line for plotting
            min_val, max_val = min(ads_ref_values + ads_synth_values), max(ads_ref_values + ads_synth_values)
            range_val = max_val - min_val
            buffer = range_val * 0.1
            x_range = np.linspace(min_val - buffer, max_val + buffer, 100)
            line_fit_ads = slope_ads * x_range + intercept_ads

            # Plot the extended linear fit line for ADS
            ax.plot(x_range, line_fit_ads, color='black', linestyle='--', label=f'ADS Fit Line (r={r_ads:.2f}, p={p_value_ads:.4f})')

        # Set labels and title for the plot
        ax.set_xlabel("References")
        ax.set_ylabel("Syntheses")
        ax.set_title(f"{feature_name.replace('_', ' ').title()} - Scatter - Enhanced References vs. Enhanced Syntheses, x for ADS, o for IDS")

        # Set limits with buffer
        ax.set_xlim(min_val - buffer, max_val + buffer)
        ax.set_ylim(min_val - buffer, max_val + buffer)

        # Combine scatter handles into one legend
        combined_handles = scatter_handles + [
            Line2D([0], [0], color='black', linestyle='-', label=f'IDS Fit Line (r={r_ids:.2f}, p={p_value_ids:.4f})'),
            Line2D([0], [0], color='black', linestyle='--', label=f'ADS Fit Line (r={r_ads:.2f}, p={p_value_ads:.4f})')
        ]
        combined_labels = [f'GILES {i}' for i in range(1, 11)] + [f'IDS Fit Line (r={r_ids:.2f}, p={p_value_ids:.4f})', f'ADS Fit Line (r={r_ads:.2f}, p={p_value_ads:.4f})']

        # Create a single legend
        ax.legend(combined_handles, combined_labels, loc='best', fontsize='small')

    # Plot Enhanced vs Enhanced
    plot_combined_data(ax1, ref_data, synth_data, variable, is_enhanced=True)

    # Plot Original vs Enhanced
    plot_combined_data(ax2, ref_data, synth_data, variable, is_enhanced=False)

    # Adjust layout and save the plot as a PDF
    pdf_filename = os.path.join(OUTPUT_DIR, f"AllSpeakers_{variable}_ref_synth_scatter.pdf")
    plt.tight_layout()
    plt.savefig(pdf_filename, format='pdf')
    plt.close(fig)  # Close the figure to free memory


# Process all variables with progress tracking using tqdm
for variable in tqdm(variables_list):
    draw_combined_ref_synth_scatter(ref_data, synth_data, variable)
