import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2
import os

# Define directories and subdirectories
SYNTHESIZED_FILE = r"G:\Research\XTTS_Test\DATA\IDS-ADS\IDS-ADS_syntheses.csv"
REFERENCES_FILE = r"G:\Research\XTTS_Test\DATA\IDS-ADS\IDS-ADS_references.csv"
SCATTER_OUTPUT_DIR = r"G:\Research\XTTS_Test\DATA\IDS-ADS\figures\scatter\other"

# Function to add ellipses for 95% confidence interval
def add_ellipse(ax, x_data, y_data, color, linestyle, label=None):
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

def draw_subplot(ax, data, x_var, y_var, category_labels, color_map):
    handles = []
    for category in category_labels:
        category_data = data[data['category'] == category]
        x_data = category_data[x_var].values
        y_data = category_data[y_var].values

        # Determine marker and linestyle based on category
        if 'synthesis' in category:
            marker = 'x'
            linestyle = '--'  # Dashed for synthesized data ellipses
        else:
            marker = 'o'  # Default marker for reference
            linestyle = '-'  # Solid for reference data ellipses

        color = color_map.get(category, 'tab:gray')
        scatter = ax.scatter(x_data, y_data, color=color, alpha=0.7, label=category, marker=marker)
        add_ellipse(ax, x_data, y_data, color, linestyle=linestyle)
        handles.append(scatter)

    ax.set_xlabel(x_var.replace('_', ' ').title())
    ax.set_ylabel(y_var.replace('_', ' ').title())
    ax.legend(handles, category_labels, title="Category")
    ax.grid(True)

def plot_data_for_speaker(speaker, synthesized_df, references_df, variables):
    for var_name, (x_var, y_var) in variables.items():
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f"Data for Speaker {speaker} - {var_name.replace('_', ' ').title()}", fontsize=16)

        # Original Data
        ax = axs[0, 0]
        ax.set_title("Original Data")

        original_synthesized = synthesized_df[(synthesized_df['speaker'] == speaker) &
                                              (synthesized_df['category'].str.contains('original'))]
        original_references = references_df[(references_df['speaker'] == speaker) &
                                            (references_df['category'].str.contains('original'))]

        draw_subplot(ax, pd.concat([original_synthesized, original_references]), x_var, y_var,
                     ['ADS_original_reference', 'IDS_original_reference',
                      'ADS_original_synthesis', 'IDS_original_synthesis'], color_map)

        # Denoised Data
        ax = axs[0, 1]
        ax.set_title("Denoised Data")

        denoised_synthesized = synthesized_df[(synthesized_df['speaker'] == speaker) &
                                              (synthesized_df['category'].str.contains('denoised'))]
        denoised_references = references_df[(references_df['speaker'] == speaker) &
                                            (references_df['category'].str.contains('denoised'))]

        draw_subplot(ax, pd.concat([denoised_synthesized, denoised_references]), x_var, y_var,
                     ['ADS_denoised_reference', 'IDS_denoised_reference',
                      'ADS_denoised_synthesis', 'IDS_denoised_synthesis'], color_map)

        # Enhanced Data
        ax = axs[1, 0]
        ax.set_title("Enhanced Data")

        enhanced_synthesized = synthesized_df[(synthesized_df['speaker'] == speaker) &
                                              (synthesized_df['category'].str.contains('enhanced'))]
        enhanced_references = references_df[(references_df['speaker'] == speaker) &
                                            (references_df['category'].str.contains('enhanced'))]

        draw_subplot(ax, pd.concat([enhanced_synthesized, enhanced_references]), x_var, y_var,
                     ['ADS_enhanced_reference', 'IDS_enhanced_reference',
                      'ADS_enhanced_synthesis', 'IDS_enhanced_synthesis'], color_map)

        # Original vs Enhanced
        ax = axs[1, 1]
        ax.set_title("Original vs Enhanced")

        original_enhanced_data = pd.concat([
            synthesized_df[(synthesized_df['speaker'] == speaker) & (synthesized_df['category'].str.contains('original'))],
            synthesized_df[(synthesized_df['speaker'] == speaker) & (synthesized_df['category'].str.contains('enhanced'))],
            references_df[(references_df['speaker'] == speaker) & (references_df['category'].str.contains('original'))],
            references_df[(references_df['speaker'] == speaker) & (references_df['category'].str.contains('enhanced'))]
        ])

        draw_subplot(ax, original_enhanced_data, x_var, y_var,
                     ['ADS_original_reference', 'IDS_original_reference',
                      'ADS_enhanced_reference', 'IDS_enhanced_reference',
                      'ADS_original_synthesis', 'IDS_original_synthesis',
                      'ADS_enhanced_synthesis', 'IDS_enhanced_synthesis'], color_map)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
        plt.savefig(os.path.join(SCATTER_OUTPUT_DIR, f"Baby{speaker}_{var_name}_scatter.pdf"))
        plt.close()

def plot_all_data(synthesized_df, references_df, variables):
    speakers = synthesized_df['speaker'].unique()

    for speaker in speakers:
        plot_data_for_speaker(speaker, synthesized_df, references_df, variables)

def plot_all_data_combined(synthesized_df, references_df, variables):
    for var_name, (x_var, y_var) in variables.items():
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f"Data for All Speakers - {var_name.replace('_', ' ').title()}", fontsize=16)

        # Original Data
        ax = axs[0, 0]
        ax.set_title("Original Data")

        original_synthesized = synthesized_df[synthesized_df['category'].str.contains('original')]
        original_references = references_df[references_df['category'].str.contains('original')]

        draw_subplot(ax, pd.concat([original_synthesized, original_references]), x_var, y_var,
                     ['ADS_original_reference', 'IDS_original_reference',
                      'ADS_original_synthesis', 'IDS_original_synthesis'], color_map)

        # Denoised Data
        ax = axs[0, 1]
        ax.set_title("Denoised Data")

        denoised_synthesized = synthesized_df[synthesized_df['category'].str.contains('denoised')]
        denoised_references = references_df[references_df['category'].str.contains('denoised')]

        draw_subplot(ax, pd.concat([denoised_synthesized, denoised_references]), x_var, y_var,
                     ['ADS_denoised_reference', 'IDS_denoised_reference',
                      'ADS_denoised_synthesis', 'IDS_denoised_synthesis'], color_map)

        # Enhanced Data
        ax = axs[1, 0]
        ax.set_title("Enhanced Data")

        enhanced_synthesized = synthesized_df[synthesized_df['category'].str.contains('enhanced')]
        enhanced_references = references_df[references_df['category'].str.contains('enhanced')]

        draw_subplot(ax, pd.concat([enhanced_synthesized, enhanced_references]), x_var, y_var,
                     ['ADS_enhanced_reference', 'IDS_enhanced_reference',
                      'ADS_enhanced_synthesis', 'IDS_enhanced_synthesis'], color_map)

        # Original vs Enhanced
        ax = axs[1, 1]
        ax.set_title("Original vs Enhanced")

        original_enhanced_data = pd.concat([
            synthesized_df[synthesized_df['category'].str.contains('original')],
            synthesized_df[synthesized_df['category'].str.contains('enhanced')],
            references_df[references_df['category'].str.contains('original')],
            references_df[references_df['category'].str.contains('enhanced')]
        ])

        draw_subplot(ax, original_enhanced_data, x_var, y_var,
                     ['ADS_original_reference', 'IDS_original_reference',
                      'ADS_enhanced_reference', 'IDS_enhanced_reference',
                      'ADS_original_synthesis', 'IDS_original_synthesis',
                      'ADS_enhanced_synthesis', 'IDS_enhanced_synthesis'], color_map)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
        plt.savefig(os.path.join(SCATTER_OUTPUT_DIR, f"AllSpeakers_{var_name}_scatter.pdf"))
        plt.close()

if __name__ == "__main__":
    # Load synthesized and reference data
    synthesized_df = pd.read_csv(SYNTHESIZED_FILE)
    references_df = pd.read_csv(REFERENCES_FILE)

    # Extract categories and speakers for synthesized and reference data
    synthesized_df['category'] = synthesized_df['file_name'].str.extract(r'(ADS|IDS)_(original|enhanced|denoised)')[0] + '_' + synthesized_df['file_name'].str.extract(r'(ADS|IDS)_(original|enhanced|denoised)')[1] + '_synthesis'
    synthesized_df['speaker'] = synthesized_df['file_name'].str.extract(r'Baby (\d+)')[0]

    references_df['category'] = references_df['file_name'].str.extract(r'(ADS|IDS)_(original|enhanced|denoised)')[0] + '_' + references_df['file_name'].str.extract(r'(ADS|IDS)_(original|enhanced|denoised)')[1] + '_reference'
    references_df['speaker'] = references_df['file_name'].str.extract(r'Baby (\d+)')[0]

    # Define color mappings
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

    variables = {
        'f0_log_sd_mean': ('f0_log_std', 'f0_log_mean'),
        'f0_delta_abs_log_sd_mean': ('f0_delta_abs_log_std', 'f0_delta_abs_log_mean'),
        'spectral_tilt_sd_mean': ('spectral_tilt_std', 'spectral_tilt_mean'),
        'syllable_duration_log_sd_mean': ('syllable_durations_log_std', 'syllable_durations_log_mean'),
        'f0_log_range': ('f0_log_min5', 'f0_log_max95'),
        'f0_delta_abs_log_range': ('f0_delta_abs_log_min_5', 'f0_delta_abs_log_max_95'),
        'spectral_tilt_range': ('spectral_tilt_min5', 'spectral_tilt_max95'),
        'syllable_duration_log_range': ('syllable_durations_log_min5', 'syllable_durations_log_max95'),
        'f0_log_sd_mean_phrase': ('f0_log_std_phrase', 'f0_log_mean_phrase'),
        'f0_delta_abs_log_sd_mean_phrase': ('f0_delta_abs_log_std_phrase', 'f0_delta_abs_log_mean_phrase'),
        'spectral_tilt_sd_mean_phrase': ('spectral_tilt_std_phrase', 'spectral_tilt_mean_phrase'),
        'syllable_duration_log_sd_mean_phrase': ('syllable_durations_log_std_phrase', 'syllable_durations_log_mean_phrase'),
        'f0_log_range_phrase': ('f0_log_min5_phrase', 'f0_log_max95_phrase'),
        'f0_delta_abs_log_range_phrase': ('f0_delta_abs_log_min_5_phrase', 'f0_delta_abs_log_max_95_phrase'),
        'spectral_tilt_range_phrase': ('spectral_tilt_min5_phrase', 'spectral_tilt_max95_phrase'),
        'syllable_duration_log_range_phrase': ('syllable_durations_log_min5_phrase', 'syllable_durations_log_max95_phrase'),
    }

    plot_all_data(synthesized_df, references_df, variables)
    plot_all_data_combined(synthesized_df, references_df, variables)