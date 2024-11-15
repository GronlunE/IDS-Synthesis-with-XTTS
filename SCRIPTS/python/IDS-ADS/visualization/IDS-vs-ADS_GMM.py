import os

# Set the number of threads to avoid memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# Load the data from CSV files
syntheses_df = pd.read_csv(r"G:\Research\XTTS_Test\DATA\IDS-ADS\IDS-ADS_syntheses.csv")
references_df = pd.read_csv(r"G:\Research\XTTS_Test\DATA\IDS-ADS\IDS-ADS_references.csv")

# Select the relevant variables
variables_list = [
    'f0_log_std', 'f0_log_mean',
    'f0_delta_abs_log_std', 'f0_delta_abs_log_mean',
    'spectral_tilt_std', 'spectral_tilt_mean',
    'syllable_durations_log_std', 'syllable_durations_log_mean',
    'f0_log_std_phrase', 'f0_log_mean_phrase',
    'f0_delta_abs_log_std_phrase', 'f0_delta_abs_log_mean_phrase',
    'spectral_tilt_std_phrase', 'spectral_tilt_mean_phrase',
    'syllable_durations_log_std_phrase', 'syllable_durations_log_mean_phrase'
]

# Create or overwrite the text file to save results
with open(r"G:\Research\XTTS_Test\DATA\IDS-ADS\clustering_results.txt", "w") as text_file:

    def perform_clustering(dataframe, dataset_name, label, ax, cluster_index):
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        dataframe = dataframe.copy(deep=True)  # Ensure a deep copy

        # Extract the relevant variables for clustering
        data = dataframe[variables_list].values

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Fit Gaussian Mixture Model (GMM) with a specified number of clusters
        n_clusters = 2  # Adjust this number as needed
        gmm = GaussianMixture(n_components=n_clusters, random_state=123)
        gmm.fit(scaled_data)

        # Predict the clusters
        cluster_labels = gmm.predict(scaled_data)

        # Add cluster labels to the dataframe using .loc
        dataframe.loc[:, 'cluster_label'] = cluster_labels

        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        # Plot the clusters
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=50)
        ax.set_title(f'GMM Clustering of {dataset_name} ({label}) Data')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')

        # Print the clusters in a structured format
        total_files = len(dataframe)
        text_file.write(f"\n{dataset_name} ({label}) Clusters:\n")

        for cluster in range(n_clusters):
            text_file.write(f"Cluster {cluster + 1}:\n" + "-" * 10 + "\n")
            cluster_files = dataframe[dataframe['cluster_label'] == cluster]['file_name']
            cluster_count = len(cluster_files)

            # Count the IDS and ADS files based on file naming convention
            ids_files = cluster_files[cluster_files.str.contains("IDS")]
            ads_files = cluster_files[cluster_files.str.contains("ADS")]

            ids_count = len(ids_files)
            ads_count = len(ads_files)

            # Calculate percentages
            ids_percentage = (ids_count / total_files) * 100 if total_files > 0 else 0
            ads_percentage = (ads_count / total_files) * 100 if total_files > 0 else 0
            cluster_ids_percentage = (ids_count / cluster_count) * 100 if cluster_count > 0 else 0
            cluster_ads_percentage = (ads_count / cluster_count) * 100 if cluster_count > 0 else 0

            text_file.write(f"\nCounts in Cluster {cluster + 1}:\n")
            text_file.write(f"  IDS files: {ids_count} ({cluster_ids_percentage:.2f}% of cluster, {ids_percentage:.2f}% of total)\n")
            text_file.write(f"  ADS files: {ads_count} ({cluster_ads_percentage:.2f}% of cluster, {ads_percentage:.2f}% of total)\n")

            # Calculate Pearson correlation for the first two relevant variables
            if len(dataframe) > 1:  # Avoid issues with single point correlation
                r, p = pearsonr(dataframe[variables_list[0]], dataframe[variables_list[1]])
                text_file.write(f"  Pearson r: {r:.2f}, p-value: {p:.4f}\n")
            else:
                text_file.write("  Pearson correlation not applicable (insufficient data).\n")

    # Step 1: Create subplots for enhanced and original syntheses and references
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Perform clustering for each combination
    for dataset_df, dataset_name, row, col in [
        (syntheses_df, "Syntheses", 0, 0),
        (references_df, "References", 0, 1),
        (syntheses_df, "Syntheses", 1, 0),
        (references_df, "References", 1, 1)
    ]:
        for label in ["enhanced", "original"]:  # Exclude "denoised"
            # Filter the dataframe based on the current label and ensure we are checking for enhanced
            filtered_df = dataset_df[dataset_df['file_name'].str.contains(label) & ~dataset_df['file_name'].str.contains("denoised")]
            if not filtered_df.empty:
                perform_clustering(filtered_df, dataset_name, label, axs[row, col], (row, col))

    plt.tight_layout()
    plt.savefig(r"G:\Research\XTTS_Test\DATA\IDS-ADS\clustering_results.pdf")
    plt.show()

print("Clustering completed and results saved to text file.")
