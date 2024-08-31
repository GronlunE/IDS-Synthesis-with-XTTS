"""
Created on 31.8. 2024

@author: GronlunE

Description:

This script handles the normalization of data in CSV files. It includes functions to normalize numerical columns of data to a range between 0 and 1 and save the results to new CSV files.

The script provides:
- `normalize_column`: A function to normalize a single column of data.
- `normalize_data`: A function to read data from an input CSV file, normalize all numeric columns, and save the results to an output CSV file.
- `normalize`: A high-level function that specifies file paths for input and output CSV files and performs the normalization for specified datasets.

Usage:
- Ensure the required CSV files are available at the specified paths.
- Run the script to normalize the data in the CSV files. The normalized data will be saved to new files as specified.

Dependencies:
- `pandas` for handling CSV file operations and data manipulation.

"""

import pandas as pd


def normalize_column(col):
    """ Normalize a column between 0 and 1 """
    min_val = col.min()
    max_val = col.max()
    return (col - min_val) / (max_val - min_val)


def normalize_data(input_csv, output_csv):
    """
    Normalize numerical columns in a CSV file and save to a new file.

    :param input_csv: Path to the input CSV file with data to be normalized.
    :param output_csv: Path to the output CSV file to save the normalized data.
    :return: None
    """
    # Load the data from CSV
    df = pd.read_csv(input_csv)

    # Specify non-numeric columns to exclude from normalization
    non_numeric_columns = ['file_name']

    # Separate numeric columns for normalization
    numeric_columns = [col for col in df.columns if col not in non_numeric_columns]

    # Normalize each numeric column
    normalized_df = df.copy()
    for col in numeric_columns:
        normalized_df[col] = normalize_column(df[col])

    # Save the normalized data to a new CSV file
    normalized_df.to_csv(output_csv, index=False)


def normalize():
    """
    Normalize datasets and save the results to new CSV files.

    :return: None
    """
    # File paths
    reference_csv = r"plot_data/scatter/references.csv"
    synthesized_csv = r"plot_data/scatter/synthesized.csv"
    normalized_reference_csv = r"plot_data/radar/references_normalized.csv"
    normalized_synthesized_csv = r"plot_data/radar/synthesized_normalized.csv"

    # Normalize and save the data
    normalize_data(reference_csv, normalized_reference_csv)
    normalize_data(synthesized_csv, normalized_synthesized_csv)
