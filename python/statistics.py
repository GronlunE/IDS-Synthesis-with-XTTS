"""
Created on 31.8. 2024

@author: GronlunE

Description:

This script performs data extraction and normalization operations by utilizing functions from the `data_extraction` module. It handles tasks related to measuring data from CSV files, calculating kernel density estimates (KDE) data, and normalizing datasets.

The script performs the following actions:
- Extracts and calculates measurements from CSV files using the `csv_data_extraction` function.
- Computes KDE data using the `kde_data_extraction` function.
- Normalizes data using the `normalize` function.

Usage:
- Ensure the `data_extraction` module is available and includes the necessary functions.
- Run the script to execute data extraction and normalization processes. Each function will perform its designated task sequentially.

Dependencies:
- `data_extraction` module, which should include `csv_data_extraction`, `kde_data_extraction`, and `normalize` functions.

"""

from data_extraction import csv_data_extraction, kde_data_extraction, normalize

# Calculate measurements from CSV files
csv_data_extraction.calculate_measurements()

# Calculate KDE data
kde_data_extraction.calculate_kde_data()

# Normalize the dataset
normalize.normalize()
