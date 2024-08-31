import pandas as pd


def normalize_column(col):
    """ Normalize a column between 0 and 1 """
    min_val = col.min()
    max_val = col.max()
    return (col - min_val) / (max_val - min_val)


def normalize_data(input_csv, output_csv):
    """

    :param input_csv:
    :param output_csv:
    :return:
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

    :return:
    """
    # File paths
    reference_csv = r"plot_data/scatter/references.csv"
    synthesized_csv = r"plot_data/scatter/synthesized.csv"
    normalized_reference_csv = r"plot_data/radar/references_normalized.csv"
    normalized_synthesized_csv = r"plot_data/radar/synthesized_normalized.csv"

    # Normalize and save the data
    normalize_data(reference_csv, normalized_reference_csv)
    normalize_data(synthesized_csv, normalized_synthesized_csv)
