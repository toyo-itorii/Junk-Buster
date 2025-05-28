# data_loader.py

import pandas as pd
import os

def load_dataset(file_path):
    """
    Loads a dataset into a Pandas DataFrame, handling CSV and TSV formats.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If the file format is not supported or the file does not exist.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"Error: File not found at '{file_path}'")

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
        except UnicodeDecodeError:
            print("Warning: 'latin-1' encoding failed, trying 'utf-8'.")
            df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Successfully loaded CSV: {file_path}")
    elif file_extension == '.tsv':
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='latin-1')
        except UnicodeDecodeError:
            print("Warning: 'latin-1' encoding failed, trying 'utf-8'.")
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        print(f"Successfully loaded TSV: {file_path}")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only .csv and .tsv are supported.")

    return df

if __name__ == "__main__":
    # Example Usage:

    # --- For CSV file ---
    csv_file_path = 'data/emails.csv' # Adjust this path to your CSV file
    try:
        df_csv = load_dataset(csv_file_path)
        print("\nFirst 5 rows of the CSV DataFrame:")
        print(df_csv.head())
        print(f"\nShape of the CSV DataFrame: {df_csv.shape}")
    except ValueError as e:
        print(e)
    except FileNotFoundError:
        print(f"Error: Make sure '{csv_file_path}' exists in your 'data' directory or specify the correct path.")

    print("-" * 30)

    # # --- For TSV file (example, you might need to create a dummy TSV or adjust path) ---
    # tsv_file_path = 'data/my_text_data.tsv' # Adjust this path to your TSV file
    # # To run this example, you might need to create a dummy TSV file for testing:
    # # with open(tsv_file_path, 'w') as f:
    # #     f.write("col1\tcol2\n")
    # #     f.write("value1a\tvalue1b\n")
    # #     f.write("value2a\tvalue2b\n")

    # try:
    #     df_tsv = load_dataset(tsv_file_path)
    #     print("\nFirst 5 rows of the TSV DataFrame:")
    #     print(df_tsv.head())
    #     print(f"\nShape of the TSV DataFrame: {df_tsv.shape}")
    # except ValueError as e:
    #     print(e)
    # except FileNotFoundError:
    #     print(f"Error: Make sure '{tsv_file_path}' exists in your 'data' directory or specify the correct path.")

    # print("-" * 30)

    # # --- Example of an unsupported file format ---
    # unsupported_file_path = 'data/image.jpg'
    # try:
    #     df_unsupported = load_dataset(unsupported_file_path)
    # except ValueError as e:
    #     print(e)
    # except FileNotFoundError:
    #     print(f"Error: Make sure '{unsupported_file_path}' exists or specify the correct path.")