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
        df = load_dataset(csv_file_path)
        print("\nInitial DataFrame loaded successfully for EDA.")
    except Exception as e:
        print(f"Could not load data for EDA: {e}")
        # Exit or handle the error if data loading fails
        exit() # Exiting if data isn't loaded for subsequent steps

    print("\n" + "="*50)
    print("Beginning Initial Data Exploration (EDA)")
    print("="*50)

    # 1. Inspect Data
    print("\n--- 1. Inspecting Data ---")
    print("\n1.1. df.head(): First 5 rows of the DataFrame")
    print(df.head())

    print("\n1.2. df.info(): Summary of DataFrame, including data types and non-null values")
    df.info()

    print("\n1.3. df.describe(): Descriptive statistics for numerical columns")
    print(df.describe())

    # 2. Check for Missing Values
    print("\n--- 2. Checking for Missing Values ---")
    print("\n2.1. df.isnull().sum(): Count of missing values per column")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0]) # Only show columns with missing values

    # Strategy for missing values (example for 'sms_spam_collection.csv'):
    # The 'sms_spam_collection.csv' often has several unnamed columns that are entirely empty.
    # We will decide to drop them.
    # Identify columns that are mostly or entirely null (often unnamed columns in this dataset)
    print("\n2.2. Deciding on Missing Value Strategy:")
    # For 'sms_spam_collection.csv', columns like 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4' are often empty.
    # Let's drop columns that have a high percentage of missing values (e.g., > 90%)
    threshold = len(df) * 0.90 # 90% missing values
    cols_to_drop_missing = missing_values[missing_values > threshold].index.tolist()

    if cols_to_drop_missing:
        print(f"Dropping columns with > 90% missing values: {cols_to_drop_missing}")
        df.drop(columns=cols_to_drop_missing, inplace=True)
        print("Columns dropped. Re-checking missing values:")
        print(df.isnull().sum())
    else:
        print("No columns with > 90% missing values found to drop. Proceeding.")


    # 3. Understand Target Distribution
    print("\n--- 3. Understanding Target Distribution ---")
    # For 'sms_spam_collection.csv', the target column is usually 'v1'
    target_column = 'v1' # Assuming 'v1' is the original target column name

    if target_column in df.columns:
        print(f"\n3.1. df['{target_column}'].value_counts(): Distribution of the target variable")
        target_distribution = df[target_column].value_counts()
        print(target_distribution)

        print("\n3.2. Checking for Imbalance:")
        if len(target_distribution) > 1:
            # Calculate percentage for better understanding
            total_samples = target_distribution.sum()
            for value, count in target_distribution.items():
                percentage = (count / total_samples) * 100
                print(f"'{value}': {count} samples ({percentage:.2f}%)")

            # Simple check for imbalance (e.g., if one class is less than 20% of the other)
            if target_distribution.min() / target_distribution.max() < 0.2:
                print("Warning: The target variable shows significant class imbalance.")
                print("This might require handling (e.g., oversampling, undersampling, or using specific evaluation metrics) in later steps.")
            else:
                print("The target variable appears relatively balanced or imbalance is not severe.")
        else:
            print(f"Only one unique value found in '{target_column}'. Cannot assess distribution.")
    else:
        print(f"Target column '{target_column}' not found. Please check column names after loading.")


    # 4. Rename Columns (if necessary)
    print("\n--- 4. Renaming Columns ---")
    # For 'sms_spam_collection.csv', 'v1' is typically the label and 'v2' is the message.
    # Check if 'v1' and 'v2' exist before renaming
    columns_to_rename = {}
    if 'v1' in df.columns:
        columns_to_rename['v1'] = 'label'
    if 'v2' in df.columns:
        columns_to_rename['v2'] = 'message'

    if columns_to_rename:
        print(f"Renaming columns: {columns_to_rename}")
        df.rename(columns=columns_to_rename, inplace=True)
        print("Columns renamed. New column names:")
        print(df.columns.tolist())
    else:
        print("No 'v1' or 'v2' columns found for default renaming.")

    # 5. Drop Unnecessary Columns (after renaming, if applicable)
    print("\n--- 5. Dropping Unnecessary Columns ---")
    # This step is often implicitly handled by the "check for missing values" step
    # if the unnecessary columns are empty.
    # However, some datasets might have ID columns or other irrelevant features that are not empty.
    # For 'sms_spam_collection.csv', after dropping empty 'Unnamed' columns and renaming,
    # there usually aren't other obviously unnecessary columns.

    # Example of dropping if you specifically knew a column name (e.g., 'id_column')
    # if 'id_column' in df.columns:
    #     print("Dropping 'id_column'.")
    #     df.drop(columns=['id_column'], inplace=True)
    # else:
    #     print("No specific 'id_column' found to drop.")

    print("Reviewing final DataFrame head after EDA steps:")
    print(df.head())
    print(f"Final DataFrame shape: {df.shape}")

    print("\n" + "="*50)
    print("Initial Data Exploration (EDA) Complete.")
    print("="*50)

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