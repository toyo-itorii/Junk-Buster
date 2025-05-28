# utils.py
import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- NLTK Data Downloads (run once to ensure all necessary data is available) ---
# These checks prevent re-downloading if already present.
# def check_and_download_nltk_data():
#     """Checks if necessary NLTK data is downloaded and downloads it if not."""
#     nltk_data_packages = {
#         'punkt': 'tokenizers/punkt',
#         'wordnet': 'corpora/wordnet',
#         'stopwords': 'corpora/stopwords',
#         'omw-1.4': 'corpora/omw-1.4'
#     }
#     for package_name, package_path in nltk_data_packages.items():
#         try:
#             nltk.data.find(package_path)
#         except nltk.downloader.DownloadError:
#             print(f"Downloading NLTK package: {package_name}...")
#             nltk.download(package_name)
#             print(f"Downloaded {package_name}.")
#     print("NLTK data check complete.")

# --- Data Loading Function ---
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

# --- Text Preprocessing Function ---
def preprocess_text(text, use_stemming=False, use_lemmatization=True):
    """
    Cleans and preprocesses a raw text string for natural language processing.

    Args:
        text (str): The raw input text string (e.g., an email message).
        use_stemming (bool): If True, applies Porter Stemming.
        use_lemmatization (bool): If True, applies WordNet Lemmatization.
                                  (Cannot be True if use_stemming is True).

    Returns:
        str: The cleaned and preprocessed text string.
    """
    if use_stemming and use_lemmatization:
        raise ValueError("Cannot use both stemming and lemmatization simultaneously. Choose one.")

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d+', '', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    processed_tokens = []
    if use_stemming:
        stemmer = PorterStemmer()
        for word in filtered_tokens:
            processed_tokens.append(stemmer.stem(word))
    elif use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        for word in filtered_tokens:
            processed_tokens.append(lemmatizer.lemmatize(word))
    else:
        processed_tokens = filtered_tokens

    cleaned_text = ' '.join(processed_tokens)
    return cleaned_text