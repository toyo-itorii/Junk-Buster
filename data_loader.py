import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # Still imported, but won't be used for this specific dataset type
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- NLTK Data Downloads (run once to ensure all necessary data is available) ---
# It's good practice to keep these, even if preprocessing is skipped, in case
# you switch back to a raw text dataset or use other NLTK features.
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords')
# try:
#     nltk.data.find('corpora/omw-1.4') # Open Multilingual Wordnet, required by WordNetLemmatizer
# except nltk.downloader.DownloadError:
#     nltk.download('omw-1.4')


# --- Data Loading Function (no change needed here) ---
def load_dataset(file_path):
    """
    Loads a dataset into a Pandas DataFrame, handling CSV and TSV formats.
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

# --- Text Preprocessing Function (no change, but will be skipped for this dataset) ---
def preprocess_text(text, use_stemming=False, use_lemmatization=True):
    """
    Cleans and preprocesses a raw text string for natural language processing.
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


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the path to your dataset
    csv_file_path = 'data/emails.csv' # This is the dataset you are currently using

    try:
        # Phase 1: Load the Dataset
        print("="*50)
        print("Phase 1: Loading Dataset")
        print("="*50)
        df = load_dataset(csv_file_path)
        print("Dataset loaded successfully.\n")

        # Phase 2: Initial Data Exploration (EDA)
        print("="*50)
        print("Phase 2: Initial Data Exploration (EDA)")
        print("="*50)

        # In this dataset, the columns are already words, and the last column is 'Prediction'.
        # So, 'v1' and 'v2' renaming is not applicable.
        print("Dataset already contains pre-vectorized features (word counts) and a 'Prediction' column.")
        print("Skipping 'v1'/'v2' renaming and 'Unnamed' column dropping based on observed schema.")

        # Display basic info and head
        print("\nDataFrame Info:")
        df.info()
        print("\nDataFrame Head:")
        print(df.head())
        print(f"\nDataFrame Shape: {df.shape}")

        # Check for any remaining missing values (should be none based on previous output)
        print("\nMissing Values Check:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

        # Understand Target Distribution
        target_column_name = 'Prediction' # Correct target column for this dataset
        if target_column_name in df.columns:
            print(f"\nTarget ('{target_column_name}') Distribution:")
            target_distribution = df[target_column_name].value_counts()
            print(target_distribution)
            print("\nTarget Distribution (Percentage):")
            print(df[target_column_name].value_counts(normalize=True) * 100)

            if target_distribution.min() / target_distribution.max() < 0.2:
                print("\nWarning: Significant class imbalance detected in 'Prediction' column.")
                print("Consider strategies like oversampling/undersampling or specific evaluation metrics later.")
        else:
            raise ValueError(f"Target column '{target_column_name}' not found. Please check dataset schema.")
        print("\nInitial EDA complete.\n")


        # Phase 3: Text Preprocessing (SKIPPING for this dataset type)
        print("="*50)
        print("Phase 3: Text Preprocessing (Skipping - Data already vectorized)")
        print("="*50)
        # This dataset contains pre-vectorized word counts, not raw email strings.
        # So, the 'preprocess_text' function is not applicable here.
        # We will directly use the numerical word count columns as features.
        # No 'processed_message' column will be created.


        # Phase 4: Split Data
        print("="*50)
        print("Phase 4: Splitting Data into Training and Testing Sets")
        print("="*50)

        # Features (X) are all columns except 'Email No.' and 'Prediction'
        # Target (y) is the 'Prediction' column
        X = df.drop(columns=['Email No.', 'Prediction'])
        y = df['Prediction']

        # Convert target labels to 0/1 if they are not already (e.g., if 'spam'/'ham')
        # Based on your output, 'Prediction' seems to be 0 or 1, which is good.
        # If it were 'spam'/'ham', you would map them: y = y.map({'ham': 0, 'spam': 1})

        # Split data (80% train, 20% test) with stratification for imbalance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"Shape of X_train (training features): {X_train.shape}")
        print(f"Shape of X_test (testing features):  {X_test.shape}")
        print(f"Shape of y_train (training labels):  {y_train.shape}")
        print(f"Shape of y_test (testing labels):    {y_test.shape}")

        print("\nDistribution of labels in training set (y_train):")
        print(y_train.value_counts(normalize=True) * 100)
        print("\nDistribution of labels in testing set (y_test):")
        print(y_test.value_counts(normalize=True) * 100)
        print("\nData splitting complete.\n")


        # Phase 5: Feature Engineering (Vectorization) - SKIPPING for this dataset type
        print("="*50)
        print("Phase 5: Feature Engineering (Skipping - Data already numerical)")
        print("="*50)
        # Since your input data columns are already numerical word counts,
        # TF-IDF vectorization is not needed here. X_train and X_test are already
        # in a suitable numerical format for the models.
        X_train_vectorized = X_train # Assign directly as no further vectorization is needed
        X_test_vectorized = X_test   # Assign directly

        print("Data is already in numerical feature format. Skipping TF-IDF vectorization step.")
        print(f"Shape of X_train_vectorized (same as X_train): {X_train_vectorized.shape}")
        print(f"Shape of X_test_vectorized (same as X_test):  {X_test_vectorized.shape}")
        print("\nFeature Engineering phase complete.\n")


        # Phase 6: Model Training & Evaluation
        print("="*50)
        print("Phase 6: Model Training & Evaluation")
        print("="*50)

        # --- Model 1: Multinomial Naive Bayes ---
        print("\n--- Training Multinomial Naive Bayes Model ---")
        nb_model = MultinomialNB()
        nb_model.fit(X_train_vectorized, y_train)
        y_pred_nb = nb_model.predict(X_test_vectorized)

        print("\nEvaluation for Multinomial Naive Bayes:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
        print("-" * 30)

        # --- Model 2: Logistic Regression ---
        print("\n--- Training Logistic Regression Model ---")
        # solver='liblinear' is good for sparse data and L1/L2 regularization
        # max_iter is increased to ensure convergence
        logreg_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        logreg_model.fit(X_train_vectorized, y_train)
        y_pred_logreg = logreg_model.predict(X_test_vectorized)

        print("\nEvaluation for Logistic Regression:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred_logreg))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
        print("-" * 30)

        print("\nModel training and evaluation complete. You can now compare the performance of both models.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the script execution: {e}")