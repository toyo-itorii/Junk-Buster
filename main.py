# main.py
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # Kept for potential future use with raw text
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Import reusable functions from utils.py
from utils import load_dataset, preprocess_text # check_and_download_nltk_data Assuming you add check_and_download_nltk_data to utils.py


def perform_eda(df, target_column_name='Prediction'):
    """Performs initial Exploratory Data Analysis (EDA) on the DataFrame."""
    print("="*50)
    print("Phase 2: Initial Data Exploration (EDA)")
    print("="*50)

    print("Dataset already contains pre-vectorized features (word counts) and a 'Prediction' column.")
    print("Skipping 'v1'/'v2' renaming and 'Unnamed' column dropping based on observed schema.")

    print("\nDataFrame Info:")
    df.info()
    print("\nDataFrame Head:")
    print(df.head())
    print(f"\nDataFrame Shape: {df.shape}")

    print("\nMissing Values Check:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

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
    return df # Return df, though it's modified in place, for consistency


def split_data(df, feature_cols, target_col):
    """Splits the data into training and testing sets."""
    print("="*50)
    print("Phase 4: Splitting Data into Training and Testing Sets")
    print("="*50)

    X = df[feature_cols]
    y = df[target_col]

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
    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train_vectorized, X_test_vectorized, y_train, y_test):
    """Trains and evaluates Multinomial Naive Bayes and Logistic Regression models."""
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
    logreg_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    logreg_model.fit(X_train_vectorized, y_train)
    y_pred_logreg = logreg_model.predict(X_test_vectorized)

    print("\nEvaluation for Logistic Regression:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_logreg))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
    print("-" * 30)

    print("\nModel training and evaluation complete. You can now compare the performance of both models.")
    return nb_model, logreg_model


def save_models(nb_model, logreg_model, models_dir):
    """Saves the trained models to disk."""
    print("="*50)
    print("Phase 7: Model Persistence")
    print("="*50)

    nb_model_path = os.path.join(models_dir, 'multinomial_nb_model.joblib')
    logreg_model_path = os.path.join(models_dir, 'logistic_regression_model.joblib')

    # Note: tfidf_vectorizer is NOT saved here because the input data (emails.csv)
    # is already vectorized (numerical word counts), so no TfidfVectorizer was fitted.
    # If you were working with raw text, you would save the vectorizer here too.

    joblib.dump(nb_model, nb_model_path)
    print(f"Multinomial Naive Bayes model saved to {nb_model_path}")

    joblib.dump(logreg_model, logreg_model_path)
    print(f"Logistic Regression model saved to {logreg_model_path}")

    print("\nModel persistence complete. Models are saved for future use.")


def load_and_predict_example(models_dir):
    """Demonstrates loading models and making a dummy prediction."""
    print("="*50)
    print("Loading Models and Making Dummy Prediction")
    print("="*50)

    logistic_regression_model_path = os.path.join(models_dir, 'logistic_regression_model.joblib')
    multinomial_nb_model_path = os.path.join(models_dir, 'multinomial_nb_model.joblib')

    loaded_logreg_model = None
    loaded_nb_model = None

    try:
        loaded_logreg_model = joblib.load(logistic_regression_model_path)
        print(f"Logistic Regression model loaded successfully from {logistic_regression_model_path}")
    except FileNotFoundError:
        print(f"Error: Logistic Regression model file not found at {logistic_regression_model_path}")

    try:
        loaded_nb_model = joblib.load(multinomial_nb_model_path)
        print(f"Multinomial Naive Bayes model loaded successfully from {multinomial_nb_model_path}")
    except FileNotFoundError:
        print(f"Error: Multinomial Naive Bayes model file not found at {multinomial_nb_model_path}")

    if loaded_logreg_model and loaded_nb_model:
        # Create a dummy example of new input data (e.g., one row of 3000 features)
        # This assumes your 'emails.csv' has 3000 feature columns.
        dummy_new_data = pd.DataFrame([[0] * 3000]) # Example: a single "new email" with 3000 zero counts

        try:
            logreg_prediction = loaded_logreg_model.predict(dummy_new_data)
            nb_prediction = loaded_nb_model.predict(dummy_new_data)

            print(f"\nDummy new data shape: {dummy_new_data.shape}")
            print(f"Logistic Regression prediction: {logreg_prediction[0]} (0: Not Spam, 1: Spam)")
            print(f"Multinomial Naive Bayes prediction: {nb_prediction[0]} (0: Not Spam, 1: Spam)")

        except Exception as e:
            print(f"Error during dummy prediction: {e}")
            print("Please ensure your new data has the same number of features (columns) as the training data (3000 for 'emails.csv').")
    else:
        print("\nCould not load one or both models. Skipping prediction example.")


if __name__ == "__main__":
    # Ensure NLTK data is available
    # check_and_download_nltk_data()

    # Define paths
    csv_file_path = 'data/emails.csv'
    models_dir = 'trained_models'
    os.makedirs(models_dir, exist_ok=True) # Create models directory if it doesn't exist

    try:
        # Phase 1: Load the Dataset
        print("="*50)
        print("Phase 1: Loading Dataset")
        print("="*50)
        df = load_dataset(csv_file_path)
        print("Dataset loaded successfully.\n")

        # Phase 2: Initial Data Exploration (EDA)
        # (This function modifies df in place and prints info)
        df = perform_eda(df) # Pass df and get potential modifications back

        # Phase 3: Text Preprocessing (Skipped for this dataset type)
        print("="*50)
        print("Phase 3: Text Preprocessing (Skipping - Data already vectorized)")
        print("="*50)
        print("This dataset contains pre-vectorized word counts, so no raw text preprocessing is needed.\n")

        # Determine feature columns based on your specific dataset (emails.csv)
        # Assuming 'Email No.' and 'Prediction' are NOT features.
        feature_columns = [col for col in df.columns if col not in ['Email No.', 'Prediction']]
        target_column = 'Prediction'

        # Phase 4: Split Data
        X_train, X_test, y_train, y_test = split_data(df, feature_columns, target_column)

        # Phase 5: Feature Engineering (Skipped for this dataset type)
        print("="*50)
        print("Phase 5: Feature Engineering (Skipping - Data already numerical)")
        print("="*50)
        print("Data is already in numerical feature format. Skipping TF-IDF vectorization step.\n")
        # Assign directly as no further vectorization is needed
        X_train_vectorized = X_train
        X_test_vectorized = X_test

        # Phase 6: Model Training & Evaluation
        nb_model, logreg_model = train_and_evaluate_models(X_train_vectorized, X_test_vectorized, y_train, y_test)

        # Phase 7: Model Persistence
        save_models(nb_model, logreg_model, models_dir)

        # Demonstration of loading models and making new predictions
        load_and_predict_example(models_dir)


    except Exception as e:
        print(f"\nAn unexpected error occurred during the script execution: {e}")