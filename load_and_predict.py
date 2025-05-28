import joblib
import pandas as pd
import os

# --- Define the directory where your models are saved ---
models_dir = 'trained_models'

# --- Define the paths to your saved models ---
# (These should match the paths used during saving)
logistic_regression_model_path = os.path.join(models_dir, 'logistic_regression_model.joblib')
multinomial_nb_model_path = os.path.join(models_dir, 'multinomial_nb_model.joblib')

# --- Load the Logistic Regression model ---
try:
    loaded_logreg_model = joblib.load(logistic_regression_model_path)
    print(f"Logistic Regression model loaded successfully from {logistic_regression_model_path}")
except FileNotFoundError:
    print(f"Error: Logistic Regression model file not found at {logistic_regression_model_path}")
    loaded_logreg_model = None # Set to None if loading fails

# --- Load the Multinomial Naive Bayes model ---
try:
    loaded_nb_model = joblib.load(multinomial_nb_model_path)
    print(f"Multinomial Naive Bayes model loaded successfully from {multinomial_nb_model_path}")
except FileNotFoundError:
    print(f"Error: Multinomial Naive Bayes model file not found at {multinomial_nb_model_path}")
    loaded_nb_model = None # Set to None if loading fails

# --- Example of how to use the loaded models for prediction ---
if loaded_logreg_model and loaded_nb_model:
    print("\n--- Making a New Prediction (Example) ---")

    # For your 'emails.csv' dataset, your input features are numerical word counts.
    # To make a new prediction, you need new data in the *exact same numerical format*
    # as your training data (the 3000 columns of word counts).

    # ***IMPORTANT: THIS IS A DUMMY EXAMPLE!***
    # In a real scenario, you'd feed new, pre-processed and vectorized email data here.
    # Since your 'emails.csv' dataset didn't use TF-IDF,
    # you don't need a TF-IDF vectorizer to transform new input.
    # You'd need to convert a new email message into its 3000-word-count representation
    # using the same method that created your original dataset's features.

    # Let's create a dummy example of new input data (e.g., one row of 3000 features)
    # Each value would typically represent the count of a specific word.
    # For a realistic new prediction, you'd have to create a function
    # that takes a raw email, tokenizes it, counts words, and matches them
    # to the columns your models were trained on.

    # Example: A single "new email" (represented as a 1x3000 feature array/DataFrame row)
    # This example assumes a sparse matrix or DataFrame with 3000 columns of zeros.
    # In practice, you'd populate this with actual word counts from a new email.
    try:
        # Create a DataFrame with the correct number of columns (3000 from your original data)
        # and fill it with zeros or example values for demonstration.
        # Your original X had 3000 features (columns) excluding 'Email No.' and 'Prediction'
        dummy_new_data = pd.DataFrame([[0] * 3000]) # A single "new email" represented as 3000 zeros

        # Make predictions
        logreg_prediction = loaded_logreg_model.predict(dummy_new_data)
        nb_prediction = loaded_nb_model.predict(dummy_new_data)

        print(f"Dummy new data shape: {dummy_new_data.shape}")
        print(f"Logistic Regression prediction: {logreg_prediction[0]} (0: Not Spam, 1: Spam)")
        print(f"Multinomial Naive Bayes prediction: {nb_prediction[0]} (0: Not Spam, 1: Spam)")

    except Exception as e:
        print(f"Error during dummy prediction: {e}")
        print("Please ensure your new data has the same number of features (columns) as the training data.")