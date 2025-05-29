import joblib
import pandas as pd
import os

# --- Define the directory where your models are saved ---
models_dir = 'trained_models'

# --- Define the paths to your saved models ---
logistic_regression_model_path = os.path.join(models_dir, 'logistic_regression_model.joblib')
multinomial_nb_model_path = os.path.join(models_dir, 'multinomial_nb_model.joblib')

# --- Load the original dataset to get feature names ---
def get_feature_names(csv_path='data/emails.csv'):
    """Load the original dataset to extract feature column names."""
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
        feature_columns = [col for col in df.columns if col not in ['Email No.', 'Prediction']]
        print(f"Extracted {len(feature_columns)} feature names from original dataset")
        return feature_columns
    except Exception as e:
        print(f"Error loading dataset to get feature names: {e}")
        return None

# --- Load the Logistic Regression model ---
try:
    loaded_logreg_model = joblib.load(logistic_regression_model_path)
    print(f"Logistic Regression model loaded successfully from {logistic_regression_model_path}")
except FileNotFoundError:
    print(f"Error: Logistic Regression model file not found at {logistic_regression_model_path}")
    loaded_logreg_model = None

# --- Load the Multinomial Naive Bayes model ---
try:
    loaded_nb_model = joblib.load(multinomial_nb_model_path)
    print(f"Multinomial Naive Bayes model loaded successfully from {multinomial_nb_model_path}")
except FileNotFoundError:
    print(f"Error: Multinomial Naive Bayes model file not found at {multinomial_nb_model_path}")
    loaded_nb_model = None

# --- Example of how to use the loaded models for prediction ---
if loaded_logreg_model and loaded_nb_model:
    print("\n--- Making a New Prediction (Example) ---")
    
    # Get the correct feature names from the original dataset
    feature_names = get_feature_names()
    
    if feature_names:
        # Create dummy data with proper feature names
        # This creates a DataFrame with the same column structure as training data
        dummy_data_dict = {feature: [0] for feature in feature_names}
        dummy_new_data = pd.DataFrame(dummy_data_dict)
        
        # Optional: Add some realistic word counts for demonstration
        # You can modify these to test different scenarios
        if 'the' in dummy_new_data.columns:
            dummy_new_data.loc[0, 'the'] = 5  # Common word
        if 'free' in dummy_new_data.columns:
            dummy_new_data.loc[0, 'free'] = 3  # Spam indicator
        if 'money' in dummy_new_data.columns:
            dummy_new_data.loc[0, 'money'] = 2  # Spam indicator
        if 'click' in dummy_new_data.columns:
            dummy_new_data.loc[0, 'click'] = 1  # Spam indicator
            
        print(f"Dummy new data shape: {dummy_new_data.shape}")
        print("Sample of dummy data (non-zero values):")
        non_zero_features = dummy_new_data.loc[0, dummy_new_data.loc[0] > 0]
        if len(non_zero_features) > 0:
            print(non_zero_features.to_dict())
        else:
            print("All features are zero")

        try:
            # Make predictions
            logreg_prediction = loaded_logreg_model.predict(dummy_new_data)
            nb_prediction = loaded_nb_model.predict(dummy_new_data)
            
            # Get prediction probabilities for more insight
            logreg_proba = loaded_logreg_model.predict_proba(dummy_new_data)
            nb_proba = loaded_nb_model.predict_proba(dummy_new_data)

            print(f"\n--- Prediction Results ---")
            print(f"Logistic Regression prediction: {logreg_prediction[0]} (0: Not Spam, 1: Spam)")
            print(f"Logistic Regression probabilities: Not Spam={logreg_proba[0][0]:.3f}, Spam={logreg_proba[0][1]:.3f}")
            
            print(f"Multinomial Naive Bayes prediction: {nb_prediction[0]} (0: Not Spam, 1: Spam)")
            print(f"Multinomial Naive Bayes probabilities: Not Spam={nb_proba[0][0]:.3f}, Spam={nb_proba[0][1]:.3f}")
            
            # Interpretation
            print(f"\n--- Interpretation ---")
            if logreg_prediction[0] == nb_prediction[0]:
                print(f"✓ Both models agree: This email is {'SPAM' if logreg_prediction[0] == 1 else 'NOT SPAM'}")
            else:
                print("⚠ Models disagree:")
                print(f"  - Logistic Regression says: {'SPAM' if logreg_prediction[0] == 1 else 'NOT SPAM'}")
                print(f"  - Naive Bayes says: {'SPAM' if nb_prediction[0] == 1 else 'NOT SPAM'}")

        except Exception as e:
            print(f"Error during prediction: {e}")
            print("Please ensure your new data has the same number of features (columns) as the training data.")
    else:
        print("Could not load feature names. Using basic approach with potential warnings...")
        # Fallback to original approach
        dummy_new_data = pd.DataFrame([[0] * 3000])
        try:
            logreg_prediction = loaded_logreg_model.predict(dummy_new_data)
            nb_prediction = loaded_nb_model.predict(dummy_new_data)
            print(f"Dummy new data shape: {dummy_new_data.shape}")
            print(f"Logistic Regression prediction: {logreg_prediction[0]} (0: Not Spam, 1: Spam)")
            print(f"Multinomial Naive Bayes prediction: {nb_prediction[0]} (0: Not Spam, 1: Spam)")
        except Exception as e:
            print(f"Error during dummy prediction: {e}")

else:
    print("\nCould not load one or both models. Please ensure models are saved correctly.")

# --- Function to predict on new email with proper feature names ---
def predict_email(email_word_counts, feature_names):
    """
    Predict spam/not spam for a new email.
    
    Args:
        email_word_counts (dict): Dictionary with word as key, count as value
                                 e.g., {'free': 3, 'money': 2, 'click': 1}
        feature_names (list): List of feature names from training data
    
    Returns:
        dict: Predictions from both models
    """
    if not (loaded_logreg_model and loaded_nb_model and feature_names):
        return {"error": "Models or feature names not loaded"}
    
    # Create DataFrame with proper structure
    new_data_dict = {feature: [0] for feature in feature_names}
    new_email_df = pd.DataFrame(new_data_dict)
    
    # Fill in the word counts
    for word, count in email_word_counts.items():
        if word in feature_names:
            new_email_df.loc[0, word] = count
    
    # Make predictions
    logreg_pred = loaded_logreg_model.predict(new_email_df)[0]
    nb_pred = loaded_nb_model.predict(new_email_df)[0]
    logreg_proba = loaded_logreg_model.predict_proba(new_email_df)[0]
    nb_proba = loaded_nb_model.predict_proba(new_email_df)[0]
    
    return {
        'logistic_regression': {
            'prediction': logreg_pred,
            'spam_probability': logreg_proba[1],
            'not_spam_probability': logreg_proba[0]
        },
        'naive_bayes': {
            'prediction': nb_pred,
            'spam_probability': nb_proba[1],
            'not_spam_probability': nb_proba[0]
        }
    }

# Example usage of the prediction function
print("\n" + "="*50)
print("Example: Predicting a new email")
print("="*50)

# Example 1: Email with spam-like words
spam_like_email = {
    'free': 5,
    'money': 3,
    'click': 2,
    'now': 2,
    'offer': 1,
    'limited': 1
}

# Example 2: Normal email words
normal_email = {
    'meeting': 2,
    'schedule': 1,
    'tomorrow': 1,
    'office': 1,
    'please': 1
}

feature_names = get_feature_names()
if feature_names:
    print("\nExample 1 - Spam-like email:")
    spam_result = predict_email(spam_like_email, feature_names)
    if 'error' not in spam_result:
        print(f"Input words: {spam_like_email}")
        print(f"Logistic Regression: {'SPAM' if spam_result['logistic_regression']['prediction'] == 1 else 'NOT SPAM'} "
              f"(confidence: {spam_result['logistic_regression']['spam_probability']:.3f})")
        print(f"Naive Bayes: {'SPAM' if spam_result['naive_bayes']['prediction'] == 1 else 'NOT SPAM'} "
              f"(confidence: {spam_result['naive_bayes']['spam_probability']:.3f})")
    
    print("\nExample 2 - Normal email:")
    normal_result = predict_email(normal_email, feature_names)
    if 'error' not in normal_result:
        print(f"Input words: {normal_email}")
        print(f"Logistic Regression: {'SPAM' if normal_result['logistic_regression']['prediction'] == 1 else 'NOT SPAM'} "
              f"(confidence: {normal_result['logistic_regression']['spam_probability']:.3f})")
        print(f"Naive Bayes: {'SPAM' if normal_result['naive_bayes']['prediction'] == 1 else 'NOT SPAM'} "
              f"(confidence: {normal_result['naive_bayes']['spam_probability']:.3f})")
        
    input("\nPress Enter to exit the script.")
