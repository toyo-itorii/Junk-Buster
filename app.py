from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store loaded models and feature names
loaded_logreg_model = None
loaded_nb_model = None
feature_names = None

def load_models():
    """Load the trained models and feature names on startup"""
    global loaded_logreg_model, loaded_nb_model, feature_names
    
    models_dir = 'trained_models'
    logistic_regression_model_path = os.path.join(models_dir, 'logistic_regression_model.joblib')
    multinomial_nb_model_path = os.path.join(models_dir, 'multinomial_nb_model.joblib')
    
    try:
        loaded_logreg_model = joblib.load(logistic_regression_model_path)
        print(f"✓ Logistic Regression model loaded from {logistic_regression_model_path}")
    except FileNotFoundError:
        print(f"✗ Error: Logistic Regression model not found at {logistic_regression_model_path}")
    
    try:
        loaded_nb_model = joblib.load(multinomial_nb_model_path)
        print(f"✓ Multinomial Naive Bayes model loaded from {multinomial_nb_model_path}")
    except FileNotFoundError:
        print(f"✗ Error: Multinomial Naive Bayes model not found at {multinomial_nb_model_path}")
    
    # Load feature names from original dataset
    try:
        df = pd.read_csv('data/emails.csv', encoding='latin-1')
        feature_names = [col for col in df.columns if col not in ['Email No.', 'Prediction']]
        print(f"✓ Loaded {len(feature_names)} feature names from dataset")
    except Exception as e:
        print(f"✗ Error loading feature names: {e}")

def preprocess_email_text(text):
    """
    Convert raw email text to word counts that match the training data format
    """
    if not text:
        return {}
    
    # Basic text preprocessing
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    text = re.sub(r'\d+', '', text)  # Remove numbers
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    except:
        tokens = [word for word in tokens if len(word) > 2]
    
    # Count word frequencies
    word_counts = Counter(tokens)
    
    return dict(word_counts)

def predict_email(email_text):
    """
    Predict spam/not spam for email text using the loaded models
    """
    global loaded_logreg_model, loaded_nb_model, feature_names
    
    if not (loaded_logreg_model and loaded_nb_model and feature_names):
        return {"error": "Models not properly loaded"}
    
    # Convert email text to word counts
    word_counts = preprocess_email_text(email_text)
    
    # Create DataFrame with proper structure matching training data
    new_data_dict = {feature: [0] for feature in feature_names}
    new_email_df = pd.DataFrame(new_data_dict)
    
    # Fill in the word counts for words that exist in our feature set
    for word, count in word_counts.items():
        if word in feature_names:
            new_email_df.loc[0, word] = count
    
    try:
        # Make predictions
        logreg_pred = loaded_logreg_model.predict(new_email_df)[0]
        nb_pred = loaded_nb_model.predict(new_email_df)[0]
        logreg_proba = loaded_logreg_model.predict_proba(new_email_df)[0]
        nb_proba = loaded_nb_model.predict_proba(new_email_df)[0]
        
        # Get some insights about the features
        non_zero_features = new_email_df.loc[0, new_email_df.loc[0] > 0]
        detected_words = non_zero_features.to_dict() if len(non_zero_features) > 0 else {}
        
        return {
            'success': True,
            'logistic_regression': {
                'prediction': int(logreg_pred),
                'spam_probability': float(logreg_proba[1]),
                'not_spam_probability': float(logreg_proba[0])
            },
            'naive_bayes': {
                'prediction': int(nb_pred),
                'spam_probability': float(nb_proba[1]),
                'not_spam_probability': float(nb_proba[0])
            },
            'features': {
                'total_words': len(email_text.split()) if email_text else 0,
                'unique_words': len(set(email_text.lower().split())) if email_text else 0,
                'detected_features': len(detected_words),
                'feature_words': detected_words
            }
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "models_loaded": {
            "logistic_regression": loaded_logreg_model is not None,
            "naive_bayes": loaded_nb_model is not None,
            "features": feature_names is not None
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_email():
    """Main endpoint for email spam analysis"""
    try:
        data = request.get_json()
        
        if not data or 'email_text' not in data:
            return jsonify({"error": "Missing email_text in request"}), 400
        
        email_text = data['email_text']
        
        if not email_text.strip():
            return jsonify({"error": "Email text cannot be empty"}), 400
        
        # Predict using the trained models
        result = predict_email(email_text)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "models": {
            "logistic_regression_loaded": loaded_logreg_model is not None,
            "naive_bayes_loaded": loaded_nb_model is not None,
            "feature_count": len(feature_names) if feature_names else 0
        }
    })

if __name__ == '__main__':
    print("Loading models...")
    load_models()
    
    if loaded_logreg_model and loaded_nb_model and feature_names:
        print("✓ All models loaded successfully!")
        print(f"✓ Ready to analyze emails using {len(feature_names)} features")
        print("✓ Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("✗ Failed to load models. Please ensure:")
        print("  1. trained_models/ directory exists")
        print("  2. Model files are present: logistic_regression_model.joblib, multinomial_nb_model.joblib")
        print("  3. data/emails.csv file exists for feature names")
        print("  4. Run main.py first to train and save the models")