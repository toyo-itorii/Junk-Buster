# app.py
from flask import Flask, request, jsonify
from spam_classifier import analyze_email  # your function

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get email text from the frontend
    data = request.get_json()
    text = data.get('text', '')

    # Run the AI classifier
    result = analyze_email(text)

    # Return the results as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
