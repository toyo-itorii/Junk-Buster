---
# Spam Email Classifier

## Project Overview

This project implements a machine learning-based spam email classifier. It demonstrates a typical end-to-end machine learning workflow, including **data loading**, **exploratory data analysis (EDA)**, text preprocessing (though skipped for the provided dataset which is pre-vectorized), **data splitting**, **model training**, **evaluation**, and **persistence**.

The classifier is built using Python and popular machine learning libraries like `pandas`, `scikit-learn`, and `nltk`.

---
## Dataset

The project currently uses the `emails.csv` dataset, located in the `data/` directory. This dataset is unique because it's already pre-processed and vectorized. Each column represents a word from the email vocabulary, and the values are their respective counts within an email. The final column, `Prediction`, serves as the target variable (0 for not spam, 1 for spam).
[**Kaggle**](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

---
## Project Structure

```
spam_classifier_project/
├── data/
│   └── emails.csv          # The email dataset
├── trained_models/
│   ├── logistic_regression_model.joblib  # Saved Logistic Regression model
│   └── multinomial_nb_model.joblib     # Saved Multinomial Naive Bayes model
├── utils.py                # Reusable functions for data loading and preprocessing
└── main.py                 # Main script orchestrating the ML pipeline
└── README.md               # This README file
└── requirements.txt        # List of project dependencies
```

---
## Setup and Installation

To get this project up and running on your local machine, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/toyo-itorii/Junk-Buster
    cd spam_classifier_project
    ```
    (If you don't have a repository, simply navigate to your project directory.)

2.  **Create a Virtual Environment (Recommended):**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    Install all the required Python libraries using `pip` and the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download NLTK Data:**
    The `nltk` library requires specific data packages. The `main.py` script will automatically check for and download these the first time it runs.

---
## How to Run

After setting up the environment and installing dependencies, you can run the main script to perform the entire ML pipeline:

```bash
python main.py
```

This script will:
1.  **Load** the `emails.csv` dataset.
2.  Perform **Exploratory Data Analysis (EDA)**.
3.  *Skip* text preprocessing (as the data is already vectorized).
4.  **Split** the data into training and testing sets.
5.  *Skip* TF-IDF vectorization (as the data is already numerical).
6.  **Train and evaluate** two models: **Multinomial Naive Bayes** and **Logistic Regression**.
7.  **Save** the trained models (`.joblib` files) into the `trained_models/` directory.
8.  Demonstrate **loading** the saved models and making a dummy prediction.

---
## Modules

* **`main.py`**: The entry point of the application. It orchestrates the entire machine learning pipeline from data loading to model persistence.
* **`utils.py`**: Contains utility functions such as `load_dataset` for handling various file formats and `preprocess_text` for cleaning raw text (though not directly used on this dataset due to its pre-vectorized nature).
* **`requirements.txt`**: Lists all the necessary Python packages and their versions to run this project.

---
## Models Trained

This project trains and evaluates two common classification algorithms:

* **Multinomial Naive Bayes (MNB)**: A probabilistic classifier often used for text classification.
* **Logistic Regression**: A linear model used for binary classification, providing insights into feature importance.

Both models are evaluated using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix** to provide a comprehensive understanding of their performance.

---
## Model Persistence

The trained `Multinomial Naive Bayes` and `Logistic Regression` models are saved using `joblib` in the `trained_models/` directory. This allows for quick loading and reuse of the models for making predictions without needing to retrain them every time the application runs.

---
## Future Enhancements

* **Hyperparameter Tuning:** Implement techniques like GridSearchCV or RandomizedSearchCV to find optimal hyperparameters for the models.
* **More Advanced Models:** Experiment with other classification algorithms (e.g., SVM, Gradient Boosting).
* **Raw Text Support:** Modify the pipeline to handle raw email text directly, incorporating the `preprocess_text` function and `TfidfVectorizer` for datasets that are not pre-vectorized.
* **Deployment:** Create a simple API (e.g., with Flask or FastAPI) to expose the trained model for real-time spam prediction.
* **Logging:** Add comprehensive logging for better tracking of the pipeline's execution.
* **Data Versioning:** Use tools like DVC for managing dataset versions.
* **Web UI:** Create a responsive web interface, allowing user to input email text and receive instant predictions with confidence scores.
---