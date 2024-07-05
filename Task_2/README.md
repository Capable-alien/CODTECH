# Task 2: Credit Card Fraud Detection

Develop a fraud detection model to identify fraudulent credit card transactions. Use techniques like anomaly detection or supervised learning with imbalanced data.

## Credit Card Fraud Detection with Logistic Regression

This project implements a credit card fraud detection model using logistic regression, a supervised learning algorithm, on the credit card transaction dataset. The goal is to predict fraudulent transactions using features derived from transaction data. SMOTE (Synthetic Minority Over-sampling Technique) is used to handle class imbalance in the dataset.

## Project Structure

- `data/`: Directory containing the dataset used (`creditcard.csv`).
- `main.py`: Script to load data, preprocess, train a logistic regression model with SMOTE, and save the trained model.
- `app.py`: Streamlit application script to upload a CSV file, predict fraudulent transactions using the trained model, and display results interactively.
- `logistic_regression_model.joblib`: Saved logistic regression model file.

## Usage
1. Install the required dependencies listed in `requirements.txt` using pip:
    `pip install -r requirements.txt`
   
2. Run the code for training using the following command:
    `python main.py`

3. Launch the Streamlit application using the command:
    `streamlit run app.py`
