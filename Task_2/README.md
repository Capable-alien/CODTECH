# Task 2: Credit Card Fraud Detection

## Develop a fraud detection model to identify fraudulent credit card transactions. Use techniques like anomaly detection or supervised learning with imbalanced data.

## Credit Card Fraud Detection with Random Forest

This project implements a credit card fraud detection model using Random Forest classifiers, which is a supervised learning algorithm, on the credit card transaction dataset. The goal is to identify fraudulent transactions. SMOTE (Synthetic Minority Over-sampling Technique) is used to handle class imbalance.

## Project Structure

- `data/`: Directory containing the dataset used (`creditcard.csv`).
- `model.py`: Script defining the Random Forest model architecture and training functions.
- `preprocess.py`: Script for data preprocessing, including scaling and handling class imbalance.
- `main.py`: Main script to load data, preprocess, build, train, and evaluate the Random Forest model.

## Usage
1. Install the required dependencies listed in `requirements.txt` using pip:
    `pip install -r requirements.txt`
   
2. Run the code for training using the following command:
    `python main.py`
