# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained logistic regression model
model = joblib.load('Task_2/logistic_regression_model.joblib')

def predict_fraud(csv_file):
    df = pd.read_csv(csv_file)
    
    # Assuming the columns are in the order: Time, V1-V28, Amount
    X = df.drop(columns=['Class'])  # Exclude 'Class' column from features
    predictions = model.predict(X)
    
    df['Predicted_Class'] = np.where(predictions == 1, 'fraudulent', 'non-fraudulent')
    
    return df

def main():
    st.title('Upload a CSV file for fraud detection:')
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        
        # Perform prediction using the uploaded file
        predictions_df = predict_fraud(uploaded_file)
        
        # Display results without 'Class' column
        st.dataframe(predictions_df.drop(columns=['Class']))

if __name__ == '__main__':
    main()
