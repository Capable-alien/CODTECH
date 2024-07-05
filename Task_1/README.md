# CODTECH Internship

**Name:** Sukash L

**Company:** CODTECH IT SOLUTIONS

**ID:** CT4ML2809

**Domain:** Machine Learning

**Duration:** June to July 2024 


# Task 1: Analysis on Movie Reviews

Develop a sentiment analysis model to classify movie reviews as positive and negative. Use a dataset like the IMDB Movie Reviews dataset for training and testing.

## Sentiment Analysis with LSTM

This project implements a sentiment analysis model using LSTM neural networks on the IMDB movie reviews dataset. The goal is to classify movie reviews as positive or negative based on their sentiment.

## Project Structure

- `data/`: Directory containing the dataset used (`IMDB Dataset.csv`).
- `model.py`: Script defining the LSTM model architecture and training functions.
- `preprocess.py`: Script for data preprocessing, including text tokenization and padding.
- `main.py`: Main script to load data, preprocess, build, train, and evaluate the LSTM model.
- `app.py`: Streamlit application for real-time sentiment prediction of user-provided movie reviews.
- `sentiment_model.h5`: Saved sentiment model file.
  
## Usage
1. Install the required dependencies listed in `requirements.txt` using pip:
    `pip install -r requirements.txt`
   
2. Run the code for training using the following command:
    `python main.py`

3. Launch the Streamlit application using the command:
    `streamlit run app.py`
