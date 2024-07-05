# main.py
from keras.models import load_model
import preprocess
from model import build_model, train_model, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Define constants
    DATA_PATH = 'Task_1\IMDB Dataset.csv'
    MAX_WORDS = 10000
    MAX_LENGTH = 100
    EMBEDDING_DIM = 128
    LSTM_UNITS = 128
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 32
    EPOCHS = 10

    # Load and preprocess data
    X_train, X_test, y_train, y_test, word_index, num_words, max_length = preprocess.load_and_preprocess_data(
        data_path=DATA_PATH, max_words=MAX_WORDS, max_length=MAX_LENGTH)

    # Split validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Build model
    model = build_model(num_words=num_words, embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE, max_length=max_length)

    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save the model
    model.save('Task_1\sentiment_model.h5')
    print("Model saved as sentiment_model.h5")

    # Print classification report
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
