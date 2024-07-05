# app.py
import streamlit as st
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import preprocess
import numpy as np

# Load the model
model = load_model('Task_1\sentiment_model.h5')

# Load the tokenizer
_, _, _, _, word_index, max_words, max_length = preprocess.load_and_preprocess_data(data_path='Task_1\IMDB Dataset.csv', max_words=10000, max_length=100)
tokenizer = Tokenizer(num_words=max_words)
tokenizer.word_index = word_index

# Streamlit app
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review and get the sentiment prediction:")

user_input = st.text_area("Movie Review")

if st.button("Predict"):
    if user_input:
        # Preprocess the input
        sequences = tokenizer.texts_to_sequences([user_input])
        padded_sequences = pad_sequences(sequences, maxlen=max_length)
        
        # Predict sentiment
        prediction = model.predict(padded_sequences)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a movie review.")
