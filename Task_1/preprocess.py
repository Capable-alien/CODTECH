# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_data(data_path, max_words=10000, max_length=100):
    # Load dataset
    df = pd.read_csv(data_path)

    # Preprocess data
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

    # Tokenize text
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df['review'])
    X = tokenizer.texts_to_sequences(df['review'])
    X = pad_sequences(X, maxlen=max_length)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, df['sentiment'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, tokenizer.word_index, max_words, max_length
