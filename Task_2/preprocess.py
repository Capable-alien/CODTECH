# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(data_path='Task_2\creditcard.csv'):
    # Load dataset
    df = pd.read_csv(data_path)

    # Preprocess data
    df.dropna(inplace=True)

    # Scale numerical features
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

    # Split into training and testing sets
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Handling class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
