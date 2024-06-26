# main.py

import preprocess
from model import build_model, train_model, evaluate_model

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = preprocess.load_and_preprocess_data()

    # Build model
    model = build_model()

    # Train model
    model = train_model(model, X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
