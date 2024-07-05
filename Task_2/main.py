# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump

# Load your dataset
df = pd.read_csv('Task_2\creditcard.csv')

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_resampled, y_resampled)

# Save the model
dump(model, 'logistic_regression_model.joblib')

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
