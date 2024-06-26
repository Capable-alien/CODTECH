# model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def build_model():
    model = RandomForestClassifier(random_state=42)
    return model

def train_model(model, X_train, y_train):
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [200],
        'max_depth': [20],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [True]
    }
    
    # Perform GridSearchCV to find best parameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    return best_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
