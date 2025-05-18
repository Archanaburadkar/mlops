import numpy as np
import pandas as pd
import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score
import mlflow.sklearn
# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
# Initialize classifier
rf = RandomForestClassifier(random_state=42)
# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
# Run grid search
grid_search.fit(X_train, y_train)
# Results Reporting
print("Best Hyperparameter:", grid_search.best_params_)
print("Best Cross-validated Accuracy:",grid_search.best_score_)
# The best model found by GridSearchCV 
best_rf_model = grid_search.best_estimator_
# Evaluate the best model on the test set
y_pred = best_rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy of the Best Model:", test_accuracy)
# Log best parameters
mlflow.set_experiment("RandomForest_Hyperparameter_Tuning")
with mlflow.start_run():
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
# Initialize and run GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
#Log individual cross-validation scores for each parameter combination
cv_results = grid_search.cv_results_ 
for i in range(len(cv_results['params'])):
        params = cv_results['params'][i]
        mean_score = cv_results['mean_test_score'][i]
# Evaluate the best model on the test set and log the metric
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
# Log metrics
mlflow.log_metric("test_accuracy", test_accuracy)
# Log model
mlflow.sklearn.log_model(best_rf_model, "best_random_forest_model")

print("MLflow Run ID:", mlflow.active_run().info.run_id)
print("Test Accuracy:", test_accuracy)