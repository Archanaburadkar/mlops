from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
# Load data
iris = load_iris()
X, y = iris.data, iris.target
# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
# Model
clf = RandomForestClassifier(random_state=42)
# GridSearch
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)
# Print results
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validated Accuracy:", grid_search.best_score_)
# MLflow Logging
mlflow.set_experiment("RandomForest_Hyperparameter_Tuning")
with mlflow.start_run():
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
print("Logged to MLflow successfully.")
