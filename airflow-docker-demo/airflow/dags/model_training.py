import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import logging, json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO)

class RandomForestModel:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def evaluate_metrics(self, actual, predicted):
        """Evaluate regression metrics"""
        MSE = mean_squared_error(actual, predicted)
        R2 = r2_score(actual, predicted)

        logging.info(f"MSE: {MSE}")
        logging.info(f"R2 Score: {R2}")

        return MSE, R2

    def random_forest(self):
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=0.25, random_state=42
        )

        # Hyperparameter search space
        param_dict = {
            'n_estimators': [int(x) for x in np.linspace(10, 300, 100)],
            'max_depth': [int(x) for x in np.linspace(2, 10, 5)],
            'min_samples_split': [int(x) for x in np.linspace(5, 25, 10)],
            'min_samples_leaf': [int(x) for x in np.linspace(1, 10, 5)],
            'max_features': [int(x) for x in np.linspace(3, 9, 6)]
        }

        rf_model = RandomForestRegressor()
        rf_grid = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_dict,
            cv=5, verbose=2, scoring='r2'
        )

        # ✅ Step 7: Connect to MLflow Tracking Server + Set Experiment
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("RandomForest_Experiments")

        # MLflow tracking
        with mlflow.start_run():
            # ✅ Step 1: Autolog everything
            mlflow.sklearn.autolog()

            # Train
            rf_grid.fit(X_train, Y_train)

            # Best model
            rf_optimal_model = rf_grid.best_estimator_
            logging.info(f"Best Model: {rf_optimal_model}")
            logging.info(f"Best Params: {rf_grid.best_params_}")

            # Predictions
            Y_train_pred = rf_optimal_model.predict(X_train)
            Y_test_pred = rf_optimal_model.predict(X_test)

            # Metrics
            logging.info("Train Set Metrics:")
            self.evaluate_metrics(Y_train, Y_train_pred)

            logging.info("Test Set Metrics:")
            MSE, R2 = self.evaluate_metrics(Y_test, Y_test_pred)

            # Save predictions locally
            pred = pd.DataFrame({'Actual Value': Y_test, 'Predicted Value': Y_test_pred})
            pred.to_csv("predictions.csv", index=False)
            logging.info("Predictions saved in predictions.csv")

            # ✅ Step 2: Log predictions as MLflow artifact
            mlflow.log_artifact("predictions.csv")

            # ✅ Step 3: Save metrics in JSON (useful for Airflow/Jenkins)
            results = {"MSE": MSE, "R2": R2}
            with open("metrics.json", "w") as f:
                json.dump(results, f)
            mlflow.log_artifact("metrics.json")

            # ✅ Step 4: Register model in MLflow Model Registry
            mlflow.sklearn.log_model(
                sk_model=rf_optimal_model,
                artifact_path="random_forest_model",
                registered_model_name="RandomForestRegressor"
            )

        # ✅ Step 5: Return clean results
        return rf_optimal_model, Y_test, Y_test_pred, R2
