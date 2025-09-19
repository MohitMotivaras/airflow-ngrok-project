import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class RandomForestModel:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def evaluate_metrics(self, actual, predicted):
        MSE = mean_squared_error(actual, predicted)
        R2 = r2_score(actual, predicted)

        print('MSE is {}'.format(MSE))
        print('R2 score is {}'.format(R2))

        return MSE, R2

    def random_forest(self):
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=42)

        param_dict = {
            'n_estimators': [int(x) for x in np.linspace(10, 300, 100)],
            'max_depth': [int(x) for x in np.linspace(2, 10, 5)],
            'min_samples_split': [int(x) for x in np.linspace(5, 25, 10)],
            'min_samples_leaf': [int(x) for x in np.linspace(1, 10, 5)],
            'max_features': [int(x) for x in np.linspace(3, 9, 6)]
        }

        rf_model = RandomForestRegressor()
        rf_grid = RandomizedSearchCV(estimator=rf_model,
                                     param_distributions=param_dict,
                                     cv=5, verbose=2, scoring='r2')

        # MLflow tracking
        with mlflow.start_run():
            rf_grid.fit(X_train, Y_train)

            # Best model
            rf_optimal_model = rf_grid.best_estimator_
            print("Best Model:", rf_optimal_model)
            print("Best Params:", rf_grid.best_params_)

            # Predictions
            Y_train_pred = rf_optimal_model.predict(X_train)
            Y_test_pred = rf_optimal_model.predict(X_test)

            # Metrics
            print("Train Set Metrics:")
            self.evaluate_metrics(Y_train, Y_train_pred)

            print("\nTest Set Metrics:")
            MSE, R2 = self.evaluate_metrics(Y_test, Y_test_pred)

            # Save predictions
            pred = pd.DataFrame({'Actual Value': Y_test,
                                  'Predicted Value': Y_test_pred})
            pred.to_csv("predictions.csv")
            print("Predictions saved in a csv file!")

            # 🔹 Log to MLflow
            mlflow.log_params(rf_grid.best_params_)
            mlflow.log_metric("MSE", MSE)
            mlflow.log_metric("R2", R2)

            # Save the model in MLflow
            mlflow.sklearn.log_model(rf_optimal_model, "random_forest_model")

        return Y_test, Y_test_pred, R2
