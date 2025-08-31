import numpy as np
import pandas as pd
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
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=42)
        print("The shape of training set is", X_train.shape, Y_train.shape)
        print("The shape of testing set is", X_test.shape, Y_test.shape)
        print("\n")

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

        rf_grid.fit(X_train, Y_train)

        print(rf_grid.best_estimator_)
        rf_optimal_model = rf_grid.best_estimator_
        print(rf_grid.best_params_)

        Y_train_pred = rf_optimal_model.predict(X_train)
        Y_test_pred = rf_optimal_model.predict(X_test)

        print("Train Set Metrics:")
        print("----------------------------------------------")
        self.evaluate_metrics(Y_train, Y_train_pred)
        print("\n")

        print("Test Set Metrics")
        print("----------------------------------------------")
        metrics = self.evaluate_metrics(Y_test, Y_test_pred)
        print("\n")
        r2_rf = metrics[1]

        pred = pd.DataFrame({'Actual Value': Y_test,
                              'Predicted Value': Y_test_pred})
        print("The top 5 rows of actual vs predicted values\n", pred.head())
        pred.to_csv("predictions.csv")
        print("Predictions saved in a csv file!")

        return Y_test, Y_test_pred, r2_rf

# Example usage:
# You would need to create an instance of RandomForestModel with your X and Y data and then call the random_forest method.
# rf_model = RandomForestModel(X, Y)
# rf_model.random_forest()