def train_model(**kwargs):
    """Train model, evaluate & log in MLflow."""
    # Load transformed dataset
    with open(TRANSFORMED_FILE, 'rb') as f:
        X, Y = pickle.load(f)

    # Initialize and train
    rf_model_obj = RandomForestModel(X, Y)
    best_model, Y_test, Y_pred, r2 = rf_model_obj.random_forest()

    # Save predictions
    predictions_path = '/opt/airflow/dags/predictions.csv'
    import pandas as pd
    pred_df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
    pred_df.to_csv(predictions_path, index=False)

    # MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Travel_Price_Prediction")

    # Log metrics, model, and artifacts
    with mlflow.start_run(run_name="airflow_random_forest"):
        # Log metrics
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(Y_test, Y_pred)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="models",
            registered_model_name="TravelPriceRF"
        )

        # Log predictions
        mlflow.log_artifact(predictions_path)

    return f"✅ Model trained | R2: {r2:.4f} | MSE: {mse:.4f}"
