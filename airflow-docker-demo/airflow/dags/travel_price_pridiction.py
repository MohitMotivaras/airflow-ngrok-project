from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import pandas as pd
import pickle
import os

import mlflow
import mlflow.sklearn

from data_ingestion import DataLoader
from data_transformation import DataTransformer
from model_training import RandomForestModel

# -------------------------------
# Configurations
# -------------------------------
DATA_FILE_PATH = '/opt/airflow/dags/flights.csv'
TRANSFORMED_FILE = '/opt/airflow/dags/transformed.pkl'
MODEL_DIR = '/opt/airflow/models/'

# Make sure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

default_args = {
    'owner': 'Admin',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),  # past date for testing
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'travel_price_prediction',
    default_args=default_args,
    description='A DAG for travel price prediction using RandomForest + MLflow',
    schedule_interval=None,  # Change to '@daily' for auto run
    catchup=False
)

# -------------------------------
# Tasks
# -------------------------------

def load_data(**kwargs):
    """Load raw data and push via XCom."""
    loader = DataLoader(DATA_FILE_PATH)
    df = loader.load_data()
    kwargs['ti'].xcom_push(key='raw_data', value=df.to_json())
    return "✅ Data Loaded"


def transform_data(**kwargs):
    """Transform raw dataset into X, Y."""
    ti = kwargs['ti']
    df_json = ti.xcom_pull(task_ids='load_data_task', key='raw_data')
    df = pd.read_json(df_json)

    transformer = DataTransformer(df)
    X, Y = transformer.transform()

    # Save transformed dataset
    with open(TRANSFORMED_FILE, 'wb') as f:
        pickle.dump((X, Y), f)

    return "✅ Data Transformed"


def train_model(**kwargs):
    """Train model, evaluate & log in MLflow."""
    with open(TRANSFORMED_FILE, 'rb') as f:
        X, Y = pickle.load(f)

    model = RandomForestModel(X, Y)
    Y_test, Y_pred, r2 = model.random_forest()

    model_name = f"random_forest_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    model_path = os.path.join(MODEL_DIR, model_name)

    # 🔹 Start MLflow experiment logging
    mlflow.set_tracking_uri("http://localhost:5000")  # change if MLflow runs elsewhere
    mlflow.set_experiment("Travel_Price_Prediction")

    with mlflow.start_run(run_name="airflow_random_forest"):
        mlflow.log_metric("r2_score", r2)

        # Save and log the trained model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models",
            registered_model_name="TravelPriceRF"
        )

    return f"✅ Model trained with R2 score: {r2}, saved at {model_path}"


# -------------------------------
# Airflow Operators
# -------------------------------
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

transform_data_task = PythonOperator(
    task_id='transform_data_task',
    python_callable=transform_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    dag=dag,
)

# DAG Dependencies
load_data_task >> transform_data_task >> train_model_task
