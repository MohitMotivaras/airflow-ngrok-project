from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import pandas as pd
import pickle

from data_ingestion import DataLoader
from data_transformation import DataTransformer
from model_training import RandomForestModel

DATA_FILE_PATH = '/opt/airflow/dags/flights.csv'

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
    description='A DAG for travel price prediction using RandomForest',
    schedule=None,
)

def load_data(**kwargs):
    loader = DataLoader(DATA_FILE_PATH)
    df = loader.load_data()
    kwargs['ti'].xcom_push(key='raw_data', value=df.to_json())
    return "Data Loaded"

def transform_data(**kwargs):
    ti = kwargs['ti']
    df_json = ti.xcom_pull(task_ids='load_data_task', key='raw_data')
    df = pd.read_json(df_json)
    transformer = DataTransformer(df)
    X, Y = transformer.transform()
    with open('/tmp/transformed.pkl', 'wb') as f:
        pickle.dump((X, Y), f)
    return "Data Transformed"

def train_model(**kwargs):
    with open('/tmp/transformed.pkl', 'rb') as f:
        X, Y = pickle.load(f)
    model = RandomForestModel(X, Y)
    Y_test, Y_pred, r2 = model.random_forest()
    return f"Model trained with R2 score: {r2}"

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

load_data_task >> transform_data_task >> train_model_task
