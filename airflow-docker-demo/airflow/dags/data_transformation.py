import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataTransformer:
    """
    A class for cleaning and transforming travel datasets 
    into model-ready features and labels.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the transformer with raw data.
        
        Parameters
        ----------
        data : pd.DataFrame
            The input dataset containing travel records.
        """
        self.data = data.copy()

    def transform(self):
        """
        Perform preprocessing, feature engineering, 
        and outlier handling on the dataset.
        
        Returns
        -------
        X : pd.DataFrame
            Feature matrix after preprocessing.
        y : pd.Series
            Target variable (price).
        """
        df = self.data

        # Drop columns with too many missing values or irrelevant identifiers
        df = df.drop(columns=['travelCode', 'userCode'], errors='ignore')

        # Handle missing values
        df = df.dropna(subset=['date', 'price'])

        # Convert to datetime format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # Extract date-related features
        df['day_of_journey'] = df['date'].dt.day
        df['month_of_journey'] = df['date'].dt.month
        df['year_of_journey'] = df['date'].dt.year

        # Outlier removal using IQR method on 'price'
        Q1, Q3 = df['price'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

        # Encode categorical features
        categorical_cols = ['from', 'to', 'flightType', 'agency']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # Define features and target
        feature_cols = [
            'from', 'to', 'flightType', 'time', 'distance', 'agency',
            'day_of_journey', 'month_of_journey', 'year_of_journey'
        ]
        feature_cols = [col for col in feature_cols if col in df.columns]

        X = df[feature_cols]
        y = df['price']

        return X, y
