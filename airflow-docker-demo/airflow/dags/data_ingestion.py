import numpy as np
import pandas as pd
from data_transformation import DataTransformer
from model_training import RandomForestModel

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_data(self):
        df = pd.read_csv(self.file_path, on_bad_lines='skip')

        # Check for missing values
        if df.isnull().sum().sum() == 0:
            print("There are no missing values present")
        else:
            print("There are missing values present and we need to handle them!")

        return df
