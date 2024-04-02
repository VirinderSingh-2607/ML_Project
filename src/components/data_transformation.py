import sys
import os
import pandas as pd 
from  src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

import numpy as np

from dataclasses import dataclass

@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [""]

            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="mean")),
                    ("Scaler",StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder())
                    ("Scaler",StandardScaler())
                ]
            )

            preprocessor  = ColumnTransformer(
                [("num_pipeline",numerical_pipeline,numerical_columns),
                 ("Cat_pipeline",categorical_pipeline,categorical_columns)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = []