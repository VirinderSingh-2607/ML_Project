import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class DataIngestionConfig:
    Train_data_path: str=os.path.join('artifacts',"Train.csv")
    Test_data_path: str=os.path.join('artifacts',"Test.csv")
    Raw_data_path: str=os.path.join('artifacts',"Data.csv")

class DataIngestion:
    def __init__(self):
        self.Ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\Data\stud.csv')
            logging.info("Read the dataset as a dataframe")
            os.makedirs(os.path.dirname(self.Ingestion_config.Train_data_path),exist_ok = True)
            logging.info("Train test split is initiated")
            Train_set,Test_set = train_test_split(df,test_size = 0.2,random_state = 42)
            Train_set.to_csv(self.Ingestion_config.Train_data_path,index = False,header = True)
            Test_set.to_csv(self.Ingestion_config.Test_data_path,index = False,header = True)
            df.to_csv(self.Ingestion_config.Raw_data_path,index=False,header=True)
            logging.info("ingestion of the data is completed")
            return (self.Ingestion_config.Train_data_path,
                    self.Ingestion_config.Test_data_path) 
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()