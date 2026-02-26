import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


##All your file paths / settings live in one place
##Instead of hardcoding 'artifacts/train.csv' scattered everywhere, you centralize it
@dataclass
class DataIngestionConfig:
    ##defining path names
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # consuming the dataIngestionConfig

    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component")
        try:
            ##read from the relative path
            df = pd.read_csv(os.path.join('notebook', 'data', 'stud.csv'))
            logging.info("Read the dataset as dataframe")
            ##make the directory artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            ##save raw data to artifacts/data.csv
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test initiated")
            ##train test split
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            ## save train  and test data to their directory
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data completed')

            ##return the path names
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)



