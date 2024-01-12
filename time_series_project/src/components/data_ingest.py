import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import remove_cols, get_dates

#Paths to saving the data
@dataclass
class DataIngestionCinfig:
    train_data_path: str= os.path.join('artifacts', 'train.csv')
    test_data_path: str= os.path.join('artifacts', 'test.csv')
    raw_data_path: str= os.path.join('artifacts', 'data.csv')

#Data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionCinfig()

    def get_data(self):
        logging.info('Reading data from raw data path')
        try:
            #Reading data from the csv file
            df = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')
            logging.info('Data read successfully')

            #Removing columns
            logging.info('Removing columns from the dataframe')
            df = remove_cols(df)
            df = df.drop(['Product_Code'], axis=1)
            logging.info('Columns removed successfully')

            df = df.transpose()
            # Getting dates from the dataframe
            df = get_dates(df)

            #Saving the data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Data saved successfully')

            #Splitting the data
            logging.info('Splitting data into train and test')
            split_point = int(len(df)*0.7)
            train = df.iloc[:split_point, :]
            test = df.iloc[split_point:, :]
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data split successfully')

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
                )
        except Exception as e:
            raise CustomException(error_ms=e, detail=sys)

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    data_ingestion.get_data()