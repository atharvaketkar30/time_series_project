import sys
import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgbm

from src.utils import save_object, get_dates, convert_df_to_dict, \
    get_lagged_features, get_rolling_features, remove_cols, \
    get_ewm_features, convert_dict_to_df, add_product_num_column

@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    preprocessed_train_data_path: str = os.path.join('artifacts', 'preprocessed_train.csv')
    preprocessed_test_data_path: str = os.path.join('artifacts', 'preprocessed_test.csv')

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # self.x_path = x_path
        # self.df = pd.read_csv(self.x_path)
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X.copy()
        # X_new = remove_cols(X_new)
        # logging.info('Getting dates from the dataframe')
        # X_new = get_dates(X_new)
        logging.info('Dates extracted successfully. Now converting dataframe to dictionary')
        X_new_dict = convert_df_to_dict(X_new)

        # print(X_new_dict['P_0'].head())

        logging.info('Dictionary created successfully. Now extracting features')
        lags = [1,2,3,4,5]
        logging.info('Extracting lag features')
        for k,ts in X_new_dict.items():
            X_new_dict[k] = get_lagged_features(ts, lags=lags, target = k)
        # print(X_new_dict['P_0'].head())
        logging.info('Extracting rolling features')
        windows = list(range(1, 6))
        for k,ts in X_new_dict.items():
            X_new_dict[k] = get_rolling_features(ts, windows=windows, target = k)
        # print(X_new_dict['P_0'].head())
        logging.info('Extracting ewm features')
        alphas = [0.9, 0.95, 0.5]
        for k,ts in X_new_dict.items():
            X_new_dict[k] = get_ewm_features(ts, alphas=alphas, lags=lags, target = k)
        
        # print(X_new_dict['P_0'].head())
        logging.info('Features extracted successfully. Now converting dictionary to dataframe')
        for k,ts in X_new_dict.items():
            X_new_dict[k] = add_product_num_column(ts, k)
        
        
        X_final = convert_dict_to_df(X_new_dict)
        
        return X_final


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor(self):
        '''
        This function returns the preprocessor object
        '''
        logging.info('Getting preprocessor object')
        try:
            # Preprocessor Pipeline
            # preprocessor = Pipeline(steps=[
            #     ('custom_transformer', CustomTransformer())
            # ])

            preprocessor = CustomTransformer()

            return preprocessor

            # return preprocessor
        except Exception as e:
            raise CustomException(error_ms=e, detail=sys)
            

    def transform_data(self, train_path: str, test_path: str):
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # logging.info('Train and Test Data read successfully')

            logging.info('Getting the preprocessor object')
            preprocessor_train = self.get_preprocessor()
            preprocessor_test = self.get_preprocessor()

            logging.info('Transforming the data')
            preprocessed_train = preprocessor_train.transform(train_df)
            preprocessed_test = preprocessor_test.transform(test_df)

            logging.info('Saving the preprocessed data')
            preprocessed_train.to_csv(self.data_transformation_config.preprocessed_train_data_path, index=False, header=True)
            preprocessed_test.to_csv(self.data_transformation_config.preprocessed_test_data_path, index=False, header=True)

            logging.info('Saving the preprocessor object')
            save_object(preprocessor_train, self.data_transformation_config.preprocessor_file_path)

            return (
                preprocessed_train,
                preprocessed_test,
                self.data_transformation_config.preprocessor_file_path
            )


        except Exception as e:
            raise CustomException(error_ms=e, detail=sys)

if __name__ == '__main__':
    obj = DataTransformation()
    obj.transform_data('artifacts/train.csv', 'artifacts/test.csv')