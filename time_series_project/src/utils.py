import os
import sys
import dill
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import mean_squared_error, \
    mean_absolute_error, r2_score
from typing import List, Dict

def save_object(obj, file_path: str):
    '''
    This function saves the object as a pickle file
    '''
    logging.info('Saving object as pickle file')
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
        logging.info('Object saved successfully')
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)
    
def remove_cols(df:pd.DataFrame):
    '''
    This function removes the columns from the dataframe
    '''
    logging.info('Removing columns from the dataframe')
    try:
        cols_to_remove = df.columns[df.columns.str.contains('MIN')| df.columns.str.contains('MAX')| df.columns.str.startswith('W')]
        df = df.drop(cols_to_remove, axis=1)
        logging.info('Columns removed successfully')
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)

def get_dates(df: pd.DataFrame):
    '''
    This function returns the dates from the dataframe
    '''
    logging.info('Getting dates from the dataframe')
    try:
        dates = pd.date_range(start='2019-01-01', periods = len(df), freq='1w')
        df = df.set_index(dates, inplace=False)
        logging.info('Dates extracted successfully')
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)
    
def convert_df_to_dict(df: pd.DataFrame):
    '''
    This function takes the entire dataset and separates it into 
    a dictionary of dataframes for each product
    '''
    logging.info('Converting dataframe to dictionary')
    try:
        # num_prods = len(df)
        df.columns = [f'P_{i}' for i in range(811)]

        df_dict = {}
        for col in df.columns:
            df_dict[col] = pd.DataFrame(df[col], columns=[col])
        return df_dict
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)

def get_lagged_features(df: pd.DataFrame, lags: List[int], target: str):
    '''
    This function returns the lagged columns
    '''
    try:
        for l in lags:
            df['lag_{}'.format(l)] = df[target].shift(l)
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)
        # raise KeyError('The column {} does not exist'.format(target))

def get_rolling_features(df: pd.DataFrame, windows: List[int], target: str):
    '''
    This function returns the rolling features
    '''
    try:
        for w in windows:
            df['rolling_mean_{}'.format(w)] = df[target].shift(1).rolling(w).mean()
            # df['rolling_std_{}_{}'.format(df.columns[0], w)] = df[df.columns[0]].shift(1).rolling(w).std()
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)

def get_ewm_features(df: pd.DataFrame, alphas: List[float], lags: List[int], target: str):
    '''
    This function returns the ewm features
    '''
    try:
        for l in lags:
            for a in alphas:
                df['ewm_mean_{}_{}'.format(l, a)] = df[target].shift(l).ewm(alpha=a).mean()
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)

def add_product_num_column(df: pd.DataFrame, k: str):
    '''
    args: dataframe, product number(key)
    '''
    try:
        df['Product'] = int(k[2:])
        df = df.rename({k: 'Target'}, axis = 'columns')
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)
    
def convert_dict_to_df(df_dict: Dict[str, pd.DataFrame]):
    '''
    This function converts the dictionary to dataframe
    args: dictionary of dataframes
    '''
    logging.info('Converting dictionary to dataframe')
    try:
        # df = pd.DataFrame()
        # for k, v in df_dict.items():
        #     df = pd.concat([df, v], axis=0)
        df = pd.concat(list(df_dict.values()), axis=0, ignore_index=True)
        logging.info('Dictionary converted to dataframe successfully')
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)

def evaluate_models(models: Dict[str, object], X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2 = r2_score(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            evals:dict = {'rmse': rmse,
                        'r2': r2,
                        'mae': mae
                        }
            report[name] = evals
        return report
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)