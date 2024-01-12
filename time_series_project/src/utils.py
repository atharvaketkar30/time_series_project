import os
import sys
import dill
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
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
        logging.info('Dataframe converted to dictionary successfully')
        return df_dict
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)

def get_lagged_features(df: pd.DataFrame, lags: List[int], target: str):
    '''
    This function returns the lagged columns
    '''
    logging.info('Getting lagged columns')
    try:
        for l in lags:
            df['lag_{}_{}'.format(target, l)] = df[target].shift(l)
            df = df.dropna().reset_index(drop=True)
        logging.info('Lagged columns extracted successfully')
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)
        # raise KeyError('The column {} does not exist'.format(target))

def get_rolling_features(df: pd.DataFrame, windows: List[int], target: str):
    '''
    This function returns the rolling features
    '''
    logging.info('Getting rolling features')
    try:
        for w in windows:
            df['rolling_mean_{}_{}'.format(target, w)] = df[target].shift(1).rolling(w).mean()
            # df['rolling_std_{}_{}'.format(df.columns[0], w)] = df[df.columns[0]].shift(1).rolling(w).std()
            df = df.dropna().reset_index(drop=True)
        logging.info('Rolling features extracted successfully')
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)

def get_ewm_features(df: pd.DataFrame, alphas: List[float], lags: List[int], target: str):
    '''
    This function returns the ewm features
    '''
    logging.info('Getting ewm features')
    try:
        for l in lags:
            for a in alphas:
                df['ewm_mean_{}_{}_{}'.format(target, l, a)] = df[target].shift(1).ewm(alpha=a).mean()
                df = df.dropna().reset_index(drop=True)
        logging.info('Ewm features extracted successfully')
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)

def add_product_num_column(df: pd.DataFrame, k: str):
    '''
    args: dataframe, product number(key)
    '''
    try:
        df['Product'] = int(k[2:])
        df = df.rename({k: 'Target'})
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
        df = pd.DataFrame()
        for k, v in df_dict.items():
            df = pd.concat([df, v], axis=0)
        logging.info('Dictionary converted to dataframe successfully')
        return df
    except Exception as e:
        raise CustomException(error_ms=e, detail=sys)