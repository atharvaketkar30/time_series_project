import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_model(self, train:str, test:str):
        logging.info('Training the model')
        try:
            train = pd.read_csv(train)
            test = pd.read_csv(test)
            X_train, y_train = train.drop(['Target'], axis=1), train['Target']
            X_test, y_test = test.drop(['Target'], axis=1), test['Target']

            models = {
                'catboost': CatBoostRegressor(),
                'xgboost': XGBRegressor(),
                'lightgbm': LGBMRegressor(),
                'random_forest': RandomForestRegressor(),
                'linear_regression': LinearRegression()
            }

            model_report:dict = evaluate_models(models, X_train, y_train, X_test, y_test)

            #Best model
            best_model_score = np.max([sub_dict['mae'] for sub_dict in model_report.values()])
            best_model_name = list(model_report.keys())[np.argmax([sub_dict['mae'] for sub_dict in model_report.values()] )]
            best_model = models[best_model_name]
            
            #Saving the model
            logging.info('Saving the model')
            save_object(best_model, self.model_trainer_config.trained_model_path)

            print('Best model is {} with MAE {:.2f}'.format(best_model_name, best_model_score))
        except Exception as e:
            raise CustomException(error_ms=e, detail=sys)

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train_model('artifacts/preprocessed_train.csv', 'artifacts/preprocessed_test.csv')