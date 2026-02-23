import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split train and test input data")

            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Dictionary of regression models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Random Forest": {
                    'n_estimators' : [8,16,32,64,128,256]

                },

                "Decision Tree": {
                    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },

                "Gradient Boosting": {
                    'learning_rate' : [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators' : [8,16,32,64,128,256]
                },

                "K Neighbors Regressor": {
                    'n_neighbors' : [5,7,9,11]
                },

                "XGB Regressor": {
                    'learning_rate' : [.1, .01, .05, .001],
                    'n_estimators' : [8, 16, 32, 64, 128, 256]
                },

                "CatBoost Regressor": {
                    'depth' : [6,8,10],
                    'learning_rate' : [0.01, 0.05, 0.1],
                    'iterations' : [30,50,100]
                },

                "AdaBoost Regressor": {
                    'learning_rate' : [.1, .01, 0.5, .001],
                    'n_estimators' : [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params = params
            )

            # Get best model score
            best_model_score = max(model_report.values())

            # Get best model name
            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model found: {best_model_name}")

            # Fit best model again before saving
            best_model.fit(x_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)


