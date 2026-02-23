import os
import sys # for Exception handling
import dill

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from src.exception import CustomException

from sklearn.model_selection import GridSearchCV

# It can write train_test_split over here instead of data_transformation.py

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:

            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(x_train ,y_train, x_test, y_test, models, params):
    try:
        
        report = {}

        for model_name, model in models.items():
        #for i in range(len(list(models))):
        #    model = list(models.values())[i]
            #params = params[list(models.keys())[i]]

            param = params[model_name]

            gs= GridSearchCV(model, param, cv=3, scoring="r2", n_jobs=-1)
            gs.fit(x_train, y_train)

            model = gs.best_estimator_

            #model.set_params(**gs.best_params_)
            model.fit(x_train, y_train) #Train model

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
          
            report[model_name] = test_model_score
            #report[list(model.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)


    