import os
import sys # for Exception handling
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException

# It can write train_test_split over here instead of data_transformation.py

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:

            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)

    