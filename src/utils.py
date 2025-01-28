import os
import sys
import pandas as pd
import numpy as np

import dill
from exception import CustomExeception

def save_object(file_path,obj):
    try:
        dir_path=os.path.join(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomExeception(e,sys)
    