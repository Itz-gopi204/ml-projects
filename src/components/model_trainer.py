import sys
import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from src.exception import CustomExeception
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config =ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and est input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "random forest": RandomForestRegressor(),
                'decision tree': DecisionTreeRegressor(),
                "gradient boosting": GradientBoostingRegressor(),
                "linear regression": LinearRegression(),
                "XGBREGRESSION": XGBRegressor(),
                "K-neighbors": KNeighborsRegressor(),
                "adaboost regression": AdaBoostRegressor(),
                "catboosting regression": CatBoostRegressor(verbose=False),
            }

            model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,model=models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            if best_model_score < 0.6:
                raise CustomExeception("not best model found")
            logging.info('best model found on both training and testing dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            predicted=best_model.predict(x_test)
            r2_sc=r2_score(y_test, predicted)



        except Exception as e:
            raise CustomExeception(e,sys)