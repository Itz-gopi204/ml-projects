import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomExeception
from src.logger import logging
import os
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot",OneHotEncoder(drop="first")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed")
            logging.info("categorical columns encoding cokmpleted")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
                )


            return preprocessor
        except Exception as e:
            raise CustomExeception(e,sys)



    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Data read from csv completed")

            logging.info("obtaining preprocessor object")
            preproccessing_object=self.get_data_transformation_object()

            target_column_name="math_score"
            numerical_columns=["writing_score", "reading_score"]
            categorical_columns=["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(f"applying preprocceing object on train df and test df")
            input_feature_train_arr=preproccessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preproccessing_object.transform(input_feature_test_df)
            train_arr=np.c_(
                input_feature_train_arr,
                np.array(target_feature_train_df)
            )
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preproccessing_object()
            )
            test_arr=np.c_(
                input_feature_test_arr,
                np.array(target_feature_test_df)
            )
            return (train_arr, test_arr,self.data_transformation_config.preprocessor_obj_file_path)






        except Exception as e:
            raise CustomExeception(e,sys)