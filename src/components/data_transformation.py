import os 
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from src.utils import save_object
from src.components.data_ingestion import DataIngestion



## data transformation paths
@dataclass
##@dataclass auto-generates __init__, __repr__ etc. for you
##All your file paths / settings live in one place — easy to change later
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    ## to create all pkl files responsible for the transformation
    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_features = ['gender','race_ethnicity','test_preparation_course',
                                    'parental_level_of_education','lunch']
            
            ##creating pipelines
            num_pipeline = Pipeline(
                steps= [
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps= [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OHE',OneHotEncoder()),
                ]
            )
            logging.info(f"Numerical  columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_features} ")


            ##using column transformer to combine the two pipelines
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_features)
                ]
            )

            return preprocessor

        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            ## read train and test date from various path
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            ##logs
            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessor object')


            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score','reading_score']

            ##droping the target column from the train data set  and test data set before the preprocessing
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            ##logs
            logging.info('Applying preprocessing object on the train and test data set')

            ##applying preprocessing
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            ## combining the target column from the train data set  and test data set after the preprocessing so that it  can be saved and passed as a single object to the next component.
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            ##logs 
            logging.info("Saved preprocessing object")

            ##saving the pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
    


if __name__ == "__main__":
    data= DataIngestion() 
    train_data_path,test_data_path = data.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path,test_data_path)



