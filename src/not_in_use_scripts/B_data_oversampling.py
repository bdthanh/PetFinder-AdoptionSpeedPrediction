import os
import sys
import pandas as pd

from imblearn.over_sampling import SMOTENC
from A_data_ingestion import DataIngestion
from U_exception import CustomException
from U_logger import logging

TARGET = "AdoptionSpeed"
    
class DataOversampling:
    def __init__(self):
        self.type = type
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(parent_dir, 'data')
        self.train_data_path = os.path.join(data_dir, f"train.csv")
        
    def initiate_data_oversampling(self):
        logging.info("Initiating data oversampling with SMOTE_NC")
        try:
            df = pd.read_csv(self.train_data_path, index_col=False)
            logging.info("Found dataset. Read the data as dataframe")
            
            X = df.drop(TARGET, axis=1)
            y = df[TARGET]

            categorical_features = X.select_dtypes(include=['object']).columns
            cat_mask = [col in categorical_features for col in X.columns]
            smote_nc = SMOTENC(
                sampling_strategy={0: 800},
                categorical_features=cat_mask, 
                random_state=20
            )
            X_smote, y_smote = smote_nc.fit_resample(X, y)
            train_smote_set = pd.concat([X_smote, y_smote], axis=1)
            
            train_smote_set.to_csv(self.train_data_path, index=False, header=True)
            
            return self.train_data_path            
            
        except FileNotFoundError:
            print(f'File {self.train_data_path} not found. Please give the correct path to the dataset')    
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data, valid_data = data_ingestion.initiate_data_ingestion()
    data_oversampling = DataOversampling()
    train_smote_set = data_oversampling.initiate_data_oversampling()
        