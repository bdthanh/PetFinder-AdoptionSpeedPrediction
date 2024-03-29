import os
import sys
import pandas as pd

from U_exception import CustomException
from U_logger import logging
from U_utils import create_if_missing_folder, get_filepath, has_inference_data
from U_constants import TARGET, NEEDED_COLUMNS, RANDOM_STATE, MODEL_DIR, DATA_DIR, ORIGINAL_VALID_PATH, RAW_DATA_PATH, ORIGINAL_DATA, INFERENCE_DATA_PATH
from sklearn.model_selection import train_test_split, StratifiedKFold
    
class DataIngestion:
    def __init__(self):
        return
    
    def initiate_data_ingestion(self, save: bool = True):
        """ This function is used to initiate the data ingestion process. It reads the dataset and splits it into train, test, valid sets.
        The valid set is 10% of the dataset and the train, test sets are split using StratifiedKFold with 6 folds. So, the train set is 
        around 75% of the dataset. The data is then saved into csv files.

        Args:
            save (bool, optional): True to save the data as csv files. Defaults to True.

        Returns:
            train_sets, test_sets, valid_set: Data after being split
        """
        logging.info("Initiating data ingestion")
        try:
            df = pd.read_csv(ORIGINAL_DATA)
            logging.info("Found dataset. Read the data as dataframe")
            
            logging.info("Making missing dataset directory")
            os.makedirs(DATA_DIR, exist_ok=True)
            
            df = df[NEEDED_COLUMNS]
            df.to_csv(RAW_DATA_PATH,index=False,header=True)
        
            logging.info("Splitting the data into train, test sets with 6 fold and stratify the target variable")
            
            train_test_set, valid_set = train_test_split(df, test_size=0.1, random_state=RANDOM_STATE, stratify=df[TARGET])
            skf = StratifiedKFold(n_splits=6, random_state=RANDOM_STATE, shuffle=True)
            
            train_sets, test_sets = [], []
            for fold, [train_index, test_index] in enumerate(skf.split(train_test_set, train_test_set[TARGET].values)):
                _, train_path, test_path, _ = get_filepath(fold)
                train_set = train_test_set.iloc[train_index, :]
                test_set = train_test_set.iloc[test_index, :]
                train_sets.append(train_set)
                test_sets.append(test_set)
                if save: 
                    train_set.to_csv(train_path, index=False, header=True)
                    test_set.to_csv(test_path, index=False, header=True)
            
            logging.info("Finised splitting. Saving the data into csv files")
            if save:
                valid_set.to_csv(ORIGINAL_VALID_PATH, index=False, header=True)
                if not has_inference_data():
                    valid_set.to_csv(INFERENCE_DATA_PATH, index=False, header=True)
            return train_sets, test_sets, valid_set
        except FileNotFoundError:
            print(f'File {self.config.original_data_path} not found. Please give the correct path to the dataset') 
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    create_if_missing_folder(MODEL_DIR)
    create_if_missing_folder(DATA_DIR)
    data_ingestion = DataIngestion()
    train_sets, test_sets, valid_set = data_ingestion.initiate_data_ingestion()
        