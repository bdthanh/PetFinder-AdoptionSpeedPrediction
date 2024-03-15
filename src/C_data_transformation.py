import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from U_exception import CustomException
from U_logger import logging
from U_utils import save_obj, get_filepath, create_if_missing_folder
from U_constants import ORIGINAL_VALID_PATH, NUM_COLS, NORMAL_CAT_COLS_TO_PROCESS, NORMAL_CAT_COLS_NOT_TO_PROCESS, ORDINAL_CAT_COLS_TO_PROCESS, ORDINAL_CAT_COLS_NOT_TO_PROCESS, TEXT_COLS, TARGET, TEXT_FILL, TOP12_BREEDNAME, RANDOM_STATE, MODEL_DIR, DATA_DIR
from A_data_ingestion import DataIngestion
      

class DataTransformation:
    def __init__(self) -> None:
        pass
    
    def get_data(self, fold: int):
        """Get the data used in train pipeline from the csv files. Read the train, test, valid sets from the csv files and returns them.

        Args:
            fold (int): The fold number of the data used in that fold
        """
        train_sets, test_sets = [], []
        for fold in range(fold):
            _, train_path, test_path, _ = get_filepath(fold)
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            valid_data = pd.read_csv(ORIGINAL_VALID_PATH)
            train_sets.append(train_data)
            test_sets.append(test_data)
        return train_sets, test_sets, valid_data
        
    def get_data_transformer_obj(self):
        """Get the data transformer object. This object is used to transform the data. It is a pipeline that contains the preprocessing steps for the data.
           It includes pipeline for numerical, categorical, and text data (Description).
        """
        try: 
            agebins_order = [['[  0,  2)', '2', '[  3,  6)', '[  6, 12)', '[ 12, 24)', '[ 24, 60)', '[ 60,255]']]
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('minmax_scaler', MinMaxScaler())
            ])
            cat_normal_to_process = Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('ordinal_encoder', OrdinalEncoder())
            ])
            cat_ordinal_to_process = Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('ordinal_encoder', OrdinalEncoder(categories=agebins_order, handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            cat_not_to_process = Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
            ])
            text = Pipeline([
                ('count_vectorizer', CountVectorizer(ngram_range=(1,2), stop_words=stopwords.words('english'), max_features=1000)),
                ('tfidf', TfidfTransformer(sublinear_tf=True)),
                ('svd', TruncatedSVD(n_components=10, random_state=RANDOM_STATE))
            ])
            pre_processor = ColumnTransformer([
                ('num', num_pipeline, NUM_COLS),
                ('cat_normal_to_process', cat_normal_to_process, NORMAL_CAT_COLS_TO_PROCESS),
                ('cat_ordinal', cat_ordinal_to_process, ORDINAL_CAT_COLS_TO_PROCESS),
                ('other_cat', cat_not_to_process, NORMAL_CAT_COLS_NOT_TO_PROCESS + ORDINAL_CAT_COLS_NOT_TO_PROCESS),
                ('text', text, TEXT_COLS)
            ], remainder='passthrough')
            
            return pre_processor
        except Exception as e:
            raise CustomException(e, sys)
      
    def save_transformed_data(self, train_data, test_data, valid_data, train_path, test_path, valid_path) -> None:
        """
        Save the transformed data into csv files. The train, test, valid sets are saved into csv files.
        """
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        valid_data.to_csv(valid_path, index=False)
    
    def save_preprocessor_obj(self, preprocessor, preprocessor_file_path):
        """
        Save the preprocessor object into a pickle file, which will be used in inference pipeline later. 
        """
        save_obj(preprocessor_file_path, preprocessor)
        return preprocessor_file_path

    def replace_limit(self, data, column, limit):
        data[column] = data[column].apply(lambda x: limit if x > limit else x)
        return data
      
    def replace_limit_train_test_valid(self, train_data, test_data, valid_data):
        """
        Replace the limit of the column 'Quantity', 'VideoAmt', 'PhotoAmt' in the train, test, valid sets. 
        The limit is set to the limit value of the column, refer to README.md for how these values are chose.
        """
        train_data = self.replace_limit(train_data, "Quantity", 6)
        test_data = self.replace_limit(test_data, "Quantity", 6)
        valid_data = self.replace_limit(valid_data, "Quantity", 6)
        train_data = self.replace_limit(train_data, "VideoAmt", 2)
        test_data = self.replace_limit(test_data, "VideoAmt", 2)
        valid_data = self.replace_limit(valid_data, "VideoAmt", 2)
        train_data = self.replace_limit(train_data, "PhotoAmt", 10)
        test_data = self.replace_limit(test_data, "PhotoAmt", 10)
        valid_data = self.replace_limit(valid_data, "PhotoAmt", 10)
        return train_data, test_data, valid_data
    
    def replace_breedname_minority(self, train_data, test_data, valid_data):
        """
        Replace the breedname that is not in the top 12 breed names with 'Other' in the train, test, valid sets.
        Refer to README.md for how these values are chose.
        """ 
        train_data['BreedName'] = train_data['BreedName'].apply(lambda x: x if x in TOP12_BREEDNAME else 'Other')
        test_data['BreedName'] = test_data['BreedName'].apply(lambda x: x if x in TOP12_BREEDNAME else 'Other') 
        valid_data['BreedName'] = valid_data['BreedName'].apply(lambda x: x if x in TOP12_BREEDNAME else 'Other')
        return train_data, test_data, valid_data
    
    def fill_na_text_description(self,train_data, test_data, valid_data):
        """
        Fill the missing value in the text column 'Description' with 'MISSING' in the train, test, valid sets.
        """
        train_data[TEXT_COLS].fillna(TEXT_FILL, inplace=True)
        test_data[TEXT_COLS].fillna(TEXT_FILL, inplace=True)
        valid_data[TEXT_COLS].fillna(TEXT_FILL, inplace=True)
        return train_data, test_data, valid_data
    
    def initiate_data_transformation(self, train_sets=None, test_sets=None, valid_set=None, save: bool=True): 
        """Initiate the data transformation process. This function is used to transform the data. 
        It replaces the breedname that is not in the top 12 breed names with 'Other', replaces the limit of the column 'Quantity', 'VideoAmt', 'PhotoAmt', 
        fills the missing value in the text column 'Description' with 'MISSING', and saves the transformed data into csv files.

        Args:
            train_sets (_type_, optional): list of train sets used in each fold. Defaults to None.
            test_sets (_type_, optional): list of test sets used in each fold. Defaults to None.
            valid_set (_type_, optional): valid set
            save (bool, optional): _description_. Defaults to True.
        """
        try:
            if train_sets is None or test_sets is None or valid_set is None:
                train_sets, test_sets, valid_set = self.get_data(fold=6)
            for fold in range(6):
                preprocessor_file_path, train_path, test_path, valid_path = get_filepath(fold)
                train_data = train_sets[fold]
                test_data = test_sets[fold]
                valid_data = valid_set
                
                train_data, test_data, valid_data = self.replace_breedname_minority(train_data, test_data, valid_data)
                train_data, test_data, valid_data = self.replace_limit_train_test_valid(train_data, test_data, valid_data)
                train_data, test_data, valid_data = self.fill_na_text_description(train_data, test_data, valid_data)
                
                X_train, y_train = train_data.drop(TARGET, axis=1), train_data[TARGET]
                X_test, y_test = test_data.drop(TARGET, axis=1), test_data[TARGET]
                X_valid, y_valid = valid_data.drop(TARGET, axis=1), valid_data[TARGET]
                
                logging.info("Getting the preprocesser object")
                preprocessor = self.get_data_transformer_obj()
                
                logging.info("Fitting the preprocessor object")
                logging.info("Transforming the train data")

                X_train = preprocessor.fit_transform(X_train, y_train)
                X_test = preprocessor.transform(X_test)
                X_valid = preprocessor.transform(X_valid)
                
                train_data = pd.concat([pd.DataFrame(X_train).reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
                test_data = pd.concat([pd.DataFrame(X_test).reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
                valid_data = pd.concat([pd.DataFrame(X_valid).reset_index(drop=True), y_valid.reset_index(drop=True)], axis=1)
                
                logging.info("Finished transforming the data")
                
                if save: 
                    preprocessor_file_path = self.save_preprocessor_obj(preprocessor, preprocessor_file_path)
                    logging.info("Saved the preprocessor object")
                
                logging.info("Rename the columns of the transformed data")
                columns = list(preprocessor.get_feature_names_out()) + [TARGET]
                train_data.columns = columns
                test_data.columns = columns 
                valid_data.columns = columns
                train_data.columns = train_data.columns.str.replace('num__', '').str.replace('cat_normal_to_process__', '').str.replace('remainder__', '').str.replace('cat_ordinal__', '').str.replace('other_cat__', '').str.replace('text__', '')
                test_data.columns = test_data.columns.str.replace('num__', '').str.replace('cat_normal_to_process__', '').str.replace('remainder__', '').str.replace('cat_ordinal__', '').str.replace('other_cat__', '').str.replace('text__', '')
                valid_data.columns = valid_data.columns.str.replace('num__', '').str.replace('cat_normal_to_process__', '').str.replace('remainder__', '').str.replace('cat_ordinal__', '').str.replace('other_cat__', '').str.replace('text__', '')

                if save:
                    logging.info("Finished renaming the columns of the transformed data")
                    logging.info("Saving the transformed data") 
                    self.save_transformed_data(train_data, test_data, valid_data, train_path, test_path, valid_path)

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    create_if_missing_folder(MODEL_DIR)
    create_if_missing_folder(DATA_DIR)
    data_ingestion = DataIngestion()
    train_sets, test_sets, valid_set = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_sets, test_sets, valid_set, save=True)