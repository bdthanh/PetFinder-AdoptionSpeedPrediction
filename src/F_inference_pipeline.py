import json
import pandas as pd 
import numpy as np
from scipy.stats import mode
from U_constants import NUM_FOLD, INFERENCE_DATA_PATH, NEEDED_COLUMNS_INFERENCE, TOP12_BREEDNAME, TEXT_COLS, TEXT_FILL, TARGET, PREDICTED_DATA_PATH, MODEL_DIR, DATA_DIR, TUNED_COEFS_PATH
from U_utils import get_preprocessor_path, get_model_path, load_obj, get_evaluation_info, create_if_missing_folder
from U_eval_metric import OptimizedRounder
from E_train_pipeline import TrainPipeline

class InferencePipeline: 
    """This class is the pipeline for inference."""
    def __init__(self) -> None:
        self.data_path = INFERENCE_DATA_PATH
        
    def load_data(self):
        self.data_all = pd.read_csv(self.data_path)
        self.has_target = False
        if TARGET in self.data_all.columns:
            self.y_real = self.data_all['AdoptionSpeed']
            self.has_target = True
        self.data = self.data_all[NEEDED_COLUMNS_INFERENCE]
        self.processed_data = []
    
    def has_target_col(self):
        """Check if data to be predicted has target column or not.
          If yes then later when running, the evaluation will be done by compare the target and prediction.

        Returns:
            bool: True if contain the target column, else False
        """
        return self.has_target
    
    def replace_limit(self, data, column, limit):
        """Replace the value of a column in the dataframe if it is greater than the limit"""
        data[column] = data[column].apply(lambda x: limit if x > limit else x)
        return data
            
    def preprocess_data(self):
        """
        Preprocess the data before predicting. This includes replacing the breed name with 'Other' if it is not in 
        the top 12 breed names, replacing the quantity, video amount, and photo amount with a limit, and filling the 
        missing text columns with a default value. Then it goes through the preprocessor for other columns.
        """
        self.data['BreedName'] = self.data['BreedName'].apply(lambda x: x if x in TOP12_BREEDNAME else 'Other')
        self.data = self.replace_limit(self.data, 'Quantity', 6)
        self.data = self.replace_limit(self.data, 'VideoAmt', 2)
        self.data = self.replace_limit(self.data, 'PhotoAmt', 10)
        self.data[TEXT_COLS].fillna(TEXT_FILL, inplace=True)
        for fold in range(NUM_FOLD):
            preprocessor_file_path = get_preprocessor_path(fold)
            preprocessor = load_obj(preprocessor_file_path)
            processed_data = preprocessor.transform(self.data)
            self.processed_data.append(processed_data)
    
    def save_predicted_df(self, y_pred_ensemble):
        """Combine the predicted columns with the original dataframe and save it to a csv file.
            Save the predicted dataframe to a csv file.

        Args:
            y_pred_ensemble: The predicted values (ensemble) from the models
        """
        pred_df = pd.DataFrame(y_pred_ensemble, columns=['AdoptionSpeed_pred'])
        combine_df = pd.concat([self.data_all, pred_df], axis=1)
        combine_df.to_csv(PREDICTED_DATA_PATH)
        
    def evaluate(self, y_pred):
        """Evaluate the model with QWK and classification report by comparing the predicted values with the real values if has target columns.

        Args:
            y_pred (_type_): _description_
        """
        if self.has_target_col():
            get_evaluation_info(y_pred, self.y_real)  
        else: 
            print('No target column found, cannot evaluate') 
            
    def predict(self):
        """Predict the target variable using the trained models. The prediction is done by majority voting.

        Returns:
            y_pred_ensemble: The predicted values (ensemble) from the models
        """
        self.load_data()
        self.preprocess_data()
        with open(TUNED_COEFS_PATH, 'r') as f:
            coefs = json.load(f)
        y_preds = []
        for fold in range(NUM_FOLD):
            model_path = get_model_path(fold)
            model = load_obj(model_path)
            y_pred = model.predict(self.processed_data[fold])
            optR = OptimizedRounder()             
            y_pred = optR.predict(y_pred, coefs[fold])
            y_preds.append(y_pred)
        y_pred_ensemble, _ = mode(np.array(y_preds), axis=0)
        y_pred_ensemble = y_pred_ensemble.reshape(-1)
        self.save_predicted_df(y_pred_ensemble)
        self.evaluate(y_pred_ensemble)
        return y_pred_ensemble
        
if __name__ == '__main__':
    create_if_missing_folder(MODEL_DIR)
    create_if_missing_folder(DATA_DIR)
    train_pipeline = TrainPipeline()
    train_pipeline.train()
    inference_pipeline = InferencePipeline()
    y_pred = inference_pipeline.predict()
        