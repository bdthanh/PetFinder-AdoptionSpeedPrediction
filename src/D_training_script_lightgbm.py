import warnings
warnings.filterwarnings('ignore')
import os
import re
import gc
import json 
import pandas as pd
import optuna
import numpy as np
from U_eval_metric import quadratic_weighted_kappa, OptimizedRounder
import lightgbm as lgb
from A_data_ingestion import DataIngestion
from C_data_transformation import DataTransformation
from U_logger import logging
from U_utils import save_obj, load_obj, get_filepath, get_evaluation_info, has_tuned_params, has_models, create_if_missing_folder, get_model_path, get_preprocessor_path
from U_constants import NOT_ORDINAL_CAT_COLS, MODEL_DIR, NUM_FOLD, NUM_TRIALS, TUNED_PARAMS_PATH, RANDOM_STATE, DATA_DIR, TUNED_COEFS_PATH
from scipy.stats import mode
from sklearn.metrics import classification_report

TARGET = 'AdoptionSpeed'

class LightGBMTrainer:
    def __init__(self) -> None:
        """
        Initialize the non-tune parameters for LightGBM model and the best_params variable to store the best parameters after tuning.
        """
        self.non_tune_params = {            
            'early_stopping_rounds': 10,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            'seed':RANDOM_STATE,
            'feature_fraction_seed': RANDOM_STATE,
            'bagging_fraction_seed': RANDOM_STATE,
            'data_random_seed': RANDOM_STATE,
            'extra_trees': True,
            'extra_seed': RANDOM_STATE,
            'zero_as_missing': True
        }
        self.best_params = None
        
    def get_data(self, fold: int):
        """Get the data for a fold. Separate the data into X_train, y_train, X_test, y_test, X_valid, y_valid

        Args:
            fold (int): The fold number corresponding to the data
        """
        _, train_path, test_path, valid_path = get_filepath(fold)
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        valid = pd.read_csv(valid_path)
        train = train.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        test = test.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        valid = valid.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        
        X_train, y_train = train.drop(TARGET, axis=1), train[TARGET]
        X_test, y_test = test.drop(TARGET, axis=1), test[TARGET]
        X_valid, y_valid = valid.drop(TARGET, axis=1), valid[TARGET]
        return X_train, y_train, X_test, y_test, X_valid, y_valid
        
    def objective(self, trial):
        """Optuna objective function to optimize hyperparameters for LightGBM model. 
           The function returns the quadratic weighted kappa score, the metric to maximize.
        """
        tune_params = {
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 1, log=True),
            "max_depth": trial.suggest_int("max_depth", 8, 10),
            "num_leaves": trial.suggest_int("num_leaves", 120, 400),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 55,10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.0, log=True)
        }
        for fold in range(NUM_FOLD):
            X_train, y_train, X_test, y_test, _, _ = self.get_data(fold)
            
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
            model = lgb.train(
                        {**self.non_tune_params, **tune_params},
                        train_set=lgb_train, 
                        valid_sets=[lgb_eval],
                        num_boost_round=5000,
                        valid_names=['valid'],
                        categorical_feature=NOT_ORDINAL_CAT_COLS
                      )
            logging.info(f"Training model with params: {tune_params}")
            y_pred = model.predict(X_train)
            optR = OptimizedRounder()
            optR.fit(y_pred, y_train)
            coefficients = optR.coefficients()
            print(  coefficients)            
            
            y_pred = model.predict(X_test)
            y_pred = optR.predict(y_pred, coefficients)
            
            print(f'Quadratic Weighted Kappa: {quadratic_weighted_kappa(y_test, y_pred)}')
            print(classification_report(y_test, y_pred))
            gc.collect()
        return quadratic_weighted_kappa(y_test, y_pred)
    
    def optimize_hyperparameters(self):
        """Optimize hyperparameters for LightGBM model using Optuna library. If the tuned params are found, no need to optimize.

        Returns:
            dict: The best parameters for LightGBM model
        """
        if has_tuned_params():
            with open(TUNED_PARAMS_PATH, 'r') as f:
                self.best_params = json.load(f)
            print('Found tuned params. No need to optimize hyperparameters')
            return self.best_params
          
        logging.info("Start optimizing hyperparameters for LightGBM 100 trials")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=NUM_TRIALS)
        self.best_params = {**self.non_tune_params, **study.best_params}
        with open(TUNED_PARAMS_PATH, 'w') as f:
            json.dump(self.best_params, f)
        logging.info(f"Best params for LightGBM: {self.best_params}")
        gc.collect()
        return self.best_params
    
    def train_model(self):
        """
        Train the LightGBM model with the best parameters found from optimization and save into pickle files. 
        If the models are found, no need to train the model.
        """
        logging.info("Training LightGBM model with best params")
        if has_models():
            print(f'Found all {NUM_FOLD} models. No need to train the model')
            return
        coefficients_list = []
        for fold in range(NUM_FOLD):
            print(f"Training model for fold {fold}")
            X_train, y_train, X_test, y_test, _, _ = self.get_data(fold)
            
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
            model = lgb.train(
                        self.best_params,
                        train_set=lgb_train, 
                        valid_sets=[lgb_eval],
                        num_boost_round=5000,
                        valid_names=['valid'],
                        categorical_feature=NOT_ORDINAL_CAT_COLS
                      )
            logging.info("Saving LightGBM model")
            save_obj(get_model_path(fold), model)
            y_pred = model.predict(X_train)
            optR = OptimizedRounder()
            optR.fit(y_pred, y_train)
            coefficients_list.append(list(optR.coefficients()))
        with open(TUNED_COEFS_PATH, 'w') as f:
            json.dump(coefficients_list, f)

    def evaluate_ensemble_model(self, y_preds, y_valid):   
        """Evaluate the ensemble model using majority voting
           There are 6 models, each is trained with a different fold. 
           The function uses mode to get the most common prediction from the 6 models and evaluate the result.

        Args:
            y_preds: List of predictions from 6 models
            y_valid: The true target variable
        """
        y_pred_ensemble, _ = mode(np.array(y_preds), axis=0)
        y_pred_ensemble = y_pred_ensemble.reshape(-1)
        get_evaluation_info(y_pred_ensemble, y_valid)
    
    def evaluate_model(self):
        """
        Evaluate the model for each fold and the ensemble model using majority voting.
        """
        y_preds = []
        print("Evaluating the model for each fold")
        with open(TUNED_COEFS_PATH, 'r') as f:
            coefs = json.load(f)
        for fold in range(NUM_FOLD):
            _, _, _, _, X_valid, y_valid = self.get_data(fold)
            model = load_obj(get_model_path(fold))
            y_pred = model.predict(X_valid)
            optR = OptimizedRounder()             
            y_pred = optR.predict(y_pred, coefs[fold])
            y_preds.append(y_pred)
            get_evaluation_info(y_pred, y_valid)
            gc.collect()
        print("Evaluating the ensemble model using majority voting")
        self.evaluate_ensemble_model(y_preds, y_valid)
        
if __name__ == "__main__":
    create_if_missing_folder(MODEL_DIR)
    create_if_missing_folder(DATA_DIR)
    data_ingestion = DataIngestion()
    train_sets, test_sets, valid_set = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_sets, test_sets, valid_set, save=True)
    trainer = LightGBMTrainer()
    trainer.optimize_hyperparameters()
    trainer.train_model()
    trainer.evaluate_model()
    