import os
import json
import numpy as np
import pandas as pd
import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from eval_metric import quadratic_weighted_kappa
from U_logger import logging
from U_utils import save_obj, load_obj
from sklearn.metrics import f1_score, classification_report, recall_score, confusion_matrix

TARGET = 'AdoptionSpeed'

class XGBoostTrainer:
    def __init__(self, type: str) -> None:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(parent_dir, 'data')
        self.model_dir = os.path.join(parent_dir, 'saved_model')
        self.model_path = os.path.join(self.model_dir, "xgboost_model.pkl")
        self.preprocessor_file_path=os.path.join(self.data_dir, f'preprocessor_{type}.pkl')
        self.train_path=os.path.join(self.data_dir, f'train_{type}.csv')
        self.test_path=os.path.join(self.data_dir, f'test_{type}.csv')
        self.valid_path=os.path.join(self.data_dir, f'valid_{type}.csv')
        
        self.best_params = None
        self.train = pd.read_csv(self.train_path)
        self.X_train = self.train.drop(TARGET, axis=1)
        self.y_train = self.train[TARGET]
        self.test = pd.read_csv(self.test_path)
        self.X_test = self.test.drop(TARGET, axis=1)
        self.y_test = self.test[TARGET]
    
    def objective(self, trial):
        params = {
            "objective": "multi:softmax",
            "metric": "mlogloss",
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 21, 2),
            "subsample": 0.9,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1, step=0.1),
            "n_estimators": 5000,
            "gamma": trial.suggest_float("gamma", 0.1, 10, log=True)
        }

        model = XGBClassifier(**params)
        logging.info(f"Training model with params: {params}")

        model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], early_stopping_rounds=10)
        y_pred = model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        print(quadratic_weighted_kappa(self.y_test, y_pred))
        logging.info(f"recall_score: {f1}")
        print(classification_report(self.y_test, y_pred))
        return f1
    
    def optimize_hyperparameters(self):
        logging.info("Start optimizing hyperparameters for XGBoost 120 trials")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=1)
        self.best_params = study.best_params
        logging.info(f"Best params for XGBoost: {self.best_params}")
        return study.best_params
    
    def train_model(self):
        logging.info("Training XGBoost model with best params")
        self.optimize_hyperparameters()
        with open(os.path.join(self.model_dir, 'xgboost_best_params.json'), 'w') as f:
            json.dump(self.best_params, f)
        final_model = XGBClassifier(**self.best_params)
        final_model.fit(self.X_train, self.y_train)
        logging.info("Saving XGBoost model")
        save_obj(self.model_path, final_model)
        return self.model_path
    
    def evaluate_model(self):
        self.model = load_obj(self.model_path)
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        cm = confusion_matrix(self.y_test, y_pred)
        class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.2)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=class_labels, 
                    yticklabels=class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix Heatmap")
        plt.savefig(os.path.join(self.model_dir, 'xgboost_confusion_matrix.png'))
    
if __name__ == "__main__":
    trainer = XGBoostTrainer()
    trainer.train_model()
    trainer.evaluate_model()