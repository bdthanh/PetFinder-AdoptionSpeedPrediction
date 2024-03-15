import os
import json
import pandas as pd
import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from eval_metric import quadratic_weighted_kappa
from U_logger import logging
from U_utils import save_obj, load_obj
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score

TARGET = 'AdoptionSpeed'

class AdaBoostTrainer:
    def __init__(self, type: str) -> None:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(parent_dir, 'data')
        self.model_dir = os.path.join(parent_dir, 'saved_model')
        self.model_path = os.path.join(self.model_dir, "adaboost_model.pkl")
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
            "n_estimators": trial.suggest_int("n_estimators", 50, 600, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1, log=True),
            "random_state": 20
        }

        model = AdaBoostClassifier(**params)
        logging.info(f"Training model with params: {params}")

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        print(quadratic_weighted_kappa(self.y_test, y_pred))
        logging.info(f"recall_score: {f1}")
        print(classification_report(self.y_test, y_pred))
        return f1
    
    def optimize_hyperparameters(self):
        logging.info("Start optimizing hyperparameters for AdaBoost 100 trials")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=1)
        self.best_params = study.best_params
        logging.info(f"Best params for AdaBoost: {self.best_params}")
    
    def train_model(self):
        logging.info("Training AdaBoost model with best params")
        self.optimize_hyperparameters()
        with open(os.path.join(self.model_dir, 'adaboost_best_params.json'), 'w') as f:
            json.dump(self.best_params, f)
        final_model = AdaBoostClassifier(**self.best_params)
        final_model.fit(self.X_train, self.y_train)
        logging.info("Saving AdaBoost model")
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
        plt.savefig(os.path.join(self.model_dir, 'adaboost_confusion_matrix.png'))
      
if __name__ == "__main__":
    trainer = AdaBoostTrainer()
    trainer.train_model()
    trainer.evaluate_model()
  
    