import time
from training_script_random_forest import RandomForestTrainer
from training_script_xgboost import XGBoostTrainer
from D_training_script_lightgbm import LightGBMTrainer
from training_script_logistic_regression import LogisticRegressionTrainer
from training_script_gaussian_nb import GaussianNBTrainer
from training_script_adaboost import AdaBoostTrainer
from training_script_catboost import CatBoostTrainer

def train():
    types = ['dog', 'cat']
    for type in types:
        lgbm = LightGBMTrainer(type)
        lgbm.train_model()
        lgbm.evaluate_model()

if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()
    print("Training Time: ", end_time-start_time)