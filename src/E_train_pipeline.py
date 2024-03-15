from A_data_ingestion import DataIngestion
from C_data_transformation import DataTransformation
from D_training_script_lightgbm import LightGBMTrainer
from U_utils import create_if_missing_folder
from U_constants import MODEL_DIR, DATA_DIR

class TrainPipeline: 
    """
    This class is the pipeline for training the model. 
    It consists of 3 main steps: data ingestion, data transformation, and model training.
    """
    def __init__(self) -> None:
        """
        Initialize the data ingestion, data transformation and the LightGBM trainer
        """
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.trainer = LightGBMTrainer()
    
    def train(self):
        """
        The main function to train the model. 
        It consists of 4 main steps: data ingestion, data transformation, hyperparameter optimization, model training, and model evaluation.
        """
        train_sets, test_sets, valid_set = self.data_ingestion.initiate_data_ingestion()
        self.data_transformation.initiate_data_transformation(train_sets, test_sets, valid_set, save=True)
        self.trainer.optimize_hyperparameters()
        self.trainer.train_model()
        self.trainer.evaluate_model()
        
if __name__ == '__main__':
    create_if_missing_folder(MODEL_DIR)
    create_if_missing_folder(DATA_DIR)
    train_pipeline = TrainPipeline()
    train_pipeline.train()
        