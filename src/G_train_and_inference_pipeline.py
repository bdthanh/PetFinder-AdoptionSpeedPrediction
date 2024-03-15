import warnings
warnings.filterwarnings('ignore')
from U_utils import create_if_missing_folder
from U_constants import MODEL_DIR, DATA_DIR
from E_train_pipeline import TrainPipeline
from F_inference_pipeline import InferencePipeline

class TrainAndInferencePipeline:
    """This class is the pipeline for both training and inference."""
    def __init__(self) -> None:
        self.train_pipeline = TrainPipeline()
        self.inference_pipeline = InferencePipeline()
    
    def train_and_inference(self):
        """The main function to train the model and then predict the data."""
        
        print("TRAINING PIPELINE STARTS...")
        self.train_pipeline.train()
        print("TRAINING PIPELINE ENDS...")
        print("============================================================")
        print("INFERENCE PIPELINE STARTS...")
        self.inference_pipeline.predict()
        print("INFERENCE PIPELINE ENDS...")
        
if __name__ == '__main__':
    create_if_missing_folder(MODEL_DIR)
    create_if_missing_folder(DATA_DIR)
    train_and_inference_pipeline = TrainAndInferencePipeline()
    train_and_inference_pipeline.train_and_inference()