import sys
import os
import dill
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from U_eval_metric import quadratic_weighted_kappa
from sklearn.metrics import classification_report, confusion_matrix
from U_constants import MODEL_DIR, DATA_DIR, TUNED_PARAMS_FILENAME, NUM_FOLD
from U_exception import CustomException 

def save_obj(file_path: str, obj):
    """Save the object to a file using dill library.

    Args:
        file_path (str): destination of the file
        obj (_type_): object to be saved
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)

def load_obj(file_path: str):
    """Load the pickle object from a file (models and preprocessors).

    Args:
        file_path (str): The file path to load the object from

    Returns:
        _type_: The object loaded from the file
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)
      
def create_if_missing_folder(path: str):
    """Create a folder if it does not exist.

    Args:
        path (str): The path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)
      
def get_filepath(fold: int):
    """Get the file paths for the preprocessor, train, test, and valid data for a specific fold.

    Args:
        fold (int): the specific fold number 

    Returns:
        str: The file paths for the preprocessor, train, test, and valid data for a specific fold
    """
    preprocessor_file_path=os.path.join(MODEL_DIR, f'preprocessor_{fold}.pkl')
    train_path=os.path.join(DATA_DIR, f'train_{fold}.csv')
    test_path=os.path.join(DATA_DIR, f'test_{fold}.csv')
    valid_path=os.path.join(DATA_DIR, f'valid_{fold}.csv')
    return preprocessor_file_path, train_path, test_path, valid_path
  
def get_preprocessor_path(fold: int): 
    """Get the preprocessor file path alone for a specific fold.

    Args:
        fold (int): the specific fold number

    Returns:
        str : The preprocessor file path
    """
    return os.path.join(MODEL_DIR, f'preprocessor_{fold}.pkl')

def get_model_path(fold: int):
    """Get the model file path for a specific fold.

    Args:
        fold (int): the specific fold number

    Returns:
        str: the model file path
    """
    return os.path.join(MODEL_DIR, f'lightgbm_model_{fold}.pkl')
  
def has_tuned_params():
    """Check in the model directory if the tuned parameters file is present.

    Returns:
        bool: True if the tuned parameters file is present, False otherwise
    """
    if TUNED_PARAMS_FILENAME in os.listdir(MODEL_DIR):
        return True
    return False
  
def has_models():
    """Check if the all models are present in the model directory for all the folds. 

    Returns:
        bool: True if all models are present, False otherwise 
    """
    for fold in range(NUM_FOLD):
        if f'lightgbm_model_{fold}.pkl' not in os.listdir(MODEL_DIR):
            return False
    return True
  
def has_inference_data():
    """Check in the data directory if the inference data file is present.

    Returns:
        bool: True if the tuned parameters file is present, False otherwise
    """
    if 'inference_data.csv' in os.listdir(DATA_DIR):
        return True
    return False

def get_evaluation_info(y_pred, y):
    """
    Print the classification report, confusion matrix heatmap, and the quadratic weighted kappa score.
    Save the confusion matrix heatmap as an image in model dir.
    """
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.savefig(os.path.join(MODEL_DIR, 'lightgbm_confusion_matrix.png'))
    print(f'Quadratic Weighted Kappa: {quadratic_weighted_kappa(y, y_pred)}')