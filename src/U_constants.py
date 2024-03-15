import os

NUM_FOLD = 6
NUM_TRIALS = 200
RANDOM_STATE = 20
NUM_SVD_COMPONENTS = 10
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PARENT_DIR, 'data')
MODEL_DIR = os.path.join(PARENT_DIR, 'saved_model')
IMAGE_DIR = os.path.join(PARENT_DIR, 'images')
ORIGINAL_DATA = os.path.join(DATA_DIR, 'pets_prepared.csv')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'data.csv')
ORIGINAL_VALID_PATH = os.path.join(DATA_DIR, 'valid.csv')
PREDICTED_DATA_PATH = os.path.join(DATA_DIR, 'predicted_data.csv')
INFERENCE_DATA_PATH = os.path.join(DATA_DIR, 'inference_data.csv')
TUNED_PARAMS_FILENAME = 'lightgbm_best_params.json'
TUNED_PARAMS_PATH = os.path.join(MODEL_DIR, TUNED_PARAMS_FILENAME)
TUNED_COEFS_PATH = os.path.join(MODEL_DIR, 'best_coefs.json')

TOP12_BREEDNAME = ['Mixed Breed', 'Domestic Short Hair', 'Domestic Medium Hair', 'Tabby',
                    'Domestic Long Hair', 'Siamese', 'Persian', 'Labrador Retriever',
                    'Shih Tzu', 'Poodle', 'Terrier', 'Golden Retriever']
NEEDED_COLUMNS = ['Type', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3', 'Fee', 'Quantity', 'VideoAmt', 'AgeBins',
                  'PhotoAmt', 'Age', 'BreedPure', 'Gender', 'MaturitySize', 'FurLength', 'Health', 'ColorAmt',
                  'Vaccinated', 'Dewormed', 'BreedName', 'Sterilized', 'StateBins', 'ColorName', 'Description',
                  'AdoptionSpeed']
NEEDED_COLUMNS_INFERENCE = ['Type', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3', 'Fee', 'Quantity', 'VideoAmt', 'AgeBins',
                  'PhotoAmt', 'Age', 'BreedPure', 'Gender', 'MaturitySize', 'FurLength', 'Health', 'ColorAmt',
                  'Vaccinated', 'Dewormed', 'BreedName', 'Sterilized', 'StateBins', 'ColorName', 'Description',
                  ]
TARGET = 'AdoptionSpeed'
TEXT_COLS = 'Description'
TEXT_FILL = 'MISSING'
NUM_COLS = ['Quantity', 'VideoAmt', 'PhotoAmt', 'ColorAmt', 'Fee', 'Age']
NORMAL_CAT_COLS_TO_PROCESS = ['BreedName', 'BreedPure', 'ColorName', 'StateBins'] 
NORMAL_CAT_COLS_NOT_TO_PROCESS = ['Type', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3',
                                'Gender', 'Vaccinated', 'Dewormed', 'Sterilized']
ORDINAL_CAT_COLS_TO_PROCESS = ['AgeBins']
ORDINAL_CAT_COLS_NOT_TO_PROCESS = ['MaturitySize', 'FurLength', 'Health']
NOT_ORDINAL_CAT_COLS = ['BreedName', 'BreedPure', 'ColorName', 'StateBins', 'Type', 'Breed1', 'Breed2', 'Color1', 
                                     'Color2', 'Color3', 'Gender', 'Vaccinated', 'Dewormed', 'Sterilized']
